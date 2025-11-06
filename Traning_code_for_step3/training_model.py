import torch
import gc
import os
import numpy as np
import pandas as pd
import joblib
from torch import nn
from torch.utils.data import Dataset
# ⭐️ [수정] 검증에 필요한 라이브러리 추가
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from sklearn.preprocessing import StandardScaler
from safetensors.torch import save_file

print("--- [1] 라이브러리 임포트 완료 ---")

# --- 저장될 경로 설정 ---
SAVE_DIRECTORY = "my_trained_best_model"
os.makedirs(SAVE_DIRECTORY, exist_ok=True)
print(f"모델이 저장될 경로: ./{SAVE_DIRECTORY}")


try:
    train = pd.read_csv("train.csv")
    print("train.csv 로드 완료.")
except FileNotFoundError:
    print("오류: train.csv 파일이 없습니다.")
    exit()

def encode_label(row):
    if row["winner_model_a"] == 1:
        return 0
    elif row["winner_model_b"] == 1:
        return 1
    else:
        return 2

def style_counts(text):
    text = str(text)
    ex = text.count("!")
    qm = text.count("?")
    co = text.count(",")
    return ex, qm, co

train["label"] = train.apply(encode_label, axis=1)

train_df, valid_df = train_test_split(
    train,
    test_size=0.2,
    random_state=42,
    stratify=train["label"]
)
print(f"--- [2] 데이터 분리 완료: 훈련 {len(train_df)}개 / 검증 {len(valid_df)}개 ---")


try:
    sia = joblib.load("vader_model.pkl")
    print("vader_model.pkl 로드 완료.")
except FileNotFoundError:
    print("오류: vader_model.pkl 파일이 없습니다.")
    exit()

# 훈련/검증 세트 모두에 대해 피처 엔지니어링 수행
def create_features(df):
    df["a_sentiment"] = df["response_a"].apply(lambda x: sia.polarity_scores(str(x))["compound"])
    df["b_sentiment"] = df["response_b"].apply(lambda x: sia.polarity_scores(str(x))["compound"])
    df["a_len"] = df["response_a"].astype(str).apply(len)
    df["b_len"] = df["response_b"].astype(str).apply(len)
    df["a_word_cnt"] = df["response_a"].astype(str).apply(lambda x: len(x.split()))
    df["b_word_cnt"] = df["response_b"].astype(str).apply(lambda x: len(x.split()))

    a_style = df["response_a"].apply(style_counts).apply(pd.Series)
    b_style = df["response_b"].apply(style_counts).apply(pd.Series)
    a_style.columns = ["a_exclam", "a_qmark", "a_commas"]
    b_style.columns = ["b_exclam", "b_qmark", "b_commas"]
    df = pd.concat([df, a_style, b_style], axis=1)

    for col in ["len", "word_cnt", "sentiment", "exclam", "qmark", "commas"]:
        df[f"diff_{col}"] = df[f"a_{col}"] - df[f"b_{col}"]
    return df

train_df = create_features(train_df.copy())
valid_df = create_features(valid_df.copy())

bias_feats = [c for c in train_df.columns if c.startswith("diff_")]
print(f"--- [3] 총 {len(bias_feats)}개의 'diff' 피처 생성 완료 ---")


scaler = StandardScaler()
# 1. 훈련(train) 데이터로 'fit'과 'transform'을 동시에 수행
train_bias_scaled = scaler.fit_transform(train_df[bias_feats].values)
# 2. 검증(valid) 데이터는 'transform'만 수행 (데이터 유출 방지)
valid_bias_scaled = scaler.transform(valid_df[bias_feats].values)

print("--- [4] StandardScaler 적용 완료 (Train: fit_transform, Valid: transform) ---")


class LLMDataset(Dataset):
    def __init__(self, df, tokenizer, scaled_bias_array, max_len=512):
        sep = tokenizer.sep_token or "</s>"
        self.texts = (df["prompt"] + f" {sep} " +
                      df["response_a"] + f" {sep} " +
                      df["response_b"]).tolist()
        self.labels = df["label"].tolist()
        self.bias_feats = torch.tensor(scaled_bias_array, dtype=torch.float) 
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["bias_features"] = self.bias_feats[idx] 
        encoding["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return encoding


BASE_MODEL_NAME = "microsoft/deberta-v3-large" 
print(f"--- [5] 인터넷에서 '{BASE_MODEL_NAME}' 모델 다운로드 시작 ---")

try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    backbone = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=3)
    print(f"'{BASE_MODEL_NAME}' 원본 모델 로드 완료.")
except OSError as e:
    print(f"오류: '{BASE_MODEL_NAME}' 모델 다운로드 실패. 인터넷 연결을 확인하세요.")
    exit()

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.1,
    bias="none", task_type="SEQ_CLS",
    target_modules=["query_proj", "key_proj", "value_proj", "dense"]
)
backbone = get_peft_model(backbone, lora_config)
print("LoRA (PEFT) 모델 설정 완료.")


class HybridClassifier(nn.Module):
    def __init__(self, backbone, bias_dim, num_labels=3):
        super().__init__()
        self.backbone = backbone
        underlying_model = self.backbone.base_model.model 
        classifier_in_features = underlying_model.classifier.in_features
        self.bias_proj = nn.Sequential(
            nn.Linear(bias_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Linear(classifier_in_features + 64, num_labels)
        underlying_model.classifier = nn.Identity() 

    def forward(self, input_ids=None, attention_mask=None, bias_features=None, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.logits 
        bias_vec = self.bias_proj(bias_features)
        fused = torch.cat([cls_emb, bias_vec], dim=1) 
        logits = self.classifier(fused) 
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}


class CustomTrainer(Trainer):
    def _wrap_model(self, model, training=True, dataloader=None):
        return model
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            bias_features=inputs.get("bias_features"),
            labels=inputs.get("labels"),
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss




def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    # Logits를 확률(probabilities)로 변환
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    # Log Loss 계산
    score = log_loss(labels, probs)
    return {"log_loss": score}


train_dataset = LLMDataset(train_df, tokenizer, train_bias_scaled, max_len=512)
valid_dataset = LLMDataset(valid_df, tokenizer, valid_bias_scaled, max_len=512)
print(f"--- [6] 훈련/검증 Torch 데이터셋 생성 완료 ---")

args = TrainingArguments(
    output_dir="./deberta-lora-validation-checkpoint",

    per_device_train_batch_size=4,    
    gradient_accumulation_steps=2,    
    num_train_epochs=3, 
    learning_rate=3e-5,
    logging_steps=50,
    fp16=True, 
    report_to="none",
    dataloader_drop_last=True,
    save_total_limit=2,
    
        
)


args = args.set_evaluate(strategy="epoch")


args = args.set_save(strategy="epoch")


args.load_best_model_at_end = True
args.metric_for_best_model = "log_loss"
args.greater_is_better = False

args.eval_delay = 0

def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "bias_features": torch.stack([b["bias_features"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch])
    }

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HybridClassifier(backbone, bias_dim=len(bias_feats)).to(device)

trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,  
    compute_metrics=compute_metrics, 
    data_collator=collate_fn,
)

print("\n--- [7] 모델 훈련 및 검증 시작 ---")
trainer.train()
print("--- [8] 모델 훈련 완료 ---")

print("\n--- [9] 최종 검증(Validation) 결과 (Best Model) ---")
# trainer.model이 'load_best_model_at_end=True'에 의해
# 이미 최고 성능 모델로 교체되었으므로, evaluate()는 최고 성능을 출력합니다.
eval_results = trainer.evaluate()
print("최고 성능 모델의 검증 결과:")
print(eval_results)

print(f"\n--- [10] 훈련된 'Best' 모델 및 스케일러 수동 저장 시작 ({SAVE_DIRECTORY}) ---")

# 1. PEFT 어댑터 저장 (trainer.model은 이미 'best' 모델임)
model.backbone.save_pretrained(SAVE_DIRECTORY)
print(f"✅ PEFT Adapter (Best) 저장 완료.")

# 2. 커스텀 헤드 가중치 저장 (trainer.model은 이미 'best' 모델임)
head_state_dict = {}
for k, v in model.state_dict().items():
    if not k.startswith("backbone."):
        head_state_dict[k] = v

head_path = os.path.join(SAVE_DIRECTORY, "model.safetensors")
save_file(head_state_dict, head_path)
print(f"커스텀 헤드 (Best) 저장 완료.")

# 3. 토크나이저 저장
tokenizer.save_pretrained(SAVE_DIRECTORY)
print(f"토크나이저 파일 저장 완료.")

# 4. 스케일러 저장 (⭐️ 중요: 'train_df'로 훈련된 'scaler' 객체 저장)
scaler_path = os.path.join(SAVE_DIRECTORY, "scaler.joblib")
joblib.dump(scaler, scaler_path)
print(f"스케일러(scaler.joblib) 파일 저장 완료.")

print(f"\n모든 파일이 '{SAVE_DIRECTORY}' 폴더에 올바르게 저장되었습니다.")

# ================================================================
# 11. 메모리 정리
# ================================================================
del model, backbone, trainer, train_dataset, valid_dataset, train, train_df, valid_df, scaler
torch.cuda.empty_cache()
gc.collect()

print("\n--- [11] 검증 및 저장 스크립트 종료 ---")