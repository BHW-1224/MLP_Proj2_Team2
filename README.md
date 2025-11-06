# MLP_Proj2_Team2
By following the steps below, the results of each step can be reproduced.

## Step-1
To reproduce our baseline results, follow these steps:
1.	Open a new Kaggle Notebook.
2.	Upload this notebook file to the Kaggle environment.
3.	Click “Add input” on the right panel, search for “LLM Classification Finetuning”, and add the dataset.
4.	Make sure all dependencies (pandas, numpy, scikit-learn, spacy, tqdm) are available — these are preinstalled in Kaggle by default.
5.	Run all cells without modification.

## Step-2
To reproduce the SentenceTransformer-based baseline, follow these steps:
1.	Open a new Kaggle Notebook.
2.	Upload this notebook file to the Kaggle environment.
3.	Click “Add input” on the right panel, search for “LLM Classification Finetuning”, and add the dataset.
4.	Also add the pretrained model dataset: sentence-transformers/all-MiniLM-L6-v2 (https://www.kaggle.com/datasets/shinomoriaoshi/sentencetransformersallminilml6v2)
5.	Make sure all required libraries (pandas, numpy, scikit-learn, tqdm, sentence-transformers) are available — these are preinstalled in Kaggle by default.
6.	Run all cells without modification.

## Step-3
Train the model by running training_model.py in the Training_code_for_step3 folder (you can also use the vader-model.pkl file in the same directory).
The model will be saved in the same directory as my_trained_best_model. Upload these model files to Kaggle and assign that directory to the "SAVED_MODEL_PATH" value.
On Kaggle, use the "mlp-proj2-step3.ipynb" code from the Kaggle Notebooks folder. In the input settings, select LLM Classification Finetuning for the competition and debertav3large for the datasets. Upload the resulting files and the vader-model file in GitHub, and run it with the correct PATH to reproduce the model.
