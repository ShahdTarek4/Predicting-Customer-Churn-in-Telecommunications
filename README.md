# Telecom Customer Churn Prediction 📉📱

This project leverages machine learning models to predict customer churn in the telecom industry. Identifying customers at risk of leaving allows telecom providers to take proactive retention actions and improve business performance.

## 🔍 Project Overview
Customer churn is a major issue in the telecom sector. Using historical customer data, this project builds and evaluates machine learning models to predict whether a customer is likely to leave the service (churn) or stay.

The workflow includes:

- Data preprocessing & exploration
- Feature engineering
- Model training & evaluation
- Churn prediction on new data

## 📁 Files Included
- `Telecom_churn_prediction.ipynb`: Main Jupyter Notebook with data analysis, model training, and evaluation.
- `train.csv`: dataset used to train model
- `test.csv`: dataset used to test model on unseen data

  ## 🧠 Models Used
The notebook explores and compares multiple models including:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- Naive Bayes
- Gradient Boosting

## 📊 Features Considered
Features typically used in churn prediction include:

- Demographics 
- Service subscriptions 
- Contract & billing details 
- Tenure and usage behavior

## 📈 Evaluation Metrics
The models are evaluated using:

- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve and AUC

## 🛠️ How to Run

Clone this repo:
```bash
git clone https://github.com/ShahdTarek4/telecom-churn-prediction.git
cd telecom-churn-prediction
```

Install required dependencies:
```bash
pip install -r requirements.txt
```

Launch the notebook:
```bash
jupyter notebook Telecom_churn_prediction.ipynb
```

Follow the cells to preprocess the data, train the models, and view predictions.

## References 
https://www.kaggle.com/competitions/customer-churn-prediction-2020

  
