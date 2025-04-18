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

## 📊 Model Performance Analysis

This section explains each model used in the project, its working principles, and how well it performed based on evaluation metrics.

### 🔹 Logistic Regression

**About:**  
A linear model that predicts the probability of a binary outcome using the logistic function. It's easy to interpret and often used as a baseline model.

**Performance:**  
- Accuracy: **72%**  
- Precision: **30%**  
- Recall: **76%**  
- F1 Score: **43%**  
- AUC-ROC: **0.80**

**Analysis:**  
Logistic Regression achieved a high recall but extremely low precision. This means it correctly identified many churn cases but also flagged many non-churn customers incorrectly (false positives). It's a poor choice if precision is important (e.g., to avoid targeting the wrong customers), but its high recall could be useful when it's more important not to miss churners.

---

### 🔹 Decision Tree Classifier

**About:**  
A tree-based model that splits data based on feature values to make decisions. It’s easy to visualize and interpret.

**Performance:**  
- Accuracy: **87%**  
- Precision: **53%**  
- Recall: **74%**  
- F1 Score: **62%**  
- AUC-ROC: **0.84**

**Analysis:**  
The decision tree showed solid recall but mediocre precision. While it’s better than logistic regression in most aspects, it tends to overfit if not pruned or regularized. The model can be useful for initial understanding, but better generalization is needed for production.

---

### 🔹 Random Forest Classifier

**About:**  
An ensemble model that builds multiple decision trees and averages their outputs to improve generalization and reduce overfitting.

**Performance:**  
- Accuracy: **94%**  
- Precision: **82%**  
- Recall: **76%**  
- F1 Score: **79%**  
- AUC-ROC: **0.91**

**Analysis:**  
Random Forest outperformed all other models across nearly all metrics. It balances precision and recall well and provides excellent overall performance. This model is robust, handles feature interactions automatically, and is suitable for deployment in a real-world system.

---

### 🔹 Support Vector Machine (SVM)

**About:**  
SVM tries to find the best hyperplane that separates classes. It performs well on high-dimensional data and can use kernels to model non-linear decision boundaries.

**Performance:**  
- Accuracy: **88%**  
- Precision: **56%**  
- Recall: **63%**  
- F1 Score: **59%**  
- AUC-ROC: **0.86**

**Analysis:**  
SVM achieved decent accuracy but had lower precision and recall compared to Random Forest or Gradient Boosting. It’s computationally more intensive and requires careful feature scaling. Overall, it’s a fair performer but not the best for this dataset.

---

### 🔹 Naive Bayes

**About:**  
A probabilistic model based on Bayes' Theorem with the assumption of feature independence. It’s simple, fast, and works surprisingly well on some problems.

**Performance:**  
- Accuracy: **63%**  
- Precision: **25%**  
- Recall: **80%**  
- F1 Score: **38%**  
- AUC-ROC: **0.80**

**Analysis:**  
Naive Bayes had the highest recall after Gradient Boosting but extremely low precision and F1 score. It misclassified many non-churners as churners. Its naive assumptions limit its effectiveness in this scenario where feature relationships matter.

---

### 🔹 Gradient Boosting Classifier

**About:**  
An advanced ensemble technique that builds models sequentially, where each new model focuses on correcting the errors of the previous ones. It often yields top performance.

**Performance:**  
- Accuracy: **91%**  
- Precision: **63%**  
- Recall: **82%**  
- F1 Score: **71%**  
- AUC-ROC: **0.92**

**Analysis:**  
Gradient Boosting performed nearly as well as Random Forest, with better recall and AUC but slightly lower precision. It’s an excellent choice when you want to capture more churners and can tolerate a few false positives. With hyperparameter tuning, this model can potentially outperform all others.

---

## Model Performance Comparison

<img width="900" alt="model_performance" src="https://github.com/user-attachments/assets/6307d68b-c33d-4855-89ca-2cd2c747c56a" />


## Test Data on Random Forest Model

Random Forest outperformed all other models on the training set and was chosen for final evaluation on the test data. It generalized well and maintained high performance. The results show a healthy balance of conservatism (fewer false positives) and sensitivity (still catching many churners). This makes it a strong candidate for real-world deployment.

**Test Set Prediction Summary:**  
- The model was applied to the unseen test dataset (test.csv).  
- **86.3%** of customers were predicted **not to churn**.  
- **13.7%** were predicted **to churn**.


![test_result](https://github.com/user-attachments/assets/9181d82a-d2f7-479a-8669-e919d51217d0)

## 🛠️ How to Run

Clone this repo:
```bash
git clone https://github.com/ShahdTarek4/Predicting-Customer-Churn-in-Telecommunications.git
cd Predicting-Customer-Churn-in-Telecommunications
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

  
