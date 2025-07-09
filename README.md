# Hospital Readmission Prediction

A real-world machine learning project to predict whether patients will be readmitted to the hospital within 30 days of discharge. Built using Python, pandas, scikit-learn, and real healthcare data involving 30,000+ hospital records.

---

## Problem Statement

Hospital readmissions within 30 days are costly and often preventable. This project builds a predictive model to help healthcare providers identify high-risk patients at the time of discharge and reduce penalties associated with avoidable readmissions.

---

## Machine Learning Pipeline

- **Data Cleaning & Feature Engineering**
  - Converted `Yes/No` to binary
  - Extracted `systolic/diastolic` BP
  - One-hot encoded categorical variables
- **EDA (Exploratory Data Analysis)**
  - Readmission trends by age, cholesterol, BMI, conditions
  - Correlation heatmaps
- **Modeling**
  - Logistic Regression
  - Random Forest Classifier
- **Evaluation**
  - Accuracy, Recall, ROC AUC, Confusion Matrices
  - Feature Importances
- **Visualization**
  - ROC Curves, Heatmaps, Feature Impact Charts

---

## Model Results

| Metric            | Logistic Regression | Random Forest |
|-------------------|---------------------|----------------|
| **ROC AUC**       | ~0.81               | **~0.87** ✅    |
| **Recall (class 1)** | ~72%             | **~83%** ✅    |
| **Interpretability** | High             | Moderate       |

> Random Forest performed best, achieving ~87% ROC AUC and identifying high-risk patients more effectively.

---

## Key Insights

- Patients with **longer hospital stays**, **higher cholesterol**, or discharged to **nursing facilities** had significantly higher readmission rates.
- **Diabetes** and **hypertension** jointly increased readmission probability.
- **Top Predictors**: length of stay, cholesterol, systolic BP, BMI, and discharge destination.

---

## Dataset

- **Name**: `hospital_readmissions_30k.csv`
- **Size**: ~30,000 rows
- **Features**: Age, Gender, BP, Cholesterol, Comorbidities, Discharge Type, Readmission status
- **Source**: Simulated real-world dataset (structured similar to UCI/Kaggle hospital data)

---

## Technologies Used

- Python, Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib
- Google Colab (Jupyter Notebook)
