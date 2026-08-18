# Dry Bean Classification - ML Assignment 2

## a. Problem Statement

The objective of this project is to classify dry beans into one of seven bean varieties using physical and shape-based measurements extracted from bean images.

The target variable is `Class`, and the classification problem contains seven categories:

- BARBUNYA
- BOMBAY
- CALI
- DERMASON
- HOROZ
- SEKER
- SIRA

Five machine learning classification algorithms are implemented and compared using the same dataset.

## b. Dataset Description

Dataset: **Dry Bean Dataset**

The dataset contains 16 numerical input features that describe geometric and shape properties of dry beans. The target column is `Class`.

The project uses an 80/20 stratified train/test split with `random_state=42`.

Exact duplicate rows are removed before model training.

## c. GitHub Repository Link

https://github.com/MurugeshPonnusamy/dry-bean-classification/

## d. Models Used

The following five classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

Logistic Regression and KNN use StandardScaler because they are sensitive to differences in feature scale.

### Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9192 | 0.9934 | 0.9197 | 0.9192 | 0.9193 | 0.9023 |
| Decision Tree | 0.8955 | 0.9357 | 0.8954 | 0.8955 | 0.8953 | 0.8737 |
| KNN | 0.9155 | 0.9811 | 0.9163 | 0.9155 | 0.9157 | 0.8978 |
| Naive Bayes | 0.7630 | 0.9644 | 0.7647 | 0.7630 | 0.7607 | 0.7143 |
| Random Forest | 0.9192 | 0.9919 | 0.9192 | 0.9192 | 0.9191 | 0.9022 |

### Model Performance Observations

**Logistic Regression:**  
Logistic Regression produced the strongest overall results. It achieved 91.92% accuracy, the highest AUC score of 0.9934, the highest precision of 0.9197, the highest weighted F1 score of 0.9193, and the highest MCC of 0.9023. These results indicate strong and well-balanced multiclass classification performance.

**Decision Tree:**  
The Decision Tree achieved 89.55% accuracy and an F1 score of 0.8953. It performed reasonably well but was weaker than Logistic Regression, KNN, and Random Forest. Its AUC of 0.9357 and MCC of 0.8737 were also lower than those of the strongest models.

**K-Nearest Neighbors:**  
KNN achieved 91.55% accuracy and an F1 score of 0.9157. After feature scaling, KNN performed competitively and remained close to Logistic Regression and Random Forest, although its AUC and MCC were slightly lower.

**Naive Bayes:**  
Gaussian Naive Bayes had the weakest overall classification performance, with 76.30% accuracy, an F1 score of 0.7607, and MCC of 0.7143. Although its AUC remained relatively high at 0.9644, its predicted class labels were considerably less accurate than those of the other models.

**Random Forest:**  
Random Forest also performed very strongly, achieving 91.92% accuracy, an AUC of 0.9919, an F1 score of 0.9191, and an MCC of 0.9022. Its accuracy and recall matched Logistic Regression, but Logistic Regression was slightly better in AUC, precision, F1, and MCC.

### Overall Winner

**Logistic Regression is selected as the overall winner.**

It achieved the best overall balance across the evaluation metrics, including the highest AUC, precision, weighted F1 score, and MCC, while tying Random Forest for the highest accuracy and recall. Random Forest was extremely close, but Logistic Regression was marginally stronger across the complete set of metrics.

## Streamlit Application

The Streamlit application provides:

- CSV test-data upload
- Model selection dropdown
- Evaluation metrics
- Comparison of all five models
- Confusion matrix
- Classification report

### Live Streamlit App Link

https://dry-bean-classification-murugeshp.streamlit.app/

## Project Structure

```text
project-folder/
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- Dry_Bean_ML_Assignment.ipynb
`-- model/
    |-- logistic_regression.pkl
    |-- decision_tree.pkl
    |-- knn.pkl
    |-- naive_bayes.pkl
    `-- random_forest.pkl
```

## Running the App Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app.py
```
