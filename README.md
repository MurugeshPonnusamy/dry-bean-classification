# Dry Bean Classification — ML Assignment 2

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

Add your GitHub repository link here after uploading the project:

`YOUR_GITHUB_REPOSITORY_LINK`

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
| Decision Tree | 0.8966 | 0.9363 | 0.8965 | 0.8966 | 0.8964 | 0.8750 |
| KNN | 0.9155 | 0.9811 | 0.9163 | 0.9155 | 0.9157 | 0.8978 |
| Naive Bayes | 0.7630 | 0.9644 | 0.7647 | 0.7630 | 0.7607 | 0.7143 |
| Random Forest | 0.9195 | 0.9921 | 0.9196 | 0.9195 | 0.9195 | 0.9026 |

### Model Performance Observations

**Logistic Regression:**  
Logistic Regression produced very strong results, with 91.92% accuracy and the highest AUC score of 0.9934. Its weighted F1 score was 0.9193 and MCC was 0.9023, showing strong multiclass classification performance.

**Decision Tree:**  
The Decision Tree achieved 89.66% accuracy and an F1 score of 0.8964. It performed reasonably well but was weaker than Logistic Regression, KNN and Random Forest. Its lower AUC and MCC also indicate that the single-tree model was less consistent than the strongest models.

**K-Nearest Neighbors:**  
KNN achieved 91.55% accuracy and an F1 score of 0.9157. After feature scaling, KNN performed competitively and was close to Logistic Regression and Random Forest, although its AUC and MCC were slightly lower.

**Naive Bayes:**  
Gaussian Naive Bayes had the weakest overall classification performance, with 76.30% accuracy, an F1 score of 0.7607 and MCC of 0.7143. Although its AUC remained relatively high at 0.9644, its predicted class labels were considerably less accurate than the other models.

**Random Forest:**  
Random Forest achieved the highest accuracy of 91.95%, the highest weighted F1 score of 0.9195 and the highest MCC of 0.9026. Its AUC of 0.9921 was also very strong and only slightly below Logistic Regression.

### Overall Winner

**Random Forest is selected as the overall winner.**

It achieved the highest Accuracy, weighted F1 score and MCC among the five models. Logistic Regression was extremely close and produced the highest AUC and marginally higher Precision, but Random Forest had the strongest overall balance across the main classification metrics.

## Streamlit Application

The Streamlit application provides:

- CSV test-data upload
- Model selection dropdown
- Evaluation metrics
- Comparison of all five models
- Confusion matrix
- Classification report

### Live Streamlit App Link

Add the deployed Streamlit Community Cloud URL here:

`YOUR_STREAMLIT_APP_LINK`

## Project Structure

```text
project-folder/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── Dry_Bean_ML_Assignment.ipynb
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl
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
