import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🫘",
    layout="wide"
)

st.title("Dry Bean Classification using Machine Learning")
st.write(
    "This application compares five classification models on the Dry Bean dataset. "
    "Upload a labelled test CSV or use the bundled `test_data.csv` file."
)

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
}

REQUIRED_TARGET = "Class"


@st.cache_resource
def load_models():
    loaded = {}
    missing = []
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            loaded[name] = joblib.load(path)
        else:
            missing.append(path)
    return loaded, missing


def calculate_metrics(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    auc = roc_auc_score(
        y,
        y_prob,
        multi_class="ovr",
        average="weighted",
        labels=model.classes_,
    )

    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(
            y, y_pred, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y, y_pred, average="weighted", zero_division=0
        ),
        "F1": f1_score(
            y, y_pred, average="weighted", zero_division=0
        ),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return metrics, y_pred


models, missing_model_files = load_models()

if missing_model_files:
    st.error(
        "Some saved model files are missing. Make sure the `model` folder is "
        "uploaded to GitHub with all five `.pkl` files."
    )
    st.code("\n".join(missing_model_files))
    st.stop()

st.sidebar.header("Test Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload test data (CSV)",
    type=["csv"],
    help="The CSV should contain the 16 input features and the true `Class` column."
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data_source = "Uploaded CSV"
elif os.path.exists("test_data.csv"):
    data = pd.read_csv("test_data.csv")
    data_source = "Bundled test_data.csv"
else:
    st.warning(
        "Upload a labelled CSV file. The repository does not currently contain "
        "`test_data.csv`."
    )
    st.stop()

st.success(f"Using: {data_source}")

if REQUIRED_TARGET not in data.columns:
    st.error(
        "The test CSV must contain a `Class` column so that evaluation metrics "
        "and the confusion matrix can be calculated."
    )
    st.stop()

X_test = data.drop(columns=[REQUIRED_TARGET])
y_test = data[REQUIRED_TARGET]

st.subheader("Test Dataset")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(data))
c2.metric("Input Features", X_test.shape[1])
c3.metric("Classes Present", y_test.nunique())

with st.expander("Preview test data"):
    st.dataframe(data.head(20), use_container_width=True)

st.subheader("Compare All Models")

comparison_rows = []
prediction_cache = {}

for model_name, model in models.items():
    try:
        metrics, predictions = calculate_metrics(model, X_test, y_test)
        prediction_cache[model_name] = predictions

        comparison_rows.append({
            "ML Model Name": model_name,
            **{key: round(value, 4) for key, value in metrics.items()}
        })
    except Exception as exc:
        st.error(f"Could not evaluate {model_name}: {exc}")

comparison_df = pd.DataFrame(comparison_rows)

if not comparison_df.empty:
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    winner = comparison_df.loc[
        comparison_df["F1"].idxmax(), "ML Model Name"
    ]
    st.info(f"Overall winner based on weighted F1 Score: **{winner}**")

st.subheader("Detailed Model Evaluation")

selected_model_name = st.selectbox(
    "Select a model",
    list(models.keys())
)

selected_model = models[selected_model_name]
selected_metrics, y_pred = calculate_metrics(
    selected_model, X_test, y_test
)

m1, m2, m3 = st.columns(3)
m4, m5, m6 = st.columns(3)

m1.metric("Accuracy", f"{selected_metrics['Accuracy']:.4f}")
m2.metric("AUC", f"{selected_metrics['AUC']:.4f}")
m3.metric("Precision", f"{selected_metrics['Precision']:.4f}")
m4.metric("Recall", f"{selected_metrics['Recall']:.4f}")
m5.metric("F1 Score", f"{selected_metrics['F1']:.4f}")
m6.metric("MCC", f"{selected_metrics['MCC']:.4f}")

st.subheader(f"Confusion Matrix — {selected_model_name}")

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    xticks_rotation=45,
    cmap="Blues",
    ax=ax,
)
ax.set_title(f"Confusion Matrix - {selected_model_name}")
plt.tight_layout()
st.pyplot(fig)

st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.round(4), use_container_width=True)

st.caption(
    "Multiclass Precision, Recall and F1 are calculated using weighted averaging. "
    "AUC is calculated using One-vs-Rest (OvR) with weighted averaging."
)
