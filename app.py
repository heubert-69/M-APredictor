import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from io import BytesIO
from feature_engineering import EngineerFeatures
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import subprocess
import os

st.set_page_config(layout="wide", page_title="Acquisition Predictor")

# --- Load Model ---
@st.cache_resource
def load_trained_model():
    return load_model("models/AcquisitionPredictor.h5")

model = load_trained_model()

# --- Upload File ---
st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# --- TensorBoard ---
st.sidebar.header("TensorBoard Logs")
logdir = "logs"
if os.path.exists(logdir):
    subprocess.Popen(["tensorboard", "--logdir", logdir, "--port", "6006"])
    st.sidebar.markdown("[Open TensorBoard ↗️](http://localhost:6006)", unsafe_allow_html=True)

# --- PDF Download Function ---
def convert_df_to_pdf(df):
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("tight")
        ax.axis("off")
        table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        pdf.savefig(fig, bbox_inches="tight")
    buffer.seek(0)
    return buffer

# --- EDA Function ---
def show_eda(df):
    st.subheader("🧪 Exploratory Data Analysis")
    st.write("**Dataset Preview:**")
    st.dataframe(df.head())

    st.write("**Summary Statistics:**")
    st.dataframe(df.describe())

    st.write("**Missing Values:**")
    st.dataframe(df.isnull().sum()[df.isnull().sum() > 0])

    if "Acquired" in df.columns:
        st.write("**Target Distribution:**")
        fig, ax = plt.subplots()
        sns.countplot(x="Acquired", data=df, ax=ax)
        st.pyplot(fig)

# --- Main App ---
if uploaded_file:
    raw_data = pd.read_csv(uploaded_file)

    # Show EDA
    show_eda(raw_data)

    # Check if "Acquired" column is present
    has_target = "Acquired" in raw_data.columns
    if not has_target:
        raw_data["Acquired"] = 0  # Dummy placeholder for feature engineering

    # Feature Engineering
    processed_data = EngineerFeatures(raw_data)

    # Drop target before prediction
    if "Acquired" in processed_data.columns:
        processed_data = processed_data.drop(columns=["Acquired"])

    # --- Align input to model shape ---
    expected_input_shape = model.input_shape[-1]
    actual_input_shape = processed_data.shape[1]

    if actual_input_shape > expected_input_shape:
        processed_data = processed_data.iloc[:, :expected_input_shape]
    elif actual_input_shape < expected_input_shape:
        padding = np.zeros((processed_data.shape[0], expected_input_shape - actual_input_shape), dtype=np.float32)
        processed_data = np.hstack([processed_data.values, padding])
    else:
        processed_data = processed_data.values

    # --- Predict ---
    y_pred = model.predict(processed_data)
    y_pred_labels = (y_pred > 0.5).astype(int)

    # --- Output ---
    st.subheader("📊 Predictions")
    output_df = raw_data.copy()
    output_df["PredictedAcquired"] = y_pred_labels

    if has_target:
        st.markdown("### 📈 Metrics (for datasets with target)")
        y_true = raw_data["Acquired"]
        report = classification_report(y_true, y_pred_labels, output_dict=True)
        f1 = f1_score(y_true, y_pred_labels)
        cm = confusion_matrix(y_true, y_pred_labels)

        st.write("**F1 Score**:", round(f1, 4))
        st.write("**Classification Report:**")
        st.dataframe(pd.DataFrame(report).transpose())
        st.write("**Confusion Matrix:**")
        st.dataframe(pd.DataFrame(cm))

    st.write("### 🔍 Output Data with Predictions")
    st.dataframe(output_df)

    # --- Downloads ---
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=output_df.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )

    pdf_bytes = convert_df_to_pdf(output_df)
    st.download_button(
        label="📄 Download Predictions as PDF",
        data=pdf_bytes,
        file_name="predictions.pdf",
        mime="application/pdf"
    )
