import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MedTech M&A Failure Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SIDEBAR NAVIGATION HELP
# ============================================================

with st.sidebar:
    st.markdown("🏠 **Home** – View test set performance and confusion matrix.")
    st.markdown("🔬 **Explainability** – Explore SHAP insights globally and locally.")
    st.markdown("🧪 **Prediction** – Predict failure risk of any uploaded or selected deal.")

# ============================================================
# STYLING
# ============================================================

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Lato&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"] {
        font-family: 'Lato', sans-serif;
        background-color: #0B1A28;
        color: white;
    }
    footer {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

plt.style.use("dark_background")
sns.set_style("darkgrid")
sns.set_palette("dark")

st.title("🧬 MedTech M&A Failure Prediction Summary")

# ============================================================
# INTRODUCTION EXPANDER
# ============================================================

with st.expander("ℹ️ What is this app? Disclaimer & Context (Click to expand)"):
    st.markdown("""
    ### 🎓 Thesis Context and Academic Objective  
    This interactive application is part of the Bachelor’s thesis:  
    **“Exploring the Complex Landscape of MedTech M&A Setbacks Using Machine Learning”**  
    submitted at the **University of St. Gallen (HSG)**.  

    The tool translates a calibrated XGBoost classification model into a user-facing interface. It enables exploration of **predicted failure probabilities** for MedTech M&A transactions between 2014 and 2025. Failure is defined as deals that were announced but not completed (i.e., withdrawn or terminated).  

    The predictions are based on a calibrated XGBoost classifier trained using Scikit-learn and SHAP. All predictions and explanations shown here are **out-of-sample**, derived from the test data split (post-2019 deals).
    """)

# ============================================================
# LOAD MODEL AND DATA
# ============================================================

df = pd.read_csv("ONLY_RELEVANT_M&A.csv")
df["Date Announced (dateann)"] = pd.to_datetime(df["Date Announced (dateann)"], errors="coerce")
df["ann_year"] = df["Date Announced (dateann)"].dt.year
test_df = df[df["ann_year"] > 2019].copy()

X_raw = test_df.drop(columns=["Deal Status (status)"], errors="ignore")
y_true = test_df["Deal Status (status)"].astype(int)

pipeline = joblib.load("safe_pipeline_xgb_streamlit.joblib")
preprocessor = pipeline.named_steps["preprocessor"]

# ============================================================
# SANITIZE INPUT
# ============================================================

expected_cols = preprocessor.feature_names_in_
X_raw = X_raw.reindex(columns=expected_cols, fill_value=np.nan)

for col in X_raw.columns:
    if X_raw[col].dtype == "object":
        X_raw[col] = X_raw[col].astype(str).replace(["nan", "None"], "Missing")
    else:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce").fillna(0)

X_raw = X_raw.fillna("Missing")

try:
    X_preprocessed = preprocessor.transform(X_raw)
except Exception as e:
    st.error(f"⚠️ Preprocessor failed: {e}")
    st.stop()

# ============================================================
# THRESHOLD & PREDICTIONS
# ============================================================

st.sidebar.header("⚙️ Threshold Settings")
threshold = st.sidebar.slider("Select classification threshold", 0.0, 1.0, 0.352, 0.01)

proba = pipeline.predict_proba(X_raw)[:, 1]
y_pred = (proba >= threshold).astype(int)

# ============================================================
# METRICS
# ============================================================

st.subheader("📊 Prediction Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Deals (Test)", len(y_true))
col2.metric("Actual Failures", int(y_true.sum()))
col3.metric(f"Predicted Failures (≥{threshold:.2f})", int(y_pred.sum()))
col4.metric("Avg. Predicted Risk (%)", round(proba.mean() * 100, 1))

# ============================================================
# CONFUSION MATRIX
# ============================================================

st.subheader("📉 Confusion Matrix")

fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Completed", "Failed"],
            yticklabels=["Completed", "Failed"],
            ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix — Calibrated XGBoost")
st.pyplot(fig)