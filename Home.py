import shap
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
from shap import TreeExplainer

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

st.title("MedTech M&A Failure Prediction Summary")

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

    The underlying model was trained on financial, transactional, and macroeconomic variables extracted from a cleaned dataset of 630 fully disclosed MedTech acquisitions. The interface provides prediction results, local SHAP explanations, and global interpretability tools in a fully self-contained dashboard.

    ---

    ### 🧠 What This App Does  
    - Predicts the **ex-ante failure risk** of a MedTech M&A transaction  
    - Shows a **feature-level explanation** (SHAP) of the prediction  
    - Lets users interactively explore **why** certain deals are flagged as high- or low-risk  
    - Displays **global model behaviour**, feature importance, and model calibration  

    The predictions are based on a calibrated XGBoost classifier trained using Scikit-learn and SHAP. All predictions and explanations shown here are **out-of-sample**, derived from the test data split (post-2019 deals).

    ---

    ### ⚠️ Legal and Academic Disclaimer  
    This tool is intended for **academic demonstration only**.  
    - It is **not suitable** for use in commercial due diligence, investment decision-making, legal advisory, or financial forecasting.  
    - The model was trained on historical, publicly available MedTech data and is **not updated in real time**.  
    - Interpretability plots are approximations based on SHAP theory and should not be over-interpreted as causal evidence.  

    Please consult qualified professionals before using any predictive tool in real-world financial or strategic contexts.

    ---

    ### 🧬 Ready to Explore?  
    Now take your mouse and **explore some MedTech**.
    """)

# ============================================================
# LOAD MODEL AND DATA
# ============================================================

df = pd.read_csv("ONLY_RELEVANT_M&A.csv")
df["Date Announced (dateann)"] = pd.to_datetime(df["Date Announced (dateann)"], errors="coerce")
df["ann_year"] = df["Date Announced (dateann)"].dt.year
test_df = df[df["ann_year"] > 2019].copy()

# Extract features and target
X_raw = test_df.drop(columns=["Deal Status (status)"], errors="ignore")
y = test_df["Deal Status (status)"]

# Load pipeline
pipeline = joblib.load("safe_pipeline_xgb_streamlit.joblib")
preprocessor = pipeline.named_steps["preprocessor"]

# ============================================================
# SANITIZE INPUT
# ============================================================

st.write("🔍 Data cleaning before transformation...")

expected_cols = preprocessor.feature_names_in_
X_raw = X_raw.reindex(columns=expected_cols, fill_value=np.nan)

for col in X_raw.columns:
    if X_raw[col].dtype == "object":
        X_raw[col] = X_raw[col].astype(str).replace("nan", "Missing").replace("None", "Missing")
    elif pd.api.types.is_numeric_dtype(X_raw[col]):
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce").fillna(0)
    else:
        X_raw[col] = X_raw[col].astype(str).replace("nan", "Missing")

X_raw = X_raw.fillna("Missing")

try:
    X_preprocessed = preprocessor.transform(X_raw)
except Exception as e:
    st.error(f"⚠️ Preprocessor failed: {e}")
    st.stop()

# ============================================================
# PREDICTIONS
# ============================================================

test_df["predicted_failure_prob"] = pipeline.predict_proba(X_raw)[:, 1]
test_df["predicted_class"] = (test_df["predicted_failure_prob"] >= 0.60).astype(int)

# ============================================================
# METRICS
# ============================================================

st.metric("Total Deals (Test)", len(test_df))
st.metric("Actual Failures", int(y.sum()))
st.metric("Predicted Failures (≥60%)", int(test_df["predicted_class"].sum()))
st.metric("Avg. Predicted Risk (%)", round(test_df["predicted_failure_prob"].mean() * 100, 2))

# ============================================================
# VISUALIZATIONS
# ============================================================

st.subheader("🔍 Reliability, Distribution, and Confusion Matrix")

# 1. Reliability Curve
fig1, ax1 = plt.subplots()
true_prob, pred_prob = calibration_curve(y, test_df["predicted_failure_prob"], n_bins=10)
ax1.plot(pred_prob, true_prob, "o-", label="XGBoost Calibrated")
ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
ax1.set_xlabel("Predicted Probability")
ax1.set_ylabel("Observed Frequency")
ax1.set_title("Calibration Curve")
ax1.legend()
st.pyplot(fig1)

# 2. Histogram of predicted probabilities
fig2, ax2 = plt.subplots()
sns.histplot(test_df, x="predicted_failure_prob", hue="Deal Status (status)", bins=20, ax=ax2, palette="coolwarm", element="step", stat="count", common_norm=False)
ax2.set_title("Prediction Probability Histogram")
ax2.set_xlabel("Predicted Probability of Failure")
ax2.set_ylabel("Count")
st.pyplot(fig2)

# 3. Confusion Matrix
fig3, ax3 = plt.subplots()
cm = confusion_matrix(y, test_df["predicted_class"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Completed", "Failed"], yticklabels=["Completed", "Failed"], ax=ax3)
ax3.set_xlabel("Predicted label")
ax3.set_ylabel("True label")
ax3.set_title("Confusion Matrix")
st.pyplot(fig3)