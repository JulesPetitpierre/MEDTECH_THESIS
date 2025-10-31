import shap
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import numpy as np
import plotly.express as px
from shap import TreeExplainer

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MedTech M&A Failure Predictor",
    
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
# LOAD MODEL AND DATA
# ============================================================

df = pd.read_csv("ONLY_RELEVANT_M&A.csv")
df["Date Announced (dateann)"] = pd.to_datetime(df["Date Announced (dateann)"], errors="coerce")
df["ann_year"] = df["Date Announced (dateann)"].dt.year

# Temporal test split
test_df = df[df["ann_year"] > 2019].copy()

# Extract features and target
X_raw = test_df.drop(columns=["Deal Status (status)"], errors="ignore")
y = test_df["Deal Status (status)"]

# Load pipeline
pipeline = joblib.load("safe_pipeline_xgb.joblib")
preprocessor = pipeline.named_steps["preprocessor"]

# ============================================================
# HARD FIX: sanitize every column type for encoder safety
# ============================================================

st.write("🔍 Data cleaning before transformation...")

# Align with trained columns
expected_cols = preprocessor.feature_names_in_
X_raw = X_raw.reindex(columns=expected_cols, fill_value=np.nan)

# ============================================================
# FINAL SANITIZER: Force all columns into clean dtypes
# ============================================================

for col in X_raw.columns:
    if X_raw[col].dtype == "object":
        X_raw[col] = X_raw[col].astype(str).replace("nan", "Missing").replace("None", "Missing")
    elif pd.api.types.is_numeric_dtype(X_raw[col]):
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce").fillna(0)
    else:
        X_raw[col] = X_raw[col].astype(str).replace("nan", "Missing")

# Final failsafe: ensure full numeric fallback for unknowns
X_raw = X_raw.fillna("Missing")

try:
    X_preprocessed = preprocessor.transform(X_raw)
except Exception as e:
    # Dump diagnostic info to Streamlit
    st.error(f"⚠️ Preprocessor failed: {e}")
    nan_cols = [col for col in X_raw.columns if X_raw[col].isna().any()]
    st.write("Columns with NaNs:", nan_cols)
    mixed_cols = [col for col in X_raw.columns if X_raw[col].map(type).nunique() > 1]
    st.write("Columns with mixed types:", mixed_cols)
    st.stop()

# ============================================================
# PREDICTIONS
# ============================================================

test_df["predicted_failure_prob"] = pipeline.predict_proba(X_raw)[:, 1]
test_df["predicted_class"] = (test_df["predicted_failure_prob"] >= 0.60).astype(int)

# ============================================================
# SHAP EXPLAINER
# ============================================================

calibrated_clf = pipeline.named_steps["classifier"]
xgb_model = calibrated_clf.calibrated_classifiers_[0].estimator
explainer = TreeExplainer(xgb_model)
shap_values = explainer(X_preprocessed)

# ============================================================
# SUMMARY METRICS
# ============================================================

st.metric("Total Deals (Test)", len(test_df))
st.metric("Actual Failures", int(test_df["Deal Status (status)"].sum()))
st.metric("Predicted Failures (≥60%)", int(test_df["predicted_class"].sum()))
st.metric("Avg. Predicted Risk (%)", round(test_df["predicted_failure_prob"].mean() * 100, 2))

# ============================================================
# SHAP 3D VISUALIZATION
# ============================================================

st.subheader("3D SHAP Interaction Explorer")

with open("columns.json") as f:
    readable_names = json.load(f)
excluded = ["Unique Deal ID", "dateann", "Unique DEAL ID (master_deal_no)"]
readable_names = [n for n in readable_names if n not in excluded]

col3, col4, col5 = st.columns(3)
feature1 = col3.selectbox("X-axis", readable_names)
feature2 = col4.selectbox("Y-axis", readable_names)
feature3 = col5.selectbox("Z-axis (SHAP of)", readable_names)

try:
    idx_x = readable_names.index(feature1)
    idx_y = readable_names.index(feature2)
    idx_z = readable_names.index(feature3)

    def safe_col(matrix, idx):
        col = matrix[:, idx]
        return col.toarray().flatten() if hasattr(col, "toarray") else np.array(col).flatten()

    x_vals = safe_col(X_preprocessed, idx_x)
    y_vals = safe_col(X_preprocessed, idx_y)
    z_vals = shap_values[:, idx_z].values.flatten()

    fig = px.scatter_3d(
        x=x_vals, y=y_vals, z=z_vals,
        color=z_vals,
        labels={"x": feature1, "y": feature2, "z": f"SHAP: {feature3}"},
        opacity=0.7
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Could not generate SHAP plot: {e}")