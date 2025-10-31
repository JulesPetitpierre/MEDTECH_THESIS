import shap
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import plotly.express as px
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
# INTRODUCTION SECTION
# ============================================================

with st.expander("Welcome. Click to read the introduction"):
    st.markdown("""
    This application represents an **applied summarizing tool of my tuned and calibrated XGBoost model** developed as part of my Bachelor's thesis:

    **"Exploring the Complex Landscape of MedTech M&A Setbacks Using Machine Learning"**

    ---
    **Defining Failure in This Context**  
    Failure refers to **withdrawn or terminated transactions** — deals that are publicly announced but not completed. These events often result from misaligned expectations, due diligence issues, or regulatory constraints.

    ---
    **Purpose of This Tool**  
    - Generate a **predicted probability** of deal failure  
    - Understand **feature-level contributions** via SHAP values  
    - Conduct **interactive risk analysis** across transactions  

    The tool translates the thesis results into an interpretable, real-world application bridging academic insight and managerial decision-making.
    """)

# ============================================================
# LOAD MODEL AND TEST DATA ONLY
# ============================================================

# Load dataset and extract year
df = pd.read_csv("ONLY_RELEVANT_M&A.csv")
df["Date Announced (dateann)"] = pd.to_datetime(df["Date Announced (dateann)"], errors="coerce")
df["ann_year"] = df["Date Announced (dateann)"].dt.year

# Keep only post-2019 (test period)
test_df = df[df["ann_year"] > 2019].copy()

# Extract target and features
y = test_df["Deal Status (status)"].astype(int)
X_raw = test_df.drop(columns=["Deal Status (status)"], errors="ignore")

# Defensive fix: ensure categorical safety
for col in X_raw.select_dtypes(include="object").columns:
    X_raw[col] = X_raw[col].fillna("MISSING").astype(str)

# Load model and components
pipeline = joblib.load("safe_pipeline_xgb.joblib")
preprocessor = pipeline.named_steps["preprocessor"]
calibrated_clf = pipeline.named_steps["classifier"]
xgb_model = calibrated_clf.calibrated_classifiers_[0].estimator

# Predict using pipeline (no manual transform)
test_df["predicted_failure_prob"] = pipeline.predict_proba(X_raw)[:, 1]
test_df["predicted_class"] = (test_df["predicted_failure_prob"] >= 0.60).astype(int)

# Load readable feature names
with open("columns.json") as f:
    readable_names = json.load(f)
excluded = ["Unique Deal ID", "dateann", "Unique DEAL ID (master_deal_no)"]
readable_names = [name for name in readable_names if name not in excluded]

# ============================================================
# SHAP EXPLAINER
# ============================================================

# Preprocess once for SHAP
X_preprocessed = preprocessor.transform(X_raw)
feature_names = preprocessor.get_feature_names_out()

explainer = TreeExplainer(xgb_model)
shap_values = explainer(X_preprocessed)

# ============================================================
# SIDEBAR FILTER
# ============================================================

st.sidebar.header("Choose a Specific Country")
countries = sorted(test_df["Target Nation (tnation)"].dropna().unique())
selected_country = st.sidebar.selectbox("Filter by Target Country", options=[None] + countries)
filtered_df = test_df if selected_country is None else test_df[test_df["Target Nation (tnation)"] == selected_country]

# ============================================================
# SUMMARY METRICS + 3D SHAP VISUALIZATION
# ============================================================

st.markdown("### Summary Statistics and Interactive SHAP Visualization")
col1, col2 = st.columns([1, 1.5])

# --- Left Column: Metrics ---
with col1:
    st.metric("Total Deals (Test Set)", len(filtered_df))
    st.metric("Actual Failures", int(filtered_df["Deal Status (status)"].sum()))
    st.metric("Predicted Failures (≥60%)", int(filtered_df["predicted_class"].sum()))
    st.metric("Avg. Predicted Risk (%)", round(filtered_df["predicted_failure_prob"].mean() * 100, 2))
    st.metric("Deals >75% Risk", int((filtered_df["predicted_failure_prob"] > 0.75).sum()))
    st.metric("Deals <25% Risk", int((filtered_df["predicted_failure_prob"] < 0.25).sum()))

# --- Right Column: SHAP 3D ---
with col2:
    st.markdown("#### 3D SHAP Feature Interaction Explorer")

    col3, col4, col5 = st.columns(3)
    with col3:
        feature1 = st.selectbox("X-axis", readable_names, key="shap3d_x")
    with col4:
        feature2 = st.selectbox("Y-axis", readable_names, key="shap3d_y")
    with col5:
        shap_target = st.selectbox("Z-axis (SHAP of)", readable_names, key="shap3d_z")

    try:
        idx_x = readable_names.index(feature1)
        idx_y = readable_names.index(feature2)
        idx_z = readable_names.index(shap_target)

        # Safe column extraction
        def safe_column(matrix, idx):
            col = matrix[:, idx]
            return col.toarray().flatten() if hasattr(col, "toarray") else np.array(col).flatten()

        x_vals = safe_column(X_preprocessed, idx_x)
        y_vals = safe_column(X_preprocessed, idx_y)
        z_vals = shap_values[:, idx_z].values.flatten()

        fig = px.scatter_3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            color=z_vals,
            labels={"x": feature1, "y": feature2, "z": f"SHAP: {shap_target}"},
            opacity=0.7
        )
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not generate SHAP plot: {e}")