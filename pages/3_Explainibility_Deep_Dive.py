import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from shap import TreeExplainer

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Explainability Deep Dive",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Explainability Deep Dive: SHAP-Based Insights")

st.markdown("""
This section enables you to explore **global and feature-level explanations** of deal failure predictions 
from the calibrated XGBoost model used in the thesis.  

All insights are based on **SHAP values**, which estimate how much each feature contributed to the predicted probability of failure.
""")

# ============================================================
# LOAD MODEL AND TEST DATA
# ============================================================

pipeline = joblib.load("safe_pipeline_xgb.joblib")
df = pd.read_csv("ONLY_RELEVANT_M&A.csv")

# Extract temporal test split: post-2019 only
df["Date Announced (dateann)"] = pd.to_datetime(df["Date Announced (dateann)"], errors="coerce")
df["ann_year"] = df["Date Announced (dateann)"].dt.year
test_df = df[df["ann_year"] > 2019].copy()

# Extract features
X_raw = test_df.drop(columns=["Deal Status (status)"], errors="ignore")
y = test_df["Deal Status (status)"]

# Transform
preprocessor = pipeline.named_steps["preprocessor"]
X_preprocessed = preprocessor.transform(X_raw)
feature_names = preprocessor.get_feature_names_out()

# ============================================================
# COMPUTE SHAP VALUES
# ============================================================

# Get the XGB model from inside the calibrated classifier
calibrated_clf = pipeline.named_steps["classifier"]
xgb_model = calibrated_clf.calibrated_classifiers_[0].estimator  # ✅ CORRECT

explainer = TreeExplainer(xgb_model)
shap_values = explainer(X_preprocessed)

# ============================================================
# GLOBAL FEATURE IMPORTANCE VISUALIZATION
# ============================================================

st.subheader("📊 Global SHAP Summary Plot (Customizable)")

col1, col2 = st.columns(2)
with col1:
    stat_choice = st.selectbox(
        "Select importance statistic",
        options=["Mean |SHAP|", "Median SHAP", "SHAP Std Dev"]
    )
with col2:
    top_n = st.selectbox("Number of top features to display", options=[10, 20])

# Compute importance vector
shap_matrix = shap_values.values
if stat_choice == "Median SHAP":
    importance = np.median(np.abs(shap_matrix), axis=0)
elif stat_choice == "SHAP Std Dev":
    importance = np.std(shap_matrix, axis=0)
else:
    importance = np.abs(shap_matrix).mean(axis=0)

# Sort and select top features
top_indices = np.argsort(importance)[-top_n:]
sorted_idx = top_indices[np.argsort(importance[top_indices])]

# Bar plot
fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
ax_bar.barh(
    [feature_names[i] for i in sorted_idx],
    importance[sorted_idx],
    color="teal"
)
ax_bar.set_xlabel(stat_choice)
ax_bar.set_title("Global SHAP Feature Importance")
st.pyplot(fig_bar)

# ============================================================
# FEATURE-LEVEL EXPLANATION: SHAP VALUE SCATTER
# ============================================================

st.subheader("🎯 SHAP Value Scatter Plot by Feature")

# Remove identifiers from dropdown
excluded = ["deal_no", "id", "master_deal_no"]
dropdown_features = [f for f in feature_names if not any(ex in f.lower() for ex in excluded)]

selected_feature = st.selectbox("Select a feature to explain", dropdown_features)

try:
    feature_idx = list(feature_names).index(selected_feature)
    feature_vals = X_preprocessed[:, feature_idx]

    if hasattr(feature_vals, "toarray"):
        feature_vals = feature_vals.toarray().flatten()
    else:
        feature_vals = np.array(feature_vals).flatten()

    shap_vals = shap_values[:, feature_idx].values

    # Plot
    st.markdown(
        f"### SHAP Scatter Plot for: <code style='color:limegreen'>{selected_feature}</code>",
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots()
    ax.scatter(feature_vals, shap_vals, alpha=0.5, color="mediumseagreen", edgecolor="black")
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("SHAP Value")
    ax.set_title("Impact on Predicted Deal Failure Probability")
    st.pyplot(fig)

except Exception as e:
    st.error(f"Could not generate SHAP scatter plot: {e}")