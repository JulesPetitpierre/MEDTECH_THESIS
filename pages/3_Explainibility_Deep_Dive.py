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
# LOAD MODEL AND DATA
# ============================================================

pipeline = joblib.load("safe_pipeline_xgb.joblib")
df = pd.read_csv("ONLY_RELEVANT_M&A.csv")

# Filter to test data post-2019
df["Date Announced (dateann)"] = pd.to_datetime(df["Date Announced (dateann)"], errors="coerce")
df["ann_year"] = df["Date Announced (dateann)"].dt.year
test_df = df[df["ann_year"] > 2019].copy()

# ============================================================
# DATA CLEANING BEFORE TRANSFORM
# ============================================================

X_raw = test_df.drop(columns=["Deal Status (status)"], errors="ignore").copy()
preprocessor = pipeline.named_steps["preprocessor"]

# Align to trained column order
expected_cols = preprocessor.feature_names_in_
X_raw = X_raw.reindex(columns=expected_cols, fill_value=np.nan)

# Fix types
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
    st.error(f"❌ Preprocessing failed: {e}")
    st.stop()

# Predict
test_df["predicted_failure_prob"] = pipeline.predict_proba(X_raw)[:, 1]
test_df["predicted_class"] = (test_df["predicted_failure_prob"] >= 0.60).astype(int)

# SHAP setup
calibrated_clf = pipeline.named_steps["classifier"]
xgb_model = calibrated_clf.calibrated_classifiers_[0].estimator
explainer = TreeExplainer(xgb_model)
shap_values = explainer(X_preprocessed)
feature_names = preprocessor.get_feature_names_out()

# ============================================================
# GLOBAL FEATURE IMPORTANCE
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

shap_matrix = shap_values.values
if stat_choice == "Median SHAP":
    importance = np.median(np.abs(shap_matrix), axis=0)
elif stat_choice == "SHAP Std Dev":
    importance = np.std(shap_matrix, axis=0)
else:
    importance = np.abs(shap_matrix).mean(axis=0)

top_indices = np.argsort(importance)[-top_n:]
sorted_idx = top_indices[np.argsort(importance[top_indices])]

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
# SHAP VALUE SCATTER PLOT
# ============================================================

st.subheader("🎯 SHAP Value Scatter Plot by Feature")

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