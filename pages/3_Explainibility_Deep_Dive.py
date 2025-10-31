import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

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
Explore **feature-level and global explanations** of the calibrated XGBoost model used in this thesis.

Explanations are based on **SHAP values**, quantifying how each input feature influenced the predicted risk of deal failure.
""")

# ============================================================
# LOAD MODEL AND DATA
# ============================================================

pipeline = joblib.load("safe_pipeline_xgb.joblib")
df = pd.read_csv("ONLY_RELEVANT_M&A.csv")

# Use only post-2019 test data
df["Date Announced (dateann)"] = pd.to_datetime(df["Date Announced (dateann)"], errors="coerce")
df["ann_year"] = df["Date Announced (dateann)"].dt.year
test_df = df[df["ann_year"] > 2019].copy()

# Extract features and preprocess
X_raw = test_df.drop(columns=["Deal Status (status)"], errors="ignore")
expected_cols = pipeline.named_steps["preprocessor"].feature_names_in_
X_raw = X_raw.reindex(columns=expected_cols, fill_value=np.nan)

# Sanitize inputs
for col in X_raw.columns:
    if X_raw[col].dtype == "object":
        X_raw[col] = X_raw[col].astype(str).replace("nan", "Missing").replace("None", "Missing")
    elif pd.api.types.is_numeric_dtype(X_raw[col]):
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce").fillna(0)
    else:
        X_raw[col] = X_raw[col].astype(str).replace("nan", "Missing")

X_raw = X_raw.fillna("Missing")

try:
    X_preprocessed = pipeline.named_steps["preprocessor"].transform(X_raw)
except Exception as e:
    st.error(f"❌ Preprocessing failed: {e}")
    st.stop()

# Predict and explain
test_df["predicted_failure_prob"] = pipeline.predict_proba(X_raw)[:, 1]
test_df["predicted_class"] = (test_df["predicted_failure_prob"] >= 0.60).astype(int)

# SHAP Explainer setup
calibrated_clf = pipeline.named_steps["classifier"]
xgb_model = calibrated_clf.calibrated_classifiers_[0].estimator
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_preprocessed)
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

# Limit dropdown features to only a clean subset
displayable_features = [f for f in feature_names if (
    "Date Announced (dateann)" in f or
    "Target Nation (tnation)" in f or
    "Target Nation Code (tnationcode)" in f
)]

# ============================================================
# GLOBAL SHAP BAR CHART
# ============================================================

st.subheader("📊 Global SHAP Summary Plot")

col1, col2 = st.columns(2)
with col1:
    stat_choice = st.selectbox(
        "Importance metric",
        options=["Mean |SHAP|", "Median SHAP", "SHAP Std Dev"]
    )
with col2:
    top_n = st.selectbox("Top features to display", options=[10, 20, 30])

# Compute SHAP importance
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
ax_bar.set_title("SHAP Global Feature Importance")
st.pyplot(fig_bar)

# ============================================================
# FEATURE-SPECIFIC SCATTER PLOT
# ============================================================

st.subheader("🎯 Feature-Level SHAP Scatter")

selected_feature = st.selectbox("Choose a feature", displayable_features)

try:
    feature_idx = list(feature_names).index(selected_feature)
    feature_vals = X_preprocessed[:, feature_idx]

    if hasattr(feature_vals, "toarray"):
        feature_vals = feature_vals.toarray().flatten()
    else:
        feature_vals = np.array(feature_vals).flatten()

    shap_vals = shap_values[:, feature_idx].values

    fig, ax = plt.subplots()
    ax.scatter(feature_vals, shap_vals, alpha=0.5, color="deepskyblue", edgecolor="black")
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("SHAP Value")
    ax.set_title("SHAP Value vs Feature Value")
    st.pyplot(fig)

except Exception as e:
    st.error(f"⚠️ Could not render SHAP scatter plot: {e}")