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
This section enables you to explore **global and feature-level explanations** of deal failure predictions 
from the calibrated XGBoost model used in the thesis.  

All insights are based on **SHAP values**, which estimate how much each feature contributed to the predicted probability of failure.
""")

# ============================================================
# LOAD MODEL AND DATA
# ============================================================

pipeline = joblib.load("safe_pipeline_xgb.joblib")
df = pd.read_csv("ONLY_RELEVANT_M&A.csv")

# Ensure date features exist for transformation
df["Date Announced (dateann)"] = pd.to_datetime(df["Date Announced (dateann)"], errors="coerce")
df["ann_year"] = df["Date Announced (dateann)"].dt.year
df["ann_month"] = df["Date Announced (dateann)"].dt.month
df = df.drop(columns=["Date Announced (dateann)"], errors="ignore")

# Extract X for transformation
X_raw = df.drop(columns=["Deal Status (status)"], errors="ignore")

# Sanitize input to match training
expected_cols = pipeline.named_steps["preprocessor"].feature_names_in_
X_raw = X_raw.reindex(columns=expected_cols, fill_value=np.nan)

for col in X_raw.columns:
    if X_raw[col].dtype == "object":
        X_raw[col] = X_raw[col].astype(str).replace("nan", "Missing")
    elif pd.api.types.is_numeric_dtype(X_raw[col]):
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce").fillna(0)
    else:
        X_raw[col] = X_raw[col].astype(str).replace("nan", "Missing")

try:
    X_preprocessed = pipeline.named_steps["preprocessor"].transform(X_raw)
except Exception as e:
    st.error(f"❌ Preprocessing failed: {e}")
    st.stop()

# ============================================================
# COMPUTE SHAP VALUES
# ============================================================

calibrated_clf = pipeline.named_steps["classifier"]
xgb_model = calibrated_clf.calibrated_classifiers_[0].estimator
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_preprocessed)

# ============================================================
# LIMIT TO TOP 68 FEATURES (mean |SHAP|)
# ============================================================

shap_matrix = shap_values.values
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
top_k = 68
top_idx = np.argsort(mean_abs_shap)[-top_k:]
limited_features = feature_names[top_idx]

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
    top_n = st.selectbox("Number of top features to display", options=[10, 20, 30, 50, 68], index=4)

# Compute importance vector
if stat_choice == "Median SHAP":
    importance = np.median(np.abs(shap_matrix), axis=0)
elif stat_choice == "SHAP Std Dev":
    importance = np.std(shap_matrix, axis=0)
else:
    importance = mean_abs_shap

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

selected_feature = st.selectbox("Select a feature to explain", sorted(limited_features))

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