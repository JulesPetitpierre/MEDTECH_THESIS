import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Setup
st.set_page_config(page_title="Explainability Deep Dive", layout="wide")
st.title("Explainability Deep Dive")

# ================================
# Load model, preprocessor, and data
# ================================
model = joblib.load("calibrated_estimator_xgboost.joblib")
preprocessor = joblib.load("preprocessor_for_xgboost.joblib")
df = pd.read_csv("ONLY_RELEVANT_M&A.csv")

# Prepare features
X_raw = df.drop(columns=["Deal Status (status)"], errors="ignore")
X_preprocessed = preprocessor.transform(X_raw)
feature_names = preprocessor.get_feature_names_out()

# ================================
# SHAP explainer and values
# ================================
explainer = shap.Explainer(model, feature_names=feature_names)
shap_values = explainer(X_preprocessed)

# ================================
# Global SHAP Summary Plot (Customizable)
# ================================
st.subheader("🔍 Global SHAP Summary Plot (Customizable)")

col1, col2 = st.columns(2)
with col1:
    stat_choice = st.selectbox(
        "Select SHAP statistic",
        options=["Mean |SHAP|", "Median SHAP", "SHAP Std Dev"]
    )
with col2:
    top_n = st.selectbox("Number of top features to display", options=[10, 20])

# Compute global importance vector
shap_matrix = shap_values.values
if stat_choice == "Median SHAP":
    importance = np.median(np.abs(shap_matrix), axis=0)
elif stat_choice == "SHAP Std Dev":
    importance = np.std(shap_matrix, axis=0)
else:  # Mean |SHAP|
    importance = np.abs(shap_matrix).mean(axis=0)

# Sort and slice top features
top_indices = np.argsort(importance)[-top_n:]
sorted_idx = top_indices[np.argsort(importance[top_indices])]

# Plot horizontal bar chart
fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
ax_bar.barh(
    [feature_names[i] for i in sorted_idx],
    importance[sorted_idx],
    color="teal"
)
ax_bar.set_xlabel(stat_choice)
ax_bar.set_title("SHAP Feature Importance")
st.pyplot(fig_bar)

# ================================
# Feature-Level SHAP Value Scatter Plot
# ================================
st.subheader("🎯 SHAP Value Scatter Plot by Feature")

# Clean list of feature names
excluded = ["deal_no", "id", "master_deal_no"]
dropdown_features = [f for f in feature_names if not any(ex in f.lower() for ex in excluded)]

selected_feature = st.selectbox("Select a feature to explain", dropdown_features)

try:
    feature_idx = list(feature_names).index(selected_feature)
    feature_vals = X_preprocessed[:, feature_idx]

    # Handle sparse format if needed
    if hasattr(feature_vals, "toarray"):
        feature_vals = feature_vals.toarray().flatten()
    else:
        feature_vals = np.array(feature_vals).flatten()

    shap_vals = shap_values[:, feature_idx].values

    # Plot
    st.markdown(f"### SHAP Scatter for: <code style='color:limegreen'>{selected_feature}</code>", unsafe_allow_html=True)
    fig, ax = plt.subplots()
    ax.scatter(feature_vals, shap_vals, alpha=0.5, color="mediumseagreen", edgecolor="black")
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("SHAP Value")
    ax.set_title("Impact of Feature Value on Predicted Failure Probability")
    st.pyplot(fig)

except Exception as e:
    st.error(f"Could not generate SHAP scatter plot: {e}")