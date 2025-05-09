import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Setup
st.set_page_config(page_title="Explainability Deep Dive", layout="wide")
st.title("Explainability Deep Dive")

# Load model and data
model = joblib.load("xgb_pipe.pkl")
df = pd.read_csv("ONLY_RELEVANT_M&A.csv")

# Prepare features
X_raw = df.drop(columns=["Deal Status (status)"], errors="ignore")
X_preprocessed = model.named_steps["preprocessor"].transform(X_raw)
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

# SHAP explainer and values
explainer = shap.Explainer(model.named_steps["classifier"], feature_names=feature_names)
shap_values = explainer(X_preprocessed)

# --- Controls for SHAP Summary ---
st.subheader("Global SHAP Summary Plot (Customizable)")

col1, col2 = st.columns(2)
with col1:
    stat_choice = st.selectbox(
        "Select SHAP statistic",
        options=["Mean |SHAP|", "Median SHAP", "SHAP Std Dev"]
    )
with col2:
    top_n = st.selectbox("How many top features?", options=[10, 20])

# Compute importance
shap_matrix = shap_values.values
if stat_choice == "Median SHAP":
    importance = np.median(np.abs(shap_matrix), axis=0)
elif stat_choice == "SHAP Std Dev":
    importance = np.std(shap_matrix, axis=0)
else:
    importance = np.abs(shap_matrix).mean(axis=0)

# Limit number of features
if top_n != "All":
    top_n = int(top_n)
    top_indices = np.argsort(importance)[-top_n:]
else:
    top_indices = np.argsort(importance)

# Plot SHAP bar
fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
sorted_idx = top_indices[np.argsort(importance[top_indices])]
ax_bar.barh(
    [feature_names[i] for i in sorted_idx],
    importance[sorted_idx],
    color="teal"
)
ax_bar.set_xlabel(stat_choice)
ax_bar.set_title("SHAP Feature Importance")
st.pyplot(fig_bar)

# --- Scatter: Feature-level SHAP visualization ---
st.subheader("SHAP Value Scatter Plot")

# Remove ID-type features from dropdown
excluded = ["master_deal_no", "deal_no", "id"]
dropdown_features = [f for f in feature_names if not any(ex in f.lower() for ex in excluded)]
selected_feature = st.selectbox("Select a Feature", dropdown_features)

# Plot selected feature SHAP scatter
try:
    feature_idx = list(feature_names).index(selected_feature)
    feature_values = X_preprocessed[:, feature_idx]

    if hasattr(feature_values, "toarray"):
        feature_values = feature_values.toarray().flatten()
    elif hasattr(feature_values, "todense"):
        feature_values = np.array(feature_values).flatten()
    else:
        feature_values = np.array(feature_values).flatten()

    shap_vals = shap_values[:, feature_idx].values

    st.markdown(f"### 💠 SHAP Scatter for: <span style='color:limegreen'><code>{selected_feature}</code></span>", unsafe_allow_html=True)
    fig, ax = plt.subplots()
    ax.scatter(feature_values, shap_vals, alpha=0.5, color="mediumseagreen", edgecolor="black")
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("SHAP Value")
    ax.set_title("SHAP Impact on Failure Probability")
    st.pyplot(fig)

except Exception as e:
    st.error(f"Could not plot feature '{selected_feature}': {e}")