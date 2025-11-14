import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import geopandas as gpd
from shap import TreeExplainer

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MedTech Deal Risk Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("MedTech M&A Deal Failure Prediction and Explainability")

st.markdown("""
This tool predicts the **ex-ante failure risk** of MedTech M&A transactions using a **calibrated XGBoost classifier**.
It also shows a **SHAP-based local explanation** for interpretability.
""")

# ============================================================
# LOAD MODEL AND DATA
# ============================================================

pipeline = joblib.load("safe_pipeline_xgb_streamlit.joblib")
full_data = pd.read_csv("ONLY_RELEVANT_M&A.csv")

preprocessor = pipeline.named_steps["preprocessor"]
expected_cols = preprocessor.feature_names_in_

# ============================================================
# DEAL SELECTION OR UPLOAD
# ============================================================

st.sidebar.header("Deal Selection")
target_list = full_data["Target Name (tmanames)"].dropna().unique().tolist()
selected_target = st.sidebar.selectbox("Select a Deal by Target Name", target_list)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Or upload a new deal (CSV)", type=["csv"])

# Determine input source
if uploaded_file:
    user_input = pd.read_csv(uploaded_file)
    actual_status = None
else:
    selected_row = full_data[full_data["Target Name (tmanames)"] == selected_target].iloc[[0]]
    actual_status = selected_row["Deal Status (status)"].values[0]
    user_input = selected_row.drop(columns=["Deal Status (status)"], errors="ignore")

# ============================================================
# SANITIZE INPUT FOR PIPELINE COMPATIBILITY
# ============================================================

user_input = user_input.copy()

# Reindex to expected model columns
user_input = user_input.reindex(columns=expected_cols, fill_value=np.nan)

# Type cleaning
for col in user_input.columns:
    if user_input[col].dtype == "object":
        user_input[col] = user_input[col].astype(str).replace(["nan", "None"], "Missing").fillna("Missing")
    else:
        user_input[col] = pd.to_numeric(user_input[col], errors="coerce").fillna(0)

# ============================================================
# PREDICTION
# ============================================================

failure_prob = float(pipeline.predict_proba(user_input)[0][1]) * 100

# Display-friendly label
if failure_prob < 25:
    label = "Very Low Risk of Failure"
    color = "#27ae60"
elif failure_prob < 50:
    label = "Likely to Succeed"
    color = "#2ecc71"
elif failure_prob < 75:
    label = "Moderate Risk"
    color = "#e67e22"
else:
    label = "High Risk of Failure"
    color = "#e74c3c"

# ============================================================
# DISPLAY RESULTS
# ============================================================

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Predicted Failure Probability")
    st.markdown(
        f"<h2 style='color:{color};'>Predicted Failure Risk: {failure_prob:.2f}%</h2>",
        unsafe_allow_html=True
    )
    st.markdown(f"**{label}**")

with col2:
    if actual_status is not None:
        if actual_status == 1:
            st.markdown("<h4 style='color:#e74c3c;'>Actual Outcome: Deal Withdrawn</h4>", unsafe_allow_html=True)
        else:
            st.markdown("<h4 style='color:#27ae60;'>Actual Outcome: Deal Completed</h4>", unsafe_allow_html=True)

# ============================================================
# SHAP EXPLANATION (with real feature names)
# ============================================================

st.subheader("Feature Contributions to This Prediction (SHAP)")

try:
    calibrated_clf = pipeline.named_steps["classifier"]
    xgb_model = calibrated_clf.calibrated_classifiers_[0].estimator

    # Transform with preprocessing
    X_input_preprocessed = preprocessor.transform(user_input)

    # SHAP explainer
    explainer = TreeExplainer(xgb_model)
    shap_values = explainer(X_input_preprocessed)

    # Assign actual transformed feature names
    feature_names = preprocessor.get_feature_names_out()
    shap_values.feature_names = feature_names

    # Waterfall plot
    shap.plots.waterfall(shap_values[0], max_display=15, show=False)
    fig = plt.gcf()
    st.pyplot(fig)

except Exception as e:
    st.warning(f"SHAP plot could not be generated: {e}")

# ============================================================
# MAP VISUALIZATION
# ============================================================

st.subheader("Deal Geography: Acquiror and Target Countries")

try:
    world = gpd.read_file("Countries/ne_110m_admin_0_countries.shp")
    acq_country = user_input["Acquiror Nation (anation)"].values[0]
    tgt_country = user_input["Target Nation (tnation)"].values[0]

    country_data = world[world["NAME"].isin([acq_country, tgt_country])]

    fig, ax = plt.subplots(figsize=(10, 6))
    world.boundary.plot(ax=ax, linewidth=0.6, edgecolor='gray')
    country_data.plot(ax=ax, color="skyblue", edgecolor='black')
    ax.set_title("Acquiror and Target Locations", fontsize=12)

    st.pyplot(fig)

except Exception:
    st.info("Country mapping unavailable for this deal.")