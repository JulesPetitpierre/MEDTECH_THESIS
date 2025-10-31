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
It also shows a **SHAP-based local explanation** for each prediction to increase interpretability.

Prediction results are derived from the thesis:

**“Exploring the Complex Landscape of MedTech M&A Setbacks Using Machine Learning”**
""")

# ============================================================
# LOAD MODEL AND FULL DATA
# ============================================================

pipeline = joblib.load("safe_pipeline_xgb.joblib")
full_data = pd.read_csv("ONLY_RELEVANT_M&A.csv")

# ============================================================
# DEAL SELECTION OR UPLOAD
# ============================================================

st.sidebar.header("Deal Selection")
target_list = full_data["Target Name (tmanames)"].dropna().unique().tolist()
selected_target = st.sidebar.selectbox("Select a Deal by Target Name", target_list)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Or upload a new deal (CSV)", type=["csv"])

# Determine input
if uploaded_file:
    user_input = pd.read_csv(uploaded_file)
    actual_status = None
elif selected_target:
    selected_row = full_data[full_data["Target Name (tmanames)"] == selected_target].iloc[[0]]
    user_input = selected_row.drop(columns=["Deal Status (status)"], errors="ignore")
    actual_status = selected_row["Deal Status (status)"].values[0]
else:
    user_input = full_data.drop(columns=["Deal Status (status)"], errors="ignore").iloc[[0]]
    actual_status = None

# Coerce numeric types (failsafe for uploaded data)
user_input = user_input.copy()
for col in user_input.columns:
    try:
        user_input[col] = pd.to_numeric(user_input[col])
    except Exception:
        pass

# ============================================================
# PREDICTION
# ============================================================

failure_prob = pipeline.predict_proba(user_input)[0][1] * 100

# Threshold mapping
if failure_prob < 25:
    label = "✅ Very Low Risk of Failure"
    color = "#27ae60"
elif failure_prob < 50:
    label = "🟢 Likely to Succeed"
    color = "#2ecc71"
elif failure_prob < 75:
    label = "🟠 Moderate Risk"
    color = "#e67e22"
else:
    label = "🔴 High Risk of Failure"
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
            st.markdown("<h4 style='color:#e74c3c;'>🟥 Actual Outcome: Deal Withdrawn</h4>", unsafe_allow_html=True)
        else:
            st.markdown("<h4 style='color:#27ae60;'>🟩 Actual Outcome: Deal Completed</h4>", unsafe_allow_html=True)

# ============================================================
# SHAP EXPLANATION
# ============================================================

st.subheader("Feature Contributions to This Prediction (SHAP)")

try:
    # Get components from pipeline
    preprocessor = pipeline.named_steps["preprocessor"]
    calibrated_clf = pipeline.named_steps["classifier"]
    xgb_model = calibrated_clf.calibrated_classifiers_[0].estimator  # ✅ Correct access

    # Transform input
    X_input_preprocessed = preprocessor.transform(user_input)
    feature_names = preprocessor.get_feature_names_out()

    # SHAP
    explainer = TreeExplainer(xgb_model)
    shap_values = explainer(X_input_preprocessed)

    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
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

    country_data = world[
        (world["NAME"] == acq_country) | (world["NAME"] == tgt_country)
    ]

    if not country_data.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        world.boundary.plot(ax=ax, linewidth=0.6, edgecolor='gray')
        country_data.plot(ax=ax, color="skyblue", edgecolor='black')

        fig.text(0.01, 0.01, f"Acquiror: {acq_country}\nTarget: {tgt_country}",
                 fontsize=10, ha='left', va='bottom')
        ax.set_title("Acquiror and Target Locations", fontsize=12)
        st.pyplot(fig)
    else:
        st.info("Could not match countries. Ensure valid country names.")

except Exception as e:
    st.warning(f"Map could not be rendered: {e}")