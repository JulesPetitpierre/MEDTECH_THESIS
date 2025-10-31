import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import geopandas as gpd

# ============================================================
# PAGE TITLE AND INTRO
# ============================================================

st.title("MedTech M&A Deal Failure Prediction and Explainability")
st.markdown("""
This page allows you to explore individual deal predictions from the calibrated **XGBoost classifier** 
used in the thesis *“Exploring the Complex Landscape of MedTech M&A Setbacks Using Machine Learning.”*  
The model provides both a **failure probability** and a **transparent SHAP explanation** for each case.
""")

# ============================================================
# LOAD MODEL AND PREPROCESSOR
# ============================================================

# Load the preprocessor and model separately (from joblib files)
preprocessor = joblib.load("preprocessor_for_xgboost.joblib")
model = joblib.load("calibrated_estimator_xgboost.joblib")

# Load full dataset for context
full_data = pd.read_csv("ONLY_RELEVANT_M&A.csv")

# ============================================================
# DEAL SELECTION AND FILE UPLOAD
# ============================================================

st.sidebar.header("Deal Selection")

acquiror_list = full_data["Target Name (tmanames)"].dropna().unique().tolist()
selected_acquiror = st.sidebar.selectbox("Select a Deal by Target", acquiror_list)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Or upload new deal data as CSV", type=["csv"])

# Determine input data source
if uploaded_file:
    user_input = pd.read_csv(uploaded_file)
elif selected_acquiror:
    selected_row = full_data[full_data["Target Name (tmanames)"] == selected_acquiror].iloc[[0]]
    user_input = selected_row.drop(columns=["Deal Status (status)"], errors="ignore")
else:
    user_input = full_data.drop(columns=["Deal Status (status)"], errors="ignore").iloc[[0]]

# Coerce numeric types safely
user_input = user_input.copy()
for col in user_input.columns:
    try:
        user_input[col] = pd.to_numeric(user_input[col])
    except Exception:
        pass

# ============================================================
# PREDICTION PIPELINE
# ============================================================

# Apply preprocessing, then predict with calibrated XGBoost model
X_input = preprocessor.transform(user_input)
probability = model.predict_proba(X_input)[0][1] * 100  # % probability of failure

# Threshold based on thesis calibration
threshold = 0.60

# Determine label and color
if probability < 25:
    bar_color = "#27ae60"
    label = "✅ Very Low Risk of Failure"
elif probability < 50:
    bar_color = "#2ecc71"
    label = "🟢 Likely to Succeed"
elif probability < 75:
    bar_color = "#e67e22"
    label = "🟠 Moderate Risk"
else:
    bar_color = "#e74c3c"
    label = "🔴 High Risk of Failure"

# ============================================================
# DISPLAY RESULTS
# ============================================================

left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Predicted Failure Probability")
    st.markdown(
        f"<h2 style='color:{bar_color};'>Predicted Failure Risk: {probability:.2f}%</h2>",
        unsafe_allow_html=True
    )
    st.markdown(f"**{label}**")

with right_col:
    if uploaded_file is None and "Deal Status (status)" in full_data.columns:
        actual_status = selected_row["Deal Status (status)"].values[0]
        if actual_status == 1:
            st.markdown("<h4 style='color:#e74c3c;'>🟥 Actual Outcome: Deal Withdrawn</h4>", unsafe_allow_html=True)
        else:
            st.markdown("<h4 style='color:#27ae60;'>🟩 Actual Outcome: Deal Completed</h4>", unsafe_allow_html=True)

# ============================================================
# SHAP EXPLANATION
# ============================================================

st.subheader("SHAP Explanation — Feature Contributions to Predicted Risk")

# Compute SHAP values
feature_names = preprocessor.get_feature_names_out()
explainer = shap.Explainer(model, feature_names=feature_names)
shap_values = explainer(X_input)

# Waterfall plot for local explanation
try:
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    fig = plt.gcf()
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Could not generate SHAP plot: {e}")

# ============================================================
# MAP VISUALISATION
# ============================================================

st.subheader("Acquiror and Target Country Map")

try:
    world = gpd.read_file("Countries/ne_110m_admin_0_countries.shp")
    acq_country = user_input["Acquiror Nation (anation)"].values[0]
    tgt_country = user_input["Target Nation (tnation)"].values[0]

    country_data = world[(world["NAME"] == acq_country) | (world["NAME"] == tgt_country)]

    if not country_data.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        world.boundary.plot(ax=ax, linewidth=0.8, edgecolor='gray')
        country_data.plot(ax=ax, color="skyblue", edgecolor='black')

        fig.text(0.01, 0.01, f"Acquiror: {acq_country}\nTarget: {tgt_country}", fontsize=10, ha='left', va='bottom')
        ax.set_title("Acquiror and Target Locations", fontsize=12)
        st.pyplot(fig)
    else:
        st.info("Could not match countries for map — please verify country name consistency.")

except Exception as e:
    st.warning(f"Map could not be rendered: {e}")
