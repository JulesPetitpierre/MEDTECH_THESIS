import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import geopandas as gpd


# Page title
st.title("Explore the MedTech M&A Deal failures and a ML Model explain them")
st.markdown("This Machine Learning tool predicts the **failure probability** of MedTech M&A deals using an interpretable XGBoost model.")

# --- Load trained model and data ---
model = joblib.load("xgb_pipe.pkl")
full_data = pd.read_csv("ONLY_RELEVANT_M&A.csv")
features = model.named_steps["preprocessor"].get_feature_names_out()

# --- Let user select a deal by Acquiror ---
acquiror_list = full_data["Target Name (tmanames)"].dropna().unique().tolist()
selected_acquiror = st.selectbox("🧠 Select a Deal by Target", acquiror_list)

# --- Upload new data ---
st.sidebar.header("Optional: Upload New Deal")
uploaded_file = st.sidebar.file_uploader("Upload deal data as CSV", type=["csv"])

# --- Determine input data ---
if uploaded_file:
    user_input = pd.read_csv(uploaded_file)
elif selected_acquiror:
    selected_row = full_data[full_data["Target Name (tmanames)"] == selected_acquiror].iloc[[0]]
    user_input = selected_row.drop(columns=["deal_status"], errors="ignore")
else:
    default_input = full_data.drop(columns=["deal_status"], errors="ignore").iloc[[0]]
    user_input = default_input

# Coerce to correct types
user_input = user_input.copy()
for col in user_input.columns:
    try:
        user_input[col] = pd.to_numeric(user_input[col])
    except:
        pass

# --- Prediction ---
probability = model.predict_proba(user_input)[0][1] * 100

# Color + Label (corrected logic)
if probability < 25:
    bar_color = "#27ae60"   # green
    label = "✅ Very Low Risk of Failure"
elif probability < 50:
    bar_color = "#2ecc71"   # light green
    label = "🟢 Likely to Succeed"
elif probability < 75:
    bar_color = "#e67e22"   # orange
    label = "🟠 Moderate Risk"
else:
    bar_color = "#e74c3c"   # red
    label = "🔴 High Risk of Failure"

# --- Display Prediction & Actual Outcome Side by Side ---
left_col, right_col = st.columns([2, 1])  # Wider left for prediction

with left_col:
    st.subheader("📊 Prediction percentage")
    st.markdown(
        f"<h2 style='color:{bar_color};'>Predicted Failure Risk: {probability:.2f}%</h2>",
        unsafe_allow_html=True
    )
    st.markdown(f"**{label}**")

with right_col:
    if uploaded_file is None and "Deal Status (status)" in selected_row.columns:
        actual_status = selected_row["Deal Status (status)"].values[0]

        if actual_status == 1:
            st.markdown(
                "<h4 style='color:#e74c3c;'>🟥 Actual Outcome: Deal Withdrawn</h4>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<h4 style='color:#27ae60;'>🟩 Actual Outcome: Deal Completed</h4>",
                unsafe_allow_html=True
            )

# --- SHAP Explanation ---
st.subheader("🔍 Explanation (SHAP), aka the power of variables in the development of the percentage points")

# Build explainer from classifier step
explainer = shap.Explainer(model.named_steps["classifier"])

# Preprocess user input
preprocessed_input = model.named_steps["preprocessor"].transform(user_input)

# Get SHAP values
shap_values = explainer(preprocessed_input)

# Plot SHAP waterfall
shap.plots.waterfall(shap_values[0], max_display=10, show=False)
fig = plt.gcf()
st.pyplot(fig)

# --- Map Plot ---
st.subheader("🌍 Company Locations")
try:
    # Load local shapefile
    world = gpd.read_file("Countries/ne_110m_admin_0_countries.shp")

    acq_country = user_input["Acquiror Nation (anation)"].values[0]
    tgt_country = user_input["Target Nation (tnation)"].values[0]

    country_data = world[
        (world["NAME"] == acq_country) | (world["NAME"] == tgt_country)
    ]

    if not country_data.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        world.boundary.plot(ax=ax, linewidth=0.8, edgecolor='black')
        country_data.plot(ax=ax, color="skyblue", edgecolor='black')

        # Add the label in the bottom-left corner of the figure
        fig.text(0.01, 0.01,
                 f"Acquiror: {acq_country}\nTarget: {tgt_country}",
                 fontsize=10, ha='left', va='bottom')

        ax.set_title("Acquiror and Target Locations")
        st.pyplot(fig)

    else:
        st.info("Could not match countries for map. Please ensure exact country names in 'Acquiror Nation (anation)' and 'Target Nation (tnation)'.")
        
except Exception as e:
    st.warning(f"Map could not be rendered: {e}")
