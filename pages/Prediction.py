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

with st.expander("SHAP Feature Index Reference"):
    st.markdown("""
    | Index | Feature Name |
    |-------|--------------|
    | 0     | Date Announced (dateann) |
    | 1     | Unique DEAL ID (master_deal_no) |
    | 2     | Target Name (tmanames) |
    | 3     | Acquiror Name (amanames) |
    | 4     | Acquiror is a Financial Sponsor (Yes/No Flag) (afinancial) |
    | 5     | Acquiror is a Leverage Buyout Firm (albofirm) |
    | 6     | Acquiror is a Limited Partnership Flag (alp) |
    | 7     | Deal Attitude (attitude) |
    | 8     | Deals is a Divestiture Flag (divest) |
    | 9     | Division (division) |
    | 10    | Form of transaction (form) |
    | 11    | Percentage of consideration paid in cash (pct_cash) |
    | 12    | Percentage of consideration paid in other then cash or stock (p) |
    | 13    | Percentage of consideration paid in stock (pct_stk) |
    | 14    | Percentage of consideration which is unknown (pct_unknown) |
    | 15    | Percentage of Shares Sought (psought) |
    | 16    | Percentage of Shares Acquiror is Seeking to Own After Transacti |
    | 17    | Related Deals Flag (rd) |
    | 18    | Deal is a Repurchase Flag (repurch) |
    | 19    | Source of Funds Borrowing Flag (sfbor) |
    | 20    | Source of Funds Common Stock Issue Flag (sfcom) |
    | 21    | Financing via Internal Corporate Funds Flag (sfcorp) |
    | 22    | Financing via Debt Securities Flag (sfdebt) |
    | 23    | Financing via Line of Credit Flag (sflc) |
    | 24    | Ranking Value incl Net Debt of Target (USD Mil) (rankval) |
    | 25    | TR Acquiror Industry Description (atf_mid_desc) |
    | 26    | TR Target Industry Description (ttf_mid_desc) |
    | 27    | TR Acquiror Macro Description (atf_macro_desc) |
    | 28    | TR Target Macro Description (ttf_macro_desc) |
    | 29    | Acquiror Nation (anation) |
    | 30    | Acquiror Nation Code (anationcode) |
    | 31    | Target Nation (tnation) |
    | 32    | Target Nation Code (tnationcode) |
    | 33    | Target Public Status (tpublic) |
    | 34    | Acquiror Public Status (apublic) |
    | 35    | deal_value |
    | 36    | Enterprise Value ($mil) (entval) |
    | 37    | Equity Value ($mil) (eqval) |
    | 38    | Price Per Share (pr) |
    | 39    | Target Sales LTM ($mil) (salesltm) |
    | 40    | Target EBIT LTM ($mil) (ebitltm) |
    | 41    | Target Pre-Tax Income LTM ($mil) (ptincltm) |
    | 42    | Target Net Income LTM ($mil) (niltm) |
    | 43    | Target Net Assets ($mil) (netass) |
    | 44    | Target Total Assets ($mil) (tass) |
    | 45    | Target Cash Flow LTM ($mil) (cashflow) |
    | 46    | Target Book Value ($mil) (bookvalue) |
    | 47    | Target Common Equity ($mil) (commonequity) |
    | 48    | Target Earnings Per Share ($mil) (epsltm) |
    | 49    | Target Closing Stock Price 1 day Prior to Deal Announcement Day |
    | 50    | Target Closing Stock Price 1 week Prior to Deal Announcement Da |
    | 51    | Target Closing Stock Price 4 weeks Prior to Deal Announcement D |
    | 52    | Consideration Offered (considoff) |
    | 53    | Consideration Sought (considsought) |
    | 54    | Vix |
    | 55    | Interest rates |
    | 56    | GDP growth |
    | 57    | Healthcare growth |
    | 58    | Covid19 |
    | 59    | dateann |
    """)

# --- Load trained model and data ---
model = joblib.load("xgb_pipe.pkl")
full_data = pd.read_csv("ONLY_RELEVANT_M&A.csv")
features = model.named_steps["preprocessor"].get_feature_names_out()

# --- Let user select a deal by Acquiror ---
acquiror_list = full_data["Target Name (tmanames)"].dropna().unique().tolist()
selected_acquiror = st.selectbox("Select a Deal by Target", acquiror_list)

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
    st.subheader("Prediction percentage")
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
st.subheader("Explanation (SHAP), aka the power of variables in the development of the percentage points")

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
st.subheader("Company Locations")
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
