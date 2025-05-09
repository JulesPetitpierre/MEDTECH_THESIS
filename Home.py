import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Streamlit page setup
st.set_page_config(page_title="MedTech M&A Summary", layout="wide")

# Custom styling
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Lato&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"]  {
        font-family: 'Lato', sans-serif;
        background-color: #0B1A28;
        color: white;
    }
    footer {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Set Seaborn and Matplotlib styling
plt.style.use("dark_background")
sns.set_style("darkgrid")
sns.set_palette("dark")

st.title("MedTech M&A Failure Prediction Summary")

with st.expander("Welcome. Click to read the introduction"):
    st.markdown("""
    This application represents an **applied summarizing tool** developed as part of my Bachelor's thesis:

    **"Decoding Mergers & Acquisitions Failures: Exploring the Complex Landscape of MedTech M&A Setbacks Through Advanced Risk Modeling Techniques"**

    ---
    **Defining Failure in This Context**  
    In this research, failure refers specifically to **withdrawn M&A transactions** — deals that are formally announced but ultimately not completed.  

    Such withdrawals, whether driven by internal misalignment, regulatory barriers, or red flags uncovered during due diligence, often represent significant hidden costs.

    ---
    **Purpose of This Tool**  
    This interactive application enables users to:
    - Generate a **predicted risk score** for deal failure  
    - Understand **feature contributions** using SHAP explainability  
    - Conduct **scenario testing** and evaluate risk across a portfolio  

    Designed with ambition and practical impact in mind, this tool connects academic modeling with real-world decision-making in M&A.

    ---
    **Note:** This is an early version of the application and will be further refined in the coming days with enhanced visuals and extended functionality.
    """)

# Load model and data
df = pd.read_csv("ONLY_RELEVANT_M&A.csv")
model = joblib.load("xgb_pipe.pkl")

# Predict failure probabilities
X = df.drop(columns=["Deal Status (status)"])
y = df["Deal Status (status)"]
df["predicted_failure_prob"] = model.predict_proba(X)[:, 1]
df["predicted_class"] = (df["predicted_failure_prob"] >= 0.5).astype(int)

# Sidebar filters
st.sidebar.header("Choose a Specific Country")
countries = sorted(df["Target Nation (tnation)"].dropna().unique())
selected_country = st.sidebar.selectbox("Country look of", options=[None] + countries)

filtered_df = df.copy()
if selected_country:
    filtered_df = filtered_df[filtered_df["Target Nation (tnation)"] == selected_country]

# Display summary metrics
st.markdown("### Failure Statistics Overview")
col1, col2 = st.columns(2)

with col1:
    st.metric("Total Deals", len(filtered_df))
    st.metric("Actual Failures", filtered_df["Deal Status (status)"].sum())
    st.metric("Predicted Failures (≥50%)", filtered_df["predicted_class"].sum())

with col2:
    st.metric("Avg. Predicted Risk (%)"round(filtered_df["predicted_failure_prob"].mean() * 100, 2)), round(filtered_df["predicted_failure_prob"].mean() * 100, 2))
    st.metric("Deals with >75% Risk", (filtered_df["predicted_failure_prob"] > 0.75).sum())
    st.metric("Deals with <25% Risk", (filtered_df["predicted_failure_prob"] < 0.25).sum())

st.markdown("---")

# Predicted risk distribution
st.subheader("Predicted Risk Distribution")
fig1, ax1 = plt.subplots(facecolor="#0B1A28")
sns.histplot(filtered_df["predicted_failure_prob"] * 100, bins=20, ax=ax1, color="#1f77b4", edgecolor="white")
ax1.set_facecolor("#0B1A28")
ax1.set_xlabel("Predicted Failure Probability (%)", color="white")
ax1.set_ylabel("Count", color="white")
ax1.tick_params(colors="white")
st.pyplot(fig1)

# Failure count by country
st.subheader("Failure Count by Country")
country_df = (
    filtered_df.groupby("Target Nation (tnation)")
    .agg(actual_failures=("Deal Status (status)", "sum"), predicted_failures=("predicted_class", "sum"))
    .sort_values("actual_failures", ascending=False)
    .head(15)
)
fig2, ax2 = plt.subplots(facecolor="#0B1A28")
country_df.plot(kind="bar", ax=ax2, color=["#1f77b4", "#2ecc71"], edgecolor="white")
ax2.set_facecolor("#0B1A28")
ax2.set_ylabel("Count", color="white")
ax2.set_title("Actual vs Predicted Failures by Country", color="white")
ax2.tick_params(colors="white")
st.pyplot(fig2)