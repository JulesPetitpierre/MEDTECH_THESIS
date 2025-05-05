import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Setup
st.set_page_config(page_title="MedTech M&A Summary", layout="wide")
st.title("📊 MedTech M&A Failure Prediction Summary")

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
st.markdown("### 📈 Failure Statistics Overview")
col1, col2 = st.columns(2)

with col1:
    st.metric("Total Deals", len(filtered_df))
    st.metric("Actual Failures", filtered_df["Deal Status (status)"].sum())
    st.metric("Predicted Failures (≥50%)", filtered_df["predicted_class"].sum())

with col2:
    st.metric("Avg. Predicted Risk (%)", round(filtered_df["predicted_failure_prob"].mean() * 100, 2))
    st.metric("Deals with >75% Risk", (filtered_df["predicted_failure_prob"] > 0.75).sum())
    st.metric("Deals with <25% Risk", (filtered_df["predicted_failure_prob"] < 0.25).sum())

st.markdown("---")

# 📊 Predicted risk distribution
st.subheader("📊 Predicted Risk Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(filtered_df["predicted_failure_prob"] * 100, bins=20, ax=ax1, color="#1f77b4", edgecolor="black")
ax1.set_xlabel("Predicted Failure Probability (%)")
st.pyplot(fig1)

# 🌍 Failure count by country
st.subheader("🌍 Failure Count by Country")
country_df = (
    filtered_df.groupby("Target Nation (tnation)")
    .agg(actual_failures=("Deal Status (status)", "sum"), predicted_failures=("predicted_class", "sum"))
    .sort_values("actual_failures", ascending=False)
    .head(15)
)
fig2, ax2 = plt.subplots()
country_df.plot(kind="bar", ax=ax2, color=["#1f77b4", "#2ecc71"])  # blue + green
ax2.set_ylabel("Count")
ax2.set_title("Actual vs Predicted Failures by Country")
st.pyplot(fig2)