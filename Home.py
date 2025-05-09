import shap
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

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
    st.metric("Avg. Predicted Risk (%)", round(filtered_df["predicted_failure_prob"].mean() * 100, 2))
    st.metric("Deals with >75% Risk", (filtered_df["predicted_failure_prob"] > 0.75).sum())
    st.metric("Deals with <25% Risk", (filtered_df["predicted_failure_prob"] < 0.25).sum())

st.markdown("---")

# Load model and data
model = joblib.load("xgb_pipe.pkl")
df = pd.read_csv("ONLY_RELEVANT_M&A.csv")

# Prepare features
X_raw = df.drop(columns=["Deal Status (status)"], errors="ignore")
X_preprocessed = model.named_steps["preprocessor"].transform(X_raw)
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
explainer = shap.Explainer(model.named_steps["classifier"], feature_names=feature_names)
shap_values = explainer(X_preprocessed)

# Load readable names
with open("columns.json") as f:
    readable_names = json.load(f)

# Select 2 features + SHAP value dimension
st.subheader("📈 Interactive 3D SHAP Scatter")
feature1 = st.selectbox("X-axis: Feature", readable_names, key="shap3d_x")
feature2 = st.selectbox("Y-axis: Feature", readable_names, key="shap3d_y")
shap_target = st.selectbox("Z-axis: SHAP Value of", readable_names, key="shap3d_z")

# Index conversion
try:
    idx_x = readable_names.index(feature1)
    idx_y = readable_names.index(feature2)
    idx_z = readable_names.index(shap_target)

    x_vals = X_preprocessed[:, idx_x].toarray().flatten() if hasattr(X_preprocessed, "toarray") else X_preprocessed[:, idx_x]
    y_vals = X_preprocessed[:, idx_y].toarray().flatten() if hasattr(X_preprocessed, "toarray") else X_preprocessed[:, idx_y]
    z_vals = shap_values[:, idx_z].values.flatten()

    # Plotly 3D scatter
    fig = px.scatter_3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        color=z_vals,
        labels={"x": feature1, "y": feature2, "z": f"SHAP: {shap_target}"},
        opacity=0.7
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Could not generate 3D plot: {e}")