# MedTech M&A Deal Failure Prediction

This repository contains the full codebase, data schema, and deployment files for the Bachelor Thesis:

**"Decoding Mergers & Acquisitions Failures:
Exploring The Complex Landscape of MedTech M&A Setbacks Through Advanced Risk Modeling Techniques
"**  
Author: *Jules Petitpierre*  
University of St. Gallen (HSG), 2025  
Supervisor: Prof. Dr. Despoina Makariou

---

## Repository Structure
```
├── ONLY_RELEVANT_M&A.csv           # Final cleaned dataset (630 transactions)
├── columns.json                    # Column metadata for the Streamlit interface
├── Home.py                         # Streamlit home page (prediction dashboard)
├── train_safe_pipeline.py         # ML pipeline training script (with calibration, CV, SHAP)
├── safe_pipeline_xgb_streamlit.joblib # Final XGBoost model with pipeline (used in app)
├── requirements.txt               # Environment dependencies
├── pages/                         # Additional Streamlit pages (SHAP explorer, deep dives)
├── Countries/                     # Country flags / visualization assets (optional)
```

---

## Technologies Used
- **Python 3.9+**
- **pandas, scikit-learn, xgboost, shap, joblib**
- **Streamlit** for front-end web app
- **Matplotlib, Seaborn** for visuals

---

## Features Used

-**Full variable list** is available in `columns.json` and Appendix A.1 of the thesis.

---

## Model Overview
- **Primary Model**: XGBoost Classifier (calibrated)
- **Benchmark Models**: Random Forest, ElasticNet Logistic Regression
- **Interpretability**: SHAP (global and local explanations)
- **Validation**: Stratified 5-fold CV, Bootstrapping (n=1000), Temporal Testing

---

## Model Deployment
The app interface provides:
- **Deal Failure Prediction** with calibrated probabilities
- **Local SHAP Explanation** of feature impact
- **Global Feature Importance** dashboard
- **Threshold Slider** with live updates of F1-score and confusion matrix

---

## Thesis Document
For full details on:
- Methodology
- Data filtering and engineering
- Model evaluation
- SHAP analysis
- Streamlit tool deployment

Refer to: `20-613-220_Jules_Petitpierre_Thesis.pdf`

---

## License
This project is academic work submitted for the completion of a Bachelor's degree and cannot be used in any way by any third party.
---

## Acknowledgements and Aids
- Prof. Dr. Despoina Makariou (Supervisor)
- Streamlit, SHAP, WRDS, Pitchbook
- All researchers and authors cited in the thesis (see bibliography)
- Chat GPT4o for structuring, debugging and simplifying model codes and streamlit codes when needed

---
