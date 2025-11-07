# MedTech M&A Deal Failure Prediction

This repository contains the full codebase, data schema, and deployment files for the Bachelor Thesis:

**"Predicting Pre-Closing M&A Deal Failures in the MedTech Sector Using Machine Learning"**  
Author: *Jules Petitpierre*  
University of St. Gallen (HSG), 2025  
Supervisor: Prof. Dr. Despoina Makariou

---

## 🔍 Project Summary
This thesis investigates whether announced MedTech M&A transactions can be reliably predicted to fail (i.e., be withdrawn or terminated) before closing, using only publicly available deal- and firm-level data from the time of announcement.

A predictive framework was developed using machine learning models—primarily calibrated **XGBoost**, with benchmarks including **Random Forest** and **ElasticNet**—and was deployed as an interactive **Streamlit web application**.

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

## How to Run the App Locally

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/medtech-ma-failure-prediction.git
cd medtech-ma-failure-prediction
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run Streamlit App
```bash
streamlit run Home.py
```

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

## Contact
**Jules Petitpierre**  
*Bachelor in Business Administration, University of St. Gallen*  
[jules.petitpierre@students.unisg.ch](mailto:jules.petitpierre@students.unisg.ch)

---

## License
This project is academic work submitted for the completion of a Bachelor's degree and cannot be used in any way by any third party.
---

## 🙏 Acknowledgements and Aids
- Prof. Dr. Despoina Makariou (Supervisor)
- Streamlit, SHAP, WRDS, Pitchbook
- All researchers and authors cited in the thesis (see bibliography)
- Chat GPT4o for structuring, correcting and simplifying codes when needed

---