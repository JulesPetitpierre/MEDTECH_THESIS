import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

# Load data
df = pd.read_csv("ONLY_RELEVANT_M&A.csv")
y = df["Deal Status (status)"].astype(int)
X = df.drop(columns=[
    "Deal Status (status)",
    "Unique DEAL ID (master_deal_no)",
    "Target Name (tmanames)",
    "Acquiror Name (amanames)"
])

# Ensure all objects are strings
for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].astype(str)

# Define transformers
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

# Define model
xgb = XGBClassifier(
    objective="binary:logistic",
    use_label_encoder=False,
    eval_metric="logloss",
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.9,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    random_state=42
)

# Transform once
X_trans = preprocessor.fit_transform(X)

# Calibrate
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cal_clf = CalibratedClassifierCV(estimator=xgb, method="isotonic", cv=cv, n_jobs=-1)
cal_clf.fit(X_trans, y)

# Full pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", cal_clf)
])

# Save
joblib.dump(pipeline, "safe_pipeline_xgb.joblib")
print("✅ Saved: safe_pipeline_xgb.joblib")

# === SAFETY CHECK ===
pipe_check = joblib.load("safe_pipeline_xgb.joblib")
ohe = pipe_check.named_steps["preprocessor"].named_transformers_["cat"]
print("✅ OHE safety check: handle_unknown =", ohe.handle_unknown)
assert ohe.handle_unknown == "ignore"