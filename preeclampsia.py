import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

df = pd.read_csv("Maternal Health Risk Data Set.csv")

X = df.drop("RiskLevel", axis=1)

print(X.isnull().sum())
print(np.isinf(X.select_dtypes(include=[np.number])).sum())

# Map labels (High risk = 1, others = 0)
df["RiskLevel"] = df["RiskLevel"].map({
    "high risk": 1,
    "mid risk": 0,
    "low risk": 0
})

X = df.drop("RiskLevel", axis=1)
y = df["RiskLevel"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(class_weight="balanced", max_iter=1000)
)
])

rf_pipeline = Pipeline([
    ("model", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ))
])

xgb_pipeline = Pipeline([
    ("model", XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=(len(y_train[y_train==0]) / len(y_train[y_train==1]))
    ))
])

for name, model in {
    "Logistic Regression": lr_pipeline,
    "Random Forest": rf_pipeline,
    "XGBoost": xgb_pipeline
}.items():
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"{name} ROC-AUC: {scores.mean():.4f}")

best_model = rf_pipeline  # choose RF as best

# 🔥 FIRST FIT THE MODEL
best_model.fit(X_train, y_train)

# ✅ THEN compute feature importance
import matplotlib.pyplot as plt

importances = best_model.named_steps["model"].feature_importances_

plt.barh(X.columns, importances)
plt.title("Feature Importance")
plt.show()

# ✅ Now evaluate
y_pred = best_model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, best_model.predict_proba(X_test)[:,1]))

from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(best_model, X_test, y_test)

shared_features = ['Age', 'SystolicBP', 'DiastolicBP']

X_shared = X[shared_features]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_shared, y, test_size=0.2, stratify=y, random_state=42
)

rf_shared = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

rf_shared.fit(X_train_s, y_train_s)

kaggle_test = pd.read_csv("test_dataset.csv")

kaggle_test = kaggle_test.rename(columns={
    "Age (yrs)": "Age",
    "Systolic BP": "SystolicBP",
    "Diastolic BP": "DiastolicBP"
})

kaggle_test["RiskLevel"] = kaggle_test["Risk_level"].map({
    "high": 1,
    "mid": 0,
    "low": 0
})

X_kaggle = kaggle_test[shared_features]
y_kaggle = kaggle_test["RiskLevel"]

y_kaggle_pred = rf_shared.predict(X_kaggle)
y_kaggle_prob = rf_shared.predict_proba(X_kaggle)[:,1]

print("External ROC-AUC:",
      roc_auc_score(y_kaggle, y_kaggle_prob))

print(confusion_matrix(y_kaggle, y_kaggle_pred))
