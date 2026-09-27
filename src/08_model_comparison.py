from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pandas as pd

# Load dataset
cancer = load_breast_cancer()

# Create DataFrame
df = pd.DataFrame(
    cancer.data,
    columns=cancer.feature_names
)

df["target"] = cancer.target

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Create models
logistic_model = LogisticRegression(
    max_iter=10000
)

xgboost_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

# Create 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Evaluate Logistic Regression
logistic_scores = cross_val_score(
    logistic_model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

# Evaluate XGBoost
xgboost_scores = cross_val_score(
    xgboost_model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("=" * 50)
print("CROSS-VALIDATION MODEL COMPARISON")
print("=" * 50)

print("\nLogistic Regression scores:")
print(logistic_scores)

print("Logistic Regression mean accuracy:")
print(logistic_scores.mean())

print("\nXGBoost scores:")
print(xgboost_scores)

print("XGBoost mean accuracy:")
print(xgboost_scores.mean())
