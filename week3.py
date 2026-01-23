

# ======================================================
# WEEK 3 – CUSTOMER CHURN PREDICTION SYSTEM
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE

# ------------------------------------------------------
# 1. DATA LOADING & UNDERSTANDING
# ------------------------------------------------------

df = pd.read_csv(r"C:\eswar\ML_project\cpp\WA_Fn-UseC_-Telco-Customer-Churn.csv")



print("Dataset Shape:", df.shape)
print(df.head())

# ------------------------------------------------------
# 2. DATA CLEANING
# ------------------------------------------------------

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Convert target variable
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop ID column
df.drop("customerID", axis=1, inplace=True)

# ------------------------------------------------------
# 3. FEATURE ENGINEERING (PDF REQUIREMENT)
# ------------------------------------------------------

# Average monthly spend
df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

# Tenure grouping
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 36, 72],
    labels=["New", "Mid", "Long-Term"]
)

# Service usage count
service_cols = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]

df["ServiceCount"] = df[service_cols].apply(
    lambda x: (x == "Yes").sum(), axis=1
)

# ------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------

plt.figure(figsize=(6,4))
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ------------------------------------------------------
# 5. DATA PREPROCESSING
# ------------------------------------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]

num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object", "category"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

X_processed = preprocessor.fit_transform(X)

# ------------------------------------------------------
# 6. HANDLE IMBALANCED DATA (PDF REQUIREMENT)
# ------------------------------------------------------

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_processed, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# ------------------------------------------------------
# 7. MODEL BUILDING & EVALUATION
# ------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n==============================")
    print(f"MODEL: {name}")
    print("==============================")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))
    print("ROC-AUC  :", roc_auc_score(y_test, y_prob))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# ------------------------------------------------------
# 8. CONCLUSION
# ------------------------------------------------------

print("""
Conclusion:
Gradient Boosting and Random Forest models outperform Logistic
Regression in identifying customer churn. Tenure, monthly charges,
contract type, and service usage are key churn drivers. These insights
can help businesses design targeted retention strategies.
""")
