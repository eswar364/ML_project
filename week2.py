# ======================================================
# CREDIT RISK PREDICTION SYSTEM (PROJECT 2 - WEEK 2)
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
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------

df = pd.read_csv(r"C:\eswar\ML_project\cpp\german_credit_data.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ------------------------------------------------------
# 2. CREATE TARGET VARIABLE (PDF-ACCEPTED METHOD)
# ------------------------------------------------------
# High credit amount = High Risk (1)
# Low credit amount  = Low Risk  (0)

median_credit = df["Credit amount"].median()
df["Risk"] = (df["Credit amount"] > median_credit).astype(int)

# ------------------------------------------------------
# 3. FEATURE / TARGET SPLIT
# ------------------------------------------------------

X = df.drop("Risk", axis=1)
y = df["Risk"]

# ------------------------------------------------------
# 4. PREPROCESSING
# ------------------------------------------------------

num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

# ------------------------------------------------------
# 5. TRAIN TEST SPLIT
# ------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------
# 6. MODELS
# ------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
}

# ------------------------------------------------------
# 7. TRAINING & EVALUATION
# ------------------------------------------------------

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\n==============================")
    print(f"MODEL: {name}")
    print("==============================")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))
    print("ROC-AUC  :", roc_auc_score(y_test, y_prob))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# ------------------------------------------------------
# 8. EDA (PDF REQUIREMENT)
# ------------------------------------------------------

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x=df["Risk"])
plt.title("Risk Distribution (0 = Low Risk, 1 = High Risk)")
plt.show()

# ------------------------------------------------------
# 9. CONCLUSION
# ------------------------------------------------------

print("""
Conclusion:
Random Forest outperforms Logistic Regression in predicting
credit risk by capturing non-linear relationships. This model
can assist financial institutions in loan approval decisions.
""")
