# Customer Purchase Prediction - Fully Working Version

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =======================
# 1️⃣ Dataset path
# =======================
file_path = r"C:\eswar\ML_project\cpp\Shopping_data.csv"

# Check if file exists
if not os.path.exists(file_path):
    print(f"⚠️ ERROR: File not found at {file_path}")
    file_path = input("Please enter the correct full CSV path: ").strip()
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File still not found at {file_path}. Exiting program.")

# =======================
# 2️⃣ Load Dataset
# =======================
df = pd.read_csv(file_path)
print("✅ Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print(df.head())

# =======================
# 3️⃣ Rename Columns to match PDF
# =======================
df.rename(columns={
    'Genre': 'Gender',
    'Annual Income (k$)': 'EstimatedSalary',
    'Spending Score (1-100)': 'Purchased'
}, inplace=True)

# =======================
# 4️⃣ Convert Purchased to Binary
# =======================
# Binary classification: 1 if score >=50, else 0
df['Purchased'] = df['Purchased'].apply(lambda x: 1 if x >= 50 else 0)

# Encode Gender
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Male=1, Female=0

print("\nColumns after renaming, encoding, and binary conversion:\n", df.head())

# =======================
# 5️⃣ Define Features & Target
# =======================
X = df[['Gender', 'Age', 'EstimatedSalary']]
y = df['Purchased']

# =======================
# 6️⃣ Train-Test Split
# =======================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# =======================
# 7️⃣ Feature Scaling
# =======================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =======================
# 8️⃣ Logistic Regression
# =======================
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

print("\n✅ Logistic Regression Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))

# =======================
# 9️⃣ Decision Tree
# =======================
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

print("\n✅ Decision Tree Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))
print("Classification Report:\n", classification_report(y_test, y_pred_tree))

# =======================
# 🔟 Visualization - Confusion Matrix for Logistic Regression
# =======================
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =======================
# 🔹 Scatter plot of Age vs Estimated Salary colored by Purchased
# =======================
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='Age', y='EstimatedSalary', hue='Purchased', palette='coolwarm')
plt.title("Age vs Estimated Salary by Purchased")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.show()
