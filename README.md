# Customer Purchase Prediction Using Classification Algorithms

## Project Overview
This is a beginner-to-intermediate level machine learning project aimed at predicting whether a customer will purchase a product or service based on their demographic and behavioral data. The project demonstrates **data preprocessing, exploratory data analysis (EDA), model building, and evaluation using classification algorithms**.

---

## Dataset
- **File:** `Shopping_data.csv`  
- **Columns:**
  - `CustomerID`: Unique customer identifier
  - `Genre`: Gender of the customer (Male/Female)
  - `Age`: Age of the customer
  - `Annual Income (k$)`: Customer's income in thousand dollars
  - `Spending Score (1-100)`: Score representing customer behavior, converted to `Purchased` (1 if score ≥50, else 0)

> The dataset can be found in the `data/` folder.

---

## Project Workflow

### 1. Data Loading & Understanding
- Loaded the dataset using pandas
- Displayed first few rows, dataset shape, and columns
- Identified the target variable `Purchased` for classification

### 2. Data Preprocessing
- Checked and handled missing values (none in this dataset)
- Encoded categorical variables (`Genre` → `Gender`) using Label Encoding
- Converted `Spending Score` into a binary target (`Purchased`)
- Scaled numerical features (`Age` and `EstimatedSalary`) using StandardScaler
- Split dataset into training and testing sets

### 3. Exploratory Data Analysis (EDA)
- Scatter plots to visualize Age vs Estimated Salary by Purchased
- Confusion matrix plots
- Optional: Histogram distributions and correlation heatmaps

### 4. Model Building
Implemented two classification algorithms:
- **Logistic Regression**
- **Decision Tree**

> Optional: Random Forest, KNN can be added

### 5. Model Evaluation
- Compared models using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
- Logistic Regression performed slightly better than Decision Tree in this dataset

---

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
