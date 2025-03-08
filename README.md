# Python-ML-Financial-Fraud
Credit Card Fraud Detection Project
Certainly! Here's the README file in the proper Markdown format, with code from your Colab notebook included and properly formatted using `#` and `*` for headings and lists.

```markdown
# Credit Card Fraud Detection

## Overview

This project focuses on building a machine learning model to detect fraudulent credit card transactions. The dataset contains transactional data from a variety of credit card activities, including whether the transaction is marked as fraudulent or not. The goal is to analyze this dataset, process the features, and develop a model capable of classifying transactions as fraudulent (1) or non-fraudulent (0).

## Dataset Information

The dataset includes the following columns:

- **trans_date_trans_time**: Date and time of the transaction.
- **cc_num**: Credit card number.
- **merchant**: Merchant receiving the payment.
- **category**: Type of merchant's business.
- **amt**: Transaction amount in US dollars.
- **first**: First name of the cardholder.
- **last**: Last name of the cardholder.
- **gender**: Gender of the cardholder (Male/Female).
- **street**: Street address of the cardholder.
- **city**: City of the cardholder's residence.
- **state**: State of the cardholder's residence.
- **zip**: ZIP code of the cardholder's residence.
- **lat**: Latitude of the cardholder's address.
- **long**: Longitude of the cardholder's address.
- **city_pop**: Population of the cardholder's city.
- **job**: Occupation of the cardholder.
- **dob**: Date of birth of the cardholder.
- **trans_num**: Transaction ID.
- **unix_time**: Unix timestamp (seconds since 1970).
- **merch_lat**: Latitude of the merchant's address.
- **merch_long**: Longitude of the merchant's address.
- **is_fraud**: Fraud label (1 for fraud, 0 for non-fraud).

## Objectives

The project aims to achieve the following:

1. **Data Preprocessing**: Clean and preprocess the data for analysis.
2. **Exploratory Data Analysis (EDA)**: Visualize and analyze the dataset to understand key patterns, correlations, and the distribution of fraud.
3. **Feature Engineering**: Generate additional features and prepare data for modeling.
4. **Modeling**: Implement machine learning models (e.g., Logistic Regression, Decision Trees, Random Forests, etc.) to predict fraudulence.
5. **Evaluation**: Assess the model performance using relevant metrics such as accuracy, precision, recall, F1 score, and ROC-AUC.

## Setup and Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

Make sure to install the necessary libraries such as `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, and `xgboost` for data analysis and model building.

## Project Workflow

### 1. Data Preprocessing

We handle missing values, encode categorical variables, and scale the numerical features in this step. Below is the code used to preprocess the dataset:

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.read_csv('credit_card_transactions.csv')

# Handling missing values
df.isnull().sum()

# Encode categorical features
label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])

# Scale numerical features
scaler = StandardScaler()
df[['amt', 'lat', 'long']] = scaler.fit_transform(df[['amt', 'lat', 'long']])

# Preview the dataset
df.head()
```

### 2. Exploratory Data Analysis (EDA)

We perform EDA to visualize and understand the relationships in the data. This involves checking for correlations between features and visualizing the distribution of fraud labels.

```python
# Fraud distribution visualization
sns.countplot(x='is_fraud', data=df)
plt.title('Fraud Distribution')
plt.show()

# Correlation matrix visualization
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()
```

### 3. Model Training

We split the data into training and testing sets, train a Random Forest classifier, and evaluate its performance. Here's the code to train and evaluate the model:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Split the dataset into features (X) and target (y)
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print("ROC-AUC Score: ", roc_auc_score(y_test, y_pred))
```

### 4. Model Evaluation

We use classification metrics such as accuracy, precision, recall, and F1-score, in addition to visualizing the confusion matrix to understand the model's performance.

```python
from sklearn.metrics import confusion_matrix

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.title('Confusion Matrix')
plt.show()
```

## Charts and Visualizations

- **Fraud Distribution**: This shows the distribution of fraud vs. non-fraud transactions.
- **Feature Correlation**: A heatmap showing the correlations between different features in the dataset.
- **Confusion Matrix**: This matrix visualizes the performance of the model, showing true positives, false positives, true negatives, and false negatives.

## Contributing

Feel free to fork the repository and make contributions. If you find any issues, please create an issue ticket or submit a pull request. Contributions in the form of bug fixes, feature improvements, and documentation are always welcome.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

---

This Markdown file should help present your project clearly and professionally on GitHub. It includes all necessary details and the specific code that was part of your Colab notebook.
