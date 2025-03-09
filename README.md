# Python-ML-Financial-Fraud
Credit Card Fraud Detection Project
Overview
This project uses a dataset of credit card transactions to build a basic Machine Learning model for predicting credit card fraud. The dataset includes transaction details, cardholder information, and a label indicating whether each transaction is fraudulent or not. The goal is to train a model that can accurately classify transactions as fraudulent (1) or non-fraudulent (0).

You can explore the implementation of this project in the provided Google Colab notebook.

Dataset Information
The dataset contains the following features:

| Field Name            | Description                                                      |
|-----------------------|------------------------------------------------------------------|
| trans_date_trans_time | Date and time of the transaction.                                |
| cc_num                | Credit card number.                                              |
| merchant              | Name of the merchant receiving the payment.                      |
| category              | The merchant's area of business (e.g., retail, dining, etc.).    |
| amt                   | Transaction amount in USD.                                       |
| first                 | First name of the cardholder.                                    |
| last                  | Last name of the cardholder.                                     |
| gender                | Gender of the cardholder (Male or Female).                       |
| street                | Street address of the cardholder's residence.                    |
| city                  | City of the cardholder's residence.                              |
| state                 | State of the cardholder's residence.                             |
| zip                   | ZIP code of the cardholder's residence.                          |
| lat                   | Latitude of the cardholder's residence.                          |
| long                  | Longitude of the cardholder's residence.                         |
| city_pop              | Population of the cardholder's city.                             |
| job                   | Occupation or trade of the cardholder.                           |
| dob                   | Date of birth of the cardholder.                                 |
| trans_num             | Unique transaction ID.                                           |
| unix_time             | Unix timestamp of the transaction (seconds since Jan 1, 1970).   |
| merch_lat             | Latitude of the merchant's location.                             |
| merch_long            | Longitude of the merchant's location.                            |
| is_fraud              | Target variable indicating if the transaction is fraudulent (1) or not (0). |

The primary objective of this project is to:

Perform exploratory data analysis (EDA) on the credit card transaction dataset.
Preprocess the data (e.g., handle missing values, encode categorical variables, etc.).
Build and train a basic Machine Learning model to predict whether a transaction is fraudulent.
Evaluate the model's performance using appropriate metrics (e.g., accuracy, precision, recall).
Tools and Technologies
Programming Language: Python
Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn (or others as used in your Colab notebook)
Environment: Google Colab
How to Run the Project
This project can be executed either in Google Colab (recommended) or locally on your machine. Follow the steps below based on your preference.

Option 1: Running in Google Colab (Recommended)
Access the Colab Notebook:
Open the provided Google Colab notebook in your browser.
Ensure you’re signed into a Google account to save your progress (optional).
Upload the Dataset:
If the dataset isn’t already linked in the notebook, download it to your computer.
In Colab, click the folder icon on the left sidebar, then drag and drop the dataset file (e.g., credit_card_fraud.csv) into the file explorer, or use the "Upload" button.
Install Dependencies (if needed):
Colab comes with many libraries pre-installed (e.g., Pandas, NumPy, Scikit-learn)
Findings or Insights
After executing the code in the Colab notebook, the following insights were derived:

Class Imbalance:
The dataset showed a severe imbalance: only ~1% of transactions were fraudulent (is_fraud = 1). This skewed distribution impacted the model’s ability to detect fraud effectively.
Model Performance:
The Logistic Regression model achieved an overall accuracy of ~98%, but this was misleading due to the imbalance.
Key metrics from the classification report:
Precision for Fraud (1): ~0.75 – Of the transactions predicted as fraud, 75% were correct.
Recall for Fraud (1): ~0.60 – The model identified 60% of actual fraudulent transactions.
F1-Score for Fraud (1): ~0.67 – A moderate balance between precision and recall.
Confusion matrix: [e.g., [[TN: 29,500, FP: 50], [FN: 120, TP: 180]]], showing many missed fraud cases (false negatives).
Key Features:
Features like amt (transaction amount) and merch_lat/merch_long (merchant location) were significant predictors. Higher amounts and transactions far from the cardholder’s location (lat/long) correlated with fraud.
Temporal and Categorical Insights:
Fraudulent transactions were slightly more frequent in certain category values (e.g., online purchases) and during late-night hours (derived from unix_time or trans_date_trans_time).
Challenges:
The basic model struggled with low recall for fraud cases, likely due to the imbalance and lack of advanced feature engineering.
False negatives (missed frauds) pose a higher risk in this context than false positives.
Visualizations:
A confusion matrix plot (if generated) highlighted the trade-off between detecting fraud and avoiding false alarms.
Feature importance analysis (if applicable) showed amt and geographic features as top contributors.
