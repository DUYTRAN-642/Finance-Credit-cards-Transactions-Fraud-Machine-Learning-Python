# Python-ML-Financial-Fraud
Credit Card Fraud Detection Project
## Overview

This project uses a dataset of credit card transactions to build a basic Machine Learning model for predicting credit card fraud. The dataset includes transaction details, cardholder information, and a label indicating whether each transaction is fraudulent or not. The goal is to train a model that can accurately classify transactions as fraudulent (1) or non-fraudulent (0).

You can explore the implementation of this project in the provided Google Colab notebook.

## Dataset Information
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

##The primary objective of this project is to:

* Perform exploratory data analysis (EDA) on the credit card transaction dataset.
* Preprocess the data (e.g., handle missing values, encode categorical variables, etc.).
* Build and train a basic Machine Learning model to predict whether a transaction is fraudulent.
* Evaluate the model's performance using appropriate metrics (e.g., accuracy, precision, recall).

# Tools and Technologies

* Programming Language: Python
* Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn (or others as used in your Colab notebook)
* Environment: Google Colab


