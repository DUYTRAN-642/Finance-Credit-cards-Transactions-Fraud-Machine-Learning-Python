# Python-ML-Financial-Fraud
Credit Card Fraud Detection Project   
![image](https://github.com/user-attachments/assets/12a71339-0a64-46ac-a116-10a681827bad)

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

## The primary objective of this project is to:

* Perform exploratory data analysis (EDA) on the credit card transaction dataset.
* Preprocess the data (e.g., handle missing values, encode categorical variables, etc.).
* Build and train a basic Machine Learning model to predict whether a transaction is fraudulent.
* Evaluate the model's performance using appropriate metrics (e.g., accuracy, precision, recall).

## Tools and Technologies

* Programming Language: Python
* Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn and others
* Environment: Google Colab

## Preprocessing

Data overview, Check missing values, outlier detection and EDA using syntax 
```
df.info()
df.describe()
```

## Feature Engineering

* Create new features by transforming to "trans_time"; "year_of_birth" to fit the format running model
```
df["trans_time"] = pd.to_datetime(df['trans_date_trans_time']).dt.hour
df["year_of_birth"] = pd.to_datetime(df['dob']).dt.year
```
![image](https://github.com/user-attachments/assets/4aaf5a47-cf1f-424c-8a2f-8a2c8b6fbaf3)

* Endcode features "category" and "gender", encode states in US by group them into 4 regions and 'get_dummies', select top 20 jobs have highest fraud cases, encode top 20 jobs with highest fraud cases and the rest as 'others'
```
list_columns = ['category','gender']
df = pd.get_dummies(df, columns=list_columns,drop_first=True,dtype=int)
state_to_region = {
    'Northeast': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA'],
    'Midwest': ['IL', 'IN', 'MI', 'OH', 'WI', 'IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD'],
    'South': ['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV', 'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX'],
    'West': ['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY', 'AK', 'CA', 'HI', 'OR', 'WA']
}
region_to_state = {state: region for region, states in state_to_region.items() for state in states}

# Map state codes to regions
df['region']=df['state'].map(region_to_state)

# Create dummy variables for regions
dummies = pd.get_dummies(df['region'], drop_first=False).astype(int)

df = pd.concat([df, dummies], axis=1)
```
## Model Traning
```
# Select features and Split dataset
drop_features = ['Unnamed: 0.1','Unnamed: 0','trans_date_trans_time','cc_num','merchant','first','last','street','city','state','job','region','dob','trans_num','encoded_job_title','is_fraud']
X = df.drop(columns = drop_features)
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#Select model RandomForest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```
Random Forest is a smart choice for credit card fraud prediction because it balances robustness, flexibility, and performance, especially for imbalanced, noisy data, especially, Random Forest does not inherently require feature scaling (e.g., normalization or standardization)

## Model Evaluation
```
[[27050    55]
 [  315  1905]]
Accuracy: 0.99
F1 Score: 0.91
```
An accuracy of 99% sounds excellent at first glance—it means the model is correct 99% of the time. However, accuracy alone can be misleading, especially if the dataset is imbalanced (which seems to be the case here, as there are far more negatives (27,105) than positives (2,220))

The F1 score is given as 0.91, which is the harmonic mean of precision and recall, confirming the metrics are consistent and indicates the model performs well on the positive class despite the imbalance

==> This model 's performance  is quite good overall, with a strong F1 score indicating robustness despite imbalance

# Reference sources

* Dataset source: https://docs.google.com/spreadsheets/d/1FWOI4uY2_4xVn0YQnWEdevdbenxucmMs_BSGnoP-s0s/edit?gid=1430542985#gid=1430542985
* Colab notebook: https://colab.research.google.com/drive/1K8kAKJmPaOubSaCstZA_j7dWgNX1T4FL
