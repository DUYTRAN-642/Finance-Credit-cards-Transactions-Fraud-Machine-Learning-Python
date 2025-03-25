# Finance – Fraud Transactions on Credit cards – Machine Learning Python
![image](https://github.com/user-attachments/assets/12a71339-0a64-46ac-a116-10a681827bad)

## 📊 Project Title: Credit Card Fraud Detection Project

Author: DUY TRAN

Date: 2025-03-21

Tools Used: Python

## 📑 Table of Contents

📌 Background & Overview

📂 Dataset Description & Data Structure

🔎 Final Conclusion & Recommendations

## 📌 Background & Overview

### 📖 What is this project about? What Business Question will it solve?

✔️ This project builds a basic Machine Learning model for predicting credit card fraud. 

✔️ Provide actionable insights for decision making in enhancing finacial security of bank, increasing customer reliability and satisfaction.

### 👤 Who is this project for?

✔️ Data engineers: who are focused on developing, deploying, and maintaining fraud detection systems.

✔️ Risk Management Teams: Individuals or teams focused on assessing and managing financial risk for organization, ensuring that fraudulent activities are minimized.

✔️ Regulatory and Compliance Agencies: Organizations that are responsible for ensuring businesses comply with financial regulations

✔️ Research and Development: Academic or corporate research groups studying fraud detection techniques and machine learning applications in finance

## The primary objective of this project is to:

* Perform exploratory data analysis (EDA) on the credit card transaction dataset
* Preprocess the data (handle missing values, check duplicates, encode categorical variables).
* Build and train a basic Machine Learning model to predict whether a transaction is fraudulent.
* Evaluate the model's performance using appropriate metrics (accuracy, confusaion matrix, F1 Score).

## Tools and Technologies

* Programming Language: Python
* Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn and others
* Environment: Google Colab

## 📂 Dataset Description & Data Structure

### 📌 Data Source

* Source: The dataset was obtained from a cedit card organization in the US
* Size: There is one table includes more than 97700 equivalent to the same number of recorded transactions with 24 features
* Format: .csv
  
### 📊 Data Structure

* Table Schema

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

* Data snapshot

  ![image](https://github.com/user-attachments/assets/62db8904-b5ae-4bc3-ae75-3df7b2313c38)

## ⚒️ Main Process

### 1️⃣ Data Cleaning & Preprocessing
* Loading Data

```
  df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Bản sao của mini-project2 .csv')
  df.head()
```

* Data overview, Check missing values, outlier detection and EDA using syntax

```
df.info()
df.describe()
```

![image](https://github.com/user-attachments/assets/fe2df2c8-9036-4513-a0d2-75febcaaf1bb)

![image](https://github.com/user-attachments/assets/11aaa2d3-6999-40e4-918a-8dbc0180f5a4)


=> The dataset has 24 features, none of them have mising values, there are 3 different data types: Float, Interger and Object

### 2️⃣ Feature Engineering

* Create new features by transforming to "trans_time"; "year_of_birth" to fit the format of running model

```
df["trans_time"] = pd.to_datetime(df['trans_date_trans_time']).dt.hour
df["year_of_birth"] = pd.to_datetime(df['dob']).dt.year
```

![image](https://github.com/user-attachments/assets/4aaf5a47-cf1f-424c-8a2f-8a2c8b6fbaf3)

* Endcode features "category" and "gender"
  
```
  list_columns = ['category','gender']
df = pd.get_dummies(df, columns=list_columns,drop_first=True,dtype=int)
```

* Encode states in US by group them into 4 regions  ➡️ 'get_dummies' the regions 'Northeast', 'Midwest', 'South', 'West'

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

![image](https://github.com/user-attachments/assets/17cb2cd9-73b7-40f9-9cf8-3b6d0b5a028d)

* select top 20 jobs have highest fraud cases ➡️ encode top 20 jobs with highest fraud cases and the rest as 'others'

```
 #select top 20 jobs have highest fraud cases
fraud_counts = (df.groupby(['job'])['is_fraud'].sum()).sort_values(ascending=False)
job_counts = df["job"].value_counts()
df1 = pd.concat([fraud_counts,job_counts], axis=1)
df1.columns = ['fraud_counts','job_counts']
df1['fraud_rate']= (df1['fraud_counts']/df1['job_counts'])
df1 = df1.reset_index()
print(df1.head(20))
```

![image](https://github.com/user-attachments/assets/935f82ab-8fa3-4173-ac17-19d9ca2ef159)

```
# entitle top 20 jobs with highest fraud cases and the rest as 'others'
top_categories  = [
    "Materials engineer",
    "Trading standards officer",
    "Naval architect",
    "Exhibition designer",
    "Surveyor, land/geomatics",
    "Mechanical engineer",
    "Prison officer",
    "Quantity surveyor",
    "Audiological scientist",
    "Copywriter, advertising",
    "Senior tax professional/tax inspector",
    "Film/video editor",
    "Scientist, biomedical",
    "Financial trader",
    "Television production assistant",
    "Buyer, industrial",
    "Private music teacher",
    "Podiatrist",
    "Nurse, children's",
    "Magazine features editor"
]
# Create a mapping
category_mapping = {title: title for title in top_categories}
category_mapping['others'] = 'others'

# Apply the mapping
df['encoded_job_title'] = df['job'].apply(lambda x: x if x in top_categories else 'others')

print(df)
# get_dummies the 'encoded_job_title'
encoded_df=pd.get_dummies(df['encoded_job_title'], prefix='job_title',drop_first=True,dtype=int)
df = pd.concat([df, encoded_df], axis=1)

print(df)
```

### 3️⃣ Model Traning
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

### 4️⃣ Model Evaluation
```
[[27050    55]
 [  315  1905]]
Accuracy: 0.99
F1 Score: 0.91
```
An accuracy of 99% sounds excellent at first glance—it means the model is correct 99% of the time. However, accuracy alone can be misleading, especially if the dataset is imbalanced (which seems to be the case here, as there are far more negatives (27,105) than positives (2,220))

The F1 score is given as 0.91, which is the harmonic mean of precision and recall, confirming the metrics are consistent and indicates the model performs well on the positive class despite the imbalance

==> This model 's performance  is quite good overall, with a strong F1 score indicating robustness despite imbalance

## 🔎 Final Conclusion & Recommendations

👉🏻 Based on the insights and findings above, we could have conclusion to stakeholders fro decisiong-making

   ✔️ The project successfully developed a machine learning model using RandomForest and demonstrated strong potential in identifying fraudulent transactions, capturing key patterns through the process of feature engineering, model training, and evaluation
   
   ✔️ The project significantly provided an valuable tool for enhancing the detection system of credit card fraud, enable risk management team to handle large volumes of transactions in real-time.

📌 Key Takeaways:
  
  ✔️ Continue refining features, such as transaction history or user behavior, and explore domain-specific features to better capture fraud patterns.
  
  ✔️ Further fine-tune the RandomForest model by experimenting with hyperparameters. 
  
  ✔️ explore alternative models like XGBoost to assess if they offer better performance in fraud detection.


# Reference sources

* Dataset source: https://docs.google.com/spreadsheets/d/1FWOI4uY2_4xVn0YQnWEdevdbenxucmMs_BSGnoP-s0s/edit?gid=1430542985#gid=1430542985
* Colab notebook: https://colab.research.google.com/drive/1K8kAKJmPaOubSaCstZA_j7dWgNX1T4FL
