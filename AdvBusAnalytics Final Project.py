# -*- coding: utf-8 -*-
'''
## Introduction:

In March 2023, as a fresh graduate, you've embarked on an exciting journey with the Federal Reserve (the Fed). The Fed's crucial task is to maintain the overall stability of the financial system, including the regulation of key non-bank financial institutions.

## SVB Collapse - March 2023:

Bank Run and Failure: On March 10, 2023, Silicon Valley Bank (SVB) collapsed after a bank run, following its announcement to sell assets and raise funds due to significant losses. This led to a huge withdrawal of customer funds, and the California Department of Financial Protection and Innovation subsequently placed SVB under the FDIC's receivership.

Background: Founded in 1983, SVB, a major player in the tech sector, saw its deposits skyrocket from \$62 billion in 2020 to \$124 billion in 2021. The bank's strategy to invest in long-term Treasury bonds backfired as rising interest rates led to a drop in bond values, causing substantial unrealized losses.

Impact: The collapse heavily impacted startups and larger tech firms worldwide, disrupting access to funds and affecting various industries.
Regulatory and Risk Management Issues: Critiques arose over the bank's risk management, citing reduced stress testing under 2018's Economic Growth, Regulatory Relief, and Consumer Protection Act. A 2021 Federal Reserve review highlighted deficiencies in SVB's risk management.

Recovery Efforts: The FDIC, supported by the Treasury, insured all SVB deposits to mitigate broader financial instability. SVB's assets were transferred to Silicon Valley Bridge Bank, N.A., and later acquired by First Citizens BancShares.

## Your Task at the Fed:

Using AI/ML tools, your assignment is to identify financial firms at risk.

* Exploratory Data Analysis: You have gathered data on publicly traded companies, some of which operate in the Financial Sector, while others do not.

* Classification Task: The initial step is to categorize these companies to determine which are Financial firms under the Fed's supervision. In the training set, a human expert has already classified firms as 'FinancialSector' or not. This classification is missing in the test set, and your objective is to predict this categorization for the test set and automate the process for future datasets.

* Risk Prediction: Subsequently, your aim is to predict 'FinancialRisk'. This involves identifying firms not in the Financial Sector as zero risk and assigning a risk score (with 1 being the highest) to firms within the Financial Sector, based on the manual calculations by a human expert in the training set. Your goal is to predict this for the test set and automate the process for future assessments.

* Presentation:
  * Introduction
  * EDA
  * Feature selection
  * Model Selection
    * Hyper-Parameter tuning
    * Model variance vs bias
  * Final model
  * Feature Importance/Explainability
  * Conclusion



This assignment is critical for maintaining the stability of the financial markets, especially in the aftermath of significant events like the SVB collapse.

Here's a data description for your dataset:

* url: The web address providing earnings call transcript
* call_transcript: Text of earnings call transcript. Features can be engineered via Natural Language Processing techniques.
* VWAP: Volume Weighted Average Price of the company's stock.
* exchangeCountry: Country where the company's stock is listed.
* securityType: Type of security issued by the company (e.g., stock, bond).
* CIK: Central Index Key, a unique identifier assigned by the SEC.
* name: The name of the company.
* securityID: A unique identifier for the security.
* incorporationCountry: Country where the company is incorporated.
* exchangeName: Name of the stock exchange where the company is listed.
* exchangeID: Unique identifier for the stock exchange.
* Accrual Ratio: Financial metric indicating the quality of earnings.
* Assets: Total assets of the company.
* B/P: Book-to-Price ratio.
* CF/P: Cash Flow to Price ratio.
* Capital Expenditure: Funds used by a company to acquire or upgrade physical assets.
* Cash: Total cash holdings.
* Debt/Equity: Ratio comparing the company's total liabilities to its shareholder equity.
* Depreciation: The allocation of the cost of assets over time.
* Dividend: Dividend payments to shareholders.
* E/P: Earnings to Price ratio.
* EBIT: Earnings Before Interest and Taxes.
* EBIT/P: EBIT to Price ratio.
* EBIT/TEV: EBIT to Total Enterprise Value ratio.
* Earnings: Total earnings of the company.
* Earnings Growth (1Y to 5Y): Earnings growth over 1 to 5-year periods.
* Earnings Variability: Fluctuations in earnings over time.
* Equity: Shareholder's equity in the company.
* FCF: Free Cash Flow.
* FCF/P: Free Cash Flow to Price ratio.
* Income Tax: Total income tax paid.
* Interest Expense: Costs of interest paid on debts.
* Long Liabilities: Long-term financial obligations.
* Long Term Debt: Debt that is due in more than one year.
* Market Cap: Market capitalization.
* Minority Interest: Portion of subsidiaries not owned by the parent company.
* Operating Cash Flow: Cash generated from operating activities.
* Operating Expense: Expenses incurred from normal business operations.
* Operating Income: Profit realized from business operations.
* Operating Income Before Depreciation: Operating income before accounting for depreciation.
* Operating Margin: Operating income divided by net sales.
* Preferred Stock: Stock with priority over common stock in dividend payment.
* Profit Margin: Net income divided by revenue.
* R&D: Research and Development expenses.
* R&D/Sales: Ratio of R&D expenses to sales.
* ROA: Return on Assets.
* ROE: Return on Equity.
* S/P: Sales to Price ratio.
* SG&A: Selling, General, and Administrative expenses.
* SG&A/Sales: Ratio of SG&A to sales.
* Sales: Total sales revenue.
* Sales Growth (1Y to 5Y): Sales growth over 1 to 5-year periods.
* Sales Variability: Fluctuations in sales over time.
* Short Term Debt: Debt that is due within one year.
* TEV: Total Enterprise Value.
* Working Capital: Capital used in day-to-day operations.
* businessDescription: A brief description of the company's business.
* close: Closing price of the company's stock. This is useful if you need to engineer a feature, e.g. to calculate Sales-To-Price, by dividing 'Sales' by 'close'.
* dividendFactor: Factor determining the dividend payout.
* fiscalDint: Fiscal year distinguishing information.
* floatShares: Number of shares available for public trading.
* outstandingShares: Total shares currently held by shareholders.
* shortInterestFloat: Percentage of float shares held as short positions.

Target Variables: (training set only, you predict on the test set and submit on Kaggle)

* FinancialSector: Classification of whether the company is in the financial sector.
* FinancialRisk: Risk level assigned to the company, based on its financial profile.

'''
! gdown 1r3jZglYXj3Xi_v4I--86fpzLoqFk3PVA
! gdown 1_2aTbleEh-kRocoEgHjaVsjYrO9jP1Os

import pandas as pd
df_train = pd.read_parquet('20231124_Financial_Risk_Project_train.parquet')
df_test = pd.read_parquet('20231124_Financial_Risk_Project_test_public.parquet')

df_train.shape, df_test.shape

# Kaggle (Categorization)

1. Jordan Adelphi. (2023). Navigating Financial Instability (Categorization). Kaggle. https://kaggle.com/competitions/navigating-financial-instability

2. Jordan Adelphi. (2023). Navigating Financial Instability (Regression). Kaggle. https://kaggle.com/competitions/navigating-financial-instability-regression


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

#EDA of Financial Sector



# Check basic information
print(df_train.info())
print(df_train.describe())
print(df_train.head())

import seaborn as sns
import matplotlib.pyplot as plt

# Visualize the distribution of the target variable
sns.countplot(x='FinancialSector', data=df_train)
plt.title('Distribution of Financial Sector')
plt.show()

# Analyze text feature
df_train['businessDescription'].str.len().hist()
plt.title('Distribution of Business Description Length')
plt.show()

"""Feature Selection"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Assuming df_train is your training dataset

# Select relevant features for analysis
selected_features = [
    'VWAP', 'Accrual Ratio', 'B/P', 'CF/P', 'Capital Expenditure', 'Cash', 'Debt/Equity', 'Depreciation',
    'Dividend', 'E/P', 'EBIT', 'EBIT/P', 'EBIT/TEV', 'Earnings','Earnings Growth (1Y)','Earnings Growth (2Y)','Earnings Growth (3Y)', 'Earnings Growth (4Y)', 'Earnings Growth (5Y)', 'Earnings Variability',
    'Equity', 'FCF', 'FCF/P', 'Income Tax', 'Interest Expense', 'Long Liabilities', 'Long Term Debt', 'Market Cap',
    'Operating Cash Flow', 'Operating Expense', 'Operating Income', 'Operating Income Before Depreciation',
    'Operating Margin', 'Profit Margin', 'R&D', 'ROA', 'ROE', 'S/P', 'SG&A', 'Sales', 'Sales Growth (1Y)','Sales Growth (2Y)','Sales Growth (3Y)', 'Sales Growth (4Y)', 'Sales Growth (5Y)',
    'Sales Variability', 'Short Term Debt', 'TEV', 'Working Capital'
]

# Extract selected features in the training set
df_train_selected = df_train[selected_features + ['FinancialSector']]

# Handle missing values with imputation
numeric_imputer = SimpleImputer(strategy='mean')
df_train_selected[selected_features] = numeric_imputer.fit_transform(df_train_selected[selected_features])

# Separate features and target variable
X_train = df_train_selected[selected_features]
y_train = df_train_selected['FinancialSector']

# Standardize numeric features
scaler = StandardScaler()
X_train[selected_features] = scaler.fit_transform(X_train[selected_features])

# Select K best features based on correlation with the target variable
selector = SelectKBest(score_func=f_regression, k=15)  # You can adjust the value of k as needed
X_train_selected = selector.fit_transform(X_train, y_train)

# Get the selected feature names
selected_feature_names = X_train.columns[selector.get_support()]

# Train a model to see feature importance
model = RandomForestRegressor(random_state=42)
model.fit(X_train_selected, y_train)

# Get feature importances
feature_importances = model.feature_importances_

# Create a DataFrame with feature names and importances
feature_importance_df = pd.DataFrame({'Feature': selected_feature_names, 'Importance': feature_importances})

# Display or save the feature importance DataFrame
print(feature_importance_df)

"""Classification"""

# Select features; will be using all features with .1 or higher importance score as well as businessDescription
#Feature Selection

selected_features = ['businessDescription', 'Depreciation', 'Long Liabilities', 'SG&A', 'Working Capital']

# Create feature matrix and target variable
X_train = df_train[selected_features]
y_train = df_train['FinancialSector']
X_test = df_test[selected_features]

from sklearn.feature_extraction.text import TfidfVectorizer

# Text feature processing
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_text = vectorizer.fit_transform(X_train['businessDescription'])
X_test_text = vectorizer.transform(X_test['businessDescription'])

from scipy.sparse import hstack

# Combine features
X_train_combined = hstack([X_train_text, X_train.drop('businessDescription', axis=1)])
X_test_combined = hstack([X_test_text, X_test.drop('businessDescription', axis=1)])

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_combined, y_train, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_split, y_train_split)

# Validate the model
val_predictions = clf.predict(X_val_split)
accuracy = accuracy_score(y_val_split, val_predictions)
print(f'Validation Accuracy: {accuracy}')

# Make predictions on the test set
test_predictions = clf.predict(X_test_combined)

# Assign predictions to the test dataset
df_test['FinancialSector'] = test_predictions

df_test_dropcall = df_test.drop('call_transcript', axis=1)
df_test_dropcall.to_csv('Classification_try2.csv', index=True)

"""Fine Tuning Hyper Paremeters

Using Grid Search
"""

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the grid search model
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=3,  # Adjust the number of folds as needed
                           scoring='accuracy',
                           verbose=2,
                           n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train_combined, y_train)

# Print the best parameters
print("Best Parameters:", grid_search.best_params_)

best_params = grid_search.best_params_

final_model = RandomForestClassifier(random_state=42, **best_params)
final_model.fit(X_train_combined, y_train)

# Make predictions on the test set
test_predictions = final_model.predict(X_test_combined)

# Assign predictions to the test dataset
df_test['FinancialSector'] = test_predictions

df_test_dropcall = df_test.drop('call_transcript', axis=1)
df_test_dropcall.to_csv('Classification_gridsearch.csv', index=True)

"""Model Assessment"""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# the true labels for the test set in y_test
y_test = df_test['FinancialSector']

# Calculate predictions
test_predictions = final_model.predict(X_test_combined)

# Evaluate accuracy
accuracy = accuracy_score(y_test, test_predictions)
print(f"Accuracy: {accuracy:.4f}")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, test_predictions))

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, test_predictions)
print("\nConfusion Matrix:")
print(conf_matrix)

"""Moving into Kaggle Regression Competition

EDA of Financial Risk
"""

# Display basic statistics of 'FinancialRisk'
print(df_train['FinancialRisk'].describe())

# Check for class imbalance
class_counts = df_train['FinancialRisk'].value_counts()
print('Class Counts:\n', class_counts)

"""Feature Selection"""

# Visualize the correlation between 'FinancialRisk' and other numeric features
plt.figure(figsize=(12, 8))
sns.heatmap(df_train.corr()[['FinancialRisk']].sort_values(by='FinancialRisk', ascending=False),
            annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation with FinancialRisk')
plt.show()

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Assuming df_train is your training dataset

# Select relevant features for analysis
selected_features = [
    'VWAP', 'Accrual Ratio', 'B/P', 'CF/P', 'Capital Expenditure', 'Cash', 'Debt/Equity', 'Depreciation',
    'Dividend', 'E/P', 'EBIT', 'EBIT/P', 'EBIT/TEV', 'Earnings','Earnings Growth (1Y)','Earnings Growth (2Y)','Earnings Growth (3Y)', 'Earnings Growth (4Y)', 'Earnings Growth (5Y)', 'Earnings Variability',
    'Equity', 'FCF', 'FCF/P', 'Income Tax', 'Interest Expense', 'Long Liabilities', 'Long Term Debt', 'Market Cap',
    'Operating Cash Flow', 'Operating Expense', 'Operating Income', 'Operating Income Before Depreciation',
    'Operating Margin', 'Profit Margin', 'R&D', 'ROA', 'ROE', 'S/P', 'SG&A', 'Sales', 'Sales Growth (1Y)','Sales Growth (2Y)','Sales Growth (3Y)', 'Sales Growth (4Y)', 'Sales Growth (5Y)',
    'Sales Variability', 'Short Term Debt', 'TEV', 'Working Capital'
]

# Extract selected features in the training set
df_train_selected = df_train[selected_features + ['FinancialRisk']]

# Handle missing values with imputation
numeric_imputer = SimpleImputer(strategy='mean')
df_train_selected[selected_features] = numeric_imputer.fit_transform(df_train_selected[selected_features])

# Separate features and target variable
X_train = df_train_selected[selected_features]
y_train = df_train_selected['FinancialRisk']

# Standardize numeric features
scaler = StandardScaler()
X_train[selected_features] = scaler.fit_transform(X_train[selected_features])

# Select K best features based on correlation with the target variable
selector = SelectKBest(score_func=f_regression, k=15)  # You can adjust the value of k as needed
X_train_selected = selector.fit_transform(X_train, y_train)

# Get the selected feature names
selected_feature_names = X_train.columns[selector.get_support()]

# Train a model to see feature importance
model = RandomForestRegressor(random_state=42)
model.fit(X_train_selected, y_train)

# Get feature importances
feature_importances = model.feature_importances_

# Create a DataFrame with feature names and importances
feature_importance_df = pd.DataFrame({'Feature': selected_feature_names, 'Importance': feature_importances})

# Display or save the feature importance DataFrame
print(feature_importance_df)

"""Running Regression with RandomForest"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Assuming df_train contains the target variable 'FinancialRisk'
# and df_test is missing the 'FinancialRisk' column

# Selecting features with moderate or higher importance scores
selected_features = ['VWAP', 'Long Liabilities', 'Short Term Debt']

# Concatenating selected features with additional necessary columns
selected_features = ['FinancialSector'] + selected_features
df_train_selected = df_train[selected_features + ['FinancialRisk']].copy()
df_test_selected = df_test[selected_features].copy()

# Handling missing values using SimpleImputer
numeric_imputer = SimpleImputer(strategy='mean')
df_train_selected[selected_features] = numeric_imputer.fit_transform(df_train_selected[selected_features])
df_test_selected[selected_features] = numeric_imputer.transform(df_test_selected[selected_features])

# Standardizing numeric features using StandardScaler
scaler = StandardScaler()
df_train_selected[selected_features] = scaler.fit_transform(df_train_selected[selected_features])
df_test_selected[selected_features] = scaler.transform(df_test_selected[selected_features])

# Extracting features and target variable
X_train = df_train_selected[selected_features]
y_train = df_train_selected['FinancialRisk']
X_test = df_test_selected[selected_features]

# Train a RandomForestRegressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
test_predictions = model.predict(X_test)

# Set FinancialRisk to 0 for companies not in the financial sector
test_predictions[df_test['FinancialSector'] == 0] = 0

# Assigning risk score between 0 and 1
test_predictions = test_predictions.clip(0, 1)

# Assigning risk score between 0 and 1
df_test['FinancialRisk'] = test_predictions

# Display or save the updated df_test with the predicted 'FinancialRisk' column
print(df_test[['FinancialSector', 'FinancialRisk']])

# Evaluate the model performance
mse = mean_squared_error(y_train, model.predict(X_train))
r2 = r2_score(y_train, model.predict(X_train))
cv_scores = cross_val_score(model, X_train, y_train, cv=5)

print(f'Mean Squared Error on Training Set: {mse}')
print(f'R-squared on Training Set: {r2}')
print(f'Cross-validated R-squared scores: {cv_scores}')

df_test_dropcall = df_test.drop('call_transcript', axis=1)
df_test_dropcall.to_csv('Regression_Random_Forest.csv', index=True)

"""Running Regression with XGboost"""

import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Assuming df_train contains the target variable 'FinancialRisk'
# and df_test is missing the 'FinancialRisk' column

# Selecting features with moderate or higher importance scores
selected_features = ['VWAP','Long Liabilities', 'Short Term Debt', 'Preferred Stock', 'Depreciation']

# Concatenating selected features with additional necessary columns
selected_features = ['FinancialSector'] + selected_features
df_train_selected = df_train[selected_features + ['FinancialRisk']].copy()
df_test_selected = df_test[selected_features].copy()

# Handling missing values using SimpleImputer
numeric_imputer = SimpleImputer(strategy='mean')
df_train_selected[selected_features] = numeric_imputer.fit_transform(df_train_selected[selected_features])
df_test_selected[selected_features] = numeric_imputer.transform(df_test_selected[selected_features])

# Standardizing numeric features using StandardScaler
scaler = StandardScaler()
df_train_selected[selected_features] = scaler.fit_transform(df_train_selected[selected_features])
df_test_selected[selected_features] = scaler.transform(df_test_selected[selected_features])

# Extracting features and target variable
X_train = df_train_selected[selected_features]
y_train = df_train_selected['FinancialRisk']
X_test = df_test_selected[selected_features]


# Step 4: Model Selection
model = xgb.XGBRegressor(random_state=42)

# Step 5: Model Training
model.fit(X_train, y_train)

# Use the trained model to predict the 'FinancialRisk' for df_test
test_predictions = model.predict(X_test)

# Set FinancialRisk to 0 for companies not in the financial sector
test_predictions[df_test['FinancialSector'] == 0] = 0

# Assigning risk score between 0 and 1
test_predictions = test_predictions.clip(0, 1)

# Append the modified 'FinancialRisk' column to df_test
df_test['FinancialRisk'] = test_predictions

# Display or save the updated df_test with the corrected 'FinancialRisk' column
print(df_test[['FinancialSector', 'FinancialRisk']])

# Evaluate the model performance
mse = mean_squared_error(y_train, model.predict(X_train))
r2 = r2_score(y_train, model.predict(X_train))
cv_scores = cross_val_score(model, X_train, y_train, cv=5)

print(f'Mean Squared Error on Training Set: {mse}')
print(f'R-squared on Training Set: {r2}')
print(f'Cross-validated R-squared scores: {cv_scores}')

df_test_dropcall = df_test.drop('call_transcript', axis=1)
df_test_dropcall.to_csv('Regression_XGBoost.csv', index=True)

"""Fine tuning XGBoost HyperParameters"""

import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Step 4: Model Selection
model = xgb.XGBRegressor(random_state=42)

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
}

# Step 5: Hyperparameter Tuning
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Use the best model to predict the 'FinancialRisk' for df_test
best_model = grid_search.best_estimator_
test_predictions = best_model.predict(X_test)

# Set FinancialRisk to 0 for companies not in the financial sector
test_predictions[df_test['FinancialSector'] == 0] = 0

# Assigning risk score between 0 and 1
test_predictions = test_predictions.clip(0, 1)

# Append the modified 'FinancialRisk' column to df_test
df_test['FinancialRisk'] = test_predictions

# Display or save the updated df_test with the corrected 'FinancialRisk' column
print(df_test[['FinancialSector', 'FinancialRisk']])

# Analyze feature importance
feature_importances = best_model.feature_importances_
feature_names = X_train.columns

# Create a DataFrame with feature names and importances
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plotting the top features
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()

feature_importance_df.head()

import matplotlib.pyplot as plt

# Assuming y_train is a pandas Series or NumPy array
actual_values = y_train
predicted_values = best_model.predict(X_train)

plt.scatter(actual_values, predicted_values, c='blue', label='Actual vs. Predicted')
plt.plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], linestyle='--', color='red', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Plot')
plt.legend()
plt.show()

# Evaluate the model performance
mse = mean_squared_error(y_train, best_model.predict(X_train))
r2 = r2_score(y_train, best_model.predict(X_train))
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)

print(f'Mean Squared Error on Training Set: {mse}')
print(f'R-squared on Training Set: {r2}')
print(f'Cross-validated R-squared scores: {cv_scores}')



df_test_dropcall = df_test.drop('call_transcript', axis=1)
df_test_dropcall.to_csv('Regression_XGBoost_FineTuned2.csv', index=True)
