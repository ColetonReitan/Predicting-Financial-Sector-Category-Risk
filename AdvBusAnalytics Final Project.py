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
