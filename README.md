# Predicting Financial Sector Category and Risk
This was a semester long project conducted in the fall of 2023. This analysis includes data loading, EDA, & involves a classification NLP and regression machine learning models that have hyperparameter tuning applied.
Completed fully within python.  

This was a Kaggle Competition style project set up by the professor between two of the classes he was teaching - my regression model returned the best score.  

## Project Overview

This assignment is critical for maintaining the stability of the financial markets, especially in the aftermath of significant events like the SVB collapse. 
The aim of this project is to use AI/ML tools to identitfy financial firms at risk of failure. The project is broken down into 3 parts: 


**Exploratory Data Analysis**: You have gathered data on publicly traded companies, some of which operate in the Financial Sector, while others do not.

**Classification Task**: The initial step is to categorize these companies to determine which are Financial firms under the Fed's supervision. In the training set, a human expert has already classified firms as 'FinancialSector' or not. This classification is missing in the test set, and your objective is to predict this categorization for the test set and automate the process for future datasets.

**Risk Prediction**: Subsequently, your aim is to predict 'FinancialRisk'. This involves identifying firms not in the Financial Sector as zero risk and assigning a risk score (with 1 being the highest) to firms within the Financial Sector, based on the manual calculations by a human expert in the training set. Your goal is to predict this for the test set and automate the process for future assessments.

The pdf of the presentation for this project can be found within this repository and follows as shown:  

**Presentation**: 
* Introduction 
* EDA 
* Feature selection 
* Model Selection 
   * Hyper-Parameter tuning 
   * Model variance vs bias 
* Final model 
* Feature Importance/Explainability 
* Conclusion 

## Methodology
This project was broken down into two parts; a classification of financial sector and prediction analysis of financial risk.   

### Classification Model to Determine Financial Sector
**Feature Selection**: A randomforestregressor was used to determine 5 of the most important features.  
**Model Selection**: A vectorizer was used to convert the text features (such as company description) into scores that our model could read numerically.
A randomforestclassifier model was used to determine the classification of financial sector.   
**Hyperparameter Tuning**: Gridsearch was then used for hyper parameter tuning 

### Regression Model to Predict Financial Risk
**Feature Selection**: A randomforestregressor was also used to determine 5 of the most important features, as well as a heatmap analysis. 
**Model Selection**: There were no text features for this portion. A randomforestregressor and XGBoost model were both examined. The XGBoost Model gave better results. 
**Hyperparameter Tuning**: Gridsearch was also used as hyperparameter tuning

## Data Description
The dataset being used contains 73 features across 500 companies.
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
