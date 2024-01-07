
Introduction:

In March 2023, as a fresh graduate, you've embarked on an exciting journey with the Federal Reserve (the Fed). The Fed's crucial task is to maintain the overall stability of the financial system, including the regulation of key non-bank financial institutions.

SVB Collapse - March 2023:

Bank Run and Failure: On March 10, 2023, Silicon Valley Bank (SVB) collapsed after a bank run, following its announcement to sell assets and raise funds due to significant losses. This led to a huge withdrawal of customer funds, and the California Department of Financial Protection and Innovation subsequently placed SVB under the FDIC's receivership.

Background: Founded in 1983, SVB, a major player in the tech sector, saw its deposits skyrocket from \$62 billion in 2020 to \$124 billion in 2021. The bank's strategy to invest in long-term Treasury bonds backfired as rising interest rates led to a drop in bond values, causing substantial unrealized losses.

Impact: The collapse heavily impacted startups and larger tech firms worldwide, disrupting access to funds and affecting various industries.
Regulatory and Risk Management Issues: Critiques arose over the bank's risk management, citing reduced stress testing under 2018's Economic Growth, Regulatory Relief, and Consumer Protection Act. A 2021 Federal Reserve review highlighted deficiencies in SVB's risk management.

Recovery Efforts: The FDIC, supported by the Treasury, insured all SVB deposits to mitigate broader financial instability. SVB's assets were transferred to Silicon Valley Bridge Bank, N.A., and later acquired by First Citizens BancShares.

Your Task at the Fed:

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
