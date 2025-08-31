Franchise Abandonment Risk Dashboard
Overview

This project analyzes franchise store performance across multiple brands (HVAC, Plumbing, Electrical) to:

Predict the risk of franchise abandonment (when a store owner stops operating and paying royalties).

Segment stores into three categories: high-performing, consistent, and low-performing.

Forecast biweekly revenue for each store.

It combines financial data, clustering, and machine learning to:

Detect struggling stores.

Forecast future revenue and royalty streams.

Provide dashboards for decision-making and risk monitoring.

Features

🔹 Data Ingestion & Cleaning — handles royalty, revenue, and payment history.

🔹 Performance Segmentation — clusters stores by revenue history using KMeans and DBSCAN.

🔹 Predictive Modeling — logistic regression, random forest, and gradient boosting (XGBoost) models classify abandonment risk.

🔹 Explainability — SHAP values and feature-importance plots interpret the key drivers of risk.

🔹 Interactive Dashboard — store-level visuals (scatter, pie, heatmap) with filtering by brand, state, or owner.

🔹 Business Logic Integration — incorporates rules such as $750 minimum royalty, government loan usage, and store age (years open).



How to Run

Train the models:

revenue_forecast.ipynb   
abandonment_prediction.ipynb  
store_segmentation.ipynb 


Launch the dashboard (Streamlit):

bank_rec.ipynb   
eda_revenue.ipynb   
payments.ipynb  

Launch the dashboard (Streamlit):

test_all_the_models.py

Project Structure
franchise-risk-dashboard/
│-- data/                    # Raw and processed datasets (royalties, revenue, payments)
│-- notebooks/               # Jupyter notebooks (EDA, clustering, model training)
│-- Dashboard/               # Dashboard and modeling code
│   │-- bank_rec.ipynb       # Bank transaction data; highlights bounced transactions
│   │-- eda_revenue.ipynb    # Revenue analysis and reports
│   │-- payments.ipynb       # Payment analysis and reports
│   │-- revenue_forecast.ipynb     # ML models to predict next biweekly revenue
│   │-- abandonment_prediction.ipynb  # ML models to predict possible store abandonment
│   │-- store_segmentation.ipynb  # ML models to cluster stores into 3 performance categories
│   │-- test_all_the_models.py    # Streamlit dashboard entry point
│   │-- model_report.md      # Model comparison and selection summary
│   │-- README.md            # Project documentation

Example Dashboard Views

📊 Store Segmentation (PCA Scatter)
🥧 Risk Distribution by Brand
📈 Revenue & Missed Payments Trend
⚠️ Store-Level Risk Scoring with SHAP

Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss proposed improvements.

License

This project uses synthetic data that closely resembles real company data, but it is not actual company data.
