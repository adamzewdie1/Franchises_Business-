1️⃣ Store Segmentation Model
Dataset

Revenue data for 12 months, 6 months, and 3 months.

600 samples.

Models

KMeans (n_clusters=3, random_state=42)

Achieved the highest silhouette score: 0.32.

Resulted in 3 well-balanced clusters.

DBSCAN (eps=0.5, min_samples=5)

Produced 7 classes but with imbalanced sizes.

Not a suitable choice for this dataset.

Conclusion

KMeans is the best model for this dataset.

It generated 3 balanced and meaningful segments of stores.

2️⃣ Store Abandonment Prediction Model
Dataset

600 samples total:

586 class 0 (not abandoned)

14 class 1 (abandoned)

Highly imbalanced dataset.

Models

Logistic Regression (baseline)

Recall (class 1): 0.75

Precision (class 1): 0.60

XGBoost (with SMOTE + Stratified CV)

Recall (class 1): 0.93

Precision (class 1): 0.32

Accuracy: 95%

Threshold adjustment:

lo2 = model.predict_proba(X)[:,1]
lo3 = (lo2 > 0.1).astype(int)


CatBoost (with SMOTE + Stratified CV)

Recall (class 1): 0.93

Precision (class 1): 0.16

Accuracy: 89%

Conclusion

XGBoost is the best model.

High recall (0.93) ensures nearly all high-risk (abandoning) stores are detected.

Precision is lower → more false alarms, but acceptable in this business context.

📌 Note:

Using class_weight="balanced" in Logistic Regression or scale_pos_weight in XGBoost adjusts for class imbalance.

Formula:

scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)


This means positive errors are penalized ~9× more than negative errors.

In practice: “Don’t ignore the positives — each positive mistake is much more costly.”

3️⃣ Revenue Forecasting Model
Dataset

Biweekly revenue per store.

Filtered by store-level data.

Model

Facebook Prophet

Forecast errors across different horizons:

Horizon	MSE	RMSE	MAE	MAPE	sMAPE	Coverage
14 days	2202.9	46.9	46.9	5.7%	5.9%	1.0
28 days	197.6	14.1	14.1	1.8%	1.9%	1.0
42 days	2306.0	48.0	48.0	6.7%	6.5%	1.0
Key Takeaways

28-day forecast is the most accurate (MAPE ~1.8%).

14-day (~5.7%) and 42-day (~6.7%) still acceptable.

Coverage = 1.0 → Prophet’s prediction intervals always captured the true values.

Model Tuning

Added biweekly seasonality:

sa.add_seasonality(name="biweekly", period=28, fourier_order=3)


Tuned trend flexibility:

sa = Prophet(changepoint_prior_scale=0.5)


Interpretation:

Lower changepoint_prior_scale (0.05 default): very smooth trends, only major shifts create changes.

Higher values (0.5–1.0): allows sharper turns, better for sudden fluctuations.