# EV-Charging-Demand-Prediction-and-Usage-Pattern-Analysis
EV Charging Demand Prediction and Usage Pattern Analysis using Python, Machine Learning, Time Series Forecasting, SHAP, LIME, and Streamlit Dashboard.

## Project Overview
This project focuses on analyzing Electric Vehicle (EV) charging patterns and predicting charging demand using Python and Machine Learning techniques.

The objective is to understand EV charging behavior, identify usage patterns, analyze demand trends, and build a dashboard for visualizing insights.

## Objectives
- Understand and clean EV charging data.
- Perform Exploratory Data Analysis (EDA).
- Identify charging demand patterns.
- Build regression and classification models.
- Perform time series forecasting.
- Explain model predictions using SHAP and LIME.
- Develop an interactive Streamlit dashboard.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- Statsmodels
- SHAP
- LIME
- Streamlit

## Project Workflow
1. Data Understanding
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Regression Analysis
6. Classification – High Demand vs Normal Demand
7. Time Series Forecasting
8. Model Explainability – SHAP and LIME
9. Streamlit Dashboard Development

## Machine Learning Models
- Linear Regression
- Random Forest Regressor
- Classification model for high and normal charging demand
- Exponential Smoothing for time series forecasting

## Model Explainability
SHAP and LIME were used to understand the contribution of different features to model predictions.

## Dashboard
An interactive Streamlit dashboard was developed to display:
- Key Performance Indicators (KPIs)
- Charging demand distribution
- Location-wise and hourly analysis
- Monthly demand trends
- Charger type and company analysis
- Model performance
- Time series forecasts
- SHAP and LIME explanations

## Key Results
- Regression models were evaluated using MAE, MSE, RMSE, and R².
- Classification achieved approximately 58.76% accuracy.
- Time series forecasting achieved an MAE of 0.7615 and RMSE of 1.0123.

## How to Run the Dashboard

Install the required libraries and run:

```bash
pip install pandas numpy matplotlib scikit-learn statsmodels streamlit shap lime
```

```bash
streamlit run ev_charging_dashboard.py
```

## Conclusion
This project demonstrates the application of data analysis, machine learning, time series forecasting, and model explainability to understand EV charging demand and usage patterns.
