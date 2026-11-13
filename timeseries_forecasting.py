
# --------------------------------------------------
# TIME SERIES FORECASTING - EXPONENTIAL SMOOTHING
# --------------------------------------------------

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the cleaned EV charging dataset
df = pd.read_csv("dataset/cleaned_ChargingRecords.csv")

# Convert StartDatetime into datetime format
df["StartDatetime"] = pd.to_datetime(df["StartDatetime"])

# Calculate daily average charging demand
daily_demand = (
    df.groupby(df["StartDatetime"].dt.date)["Demand"]
    .mean()
    .reset_index()
)

daily_demand.columns = ["Date", "Average_Demand"]
daily_demand["Date"] = pd.to_datetime(daily_demand["Date"])

# Set Date as the time-series index
demand_series = daily_demand.set_index("Date")["Average_Demand"]
demand_series = demand_series.sort_index()

# Split data: use the last 30 days for testing
train = demand_series.iloc[:-30]
test = demand_series.iloc[-30:]

print("Training Records:", len(train))
print("Testing Records:", len(test))

# Train the Exponential Smoothing model
model = ExponentialSmoothing(
    train,
    trend="add",
    seasonal=None,
    initialization_method="estimated"
)

fitted_model = model.fit()

# Forecast demand for the testing period
test_forecast = fitted_model.forecast(len(test))

# Evaluate forecasting performance
mae = mean_absolute_error(test, test_forecast)
mse = mean_squared_error(test, test_forecast)
rmse = np.sqrt(mse)

print("\nEXPONENTIAL SMOOTHING MODEL EVALUATION")
print("MAE:", mae)
print("RMSE:", rmse)

# Compare actual and forecasted demand
comparison = pd.DataFrame({
    "Actual_Demand": test,
    "Forecasted_Demand": test_forecast
})

print("\nActual vs Forecasted Demand:")
print(comparison.head(10))

# Visualize actual and forecasted demand
plt.figure(figsize=(12, 5))

plt.plot(train.index, train, label="Training Demand")
plt.plot(test.index, test, label="Actual Demand")
plt.plot(test.index, test_forecast, label="Forecasted Demand")

plt.title("EV Charging Demand - Exponential Smoothing Forecast")
plt.xlabel("Date")
plt.ylabel("Average Demand")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Train the model using all available data
final_model = ExponentialSmoothing(
    demand_series,
    trend="add",
    seasonal=None,
    initialization_method="estimated"
).fit()

# Forecast charging demand for the next 7 days
future_forecast = final_model.forecast(7)
print("\nNEXT 7 DAYS CHARGING DEMAND FORECAST:")
print(future_forecast)

# Display forecasting accuracy

print("\nFINAL FORECASTING ACCURACY")
print("MAE:", mae)
print("RMSE:", rmse)


