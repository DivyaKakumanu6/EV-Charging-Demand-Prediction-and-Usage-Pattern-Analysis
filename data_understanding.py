# Import required libraries

import pandas as pd
import numpy as np

# Load the EV charging dataset

df = pd.read_csv("dataset/ChargingRecords.csv")

# Display the first five rows
print(df.head())

# Display dataset dimensions
print("Dataset Shape:", df.shape)

# Display column names
print("Column Names:", df.columns)

# Display information about columns and data types
print(df.info())

# Check missing values in each column
print("Missing Values:")
print(df.isnull().sum())

# Check duplicate records
print("Duplicate Rows:", df.duplicated().sum())

# Display duplicate records
duplicates = df[df.duplicated()]
print("Duplicate Records:")
print(duplicates)

# Remove duplicate records
df = df.drop_duplicates()

# Check the updated dataset shape
print("Shape after removing duplicates:", df.shape)

# Verify duplicate removal
print("Duplicates after cleaning:", df.duplicated().sum())

# Check data types of all columns
print("Column Data Types:")
print(df.dtypes)

# Convert date and time columns into datetime format

df["StartDatetime"] = pd.to_datetime(df["StartDatetime"])
df["EndDatetime"] = pd.to_datetime(df["EndDatetime"])

# Check updated data types
print("Updated Data Types:")
print(df.dtypes)

# Statistical summary of numerical columns
print("Numerical Data Summary:")
print(df[["Duration", "Demand"]].describe())

# Identify records with negative duration

negative_duration = df[df["Duration"] < 0]
print("Negative Duration Records:")
print(negative_duration)
print("Total Negative Duration Records:", len(negative_duration))

# Inspect negative duration records
print(negative_duration[
    ["StartDatetime", "EndDatetime", "Duration", "Demand"]
])

# Remove records with negative charging duration
df = df[df["Duration"] >= 0]

# Verify the cleaning
print("Shape after removing invalid durations:", df.shape)
print("Negative durations remaining:", (df["Duration"] < 0).sum())

# Identify outliers in Duration using the IQR method

Q1 = df["Duration"].quantile(0.25)
Q3 = df["Duration"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

#  Identify and Inspect Upper Outliers in Duration

# Filter charging sessions with duration above the upper limit
outliers = df[df["Duration"] > upper_limit]

# Display the total number of upper outliers
print("Total Upper Outliers:", len(outliers))

# Display the first 10 outlier records
print("\nFirst 10 Upper Outlier Records:")
print(outliers[["Duration", "Demand"]].head(10))

# Find the maximum duration among the outliers
print("\nMaximum Outlier Duration:", outliers["Duration"].max())

#  Inspect the Longest Charging Sessions

# Sort charging sessions by Duration in descending order
longest_sessions = df.sort_values(by="Duration", ascending=False)

# Display the 10 longest charging sessions
print("\nTop 10 Longest Charging Sessions:")
print(longest_sessions[["UserID", "ChargerID", "StartDatetime",
                        "EndDatetime", "Duration", "Demand"]].head(10))

# Check Zero and Negative Values in Demand

# Filter records where Demand is zero or less
invalid_demand = df[df["Demand"] <= 0]

# Count the number of such records
print("\nZero or Negative Demand Records:", len(invalid_demand))

# Display the minimum and maximum Demand values
print("Minimum Demand:", df["Demand"].min())
print("Maximum Demand:", df["Demand"].max())

# Check Unique Values in Categorical Columns

# Display unique values in ChargerType
print("\nUnique Charger Types:")
print(df["ChargerType"].unique())

# Display unique values in ChargerCompany
print("\nUnique Charger Companies:")
print(df["ChargerCompany"].unique())

# Display unique values in Location
print("\nUnique Locations:")
print(df["Location"].unique())

# Display unique values in StartDay
print("\nUnique Start Days:")
print(df["StartDay"].unique())

# Inspect Unique Values in EndDay

# Count the number of unique end dates
print("\nNumber of Unique End Dates:", df["EndDay"].nunique())

# Display the first 10 unique end dates
print("\nFirst 10 Unique End Dates:")
print(df["EndDay"].unique()[:10])

# Check Start and End Date Consistency

# Identify records where the end date is earlier than the start date
invalid_dates = df[df["EndDatetime"] < df["StartDatetime"]]

# Display the number of invalid date records
print("\nInvalid Date Records:", len(invalid_dates))

# Display the first few invalid records, if any
print(invalid_dates[["StartDatetime", "EndDatetime", "Duration"]].head())

# Verify Duration Using Start and End Datetime

# Calculate duration from the difference between end and start time

calculated_duration = (
    df["EndDatetime"] - df["StartDatetime"]
).dt.total_seconds() / 60

# Compare calculated duration with the recorded Duration
duration_mismatch = df[calculated_duration != df["Duration"]]

# Display the number of mismatched records
print("\nDuration Mismatch Records:", len(duration_mismatch))


# Inspect Duration Mismatch Records

print("\nFirst 10 Duration Mismatch Records:")

mismatch_check = df.loc[
    duration_mismatch.index,
    ["StartDatetime", "EndDatetime", "Duration"]
].copy()

mismatch_check["Calculated_Duration"] = calculated_duration.loc[
    duration_mismatch.index
]
print(mismatch_check.head(10))


# Categorize Duration Mismatches

difference = abs(calculated_duration - df["Duration"])

small_mismatches = df[difference == 1]
large_mismatches = df[difference > 1]

print("Small Mismatches (1 minute):", len(small_mismatches))
print("Large Mismatches (More than 1 minute):", len(large_mismatches))


# Inspect Large Duration Mismatches

print("\nFirst 20 Large Duration Mismatches:")

large_mismatch_check = df.loc[
    large_mismatches.index,
    ["StartDatetime", "EndDatetime", "Duration"]
].copy()

large_mismatch_check["Calculated_Duration"] = (
    calculated_duration.loc[large_mismatches.index]
)

large_mismatch_check["Difference"] = (
    difference.loc[large_mismatches.index]
)

print(large_mismatch_check.head(20))

# Check Zero Duration Records

zero_duration = df[df["Duration"] == 0]

print("Total Zero Duration Records:", len(zero_duration))

print("\nFirst 10 Zero Duration Records:")
print(zero_duration[
    ["StartDatetime", "EndDatetime", "Duration", "Demand"]
].head(10))

# Analyze Zero Duration Records

zero_duration = df[df["Duration"] == 0]

same_time = zero_duration[
    zero_duration["StartDatetime"] == zero_duration["EndDatetime"]
]

different_time = zero_duration[
    zero_duration["StartDatetime"] != zero_duration["EndDatetime"]
]

print("Total Zero Duration Records:", len(zero_duration))

print("Same Start and End Time:", len(same_time))

print("Different Start and End Time:", len(different_time))

# Inspect Zero Duration Records with Different Timestamps

print("First 15 Inconsistent Zero Duration Records:")

print(different_time[
    ["StartDatetime", "EndDatetime", "Duration", "Demand"]
].head(15))

# Calculate Charging Duration from Timestamps

df["Calculated_Duration"] = (
    df["EndDatetime"] - df["StartDatetime"]
).dt.total_seconds() / 60

# Display the updated dataset

print("\nUpdated Dataset:")
print(df[[
    "StartDatetime",
    "EndDatetime",
    "Duration",
    "Calculated_Duration"
]].head())

# Verify Calculated Duration

print("Missing Calculated Durations:",
      df["Calculated_Duration"].isnull().sum())

print("Negative Calculated Durations:",
      (df["Calculated_Duration"] < 0).sum())

print("Zero Calculated Durations:",
      (df["Calculated_Duration"] == 0).sum())

print("Calculated Duration Summary:")
print(df["Calculated_Duration"].describe())

# Compare Original and Calculated Zero Durations

print("Original Duration = 0:",
      (df["Duration"] == 0).sum())

print("Calculated Duration = 0:",
      (df["Calculated_Duration"] == 0).sum())

print("Both Durations = 0:",
      ((df["Duration"] == 0) &
       (df["Calculated_Duration"] == 0)).sum())

print("Original Non-zero but Calculated Zero:",
      ((df["Duration"] != 0) &
       (df["Calculated_Duration"] == 0)).sum())

# Inspect Records with Non-zero Original Duration
# but Zero Calculated Duration

timestamp_mismatch = df[
    (df["Duration"] != 0) &
    (df["Calculated_Duration"] == 0)
]

print("Total Records:", len(timestamp_mismatch))

print(timestamp_mismatch[
    ["StartDatetime", "EndDatetime",
     "Duration", "Calculated_Duration", "Demand"]
].head(15))

# Identify Major Duration Differences

difference = abs(
    df["Duration"] - df["Calculated_Duration"]
)

major_mismatches = df[difference > 30]

print("Major Duration Mismatches:", len(major_mismatches))

print("\nFirst 10 Major Mismatches:")
print(major_mismatches[
    ["Duration", "Calculated_Duration", "Demand"]
].head(10))

# Save the cleaned dataset with calculated duration

df.to_csv(
    "dataset/cleaned_ChargingRecords.csv",
    index=False
)

print("Cleaned dataset saved successfully!")
print("Final Dataset Shape:", df.shape)

# Check Missing Values After Initial Cleaning

print("Missing Values in Each Column:")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

# Check Zero Values in Numerical Columns

print("Zero Values in Numerical Columns:")
print((df.select_dtypes(include="number") == 0).sum())

# Inspect Data Types and Unique Values

for col in ["UserID", "ChargerCompany", "ChargerType"]:

    print("\nColumn:", col)
    print("Data Type:", df[col].dtype)
    print("Unique Values:", df[col].nunique())
    print("Value Counts:")
    print(df[col].value_counts().head(10))

    # Check Unique Values in Categorical Columns

categorical_columns = [
    "ChargerCompany",
    "ChargerType",
    "Location",
    "StartDay",
    "EndDay"
]

for col in categorical_columns:
    print("\n", col)
    print(df[col].value_counts())

# Check Date Consistency

df["StartDay"] = pd.to_datetime(df["StartDay"])
df["EndDay"] = pd.to_datetime(df["EndDay"])

start_mismatch = df[
    df["StartDay"].dt.date != df["StartDatetime"].dt.date
]

end_mismatch = df[
    df["EndDay"].dt.date != df["EndDatetime"].dt.date
]

print("Start Day Mismatches:", len(start_mismatch))
print("End Day Mismatches:", len(end_mismatch))

# Save Final Cleaned Dataset

df.to_csv("dataset/cleaned_ChargingRecords.csv", index=False)

print("Final cleaned dataset saved successfully!")
print("Final Dataset Shape:", df.shape)

## EDA

# Charging Sessions by Location
location_demand = df["Location"].value_counts()
print("Charging Sessions by Location:")
print(location_demand)

# Visualizing Charging Sessions by Location

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
location_demand.sort_values().plot(kind="barh")
plt.title("EV Charging Sessions by Location")
plt.xlabel("Number of Charging Sessions")
plt.ylabel("Location")
plt.tight_layout()
plt.show()

# Charging Sessions by Hour

df["StartHour"] = df["StartDatetime"].dt.hour
hourly_demand = df["StartHour"].value_counts().sort_index()
print("Charging Sessions by Hour:")
print(hourly_demand)

# Visualizing Hourly Charging Demand
plt.figure(figsize=(12, 5))
hourly_demand.plot(kind="line", marker="o")
plt.title("EV Charging Sessions by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Charging Sessions")
plt.xticks(range(24))
plt.grid(True)
plt.tight_layout()
plt.show()

# Charging Sessions by Day of Week
df["Weekday"] = df["StartDatetime"].dt.day_name()

weekday_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]
weekday_demand = df["Weekday"].value_counts().reindex(weekday_order)
print(weekday_demand)

# Colourful Weekday Charging Sessions Chart

plt.figure(figsize=(10, 5))
weekday_demand.plot(
    kind="bar",
    color=["skyblue", "orange", "green", "tomato",
           "purple", "gold", "teal"]
)
plt.title("EV Charging Sessions by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Number of Sessions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Charging Sessions by Charger Type

charger_type_demand = df["ChargerType"].value_counts().sort_index()
print("Charging Sessions by Charger Type:")
print(charger_type_demand)

# Visualizing Charger Type Usage
plt.figure(figsize=(7, 5))

charger_type_demand.plot(
    kind="bar",
    color=["skyblue", "coral"]
)
plt.title("EV Charging Sessions by Charger Type")
plt.xlabel("Charger Type Code")
plt.ylabel("Number of Sessions")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Charging Sessions by Charger Company

company_demand = df["ChargerCompany"].value_counts().sort_index()
print("Charging Sessions by Charger Company:")
print(company_demand)

# Visualizing Charger Company Usage

plt.figure(figsize=(7, 5))
company_demand.plot(
    kind="bar",
    color=["mediumseagreen", "cornflowerblue"]
)
plt.title("EV Charging Sessions by Charger Company")
plt.xlabel("Charger Company Code")
plt.ylabel("Number of Sessions")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Charging Duration Distribution

print("Charging Duration Statistics:")
print(df["Calculated_Duration"].describe())

# Visualizing Charging Duration

plt.figure(figsize=(10, 5))
plt.hist(
    df["Calculated_Duration"],
    bins=30,
    color="skyblue",
    edgecolor="black"
)
plt.title("Distribution of EV Charging Duration")
plt.xlabel("Charging Duration (Minutes)")
plt.ylabel("Number of Sessions")
plt.tight_layout()
plt.show()

# Charging Demand Statistics

print("Charging Demand Statistics:")
print(df["Demand"].describe())

# Visualizing Charging Demand

plt.figure(figsize=(10, 5))
plt.hist(
    df["Demand"],
    bins=30,
    color="coral",
    edgecolor="black"
)
plt.title("Distribution of EV Charging Demand")
plt.xlabel("Charging Demand")
plt.ylabel("Number of Sessions")
plt.tight_layout()
plt.show()

# Relationship Between Charging Duration and Demand

correlation = df[
    ["Calculated_Duration", "Demand"]
].corr()

print("Correlation between Duration and Demand:")
print(correlation)

# Visualizing Duration vs Demand

plt.figure(figsize=(9, 5))
plt.scatter(
    df["Calculated_Duration"],
    df["Demand"],
    color="teal",
    alpha=0.4
)
plt.title("Relationship Between Charging Duration and Demand")
plt.xlabel("Charging Duration (Minutes)")
plt.ylabel("Charging Demand")
plt.tight_layout()
plt.show()

# Average Charging Demand by Location

location_avg_demand = (
    df.groupby("Location")["Demand"]
    .mean()
    .sort_values(ascending=False)
)

print("Average Charging Demand by Location:")
print(location_avg_demand)

# Visualizing Average Demand by Location

plt.figure(figsize=(11, 6))

location_avg_demand.sort_values().plot(
    kind="barh",
    color="teal"
)

plt.title("Average EV Charging Demand by Location")
plt.xlabel("Average Charging Demand")
plt.ylabel("Location")
plt.tight_layout()
plt.show()

# Monthly Charging Sessions

df["Month"] = df["StartDatetime"].dt.to_period("M")

monthly_demand = df["Month"].value_counts().sort_index()

print("Monthly Charging Sessions:")
print(monthly_demand)

# Visualizing Monthly Trend

plt.figure(figsize=(12, 5))

monthly_demand.plot(
    kind="line",
    marker="o",
    color="darkorange"
)

plt.title("Monthly EV Charging Sessions Trend")
plt.xlabel("Month")
plt.ylabel("Number of Sessions")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Average Charging Demand by Hour

hourly_avg_demand = df.groupby("StartHour")["Demand"].mean()

print("Average Charging Demand by Hour:")
print(hourly_avg_demand)

# Visualizing Average Hourly Demand

plt.figure(figsize=(12, 5))

hourly_avg_demand.plot(
    kind="line",
    marker="o",
    color="darkorange"
)

plt.title("Average EV Charging Demand by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Charging Demand")
plt.xticks(range(24))
plt.grid(True)
plt.tight_layout()
plt.show()

# Charger Type Performance Analysis

charger_analysis = df.groupby("ChargerType").agg(
    Total_Sessions=("ChargerType", "count"),
    Average_Demand=("Demand", "mean"),
    Average_Duration=("Calculated_Duration", "mean")
)

print("Charger Type Analysis:")
print(charger_analysis)

# Comparing Average Demand by Charger Type

plt.figure(figsize=(7, 5))

charger_analysis["Average_Demand"].plot(
    kind="bar",
    color=["skyblue", "coral"]
)

plt.title("Average Charging Demand by Charger Type")
plt.xlabel("Charger Type Code")
plt.ylabel("Average Demand")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Charger Company Performance Analysis

company_analysis = df.groupby("ChargerCompany").agg(
    Total_Sessions=("ChargerCompany", "count"),
    Average_Demand=("Demand", "mean"),
    Average_Duration=("Calculated_Duration", "mean")
)

print("Charger Company Analysis:")
print(company_analysis)

# Comparing Average Demand by Charger Company

plt.figure(figsize=(7, 5))

company_analysis["Average_Demand"].plot(
    kind="bar",
    color=["skyblue", "coral"]
)

plt.title("Average Charging Demand by Charger Company")
plt.xlabel("Charger Company Code")
plt.ylabel("Average Demand")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Feature Engineering

df["Hour"] = df["StartDatetime"].dt.hour
df["Day"] = df["StartDatetime"].dt.day
df["Month"] = df["StartDatetime"].dt.month
df["Year"] = df["StartDatetime"].dt.year

df["Weekday"] = df["StartDatetime"].dt.dayofweek

df["Is_Weekend"] = (df["Weekday"] >= 5).astype(int)

df["Time_of_Day"] = pd.cut(
    df["Hour"],
    bins=[-1, 5, 11, 16, 20, 23],
    labels=["Night", "Morning", "Afternoon", "Evening", "Late Night"]
)
print("\nNew Features:")
print(df[["Hour", "Day", "Month", "Year",
          "Weekday", "Is_Weekend", "Time_of_Day"]].head())
print("\nDataset Shape:", df.shape)

# Prepare Data for Regression

features = [
    "ChargerCompany",
    "ChargerType",
    "Hour",
    "Day",
    "Month",
    "Year",
    "Weekday",
    "Is_Weekend",
    "Duration"
]

X = df[features]
y = df["Demand"]

print("\nFeature Dataset Shape:", X.shape)
print("Target Dataset Shape:", y.shape)

print("\nMissing Values in Features:")
print(X.isnull().sum().sum())

# Prepare Data for Regression

features = [
    "ChargerCompany",
    "ChargerType",
    "Hour",
    "Day",
    "Month",
    "Year",
    "Weekday",
    "Is_Weekend"
]
X = df[features]
y = df["Demand"]
print("Feature Dataset Shape:", X.shape)
print("Target Dataset Shape:", y.shape)

from sklearn.model_selection import train_test_split

# Split Data into Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)

from sklearn.linear_model import LinearRegression

# Linear Regression Model
lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

print("\nLinear Regression Model Trained Successfully")
print("First 10 Predictions:")
print(y_pred_lr[:10])

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Evaluate Linear Regression Model

mae = mean_absolute_error(y_test, y_pred_lr)
mse = mean_squared_error(y_test, y_pred_lr)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_lr)

print("\nLinear Regression Model Evaluation")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

from sklearn.ensemble import RandomForestRegressor
# Random Forest Regression Model

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Model Trained Successfully")
print("First 10 Predictions:")
print(y_pred_rf[:10])

# Evaluate Random Forest Model

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("\nRandom Forest Model Evaluation")
print("MAE:", mae_rf)
print("MSE:", mse_rf)
print("RMSE:", rmse_rf)
print("R2 Score:", r2_rf)

# Regression Model Comparison

comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [mae, mae_rf],
    "MSE": [mse, mse_rf],
    "RMSE": [rmse, rmse_rf],
    "R2 Score": [r2, r2_rf]
})

print("\nRegression Model Comparison:")
print(comparison)

# Classification: High Demand vs Normal Demand
# Use median demand as the classification threshold
threshold = df["Demand"].median()
df["Demand_Class"] = (df["Demand"] >= threshold).astype(int)
print("\nHigh Demand Threshold:", threshold)

print("\nDemand Class Distribution:")
print(df["Demand_Class"].value_counts())

print("\nDemand Class Percentage:")
print(df["Demand_Class"].value_counts(normalize=True) * 100)

# Prepare Data for Classification

X_class = df[features]
y_class = df["Demand_Class"]

print("\nClassification Features Shape:", X_class.shape)
print("Classification Target Shape:", y_class.shape)

# Split Classification Data

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class,
    y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)
print("\nClassification Training Shape:", X_train_class.shape)
print("Classification Testing Shape:", X_test_class.shape)

# Classification Model - Random Forest

from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_classifier.fit(X_train_class, y_train_class)
y_pred_class = rf_classifier.predict(X_test_class)
print("\nClassification Model Training Completed")

# Classification Model Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("\nClassification Accuracy:", accuracy_score(y_test_class, y_pred_class))
print("\nClassification Report:")
print(classification_report(y_test_class, y_pred_class))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_class, y_pred_class))

# Classification Feature Importance

importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_classifier.feature_importances_
})
importance = importance.sort_values(
    by="Importance",
    ascending=False
)
print("\nClassification Feature Importance:")
print(importance)

# Time Series Analysis - Daily Demand Trend

daily_demand = df.groupby(
    df["StartDatetime"].dt.date
)["Demand"].mean().reset_index()

daily_demand.columns = ["Date", "Average_Demand"]

daily_demand["Date"] = pd.to_datetime(daily_demand["Date"])

# Calculate 7-day moving average

daily_demand["7_Day_Moving_Average"] = (
    daily_demand["Average_Demand"].rolling(window=7).mean()
)

print("\nDaily Demand Analysis:")
print(daily_demand.head())

print("\nTotal Days:", len(daily_demand))

# Plot Daily Demand Trend

plt.figure(figsize=(12, 5))

plt.plot(
    daily_demand["Date"],
    daily_demand["Average_Demand"],
    label="Daily Average Demand",
    alpha=0.5
)
plt.plot(
    daily_demand["Date"],
    daily_demand["7_Day_Moving_Average"],
    label="7-Day Moving Average",
    linewidth=2
)
plt.title("Daily EV Charging Demand Trend")
plt.xlabel("Date")
plt.ylabel("Average Demand")
plt.legend()
plt.grid(True)
plt.show()

# Monthly Demand Analysis

monthly_demand = df.groupby(
    df["StartDatetime"].dt.to_period("M")
)["Demand"].mean().reset_index()

monthly_demand.columns = ["Month", "Average_Demand"]

monthly_demand["Month"] = monthly_demand["Month"].astype(str)

print("\nMonthly Average Demand:")
print(monthly_demand)

plt.figure(figsize=(12, 5))
plt.plot(
    monthly_demand["Month"],
    monthly_demand["Average_Demand"],
    marker="o"
)
plt.title("Monthly Average EV Charging Demand")
plt.xlabel("Month")
plt.ylabel("Average Demand")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Save Final Dataset with Engineered Features

df.to_csv("dataset/cleaned_ChargingRecords.csv", index=False)

print("\nFinal Dataset Saved Successfully!")
print("Final Dataset Shape:", df.shape)
print("Final Columns:", df.columns.tolist())

