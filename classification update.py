
# --------------------------------------------------
# CLASSIFICATION - HIGH DEMAND vs NORMAL DEMAND
# Using Median as the Threshold
# --------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
        roc_auc_score
)

# Load the cleaned dataset
df = pd.read_csv("dataset/cleaned_ChargingRecords.csv")

# Select features for classification
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

# Create demand classes using the median
threshold = df["Demand"].median()
df["Demand_Class"] = (df["Demand"] >= threshold).astype(int)

print("High Demand Threshold:", threshold)

print("\nDemand Class Distribution:")
print(df["Demand_Class"].value_counts())

print("\nDemand Class Percentage:")
print(df["Demand_Class"].value_counts(normalize=True) * 100)

# Prepare training and testing data
X = df[features]
y = df["Demand_Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Random Forest Classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)


# Evaluate classification performance
print("\nClassification Accuracy:")
print(accuracy_score(y_test, y_pred))

# Calculate ROC-AUC score
y_prob = rf_classifier.predict_proba(X_test)[:, 1]

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_prob))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display feature importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_classifier.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nClassification Feature Importance:")
print(importance)

# Save the updated demand classification
df.to_csv("dataset/cleaned_ChargingRecords.csv", index=False)

print("\nUpdated cleaned dataset saved successfully!")
