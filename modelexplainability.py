
# --------------------------------------------------
# MODEL EXPLAINABILITY - SHAP AND LIME
# --------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import shap
from lime.lime_tabular import LimeTabularExplainer

# 1. Load the cleaned dataset
df = pd.read_csv("dataset/cleaned_ChargingRecords.csv")

# 2. Select the same features used in classification
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
y = df["Demand_Class"]

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Train the Random Forest Classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_classifier.fit(X_train, y_train)

print("Random Forest model trained successfully!")

# --------------------------------------------------
# 5. SHAP - Explain overall feature contribution
# --------------------------------------------------

# Select a small sample to make SHAP faster
X_sample = X_test.sample(
    n=min(100, len(X_test)),
    random_state=42
)

# Create SHAP Tree Explainer
shap_explainer = shap.TreeExplainer(rf_classifier)

# Calculate SHAP values
shap_values = shap_explainer.shap_values(X_sample)

# Handle different SHAP output formats
if isinstance(shap_values, list):
    shap_plot_values = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_plot_values = shap_values[:, :, 1]
else:
    shap_plot_values = shap_values

# Display SHAP summary plot
shap.summary_plot(
    shap_plot_values,
    X_sample,
    show=False
)

plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.savefig("shap_summary.png", bbox_inches="tight")
plt.show()

print("SHAP analysis completed!")

# --------------------------------------------------
# 6. LIME - Explain one individual prediction
# --------------------------------------------------

lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=features,
    class_names=["Normal Demand", "High Demand"],
    mode="classification"
)

# Select one test record
sample_record = X_test.iloc[0]

# Explain the model prediction for this record
lime_explanation = lime_explainer.explain_instance(
    sample_record.values,
    rf_classifier.predict_proba,
    num_features=len(features)
)

# Display LIME explanation
print("\nLIME Explanation:")
for feature, contribution in lime_explanation.as_list():
    print(feature, ":", contribution)

# Save LIME explanation as an HTML file
lime_explanation.save_to_file("lime_explanation.html")

print("\nLIME analysis completed!")
print("SHAP and LIME results saved successfully!")
