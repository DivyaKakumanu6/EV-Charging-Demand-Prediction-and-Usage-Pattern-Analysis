
# Import Libraries

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# Page Configuration

st.set_page_config(
    page_title="EV Charging Demand Dashboard",
    page_icon="⚡",
    layout="wide"
)

# Custom Dashboard Styling

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eaf4ff, #f4edff);
}

.block-container {
    padding-top: 2rem;
}

h1 {
    color: #172554;
    font-weight: 800;
}

h2, h3 {
    color: #1e3a8a;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff, #dbeafe);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #bfdbfe;
    box-shadow: 0 4px 10px rgba(30, 64, 175, 0.10);
}

[data-testid="stMetricValue"] {
    color: #0369a1;
    font-weight: bold;
}

[data-testid="stDataFrame"] {
    border: 2px solid #bfdbfe;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Load Updated Cleaned Dataset

df = pd.read_csv("dataset/cleaned_ChargingRecords.csv")

# Dashboard Header

st.markdown("""
<div style="
    background: linear-gradient(100deg, #172554, #075985, #0f766e);
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    color: white;
">
    <h1 style="color:white;">⚡ EV Charging Demand Analysis</h1>
    <p style="font-size:18px; color:#e0f2fe;">
        Exploring Electric Vehicle Charging Patterns and Demand
    </p>
</div>
""", unsafe_allow_html=True)

st.write("")

# Key Performance Indicators

st.subheader("📊 Dashboard Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🔋 Total Charging Sessions",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "⚡ Average Demand",
        f"{df['Demand'].mean():.2f}"
    )

with col3:
    st.metric(
        "⏱️ Average Duration",
        f"{df['Calculated_Duration'].mean():.2f} min"
    )

st.write("")

# Dataset Preview

st.subheader("📁 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)

# Demand Distribution

st.subheader("📈 EV Charging Demand Distribution")

fig, ax = plt.subplots(figsize=(11, 5))

ax.hist(
    df["Demand"],
    bins=30,
    color="#0d9488",
    edgecolor="white",
    alpha=0.9
)

ax.set_xlabel("Charging Demand", fontsize=11)
ax.set_ylabel("Number of Sessions", fontsize=11)
ax.set_title(
    "Distribution of Charging Demand",
    fontsize=15,
    fontweight="bold",
    color="#172554"
)

ax.set_facecolor("#f0f9ff")
fig.patch.set_facecolor("#f0f9ff")
ax.grid(axis="y", alpha=0.2)

st.pyplot(fig)

# Dashboard Footer

st.markdown("""
<div style="
    background-color:#172554;
    padding:15px;
    border-radius:10px;
    text-align:center;
    color:white;
">
    ⚡ EV Charging Demand Prediction and Usage Pattern Analysis
    <br>
    <small>Data Science and Generative AI Project</small>
</div>
""", unsafe_allow_html=True)

# Charging Sessions by Location

st.subheader("📍 Charging Sessions by Location")

location_sessions = df["Location"].value_counts()

fig, ax = plt.subplots(figsize=(10, 5))

location_sessions.plot(
    kind="bar",
    ax=ax,
    color="#2563eb"
)

ax.set_title("Number of Charging Sessions by Location")
ax.set_xlabel("Location")
ax.set_ylabel("Total Sessions")
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# Average Demand by Hour

st.subheader("⏰ Average Charging Demand by Hour")

hourly_demand = df.groupby("Hour")["Demand"].mean()

fig, ax = plt.subplots(figsize=(10, 5))

hourly_demand.plot(
    kind="line",
    marker="o",
    ax=ax,
    color="#0d9488"
)
ax.set_title("Average Charging Demand Across Hours")
ax.set_xlabel("Hour of the Day")
ax.set_ylabel("Average Demand")
ax.grid(True)
plt.tight_layout()
st.pyplot(fig)

# Monthly Demand Trend

st.subheader("📅 Monthly Average Charging Demand")

df["StartDatetime"] = pd.to_datetime(df["StartDatetime"])

monthly_demand = df.groupby(
    df["StartDatetime"].dt.to_period("M")
)["Demand"].mean()

monthly_demand.index = monthly_demand.index.astype(str)

fig, ax = plt.subplots(figsize=(11, 5))

monthly_demand.plot(
    kind="line",
    marker="o",
    linewidth=2,
    ax=ax,
    color="#7c3aed"
)

ax.set_title("Monthly Average EV Charging Demand")
ax.set_xlabel("Month")
ax.set_ylabel("Average Demand")
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# High Demand vs Normal Demand

st.subheader("⚡ High Demand vs Normal Demand Sessions")

demand_counts = df["Demand_Class"].map({
    0: "Normal Demand",
    1: "High Demand"
}).value_counts()

fig, ax = plt.subplots(figsize=(7, 5))
demand_counts.plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90,
    ax=ax,
    colors=["#38bdf8", "#f97316"]
)
ax.set_ylabel("")
ax.set_title("Demand Category Distribution")
st.pyplot(fig)

# Charger Type Analysis

st.subheader("🔌 Charging Performance by Charger Type")

charger_type_analysis = df.groupby("ChargerType").agg(
    Total_Sessions=("Demand", "count"),
    Average_Demand=("Demand", "mean"),
    Average_Duration=("Calculated_Duration", "mean")
)

st.dataframe(charger_type_analysis.round(2), use_container_width=True)

# Charger Company Analysis

st.subheader("🏢 Charging Performance by Charger Company")
charger_company_analysis = df.groupby("ChargerCompany").agg(
    Total_Sessions=("Demand", "count"),
    Average_Demand=("Demand", "mean"),
    Average_Duration=("Calculated_Duration", "mean")
)
st.dataframe(charger_company_analysis.round(2), use_container_width=True)


# --------------------------------------------------
# MODEL EXPLAINABILITY - SHAP AND LIME
# --------------------------------------------------

st.subheader("🔍 Model Explainability")

# Get the main project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# SHAP Summary Plot
st.write("### SHAP Summary Plot")

shap_path = os.path.join(PROJECT_DIR, "shap_summary.png")

st.image(
    shap_path,
    caption="SHAP Feature Importance"
)

# LIME Explanation
st.write("### LIME Individual Prediction Explanation")

lime_path = os.path.join(PROJECT_DIR, "lime_explanation.html")

with open(lime_path, "r", encoding="utf-8") as file:
    lime_html = file.read()

st.components.v1.html(
    lime_html,
    height=600,
    scrolling=True
)

# --------------------------------------------------
# TIME SERIES FORECASTING
# --------------------------------------------------

st.subheader("📅 EV Charging Demand Forecast")

st.write(
    "The Exponential Smoothing model was used "
    "to forecast average daily charging demand."
)

# Forecasting Model Performance

col1, col2 = st.columns(2)

with col1:
    st.metric("Forecast MAE", "0.7615")

with col2:
    st.metric("Forecast RMSE", "1.0123")

# Next 7 Days Forecast

forecast_dates = pd.date_range(
    start="2022-10-01",
    periods=7,
    freq="D"
)

forecast_values = [
    17.762259,
    17.764055,
    17.765852,
    17.767648,
    17.769445,
    17.771241,
    17.773038
]

forecast_df = pd.DataFrame({
    "Date": forecast_dates,
    "Predicted Average Demand": forecast_values
})

st.dataframe(
    forecast_df.round(3),
    use_container_width=True
)

# Forecast Visualization

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    forecast_df["Date"],
    forecast_df["Predicted Average Demand"],
    marker="o",
    color="#0d9488",
    label="Forecasted Demand"
)

ax.set_title("Next 7 Days EV Charging Demand Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Predicted Average Demand")
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)


# --------------------------------------------------
# REGRESSION MODEL PERFORMANCE
# --------------------------------------------------

st.subheader("📊 Regression Model Performance")

regression_results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [10.0944, 10.6639],
    "MSE": [174.2506, 197.5850],
    "RMSE": [13.2004, 14.0565],
    "R2 Score": [0.0330, -0.0965]
})

st.dataframe(
    regression_results,
    use_container_width=True
)

st.write(
    "Linear Regression performed better than Random Forest "
    "for this dataset based on the evaluation metrics."
)
