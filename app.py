import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

st.set_page_config(page_title="Rainfall Dashboard", layout="wide")

# Load data
df = pd.read_csv("rainfall_predictions.csv")

st.title("🌧️ Rainfall Prediction Dashboard")
st.markdown("### Advanced ML Dashboard with Interactive Features")

# ===================== SIDEBAR =====================
st.sidebar.header("🔍 Controls")
year = st.sidebar.selectbox("Select Year", df["YEAR"])

filtered = df[df["YEAR"] == year]

# ===================== METRICS =====================
col1, col2, col3 = st.columns(3)

col1.metric("Actual", round(filtered['Actual Rainfall: JUN-SEPT'].values[0], 2))
col2.metric("DT Prediction", round(filtered['DT_Prediction'].values[0], 2))
col3.metric("KNN Prediction", round(filtered['KNN_Prediction'].values[0], 2))

st.divider()

# ===================== CHARTS =====================
col1, col2 = st.columns(2)

with col1:
    import matplotlib.pyplot as plt

    st.subheader("📈 Rainfall Trend")

    fig, ax = plt.subplots()

    ax.plot(df['YEAR'], df['Actual Rainfall: JUN-SEPT'], label='Actual')
    ax.plot(df['YEAR'], df['DT_Prediction'], label='DT')
    ax.plot(df['YEAR'], df['KNN_Prediction'], label='KNN')

    ax.set_xlabel("Year")
    ax.set_ylabel("Rainfall (mm)")
    ax.set_title("Rainfall Trend Over Years")
    ax.legend()

    st.pyplot(fig)
with col2:
    st.subheader("📊 Monthly Rainfall")

    months = ['JUN', 'JUL', 'AUG', 'SEPT']
    values = [
        filtered['Actual Rainfall: JUN'].values[0],
        filtered['Actual Rainfall: JUL'].values[0],
        filtered['Actual Rainfall: AUG'].values[0],
        filtered['Actual Rainfall: SEPT'].values[0]
    ]

    fig, ax = plt.subplots()

    ax.bar(months, values)

    ax.set_xlabel("Month")
    ax.set_ylabel("Rainfall (mm)")
    ax.set_title(f"Monthly Rainfall for {year}")

    st.pyplot(fig)
st.divider()

# ===================== MODEL COMPARISON =====================

import matplotlib.pyplot as plt

st.subheader("📊 Model Performance Comparison")
# Calculate errors
dt_error = np.mean(abs(df['Actual Rainfall: JUN-SEPT'] - df['DT_Prediction']))
knn_error = np.mean(abs(df['Actual Rainfall: JUN-SEPT'] - df['KNN_Prediction']))

models = ["Decision Tree", "KNN"]
errors = [dt_error, knn_error]

fig, ax = plt.subplots()

ax.bar(models, errors)

ax.set_xlabel("Model")
ax.set_ylabel("Error (mm)")
ax.set_title("Model Comparison (Lower is Better)")

st.pyplot(fig)

st.divider()

# ===================== USER INPUT PREDICTION =====================
st.subheader("🎛️ Predict Rainfall")

# Input fields (FIXED PART ✅)
col1, col2 = st.columns(2)

with col1:
    jun = st.number_input("June Rainfall", value=100.0)
    jul = st.number_input("July Rainfall", value=100.0)

with col2:
    aug = st.number_input("August Rainfall", value=100.0)
    sept = st.number_input("September Rainfall", value=100.0)

# Train model
X = df[['Actual Rainfall: JUN',
        'Actual Rainfall: JUL',
        'Actual Rainfall: AUG',
        'Actual Rainfall: SEPT']]

y = df['Actual Rainfall: JUN-SEPT']

model = DecisionTreeRegressor()
model.fit(X, y)

# Prediction
input_data = pd.DataFrame([[jun, jul, aug, sept]], columns=[
    'Actual Rainfall: JUN',
    'Actual Rainfall: JUL',
    'Actual Rainfall: AUG',
    'Actual Rainfall: SEPT'
])
prediction = model.predict(input_data)

st.success(f"🌧️ Predicted Total Rainfall: {round(prediction[0], 2)}")

st.divider()

# ===================== MAP =====================
st.subheader("🌍 Location Map")

map_data = pd.DataFrame({
    'lat': [12.97],
    'lon': [77.59]
})

st.map(map_data)

st.divider()

# ===================== DATA =====================
st.subheader("📋 Dataset")
st.dataframe(df)

st.markdown("---")
st.markdown("✅ Project By: Shirisha T S | Rainfall Prediction Dashboard")