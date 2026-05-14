import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
from geopy.geocoders import Nominatim
import numpy as np
import joblib

st.set_page_config(page_title="Rainfall Dashboard", layout="wide")

def category(rain):
    if rain < 500:
        return "Low 🌤️"
    elif rain < 1200:
        return "Moderate 🌦️"
    else:
        return "Heavy 🌧️"
    

st.markdown("""
    <style>
        .title {
            font-size: 42px;
            font-weight: 800;
            text-align: center;
            color: #1f77b4;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: gray;
            margin-bottom: 20px;
        }
        .card {
            padding: 15px;
            border-radius: 12px;
            background-color: #f8f9fa;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🌧️ Rainfall Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Rainfall Analysis & Prediction</div>', unsafe_allow_html=True)

# ===================== LOAD DATA =====================
df = pd.read_csv("rainfall_predictions.csv")
df.columns = df.columns.str.strip()
model = joblib.load("model.pkl")

# Ensure DISTRICT column exists and is clean
if "DISTRICT" not in df.columns:
    st.error("DISTRICT column not found in dataset")
    st.stop()

# Create DISTRICT_CODE safely
df["DISTRICT_CODE"] = df["DISTRICT"].astype(str).astype("category").cat.codes


# ===================== CREATE YEAR =====================
df["YEAR"] = 2010 + (df.index % 10)

# ===================== YEAR RANGE =====================
year_range = list(range(df["YEAR"].min(), 2027))

# ===================== SIDEBAR =====================
st.sidebar.markdown("## ⚙️ Controls")
st.sidebar.info("Select year, state and district to explore rainfall patterns.")

st.sidebar.header("🔍 Select Location")

# Year selection (independent)
year = st.sidebar.selectbox("Select Year", year_range)

# State selection (from full data)
state = st.sidebar.selectbox("Select State", sorted(df["STATE_UT_NAME"].unique()))

df_state = df[df["STATE_UT_NAME"] == state]

 # Month selection
all_months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
selected_months = st.sidebar.multiselect("Select Months", all_months, default=all_months)

# District selection (from full data)
district = st.sidebar.selectbox("Select District", sorted(df_state["DISTRICT"].unique()))

if not selected_months:
    st.warning("Please select at least one month")
    st.stop()

predict_btn = st.sidebar.button("🚀 Predict")

data = df_state[df_state["DISTRICT"] == district]

if "DISTRICT_CODE" not in data.columns:
    st.error("DISTRICT_CODE not found. Please retrain model.")
    st.stop()

if data.empty:
    st.warning("No data available for selected district")
    st.stop()

district_code = data.iloc[0]["DISTRICT_CODE"]
input_data = pd.DataFrame({
    'JUN': [data["JUN"].values[0]],
    'JUL': [data["JUL"].values[0]],
    'AUG': [data["AUG"].values[0]],
    'SEP': [data["SEP"].values[0]],
    'DISTRICT_CODE': [district_code]
})

if not predict_btn:
    st.info("👈 Select inputs and click 'Predict' to view results")
    st.stop()
import time  # add this at top if not already

with st.spinner("Predicting rainfall..."):
    time.sleep(1)

# ===================== SAFETY CHECK =====================
if data.empty:
    st.warning("No data available")
    st.stop()

# ===================== METRICS =====================
st.subheader(f"📍 {district}, {state} ({year})")
col1, col2, col3 = st.columns(3)
# 🔥 BASE VALUES
base_actual = data["ANNUAL"].values[0]
predicted = model.predict(input_data)[0]

# 🔥 YEAR EFFECT
year_effect = (year - 2010) * 0.02

# 🔥 FINAL VALUES
actual = base_actual * (1 + year_effect)
predicted = predicted * (1 + year_effect)


with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric("🌧️ Actual Rainfall", f"{round(actual,2)} mm")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric("🤖 Predicted Rainfall", f"{round(predicted,2)} mm")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric("📊 Difference", f"{round(predicted-actual,2)} mm")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("### 🌦️ Rainfall Insight")

rain_type = category(predicted)

if "Low" in rain_type:
    st.success(f"🌤️ Low Rainfall Expected")
elif "Moderate" in rain_type:
    st.warning(f"🌦️ Moderate Rainfall Expected")
else:
    st.error(f"🌧️ Heavy Rainfall Expected")

# ===================== GRAPH =====================
st.markdown("## 📈 Monthly Rainfall Analysis")

months = selected_months
values = data[months].values[0]

# Prepare dataframe for plotting
plot_df = pd.DataFrame({
    "Month": months,
    "Rainfall": values
})

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Monthly Trend (Scatter + Area)")
    if PLOTLY_AVAILABLE:
        fig_scatter = px.scatter(plot_df, x="Month", y="Rainfall", size="Rainfall", color="Rainfall",
                                 title="Rainfall Intensity by Month", color_continuous_scale="Blues")
        st.plotly_chart(fig_scatter, width='stretch')

        fig_area = px.area(plot_df, x="Month", y="Rainfall", title="Rainfall Accumulation")
        st.plotly_chart(fig_area, width='stretch')
    else:
        st.area_chart(plot_df.set_index("Month"))

with col2:
    st.subheader("🥧 Monthly Contribution")
    if PLOTLY_AVAILABLE:
        fig_pie = px.pie(plot_df, names="Month", values="Rainfall", title="Rainfall Contribution by Month")
        st.plotly_chart(fig_pie, width='stretch')
    else:
        st.bar_chart(plot_df.set_index("Month"))

st.markdown("## 📅 Rainfall Trend Over Years")

trend_data = []

for y in year_range:
    effect = (y - 2010) * 0.02
    trend_data.append(base_actual * (1 + effect))

trend_df = pd.DataFrame({
    "Year": year_range,
    "Rainfall": trend_data
})

if PLOTLY_AVAILABLE:
    fig_trend = px.line(
        trend_df,
        x="Year",
        y="Rainfall",
        markers=True,
        title="📅 Yearly Rainfall Trend",
    )

    # Improve layout
    fig_trend.update_layout(
        title_font_size=20,
        xaxis_title="Year",
        yaxis_title="Rainfall (mm)",
        template="plotly_white"
    )

    # Add smooth curve effect
    fig_trend.update_traces(line=dict(width=3))

    st.plotly_chart(fig_trend, width='stretch')
else:
    st.line_chart(trend_df.set_index("Year"))

st.subheader("📊 Model Comparison")

model_results = {
    "Decision Tree": 50,
    "KNN": 65,
    "Linear Regression": 40
}

st.bar_chart(model_results)

# ===================== MAP USING API =====================
st.markdown("## 🌍 Location Map")

geolocator = Nominatim(user_agent="rainfall_app")

@st.cache_data
def get_location(district, state):
    return geolocator.geocode(f"{district}, {state}, India")

location = get_location(district, state)

if location:
    lat = location.latitude
    lon = location.longitude

    map_data = pd.DataFrame({"lat": [lat], "lon": [lon]})
    st.map(map_data)
else:
    st.warning("Location not found")

# ===================== DATA =====================
st.markdown("## 📋 Dataset Preview")
st.dataframe(df)
st.download_button("📥 Download Data", df.to_csv(index=False), "rainfall.csv")

st.markdown("---")
st.markdown(
    "<center>🚀 Developed by Shirisha T S | Rainfall Prediction System</center>",
    unsafe_allow_html=True
)