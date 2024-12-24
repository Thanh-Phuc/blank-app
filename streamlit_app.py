import streamlit as st
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import altair as alt
from pathlib import Path

# Load your data
path = Path(__file__).parent/'data/Top24_short.csv'
# path = "/Users/thanhphucphan/Library/CloudStorage/GoogleDrive-phuc.phanthanh@gmail.com/My Drive/d931109001@tmu.edu.tw/Chiaki/Theses/Data/Top24_short.csv"
data = pd.read_csv(path, delimiter=",")
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%m-%Y', errors='coerce')  # Parse dates, handle errors
data = data.dropna().reset_index(drop=True)  # Drop rows with missing data
df = data.set_index('Datetime')  # Set Datetime as the index

# App layout
st.title("Wine Price Forecasting App")
selected_wine = st.selectbox("Select a Wine:", df.columns)

# EDA: Time Series Decomposition
if st.checkbox("Exploratory Data Analysis (EDA)"):
    time_series = df[selected_wine].dropna()

    # Perform decomposition
    decomposition = seasonal_decompose(time_series, model='additive', period=12)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Display Decomposition Using Streamlit Charts
    st.subheader("Original Time Series")
    st.line_chart(time_series)

    st.subheader("Trend Component")
    st.line_chart(trend)

    st.subheader("Seasonal Component")
    st.line_chart(seasonal)

    st.subheader("Residual Component")
    st.line_chart(residual)

# Forecasting
st.subheader("Forecast Wine Price")
model_type = st.radio("Select Forecasting Model:", ["ARIMA", "SARIMAX"])

# Assuming the necessary variables are already defined
# time_series, seasonal_order_p, seasonal_order_d, seasonal_order_q, seasonal_period, model_type
# SARIMAX Parameters
if model_type == "SARIMAX":
    st.sidebar.title("SARIMAX Parameters")
    seasonal_order_p = st.sidebar.number_input("Seasonal Order (P)", min_value=0, value=1, step=1)
    seasonal_order_d = st.sidebar.number_input("Seasonal Order (D)", min_value=0, value=0, step=1)
    seasonal_order_q = st.sidebar.number_input("Seasonal Order (Q)", min_value=0, value=1, step=1)
    seasonal_period = st.sidebar.number_input("Seasonal Period (s)", min_value=1, value=12, step=1)

# Fit the model based on the selected model type
if st.button("Generate Forecast"):
    time_series = df[selected_wine].dropna()
    if model_type == "ARIMA":
        model = ARIMA(time_series, order=(1, 1, 1))
        model_fit = model.fit()
    elif model_type == "SARIMAX":
        model = SARIMAX(
        time_series,
        order=(1, 1, 1),
        seasonal_order=(seasonal_order_p, seasonal_order_d, seasonal_order_q, seasonal_period)
    )
    model_fit = model.fit()

# Forecast
    forecast = model_fit.get_forecast(steps=12)
    forecast_index = pd.date_range(start=time_series.index[-1] + pd.DateOffset(1), periods=12, freq='M')
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

# Prepare data for plotting
    historical_data = pd.DataFrame({'Date': time_series.index, 'Price': time_series.values})
    forecast_data = pd.DataFrame({'Date': forecast_index, 'Forecast': forecast_mean})
    confidence_interval = pd.DataFrame({
    'Date': forecast_index,
    'Lower CI': forecast_ci.iloc[:, 0],
    'Upper CI': forecast_ci.iloc[:, 1]
    })

# Combine data for Altair
    combined_data = pd.concat([
    historical_data.set_index('Date'),
    forecast_data.set_index('Date'),
    confidence_interval.set_index('Date')
    ], axis=1).reset_index()

# Plot combined data using Altair
    st.subheader(f"{selected_wine} Price Forecast ({model_type})")
    historical_chart = alt.Chart(combined_data).mark_line(color='blue').encode(
    x='Date:T',
    y='Price:Q',
    tooltip=['Date:T', 'Price:Q'],
    color=alt.value('blue')
)

    forecast_chart = alt.Chart(combined_data).mark_line(color='orange').encode(
    x='Date:T',
    y='Forecast:Q',
    tooltip=['Date:T', 'Forecast:Q'],
    color=alt.value('orange')
)

    ci_chart = alt.Chart(combined_data).mark_area(opacity=0.3, color='lightgrey').encode(
    x='Date:T',
    y='Lower CI:Q',
    y2='Upper CI:Q',
    tooltip=['Date:T', 'Lower CI:Q', 'Upper CI:Q']
)

    chart = (historical_chart + forecast_chart + ci_chart)


    st.altair_chart(chart, use_container_width=True)
