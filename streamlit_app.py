import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Load data
path = "/Users/thanhphucphan/Library/CloudStorage/GoogleDrive-phuc.phanthanh@gmail.com/My Drive/d931109001@tmu.edu.tw/Chiaki/Theses/Data/Top24_short.csv"
data = pd.read_csv(path, delimiter=",")
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%m-%Y', errors='coerce')  # Parse dates, handle errors
data = data.dropna().reset_index(drop=True)  # Drop rows with missing data
df = data.set_index('Datetime')  # Set Datetime as the index

# App layout
st.title("Wine Price Forecasting App")
selected_wine = st.selectbox("Select a Wine:", df.columns)

# # EDA: Time Series Decomposition
# if st.checkbox("Exploratory Data Analysis (EDA)"):
#     st.subheader("Perform Time Series Decomposition")
#     time_series = df[selected_wine].dropna()

#     # Perform decomposition
#     decomposition = seasonal_decompose(time_series, model='additive', period=12)
#     trend = decomposition.trend
#     seasonal = decomposition.seasonal
#     residual = decomposition.resid

#     # Plot decomposition components using Matplotlib
#     st.write(f"{selected_wine}")
#     fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
#     axes[0].plot(time_series, label='Original Time Series', color='blue')
#     axes[0].set_title(f'{selected_wine}')
#     axes[0].legend()
#     axes[1].plot(trend, label='Trend Component', color='orange')
#     axes[1].set_title(f'Trend Component')
#     axes[1].legend()
#     axes[2].plot(seasonal, label='Seasonal Component', color='green')
#     axes[2].set_title(f'Seasonal Component')
#     axes[2].legend()
#     axes[3].plot(residual, label='Residual Component', color='red')
#     axes[3].set_title(f'Residual Component')
#     axes[3].legend()
#     plt.tight_layout()

#     # Display the plot in Streamlit
#     st.pyplot(fig)"

# EDA: Time Series Decomposition
if st.checkbox("Exploratory Data Analysis (EDA)"):
    
    time_series = df[selected_wine].dropna()

    # Perform decomposition
    decomposition = seasonal_decompose(time_series, model='additive', period=12)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
  
    # Original Time Series
    fig_original = go.Figure()
    fig_original.add_trace(go.Scatter(
        x=time_series.index, y=time_series.values,
        mode='lines', name='Original Time Series',
        line=dict(color='blue')
    ))
    fig_original.update_layout(
        title="Original Time Series",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified"
    )
    st.plotly_chart(fig_original)

    # Create interactive Plotly figures for decomposition
    st.subheader("Perform Time Series Decomposition")
    st.write(f"{selected_wine}")
    # Trend Component
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend.index, y=trend.values,
        mode='lines', name='Trend Component',
        line=dict(color='orange')
    ))
    fig_trend.update_layout(
        title="Trend Component",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified"
    )
    st.plotly_chart(fig_trend)

    # Seasonal Component
    fig_seasonal = go.Figure()
    fig_seasonal.add_trace(go.Scatter(
        x=seasonal.index, y=seasonal.values,
        mode='lines', name='Seasonal Component',
        line=dict(color='green')
    ))
    fig_seasonal.update_layout(
        title="Seasonal Component",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified"
    )
    st.plotly_chart(fig_seasonal)

    # Residual Component
    fig_residual = go.Figure()
    fig_residual.add_trace(go.Scatter(
        x=residual.index, y=residual.values,
        mode='lines', name='Residual Component',
        line=dict(color='red')
    ))
    fig_residual.update_layout(
        title="Residual Component",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified"
    )
    st.plotly_chart(fig_residual)

# Model Selection
st.subheader("Forecast Wine Price")
model_type = st.radio(
    "Select Forecasting Model:",
    options=["ARIMA", "SARIMAX"]
)

# SARIMAX Seasonal Parameters
if model_type == "SARIMAX":
    st.sidebar.title("SARIMAX Seasonal Parameters")
    seasonal_order_p = st.sidebar.number_input("Seasonal Order (P)", min_value=0, value=1, step=1)
    seasonal_order_d = st.sidebar.number_input("Seasonal Order (D)", min_value=0, value=0, step=1)
    seasonal_order_q = st.sidebar.number_input("Seasonal Order (Q)", min_value=0, value=1, step=1)
    seasonal_period = st.sidebar.number_input("Seasonal Period (s)", min_value=1, value=12, step=1)

# Forecasting
if st.button("Generate Forecast"):
    time_series = df[selected_wine].dropna()

    if model_type == "ARIMA":
        # Fit ARIMA model
        model = ARIMA(time_series, order=(1, 1, 1))
        model_fit = model.fit()

        # Forecast
        forecast = model_fit.get_forecast(steps=12)
        forecast_index = pd.date_range(start=time_series.index[-1] + pd.DateOffset(1), periods=12, freq='M')
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()
    elif model_type == "SARIMAX":
        
        # Fit SARIMAX model
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

    # Plot results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_series.index, y=time_series.values, mode='lines', name='Historical Data'))
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast_mean, mode='lines', name='Forecast'))
    fig.add_trace(go.Scatter(
        x=forecast_index, y=forecast_ci.iloc[:, 0], mode='lines', line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast_index, y=forecast_ci.iloc[:, 1],
        fill='tonexty', mode='lines', line=dict(width=0), fillcolor='rgba(255, 255, 102, 0.2)', name='Confidence Interval'
    ))
    fig.update_layout(
        title=f"{selected_wine} Price Forecast ({model_type})",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode='x unified'
    )
    st.plotly_chart(fig)
