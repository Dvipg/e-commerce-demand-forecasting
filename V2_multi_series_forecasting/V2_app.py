import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import streamlit as st
from tqdm import tqdm

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Store Item Demand Forecasting")
st.title('V2 Multi Series Forecasting')
st.markdown('A comprehensive solution for multi-series forecasting, backtesting, and visualization.')

# --- Data Loading and Forecasting (Cached) ---
@st.cache_data
def load_and_forecast_data():
    """
    Loads data, trains models for all store-item combinations, and generates forecasts.
    """
    st.info("Please wait, loading data and training models for 500 time series. This will take a few minutes...")
    
    file_path = 'V2_multi_series_forecasting/train.csv'
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error("Error: 'data/train.csv' not found. Please place the file in the 'data' subfolder inside your 'V2' directory.")
        return None, None

    df['date'] = pd.to_datetime(df['date'])
    
    # Get all unique store-item pairs
    all_pairs = df[['store', 'item']].drop_duplicates().to_records(index=False)
    
    all_forecasts = []
    
    for store, item in tqdm(all_pairs, desc="Forecasting"):
        time_series = df[(df['store'] == store) & (df['item'] == item)].copy()
        time_series = time_series.rename(columns={'date': 'ds', 'sales': 'y'})
        
        m = Prophet()
        m.fit(time_series)
        future = m.make_future_dataframe(periods=365)
        forecast = m.predict(future)
        
        forecast['store'] = store
        forecast['item'] = item
        all_forecasts.append(forecast)
        
    forecast_df = pd.concat(all_forecasts)
    
    st.success("All models trained and forecasts generated!")
    return df, forecast_df

@st.cache_data
def get_backtest_metrics(df):
    """
    Performs backtesting on a sample series and returns performance metrics.
    """
    st.info("Performing backtesting on a sample time series...")
    
    sample_df = df[(df['store'] == 1) & (df['item'] == 1)].copy()
    sample_df = sample_df.rename(columns={'date': 'ds', 'sales': 'y'})
    
    m = Prophet()
    m.fit(sample_df)
    
    df_cv = cross_validation(m, initial='1095 days', period='180 days', horizon='90 days')
    df_p = performance_metrics(df_cv)
    
    st.success("Backtesting complete!")
    return df_p


# --- Main Application Logic ---
df_data, df_forecast = load_and_forecast_data()

if df_data is not None and df_forecast is not None:
    stores = sorted(df_data['store'].unique())
    items = sorted(df_data['item'].unique())
    
    # --- Sidebar Filters ---
    with st.sidebar:
        st.header('Select Data to Visualize')
        selected_store = st.selectbox('Choose a Store:', stores)
        selected_item = st.selectbox('Choose an Item:', items)
        
        st.markdown('---')
        st.header('Backtesting')
        if st.button('Run Backtesting'):
            df_metrics = get_backtest_metrics(df_data)
            st.session_state['df_metrics'] = df_metrics
    
    # --- Dashboard Views ---
    st.markdown("---")
    
    # --- View 1: Forecasting Plot ---
    st.subheader(f'1. Sales Forecast for Store {selected_store}, Item {selected_item}')
    
    # Filter the forecast data based on user selection
    forecast_filtered = df_forecast[
        (df_forecast['store'] == selected_store) & (df_forecast['item'] == selected_item)
    ]
    
    # Filter the actual historical data
    historical_filtered = df_data[
        (df_data['store'] == selected_store) & (df_data['item'] == selected_item)
    ]
    
    # Create the Plotly figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_filtered['date'],
        y=historical_filtered['sales'],
        mode='lines',
        name='Historical Sales',
        line=dict(color='blue')
    ))
    
    # Add forecast data
    fig.add_trace(go.Scatter(
        x=forecast_filtered['ds'],
        y=forecast_filtered['yhat'],
        mode='lines',
        name='Forecasted Sales',
        line=dict(color='orange', dash='dash')
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_filtered['ds'],
        y=forecast_filtered['yhat_lower'],
        fill=None,
        mode='lines',
        line=dict(color='orange', width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_filtered['ds'],
        y=forecast_filtered['yhat_upper'],
        fill='tonexty',
        mode='lines',
        line=dict(color='orange', width=0),
        fillcolor='rgba(255,165,0,0.2)',
        showlegend=False
    ))
    
    fig.update_layout(
        title=f'Sales Forecast for Store {selected_store}, Item {selected_item}',
        xaxis_title='Date',
        yaxis_title='Sales',
        template="plotly_white",
        xaxis_rangeslider_visible=True,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- View 2: Backtesting Metrics ---
    st.subheader('2. Model Performance & Validation')
    st.markdown('Metrics from rolling-origin backtesting on a sample time series (Store 1, Item 1).')

    if 'df_metrics' in st.session_state:
        df_metrics = st.session_state['df_metrics']
        
        col_rmse, col_mape = st.columns(2)
        
        with col_rmse:
            fig_rmse = px.line(
                df_metrics,
                x="horizon",
                y="rmse",
                title="RMSE Over Forecast Horizon",
                labels={"horizon": "Forecast Horizon (days)", "rmse": "RMSE"}
            )
            st.plotly_chart(fig_rmse, use_container_width=True)
            
        with col_mape:
            fig_mape = px.line(
                df_metrics,
                x="horizon",
                y="mape",
                title="MAPE Over Forecast Horizon",
                labels={"horizon": "Forecast Horizon (days)", "mape": "MAPE"}
            )
            st.plotly_chart(fig_mape, use_container_width=True)
            
        st.dataframe(df_metrics[['horizon', 'rmse', 'mape', 'coverage']].head(10))
    else:

        st.info('Click the "Run Backtesting" button in the sidebar to generate and view the performance metrics.')
