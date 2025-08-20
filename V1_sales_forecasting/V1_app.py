import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import STL
from pyod.models.iforest import IForest
import streamlit as st

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="E-commerce Sales Analysis")

# --- Function to load and preprocess data (cached) ---
@st.cache_data
def load_and_preprocess_data():
    """Loads and preprocesses the sales data."""
    try:
        df = pd.read_csv('V1_sales_forecasting/Sample - Superstore.csv', encoding='latin1')
    except FileNotFoundError:
        st.error("Error: 'Sample - Superstore.csv' not found. Please make sure the file is in the same directory.")
        return None

    df['Order Date'] = pd.to_datetime(df['Order Date'])
    daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()

    full_date_range = pd.date_range(start=daily_sales['Order Date'].min(), end=daily_sales['Order Date'].max(), freq='D')
    daily_sales = daily_sales.set_index('Order Date').reindex(full_date_range, fill_value=0).reset_index()
    daily_sales.rename(columns={'index': 'ds', 'Sales': 'y'}, inplace=True)
    return daily_sales

# --- Function to detect anomalies (cached) ---
@st.cache_data
def detect_anomalies(data):
    """Performs both STL and IsolationForest anomaly detection."""
    data = data.copy()
    data = data.set_index('ds')
    
    try:
        res = STL(data['y'], period=7, robust=True).fit()
        data['residuals'] = res.resid
        threshold = 3 * data['residuals'].std()
        data['stl_anomaly'] = np.where(np.abs(data['residuals']) > threshold, 1, 0)
    except Exception as e:
        st.warning(f"Warning: STL decomposition failed. {e}")
        data['stl_anomaly'] = 0

    X = data[['y']].values  # Use only sales data for a simple IForest model
    if data['residuals'] is not None:
        X = data[['y', 'residuals']].values # Use residuals if available
    
    clf = IForest(contamination=0.05, random_state=42)
    clf.fit(X)
    data['iforest_anomaly_score'] = clf.decision_scores_
    data['iforest_anomaly'] = clf.labels_
    
    # Identify positive (spike) vs. negative (drop) anomalies
    data['anomaly_type'] = np.nan
    data.loc[(data['iforest_anomaly'] == 1) & (data['residuals'] > 0), 'anomaly_type'] = 'Spike'
    data.loc[(data['iforest_anomaly'] == 1) & (data['residuals'] < 0), 'anomaly_type'] = 'Drop'
    
    return data

# --- Load Data and Detect Anomalies ---
daily_sales = load_and_preprocess_data()
if daily_sales is not None:
    daily_sales_with_anomalies = detect_anomalies(daily_sales)
else:
    st.stop()


# --- Dashboard UI and Visualization ---
st.title('V1 Sales Forecasting')
st.markdown('A Sales Forecasting project showing a simple, end-to-end solution for demand analysis.')

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader('1. Historical Sales & Anomaly Detection')
    st.markdown('This view shows the daily sales trend with detected anomalies.')

    # Create the figure with Plotly
    sales_data = go.Scatter(
        x=daily_sales_with_anomalies.index,
        y=daily_sales_with_anomalies['y'],
        mode='lines',
        name='Daily Sales',
        line=dict(color='blue')
    )
    
    # Prepare anomaly data for plotting
    anomalies = daily_sales_with_anomalies[daily_sales_with_anomalies['iforest_anomaly'] == 1]
    spikes = anomalies[anomalies['anomaly_type'] == 'Spike']
    drops = anomalies[anomalies['anomaly_type'] == 'Drop']

    spike_plot = go.Scatter(
        x=spikes.index,
        y=spikes['y'],
        mode='markers',
        name='Spike Anomaly',
        marker=dict(color='orange', size=10, symbol='triangle-up'),
        hovertemplate='Date: %{x}<br>Sales: %{y}<br>Anomaly Type: Spike'
    )
    
    drop_plot = go.Scatter(
        x=drops.index,
        y=drops['y'],
        mode='markers',
        name='Drop Anomaly',
        marker=dict(color='red', size=10, symbol='triangle-down'),
        hovertemplate='Date: %{x}<br>Sales: %{y}<br>Anomaly Type: Drop'
    )
    
    fig = go.Figure(data=[sales_data, spike_plot, drop_plot])
    fig.update_layout(
        title='Daily Sales with Anomalies',
        xaxis_title='Date',
        yaxis_title='Total Sales',
        xaxis_rangeslider_visible=True,
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader('2. Top Anomalies')
    st.markdown('A list of the most significant sales spikes and drops.')
    
    top_anomalies = daily_sales_with_anomalies[
        daily_sales_with_anomalies['iforest_anomaly'] == 1
    ].sort_values('iforest_anomaly_score', ascending=False)
    
    st.dataframe(
        top_anomalies[['y', 'iforest_anomaly_score', 'anomaly_type']].head(10).style.format(
            {'y': '{:,.2f}', 'iforest_anomaly_score': '{:.4f}'}
        ).highlight_max(axis=0),
        use_container_width=True
    )
    
st.markdown("---")
st.markdown("### Project Notes")

st.info("This is Version 1 of the project, using a single-series dataset. The next steps involve migrating to a larger, multi-series dataset (V2) to showcase scalable forecasting and proper model validation techniques.")
