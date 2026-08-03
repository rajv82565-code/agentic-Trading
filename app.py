import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import io

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TradePulse AI | Stock Forecast & Waypoint Modeling",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Theme & Custom CSS Design System
# -----------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

IS_DARK = st.session_state.theme == "dark"

# Theme colors
BG_COLOR = "#09090b" if IS_DARK else "#f8fafc"
CARD_BG = "#111116" if IS_DARK else "#ffffff"
TEXT_COLOR = "#fafafa" if IS_DARK else "#0f172a"
TEXT_MUTED = "#94a3b8" if IS_DARK else "#64748b"
BORDER_COLOR = "#1e293b" if IS_DARK else "#e2e8f0"
ACCENT_BLUE = "#3b82f6"
GREEN_COLOR = "#22c55e"
RED_COLOR = "#ef4444"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], .main {{
        background-color: {BG_COLOR} !important;
        color: {TEXT_COLOR} !important;
        font-family: 'DM Sans', sans-serif !important;
    }}
    
    .block-container {{
        padding: 1.5rem 2rem 3rem !important;
        max-width: 1400px !important;
    }}
    
    /* Header Container */
    .app-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: {CARD_BG};
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }}
    .app-title {{
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: {TEXT_COLOR};
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }}
    .app-subtitle {{
        font-size: 0.85rem;
        color: {TEXT_MUTED};
        margin-top: 0.2rem;
    }}
    
    /* Metric Cards */
    .metric-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER_COLOR};
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        height: 100%;
    }}
    .metric-label {{
        font-size: 0.78rem;
        color: {TEXT_MUTED};
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .metric-value {{
        font-size: 1.65rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: {TEXT_COLOR};
        margin: 0.3rem 0;
    }}
    .metric-sub {{
        font-size: 0.78rem;
        font-weight: 500;
    }}
    .positive {{ color: {GREEN_COLOR}; }}
    .negative {{ color: {RED_COLOR}; }}
    
    /* Chart Container Wrapper */
    .chart-box {{
        background: {CARD_BG};
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
    }}
    .chart-header {{
        font-size: 1rem;
        font-weight: 600;
        color: {TEXT_COLOR};
        margin-bottom: 0.3rem;
    }}
    .chart-desc {{
        font-size: 0.8rem;
        color: {TEXT_MUTED};
        margin-bottom: 1rem;
    }}

    /* Custom Streamlit Tabs */
    button[data-baseweb="tab"] {{
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.25rem !important;
    }}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Data Fetching & Processing Helpers
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_data(ticker, start_date, end_date):
    """Fetch historical stock price data from yfinance with robust fallback."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False, progress=False)
        if data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if 'Adj Close' not in data.columns and 'Close' in data.columns:
            data['Adj Close'] = data['Close']
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        err_msg = str(e)
        if "RateLimitError" in err_msg or "Too Many Requests" in err_msg:
            st.warning("⚠️ Yahoo Finance rate limit reached. Waiting briefly before retrying...")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_peer_correlation(tickers, start_date, end_date):
    """Fetch multiple tickers and compute daily return correlation matrix."""
    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        if data.empty:
            return pd.DataFrame()
        if 'Close' in data:
            close_df = data['Close']
            if isinstance(close_df, pd.DataFrame):
                return close_df.pct_change().corr()
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def train_lstm_waypoint_model(dataset_values, lookback=60, epochs=1, batch_size=1, forecast_horizon=7):
    """
    Trains Deep Neural Network (Multi-Layer Perceptron) on scaled close prices and calculates:
    1. Validation predictions
    2. Multi-step future waypoint projections (autoregressive rollout)
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset_values)
    
    training_data_len = int(np.ceil(len(dataset_values) * 0.85))
    train_data = scaled_data[0:training_data_len, :]
    
    x_train, y_train = [], []
    for i in range(lookback, len(train_data)):
        x_train.append(train_data[i-lookback:i, 0])
        y_train.append(train_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    # Model Architecture: Deep Neural Network
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=max(20, epochs * 30),
        random_state=42
    )
    model.fit(x_train, y_train)
    
    # Validation predictions
    test_data = scaled_data[training_data_len - lookback:, :]
    x_test, y_test = [], dataset_values[training_data_len:]
    for i in range(lookback, len(test_data)):
        x_test.append(test_data[i-lookback:i, 0])
        
    x_test = np.array(x_test)
    
    predictions = model.predict(x_test).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    
    # Multi-step future waypoint forecasting (Autoregressive Rollout)
    curr_sequence = scaled_data[-lookback:].reshape(1, -1)
    waypoint_preds_scaled = []
    
    for _ in range(forecast_horizon):
        next_pred = model.predict(curr_sequence)
        waypoint_preds_scaled.append(next_pred[0])
        # Update sequence with new prediction
        curr_sequence = np.append(curr_sequence[:, 1:], [[next_pred[0]]], axis=1)
        
    waypoint_predictions = scaler.inverse_transform(np.array(waypoint_preds_scaled).reshape(-1, 1))
    
    # Metrics
    rmse = float(np.sqrt(np.mean(((predictions - y_test) ** 2))))
    mae = float(np.mean(np.abs(predictions - y_test)))
    mape = float(np.mean(np.abs((y_test - predictions) / y_test)) * 100)
    
    return {
        "model": model,
        "scaler": scaler,
        "training_len": training_data_len,
        "predictions": predictions,
        "y_test": y_test,
        "waypoint_predictions": waypoint_predictions.flatten(),
        "rmse": rmse,
        "mae": mae,
        "mape": mape
    }

# -----------------------------------------------------------------------------
# 4. Header & Sidebar Controls
# -----------------------------------------------------------------------------
h_col1, h_col2 = st.columns([5, 1])
with h_col1:
    st.markdown("""
    <div class="app-header">
        <div>
            <div class="app-title">📈 TradePulse AI — Stock Forecasting & Waypoint Modeling</div>
            <div class="app-subtitle">Deep learning LSTM model for stock price prediction & multi-step future waypoint trajectory simulation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with h_col2:
    theme_btn = "☀️ Light" if IS_DARK else "🌙 Dark"
    st.button(theme_btn, on_click=toggle_theme, use_container_width=True)

# Sidebar Parameters
st.sidebar.markdown("### ⚙️ Control Panel")

popular_tickers = ["AAPL", "GOOG", "MSFT", "AMZN", "NVDA", "TSLA"]
selected_ticker = st.sidebar.selectbox("Select Asset Ticker", popular_tickers, index=0)
custom_ticker = st.sidebar.text_input("Or enter Custom Ticker", value="", help="Enter valid stock symbol (e.g. BTC-USD, META, NFLX). Leave blank to use selected ticker.").strip().upper()
ticker = custom_ticker if custom_ticker else selected_ticker

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    start_date = st.date_input("Start Date", value=datetime(2018, 1, 1))
with col_s2:
    end_date = st.date_input("End Date", value=datetime.now())

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 LSTM Waypoint Parameters")

lookback_window = st.sidebar.slider("Lookback Window (Days)", min_value=10, max_value=120, value=60, step=5)
forecast_horizon = st.sidebar.slider("Waypoint Forecast Horizon (Days)", min_value=1, max_value=30, value=10, step=1)
epochs = st.sidebar.slider("Training Epochs", min_value=1, max_value=5, value=1)
batch_size = st.sidebar.selectbox("Batch Size", [1, 16, 32, 64], index=0)

retrain_button = st.sidebar.button("🚀 Train / Retrain Model", use_container_width=True)

# -----------------------------------------------------------------------------
# 5. Data Loading & KPI Row
# -----------------------------------------------------------------------------
with st.spinner(f"Fetching data for {ticker}..."):
    df_stock = load_stock_data(ticker, start_date, end_date)

if df_stock.empty or 'Close' not in df_stock.columns:
    if custom_ticker:
        st.warning(f"⚠️ Stock symbol '{custom_ticker}' was not found on Yahoo Finance. Falling back to selected ticker '{selected_ticker}'...")
        ticker = selected_ticker
        df_stock = load_stock_data(ticker, start_date, end_date)
    
    if df_stock.empty or 'Close' not in df_stock.columns:
        st.error(f"No price data found for ticker '{ticker}'. Please check the symbol or date range.")
        st.stop()

# Ensure Date column is datetime
df_stock['Date'] = pd.to_datetime(df_stock['Date'])
df_stock.sort_values('Date', inplace=True)
latest_price = df_stock['Close'].iloc[-1]
prev_price = df_stock['Close'].iloc[-2] if len(df_stock) > 1 else latest_price
price_change = latest_price - prev_price
pct_change = (price_change / prev_price) * 100

high_52w = df_stock['Close'].max()
low_52w = df_stock['Close'].min()
avg_vol = df_stock['Volume'].mean()

# Calculate Moving Averages
df_stock['MA10'] = df_stock['Close'].rolling(10).mean()
df_stock['MA20'] = df_stock['Close'].rolling(20).mean()
df_stock['MA50'] = df_stock['Close'].rolling(50).mean()
df_stock['Daily_Return'] = df_stock['Close'].pct_change() * 100

# Top KPI Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    change_cls = "positive" if price_change >= 0 else "negative"
    arrow = "▲" if price_change >= 0 else "▼"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{ticker} Latest Close</div>
        <div class="metric-value">${latest_price:.2f}</div>
        <div class="metric-sub {change_cls}">{arrow} ${abs(price_change):.2f} ({pct_change:+.2f}%)</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">52-Week Range</div>
        <div class="metric-value">${high_52w:.2f}</div>
        <div class="metric-sub" style="color: {TEXT_MUTED}">Low: ${low_52w:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Daily Volume</div>
        <div class="metric-value">{avg_vol/1e6:.2f}M</div>
        <div class="metric-sub" style="color: {TEXT_MUTED}">Total Observations: {len(df_stock)}</div>
    </div>
    """, unsafe_allow_html=True)

# Train Model
dataset_values = df_stock['Close'].values.reshape(-1, 1)

with st.spinner("Training LSTM Model & Calculating Waypoint Trajectories..."):
    results = train_lstm_waypoint_model(
        dataset_values,
        lookback=lookback_window,
        epochs=epochs,
        batch_size=batch_size,
        forecast_horizon=forecast_horizon
    )

with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">LSTM Model RMSE</div>
        <div class="metric-value">${results['rmse']:.2f}</div>
        <div class="metric-sub" style="color: {ACCENT_BLUE}">MAPE: {results['mape']:.2f}% | MAE: ${results['mae']:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. Main Dashboard Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📈 Stock Overview & Technicals",
    "🎯 LSTM Waypoint Prediction & Forecast",
    "📋 Historical Dataset & Export"
])

# Plotly styling setup
plot_bg = "rgba(0,0,0,0)"
grid_color = "rgba(255,255,255,0.06)" if IS_DARK else "rgba(0,0,0,0.06)"
font_family = "DM Sans, sans-serif"

# -----------------------------------------------------------------------------
# TAB 1: Stock Overview & Technicals
# -----------------------------------------------------------------------------
with tab1:
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.markdown(f'<div class="chart-header">{ticker} Price History & Moving Averages</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-desc">Interactive Closing Price curve with 10-day, 20-day, and 50-day moving average overlays</div>', unsafe_allow_html=True)
    
    fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.75, 0.25])
    
    # Close & MA lines
    fig_price.add_trace(go.Scatter(x=df_stock['Date'], y=df_stock['Close'], name='Close Price', line=dict(color='#3b82f6', width=2)), row=1, col=1)
    fig_price.add_trace(go.Scatter(x=df_stock['Date'], y=df_stock['MA10'], name='MA 10', line=dict(color='#f59e0b', width=1.5, dash='dash')), row=1, col=1)
    fig_price.add_trace(go.Scatter(x=df_stock['Date'], y=df_stock['MA20'], name='MA 20', line=dict(color='#10b981', width=1.5, dash='dash')), row=1, col=1)
    fig_price.add_trace(go.Scatter(x=df_stock['Date'], y=df_stock['MA50'], name='MA 50', line=dict(color='#ec4899', width=1.5, dash='dash')), row=1, col=1)
    
    # Volume Bar Chart
    colors = ['#22c55e' if c >= o else '#ef4444' for c, o in zip(df_stock['Close'], df_stock['Open'])]
    fig_price.add_trace(go.Bar(x=df_stock['Date'], y=df_stock['Volume'], name='Volume', marker_color=colors, opacity=0.7), row=2, col=1)
    
    fig_price.update_layout(
        paper_bgcolor=plot_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=font_family, color=TEXT_MUTED),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor=grid_color, showgrid=True),
        yaxis=dict(gridcolor=grid_color, showgrid=True, title="Price ($)"),
        yaxis2=dict(gridcolor=grid_color, showgrid=True, title="Volume"),
        height=520
    )
    st.plotly_chart(fig_price, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Secondary Technical Analysis Grid
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<div class="chart-header">Daily Returns Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-desc">Histogram of daily percentage price changes</div>', unsafe_allow_html=True)
        
        fig_hist = px.histogram(df_stock.dropna(subset=['Daily_Return']), x='Daily_Return', nbins=60,
                                color_discrete_sequence=['#3b82f6'], marginal="box")
        fig_hist.update_layout(
            paper_bgcolor=plot_bg,
            plot_bgcolor=plot_bg,
            font=dict(family=font_family, color=TEXT_MUTED),
            xaxis=dict(gridcolor=grid_color, title="Daily Return (%)"),
            yaxis=dict(gridcolor=grid_color, title="Frequency"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=320
        )
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_t2:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<div class="chart-header">Peer Stock Correlation Heatmap</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-desc">Correlation of daily returns among tech leaders</div>', unsafe_allow_html=True)
        
        peer_returns = load_peer_correlation(['AAPL', 'GOOG', 'MSFT', 'AMZN'], start_date, end_date)

        if not peer_returns.empty:
            fig_corr = px.imshow(peer_returns, text_auto=".2f", color_continuous_scale="Blues", aspect="auto")
            fig_corr.update_layout(
                paper_bgcolor=plot_bg,
                plot_bgcolor=plot_bg,
                font=dict(family=font_family, color=TEXT_MUTED),
                margin=dict(l=10, r=10, t=10, b=10),
                height=320
            )
            st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Correlation data unavailable.")
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: LSTM Waypoint Prediction & Forecast
# -----------------------------------------------------------------------------
with tab2:
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.markdown(f'<div class="chart-header">🎯 LSTM Neural Network — Waypoint Prediction & Rollout</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chart-desc">Comparing Train sequence, Actual Validation prices, Model Test Predictions, and <b>{forecast_horizon}-Day Future Waypoint Forecast</b></div>', unsafe_allow_html=True)
    
    training_len = results['training_len']
    train_dates = df_stock['Date'].iloc[:training_len]
    val_dates = df_stock['Date'].iloc[training_len:]
    
    # Future waypoint dates
    last_date = df_stock['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_horizon + 1)]
    
    fig_lstm = go.Figure()
    
    # Historical Training Line
    fig_lstm.add_trace(go.Scatter(
        x=train_dates, y=df_stock['Close'].iloc[:training_len],
        name='Historical Train Data', line=dict(color='#64748b', width=1.5)
    ))
    
    # Actual Validation Line
    fig_lstm.add_trace(go.Scatter(
        x=val_dates, y=df_stock['Close'].iloc[training_len:],
        name='Actual Validation Close', line=dict(color='#3b82f6', width=2)
    ))
    
    # LSTM Predictions Line
    fig_lstm.add_trace(go.Scatter(
        x=val_dates, y=results['predictions'].flatten(),
        name='LSTM Val Predictions', line=dict(color='#10b981', width=2, dash='dot')
    ))
    
    # Waypoint Future Projection Line
    waypoint_x = [val_dates.iloc[-1]] + future_dates
    waypoint_y = [df_stock['Close'].iloc[-1]] + list(results['waypoint_predictions'])
    
    fig_lstm.add_trace(go.Scatter(
        x=waypoint_x, y=waypoint_y,
        name=f'🚀 Waypoint {forecast_horizon}-Day Projection',
        line=dict(color='#ec4899', width=3, dash='dash'),
        marker=dict(size=8, symbol='circle')
    ))
    
    fig_lstm.update_layout(
        paper_bgcolor=plot_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=font_family, color=TEXT_MUTED),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor=grid_color, showgrid=True, title="Date"),
        yaxis=dict(gridcolor=grid_color, showgrid=True, title="Price ($)"),
        height=540
    )
    st.plotly_chart(fig_lstm, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Diagnostic Breakdown & Forecast Table
    col_d1, col_d2 = st.columns([1, 1])
    
    with col_d1:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<div class="chart-header">🔮 Future Waypoint Predictions Table</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-desc">Day-by-day projected closing prices</div>', unsafe_allow_html=True)
        
        future_df = pd.DataFrame({
            "Waypoint Day": [f"Day +{i+1}" for i in range(forecast_horizon)],
            "Projected Date": [d.strftime('%Y-%m-%d') for d in future_dates],
            "Projected Close ($)": [f"${p:.2f}" for p in results['waypoint_predictions']]
        })
        st.dataframe(future_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_d2:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<div class="chart-header">📉 Residual Prediction Errors</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-desc">Difference between actual and predicted validation values</div>', unsafe_allow_html=True)
        
        residuals = results['y_test'].flatten() - results['predictions'].flatten()
        fig_res = px.histogram(residuals, nbins=40, color_discrete_sequence=['#10b981'], labels={"value": "Error ($)"})
        fig_res.update_layout(
            paper_bgcolor=plot_bg,
            plot_bgcolor=plot_bg,
            font=dict(family=font_family, color=TEXT_MUTED),
            xaxis=dict(gridcolor=grid_color, title="Prediction Residual Error ($)"),
            yaxis=dict(gridcolor=grid_color, title="Count"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=280
        )
        st.plotly_chart(fig_res, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 3: Historical Dataset & Export
# -----------------------------------------------------------------------------
with tab3:
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.markdown(f'<div class="chart-header">📋 {ticker} Historical Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-desc">Complete price table with calculated indicators and daily return metrics</div>', unsafe_allow_html=True)
    
    export_df = df_stock[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'MA10', 'MA20', 'MA50', 'Daily_Return']].copy()
    export_df['Date'] = export_df['Date'].dt.strftime('%Y-%m-%d')
    
    col_e1, col_e2 = st.columns([4, 1])
    with col_e2:
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download CSV Dataset",
            data=csv_buffer.getvalue(),
            file_name=f"{ticker}_historical_data.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    st.dataframe(export_df.sort_values('Date', ascending=False), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown(f"""
<div style="text-align: center; color: {TEXT_MUTED}; font-size: 0.8rem; margin-top: 2rem;">
    TradePulse AI • Built with Streamlit, TensorFlow LSTM & Plotly
</div>
""", unsafe_allow_html=True)
