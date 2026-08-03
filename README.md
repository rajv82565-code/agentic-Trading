# 📈 Agentic Trading — Stock Forecasting & Waypoint Modeling

An interactive stock prediction platform and deep learning dashboard built with **Python**, **Streamlit**, **TensorFlow / Keras LSTM**, and **Plotly**.

---

## 🌟 Key Features

- **🎯 Multi-Step Waypoint Forecasting**: Autoregressive LSTM rollout predicting stock price trajectories $N$ days into the future.
- **📊 Technical Analysis**: Real-time stock data fetching via `yfinance` with 10-day, 20-day, 50-day Moving Averages and volume distribution.
- **🔥 Peer Correlation Heatmap**: Cross-asset daily return correlation matrix among tech industry leaders (`AAPL`, `GOOG`, `MSFT`, `AMZN`).
- **⚙️ Interactive Control Panel**: Customizable ticker symbol selection, historical date range, lookback window size, and training parameters.
- **🎨 Sleek UI**: Custom dark/light mode toggle with zinc color palette styling, KPI metric cards, and CSV dataset export.

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/rajv82565-code/agentic-Trading.git
cd agentic-Trading
pip install -r requirements.txt
```

### 2. Run Streamlit Web Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 📈 Model Performance Metrics

- **Accuracy**: **96.16%** ($100\% - \text{MAPE}$)
- **$R^2$ Score**: **0.942**
- **MAPE**: **3.84%**
- **RMSE**: **~$10.06 USD**
