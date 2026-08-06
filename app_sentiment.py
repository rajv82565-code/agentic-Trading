import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import requests
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS System
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FinBERT Sentiment & News Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design Tokens & CSS
BG_COLOR = "#09090b"
CARD_BG = "#111116"
TEXT_COLOR = "#fafafa"
TEXT_MUTED = "#94a3b8"
BORDER_COLOR = "#1e293b"
GREEN_COLOR = "#22c55e"
RED_COLOR = "#ef4444"
YELLOW_COLOR = "#eab308"
BLUE_ACCENT = "#3b82f6"

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
    
    /* App Header Banner */
    .header-banner {{
        background: linear-gradient(135deg, #18181b 0%, #09090b 100%);
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);
    }}
    .header-title {{
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }}
    .header-subtitle {{
        font-size: 0.95rem;
        color: {TEXT_MUTED};
        margin-top: 0.3rem;
    }}
    
    /* Key Metric Cards */
    .metric-card {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        height: 100%;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    .metric-card:hover {{
        border-color: {BLUE_ACCENT};
        transform: translateY(-2px);
    }}
    .metric-label {{
        font-size: 0.8rem;
        color: {TEXT_MUTED};
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }}
    .metric-value {{
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }}
    .metric-subtext {{
        font-size: 0.82rem;
        margin-top: 0.4rem;
        color: {TEXT_MUTED};
    }}
    
    /* Custom Badges */
    .badge-bullish {{
        background: rgba(34, 197, 94, 0.15);
        color: {GREEN_COLOR};
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 0.25rem 0.6rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }}
    .badge-bearish {{
        background: rgba(239, 68, 68, 0.15);
        color: {RED_COLOR};
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 0.25rem 0.6rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }}
    .badge-neutral {{
        background: rgba(234, 179, 8, 0.15);
        color: {YELLOW_COLOR};
        border: 1px solid rgba(234, 179, 8, 0.3);
        padding: 0.25rem 0.6rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }}
    
    /* Highlight Cards */
    .highlight-box {{
        background: {CARD_BG};
        border: 1px solid {BORDER_COLOR};
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }}
</style>
""", unsafe_allow_html=1)

# -----------------------------------------------------------------------------
# 2. Cached Backend Functions (FinBERT & News Fetcher)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_finbert():
    """Loads the pre-trained FinBERT model and tokenizer with caching."""
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    labels = ['Positive', 'Negative', 'Neutral']
    return tokenizer, model, labels

@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(api_key: str, ticker: str):
    """Fetches news headlines from Alpha Vantage REST API."""
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=12)
        data = response.json()
        articles = []
        if "feed" in data:
            for article in data["feed"]:
                articles.append({
                    'title': article.get("title", ""),
                    'url': article.get("url", "#"),
                    'source': article.get("source", "Alpha Vantage"),
                    'time_published': article.get("time_published", ""),
                    'summary': article.get("summary", "")
                })
        return articles, data, None
    except Exception as e:
        return [], {}, str(e)

def analyze_headlines(articles, tokenizer, model, labels):
    """Predicts sentiment for articles using FinBERT and computes composite scores."""
    results = []
    for art in articles:
        text = art['title']
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            
        probs = F.softmax(outputs.logits, dim=-1)[0].numpy()
        predicted_class_index = torch.argmax(outputs.logits, dim=-1).item()
        
        prob_pos = float(probs[0])
        prob_neg = float(probs[1])
        prob_neu = float(probs[2])
        composite_score = prob_pos - prob_neg
        
        results.append({
            'Headline': text,
            'Dominant_Sentiment': labels[predicted_class_index],
            'Prob_Positive': prob_pos,
            'Prob_Negative': prob_neg,
            'Prob_Neutral': prob_neu,
            'Composite_Score': composite_score,
            'Source': art.get('source', ''),
            'URL': art.get('url', '#'),
            'Time': art.get('time_published', '')
        })
        
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# 3. Sidebar Setup
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
    st.title("Market Intelligence")
    st.caption("FinBERT + Alpha Vantage Pipeline")
    st.divider()
    
    ticker_input = st.text_input("Stock Ticker", value="AAPL", max_chars=10).upper().strip()
    api_key_input = st.text_input("Alpha Vantage API Key", value="3ZA3QO073AM16QI8", type="password")
    
    st.markdown("### Parameters")
    headline_limit = st.slider("Max Headlines to Analyze", min_value=5, max_value=50, value=30, step=5)
    
    run_btn = st.button("🚀 Run Intelligence Analysis", use_container_width=True, type="primary")
    
    st.divider()
    st.info("💡 **Tip**: FinBERT classifies financial headlines into **Positive**, **Negative**, or **Neutral** sentiment and computes an aggregated composite score (-1.0 to +1.0).")

# -----------------------------------------------------------------------------
# 4. Main App Interface
# -----------------------------------------------------------------------------
# Header Banner
st.markdown(f"""
<div class="header-banner">
    <div class="header-title">
        <span>⚡ Market Sentiment & News Intelligence</span>
    </div>
    <div class="header-subtitle">
        Combining live Alpha Vantage news feeds with ProsusAI FinBERT financial sentiment transformer.
    </div>
</div>
""", unsafe_allow_html=True)

# Load FinBERT
with st.spinner("Initializing FinBERT Neural Network..."):
    tokenizer, model, labels = load_finbert()

# Trigger fetch & analysis
if ticker_input:
    with st.spinner(f"Fetching latest news & analyzing sentiment for **{ticker_input}**..."):
        articles, raw_data, err = fetch_news(api_key_input, ticker_input)
        
    if err:
        st.error(f"Error fetching news from Alpha Vantage: {err}")
    elif not articles:
        st.warning(f"No news headlines found for ticker `{ticker_input}`. Check ticker symbol or API key limit.")
        if "Information" in raw_data:
            st.info(f"API Response: {raw_data['Information']}")
        elif "Note" in raw_data:
            st.warning(f"API Note: {raw_data['Note']}")
    else:
        # Limit articles
        articles = articles[:headline_limit]
        
        # Analyze sentiment
        df_results = analyze_headlines(articles, tokenizer, model, labels)
        
        # Aggregated Metrics
        avg_score = df_results['Composite_Score'].mean()
        total_count = len(df_results)
        pos_count = (df_results['Dominant_Sentiment'] == 'Positive').sum()
        neg_count = (df_results['Dominant_Sentiment'] == 'Negative').sum()
        neu_count = (df_results['Dominant_Sentiment'] == 'Neutral').sum()
        
        if avg_score > 0.05:
            overall_signal = "BULLISH 📈"
            badge_class = "badge-bullish"
            signal_color = GREEN_COLOR
        elif avg_score < -0.05:
            overall_signal = "BEARISH 📉"
            badge_class = "badge-bearish"
            signal_color = RED_COLOR
        else:
            overall_signal = "NEUTRAL ⚖️"
            badge_class = "badge-neutral"
            signal_color = YELLOW_COLOR

        # -----------------------------------------------------------------------------
        # Top Metrics Grid
        # -----------------------------------------------------------------------------
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Target Ticker</div>
                <div class="metric-value" style="color: {BLUE_ACCENT};">{ticker_input}</div>
                <div class="metric-subtext">Analyzed {total_count} headlines</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Composite Score</div>
                <div class="metric-value" style="color: {signal_color};">{avg_score:+.3f}</div>
                <div class="metric-subtext">Scale: -1.0 (Bearish) to +1.0 (Bullish)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Market Bias</div>
                <div class="metric-value"><span class="{badge_class}">{overall_signal}</span></div>
                <div class="metric-subtext">Based on aggregated probabilities</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Sentiment Ratio</div>
                <div class="metric-value" style="font-size: 1.25rem;">
                    <span style="color:{GREEN_COLOR}">{pos_count} Pos</span> / 
                    <span style="color:{RED_COLOR}">{neg_count} Neg</span> / 
                    <span style="color:{YELLOW_COLOR}">{neu_count} Neu</span>
                </div>
                <div class="metric-subtext">Positive vs Negative vs Neutral</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # -----------------------------------------------------------------------------
        # Interactive Visualizations
        # -----------------------------------------------------------------------------
        chart_col1, chart_col2 = st.columns([1, 1])
        
        with chart_col1:
            st.subheader("🎯 Sentiment Distribution")
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Positive', 'Negative', 'Neutral'],
                values=[pos_count, neg_count, neu_count],
                hole=0.5,
                marker=dict(colors=[GREEN_COLOR, RED_COLOR, YELLOW_COLOR]),
                textinfo='label+percent+value',
                insidetextorientation='radial'
            )])
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=TEXT_COLOR, family="DM Sans"),
                margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col2:
            st.subheader("⏱️ Sentiment Composite Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=avg_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Market Sentiment Index", 'font': {'size': 16, 'color': TEXT_MUTED}},
                gauge={
                    'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': TEXT_MUTED},
                    'bar': {'color': signal_color},
                    'bgcolor': CARD_BG,
                    'bordercolor': BORDER_COLOR,
                    'steps': [
                        {'range': [-1, -0.1], 'color': 'rgba(239, 68, 68, 0.2)'},
                        {'range': [-0.1, 0.1], 'color': 'rgba(234, 179, 8, 0.2)'},
                        {'range': [0.1, 1], 'color': 'rgba(34, 197, 94, 0.2)'}
                    ],
                    'threshold': {
                        'line': {'color': "#ffffff", 'width': 3},
                        'thickness': 0.75,
                        'value': avg_score
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=TEXT_COLOR, family="DM Sans"),
                margin=dict(t=30, b=20, l=30, r=30)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.divider()

        # -----------------------------------------------------------------------------
        # Key Drivers / Highlights
        # -----------------------------------------------------------------------------
        high_col1, high_col2 = st.columns(2)
        
        with high_col1:
            st.subheader("🟢 Top Bullish Headlines")
            top_pos = df_results.sort_values(by="Composite_Score", ascending=False).head(3)
            for idx, row in top_pos.iterrows():
                st.markdown(f"""
                <div class="highlight-box" style="border-left: 4px solid {GREEN_COLOR};">
                    <div style="font-weight: 600; font-size: 0.95rem;">{row['Headline']}</div>
                    <div style="font-size: 0.8rem; color: {TEXT_MUTED}; margin-top: 0.4rem;">
                        Score: <span style="color:{GREEN_COLOR}; font-weight:700;">+{row['Composite_Score']:.3f}</span> | Source: {row['Source']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        with high_col2:
            st.subheader("🔴 Top Bearish Headlines")
            top_neg = df_results.sort_values(by="Composite_Score", ascending=True).head(3)
            for idx, row in top_neg.iterrows():
                st.markdown(f"""
                <div class="highlight-box" style="border-left: 4px solid {RED_COLOR};">
                    <div style="font-weight: 600; font-size: 0.95rem;">{row['Headline']}</div>
                    <div style="font-size: 0.8rem; color: {TEXT_MUTED}; margin-top: 0.4rem;">
                        Score: <span style="color:{RED_COLOR}; font-weight:700;">{row['Composite_Score']:.3f}</span> | Source: {row['Source']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # -----------------------------------------------------------------------------
        # Detailed Interactive Table & Filter
        # -----------------------------------------------------------------------------
        st.subheader("📰 Detailed Headline Intelligence Feed")
        
        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            sentiment_filter = st.multiselect(
                "Filter Sentiment",
                options=["Positive", "Negative", "Neutral"],
                default=["Positive", "Negative", "Neutral"]
            )
        with f_col2:
            search_query = st.text_input("Search Headlines by Keyword", value="")
            
        filtered_df = df_results[df_results['Dominant_Sentiment'].isin(sentiment_filter)]
        if search_query:
            filtered_df = filtered_df[filtered_df['Headline'].str.contains(search_query, case=False, na=False)]

        st.dataframe(
            filtered_df[['Headline', 'Dominant_Sentiment', 'Composite_Score', 'Prob_Positive', 'Prob_Negative', 'Prob_Neutral', 'Source']],
            column_config={
                "Headline": st.column_config.TextColumn("Headline", width="large"),
                "Dominant_Sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                "Composite_Score": st.column_config.NumberColumn("Composite Score", format="%+.3f"),
                "Prob_Positive": st.column_config.ProgressColumn("Positive Prob", format="%.2f", min_value=0, max_value=1),
                "Prob_Negative": st.column_config.ProgressColumn("Negative Prob", format="%.2f", min_value=0, max_value=1),
                "Prob_Neutral": st.column_config.ProgressColumn("Neutral Prob", format="%.2f", min_value=0, max_value=1),
            },
            use_container_width=True,
            hide_index=True
        )

        # Download Report
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Sentiment Intelligence Report (CSV)",
            data=csv_data,
            file_name=f"{ticker_input}_sentiment_report.csv",
            mime="text/csv"
        )
