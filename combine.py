import os
from news import fetch_alpha_vantage_headlines
from sentiment import setup_finbert_pipeline, analyze_daily_headlines

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "3ZA3QO073AM16QI8")

# Initialize FinBERT pipeline
tokenizer, model, labels = setup_finbert_pipeline()

def run_market_intelligence_pipeline(api_key, ticker):
    print(f"--- Starting Market Intelligence for {ticker} ---")
    
    # 1. Fetch the news (Your news.py code)
    raw_headlines = fetch_alpha_vantage_headlines(api_key, ticker)
    
    if not raw_headlines:
        print(f"No news found for {ticker} today.")
        return 0.0 # Return a neutral score if no news exists
        
    # 2. Run the sentiment analysis (Your FinBERT code)
    print(f"Analyzing {len(raw_headlines)} headlines with FinBERT...")
    df_sentiment = analyze_daily_headlines(raw_headlines, tokenizer, model, labels)
    
    # 3. Output a single numerical score for the Agent
    daily_score = df_sentiment['Composite_Score'].mean()
    print(f"Final Market Sentiment Score: {daily_score:.3f}")
    
    return daily_score

if __name__ == "__main__":
    agent_sentiment_input = run_market_intelligence_pipeline(ALPHA_VANTAGE_API_KEY, "AAPL")