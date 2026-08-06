import requests

def fetch_alpha_vantage_headlines(api_key, ticker):
    """
    Fetches financial news headlines using the Alpha Vantage REST API.
    """
    # The REST API base URL is https://www.alphavantage.co/.
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}"
    
    print(f"Fetching news for {ticker} from Alpha Vantage...")
    response = requests.get(url)
    data = response.json()
    
    headlines = []
    
    # Alpha Vantage stores the articles in a list called 'feed'
    if "feed" in data:
        for article in data["feed"]:
            # We only need the title string to pass to FinBERT
            headlines.append(article["title"])
            
    return headlines

import os

# 1. Insert your API key as a string or environment variable
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "3ZA3QO073AM16QI8") 
TARGET_TICKER = "AAPL"

# 2. Fetch the data
live_headlines = fetch_alpha_vantage_headlines(ALPHA_VANTAGE_API_KEY, TARGET_TICKER)

print(f"Successfully extracted {len(live_headlines)} headlines.")
print("Sample Headline:", live_headlines[0] if live_headlines else "No news found.")