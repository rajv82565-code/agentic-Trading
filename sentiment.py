import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def setup_finbert_pipeline():
    """Loads the pre-trained FinBERT model and tokenizer."""
    print("Loading FinBERT model (this may take a moment on the first run)...")
    model_name = "ProsusAI/finbert"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # FinBERT label mapping: 0=Positive, 1=Negative, 2=Neutral
    labels = ['Positive', 'Negative', 'Neutral']
    
    return tokenizer, model, labels

def analyze_daily_headlines(headlines, tokenizer, model, labels):
    """
    Takes a list of headlines, predicts sentiment, and calculates 
    a composite score for the day.
    """
    results = []
    
    for text in headlines:
        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Run through the model without calculating gradients (faster inference)
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Apply softmax to logits to get probabilities (0 to 1)
        probs = F.softmax(outputs.logits, dim=-1)[0].numpy()
        predicted_class_index = torch.argmax(outputs.logits, dim=-1).item()
        
        # Extract individual probabilities
        prob_pos = probs[0]
        prob_neg = probs[1]
        prob_neu = probs[2]
        
        # Calculate a simple composite score (-1.0 to 1.0)
        # We subtract the negative probability from the positive probability
        composite_score = prob_pos - prob_neg
        
        results.append({
            'Headline': text,
            'Dominant_Sentiment': labels[predicted_class_index],
            'Prob_Positive': prob_pos,
            'Prob_Negative': prob_neg,
            'Prob_Neutral': prob_neu,
            'Composite_Score': composite_score
        })
        
    return pd.DataFrame(results)

# --- Example Usage for Your Agent ---

if __name__ == "__main__":
    # 1. Initialize the pipeline
    tokenizer, model, labels = setup_finbert_pipeline()
    
    # 2. Example daily headlines (you would fetch these dynamically via API)
    todays_headlines = [
        "Tech giant beats earnings expectations with record Q3 revenue.",
        "Supply chain disruptions threaten to delay upcoming product launches.",
        "CEO announces internal restructuring and minor layoffs.",
        "Market analysts upgrade the stock to a strong buy rating."
    ]
    
    # 3. Process the headlines
    df_sentiment = analyze_daily_headlines(todays_headlines, tokenizer, model, labels)
    
    # 4. View the results
    print("\n--- Daily Sentiment Analysis ---")
    print(df_sentiment[['Headline', 'Dominant_Sentiment', 'Composite_Score']])
    
    # 5. Aggregate into a single daily metric for your LSTM / Agent
    daily_average_sentiment = df_sentiment['Composite_Score'].mean()
    print(f"\nFinal Aggregated Daily Sentiment Score: {daily_average_sentiment:.3f}")
    # (A score > 0 is bullish, < 0 is bearish)