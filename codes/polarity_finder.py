from transformers import pipeline

# Load sentiment analysis pipeline from Hugging Face
sentiment_analyzer = pipeline("sentiment-analysis")

def find_polarity(text):
    result = sentiment_analyzer(text)
    polarity = result[0]['label']
    score = result[0]['score']
    return polarity, score

# Example usage
text = "I feel very anxious lately"
polarity, score = find_polarity(text)
print(f"Polarity: {polarity}, Score: {score}")
