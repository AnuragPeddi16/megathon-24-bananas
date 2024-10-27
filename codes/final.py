from transformers import pipeline, TFBertForSequenceClassification, BertTokenizer
import spacy
import tensorflow as tf
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV data
data = pd.read_csv("../datasets/mental_health_dataset.csv")
print(data.head())  # Display the first few rows of the dataframe

# Prepare the data for training
category_mapping = {category: idx for idx, category in enumerate(data['Category'].unique())}
data['Category'] = data['Category'].map(category_mapping)

# Split into features and labels
X = data['User Input'].values
y = data['Category'].values
# Tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
X_tokenized = tokenizer(X.tolist(), padding=True, truncation=True, return_tensors='tf', max_length=128)

# Convert tokenized inputs to NumPy arrays
X_input_ids = X_tokenized['input_ids'].numpy()  # Convert to NumPy array
X_attention_mask = X_tokenized['attention_mask'].numpy()  # Convert to NumPy array

# Train-test split
X_train_ids, X_test_ids, y_train, y_test = train_test_split(X_input_ids, y, test_size=0.2, random_state=42)
X_train_mask, X_test_mask = train_test_split(X_attention_mask, test_size=0.2, random_state=42)

# Load the BERT model for sequence classification
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(category_mapping))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# Train the model
model.fit([X_train_ids, X_train_mask], y_train, epochs=3, batch_size=8, validation_data=([X_test_ids, X_test_mask], y_test))

# Save the trained model
model.save_pretrained("trained_model")
tokenizer.save_pretrained("trained_model")                                                                                                                                                                                      

# Define the classify_concern function
def classify_concern(text):
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=128)
    outputs = model(inputs['input_ids'])
    predicted_class = tf.argmax(outputs.logits, axis=1).numpy()[0]
    categories = list(category_mapping.keys())
    return categories[predicted_class]

# Initialize NLP components
sentiment_analyzer = pipeline("sentiment-analysis")
nlp = spacy.load("en_core_web_sm")

# To store results in a timeline
timeline_data = []

# Function to find polarity
def find_polarity(text):
    result = sentiment_analyzer(text)
    polarity = result[0]['label']
    score = result[0]['score']
    return polarity, score

# Function to extract keywords using NER
def extract_keywords(text):
    doc = nlp(text)
    keywords = [ent.text for ent in doc.ents if ent.label_ == "MENTAL_HEALTH"]
    return keywords

# Function to score intensity
def score_intensity(text):
    if "very" in text or "extremely" in text:
        return 8
    elif "slightly" in text or "a bit" in text:
        return 4
    else:
        return 6  # default intensity

# Function to track the input timeline
def track_timeline(input_text):
    timestamp = datetime.now()
    polarity, score = find_polarity(input_text)
    keywords = extract_keywords(input_text)
    concern_category = classify_concern(" ".join(keywords))
    intensity = score_intensity(" ".join(keywords))
    
    entry = {
        "timestamp": timestamp,
        "polarity": polarity,
        "score": score,
        "keywords": keywords,
        "concern_category": concern_category,
        "intensity": intensity
    }
    timeline_data.append(entry)
    
    # Analyze sentiment shift
    if len(timeline_data) > 1:
        previous_entry = timeline_data[-2]
        shift = f"Shift from {previous_entry['polarity']} to {entry['polarity']}" if entry["polarity"] != previous_entry["polarity"] else "No significant sentiment shift"
    else:
        shift = "First entry, no shift analysis"
    
    return entry, shift

# Function to analyze user input and print the results
def analyze_input(input_text):
    entry, shift = track_timeline(input_text)
    
    # Print the attributes
    print("\n--- Analysis Result ---")
    print(f"Timestamp: {entry['timestamp']}")
    print(f"Input Text: {input_text}")
    print(f"Polarity: {entry['polarity']}, Score: {entry['score']:.2f}")
    print(f"Keywords: {entry['keywords']}")
    print(f"Concern Category: {entry['concern_category']}")
    print(f"Intensity Score: {entry['intensity']}")
    print(f"Sentiment Shift: {shift}")

# Example usage
# Uncomment to test the analysis function
# analyze_input("I feel very anxious and can't sleep at night.")
# analyze_input("Today is a bit better, but I'm still feeling low.")

# Print the timeline data
print("\n--- Timeline Data ---")
for data in timeline_data:
    print(data)

# Example usage with the updated model
print("\n--- After CSV ---")
user_input = "I'm feeling quite anxious these days."
concern = classify_concern(user_input)
print("Classified Concern:", concern)
