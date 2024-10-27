from transformers import pipeline, TFBertForSequenceClassification, BertTokenizer
import spacy
import tensorflow as tf
from datetime import datetime

import pandas as pd
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.model_selection import train_test_split
import tensorflow as tf

#csv loading
# Load the CSV data
data = pd.read_csv("../datasets/mental_health_dataset.csv")

# Display the first few rows of the dataframe
print(data.head())

# Prepare the data for training
# Map categories to numeric labels
category_mapping = {category: idx for idx, category in enumerate(data['Category'].unique())}
data['Category'] = data['Category'].map(category_mapping)

# Split into features and labels
X = data['User Input'].values
y = data['Category'].values

# Tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
X_tokenized = tokenizer(X.tolist(), padding=True, truncation=True, return_tensors='tf', max_length=128)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tokenized, y, test_size=0.2, random_state=42)

# Load the BERT model for sequence classification
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(category_mapping))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# Train the model
model.fit(X_train['input_ids'], y_train, epochs=3, batch_size=8, validation_data=(X_test['input_ids'], y_test))

# Save the trained model
model.save_pretrained("trained_model")
tokenizer.save_pretrained("trained_model")

# Update the classify_concern function to use the trained model
def classify_concern(text):
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=128)
    outputs = model(inputs)
    predicted_class = tf.argmax(outputs.logits, axis=1).numpy()[0]
    categories = list(category_mapping.keys())
    return categories[predicted_class]



#the nlp partty
# Load models
sentiment_analyzer = pipeline("sentiment-analysis")
nlp = spacy.load("en_core_web_sm")
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=5)  # assuming 5 categories

# To store results in a timeline
timeline_data = []

def find_polarity(text):
    result = sentiment_analyzer(text)
    polarity = result[0]['label']
    score = result[0]['score']
    return polarity, score

def extract_keywords(text):
    doc = nlp(text)
    keywords = [ent.text for ent in doc.ents if ent.label_ == "MENTAL_HEALTH"]
    return keywords

def classify_concern(text):
    inputs = tokenizer(text, return_tensors="tf")
    outputs = model(inputs)
    predicted_class = tf.argmax(outputs.logits, axis=1).numpy()[0]
    categories = ["Anxiety", "Depression", "Stress", "Insomnia", "Eating Disorder"]
    return categories[predicted_class]

def score_intensity(text):
    if "very" in text or "extremely" in text:
        return 8
    elif "slightly" in text or "a bit" in text:
        return 4
    else:
        return 6  # default intensity

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
# user_input = "I feel very anxious and can't sleep at night."
# analyze_input(user_input)

# user_input = "Today is a bit better, but I'm still feeling low."
# analyze_input(user_input)


# Print timeline data
print("\n--- Timeline Data ---")
for data in timeline_data:
    print(data)





# Example usage with the updated model
print("\n--- after csv ---")
user_input = "I'm feeling quite anxious these days."
concern = classify_concern(user_input)
print("Classified Concern:", concern)
