from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf

# Load a pre-trained BERT model for TensorFlow
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=5) # assuming 5 categories

def classify_concern(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="tf")
    # Run the inputs through the model
    outputs = model(inputs)
    # Get the predicted class
    predicted_class = tf.argmax(outputs.logits, axis=1).numpy()[0]
    # Define the categories
    categories = ["Anxiety", "Depression", "Stress", "Insomnia", "Eating Disorder"]
    return categories[predicted_class]

# Example usage
concern = classify_concern("feeling very anxious")
print("Classified Concern:", concern)
