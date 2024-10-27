import spacy

# Load spaCy model (a pre-trained model that could be fine-tuned for specific mental health terms)
nlp = spacy.load("en_core_web_sm")

def extract_keywords(text):
    doc = nlp(text)
    keywords = [ent.text for ent in doc.ents if ent.label_ == "MENTAL_HEALTH"]
    return keywords

# Example usage
text = "I've been feeling very anxious lately."
keywords = extract_keywords(text)
print("Extracted Keywords:", keywords)
