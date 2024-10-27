from datetime import datetime
from polarity_finder import find_polarity
from keyword_extractor import extract_keywords
from concern_classifier import classify_concern
from intensity_scoring import score_intensity 

# To store results in a timeline
timeline_data = []

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
        if entry["polarity"] != previous_entry["polarity"]:
            shift = f"Shift from {previous_entry['polarity']} to {entry['polarity']}"
        else:
            shift = "No significant sentiment shift"
    else:
        shift = "First entry, no shift analysis"
    
    return entry, shift

# Example usage over a timeline
track_timeline("I feel very low and can’t sleep well.")
track_timeline("I feel a bit better but still anxious.")
for data in timeline_data:
    print(data)
