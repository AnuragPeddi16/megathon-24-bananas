from time_based import find_polarity, extract_keywords, classify_concern, score_intensity, track_timeline

def process_user_input(input_text):
    polarity, _ = find_polarity(input_text)
    keywords = extract_keywords(input_text)
    concern = classify_concern(" ".join(keywords))
    intensity = score_intensity(" ".join(keywords))
    
    entry, shift = track_timeline(input_text)
    
    return {
        "polarity": polarity,
        "keywords": keywords,
        "concern": concern,
        "intensity": intensity,
        "shift": shift
    }

# Example sequence of inputs
inputs = [
    "I can't sleep well and I feel very low.",
    "I feel a bit better but still anxious."
]

for input_text in inputs:
    result = process_user_input(input_text)
    print(result)
