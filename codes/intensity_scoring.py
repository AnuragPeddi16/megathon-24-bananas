def score_intensity(text):
    if "very" in text or "extremely" in text:
        return 8
    elif "slightly" in text or "a bit" in text:
        return 4
    else:
        return 6  # default intensity

# Example usage
intensity = score_intensity("I feel extremely anxious")
print("Intensity Score:", intensity)
