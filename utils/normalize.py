import re

# Simple normalization for English, Hindi, Marathi
def normalize_word(word: str, lang: str = "en") -> str:
    """
    Normalize word for comparison.
    - English: lowercase + remove punctuation
    - Hindi/Marathi: remove spaces + normalize unicode chars
    """
    if lang == "en":
        return re.sub(r"[^a-z]", "", word.lower())
    elif lang in ["hi", "mr"]:
        # Remove punctuation, spaces, and normalize unicode
        word = word.strip().replace(" ", "")
        # You can add more sophisticated normalization if needed
        return word
    else:
        return word.lower()
