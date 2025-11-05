import re
from unidecode import unidecode

def clean_text(text):
    """
    Clean a text string:
    - Convert to lowercase
    - Remove accents
    - Remove extra spaces
    - Remove punctuation (keep French characters)
    """
    text = str(text)
    text = unidecode(text)  # Remove accents
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s-]', '', text)  # Remove punctuation
    return text.strip()
