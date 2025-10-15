# src/cleaning.py
import re

def clean_text(text):
    """
    Clean review text: lowercase, remove numbers & punctuation
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text
