# app/utils/helpers.py

import re


STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "que", "qué", "es", "son", "en", "por",
    "para", "con", "se", "mi", "su", "lo", "a", "y", "o", "ha", "hay"
}

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñ0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    words = [word for word in text.split() if word not in STOPWORDS]
    cleaned = " ".join(words).strip()
    return cleaned
