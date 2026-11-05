
import re
import unicodedata


def normalize_unicode(text):
    """Normalize Telugu text using Unicode NFC."""
    return unicodedata.normalize("NFC", text)


def clean_whitespace(text):
    """Remove extra spaces, tabs, and line breaks."""
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text):
    """Apply preprocessing without removing Telugu characters."""
    if not isinstance(text, str):
        raise TypeError("Input text must be a string")

    text = normalize_unicode(text)
    text = clean_whitespace(text)

    return text
