import re
from typing import List

def clean_text(text: str) -> str:

    """Function to clean input text by removing unwanted characters and whitespaces."""

    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:

    """Function to split text into chunks"""

    text = clean_text(text)

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks