import tiktoken
from typing import List

"""Loading the tokenizer"""
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def token_chunk_text(
    text: str,
    max_tokens: int = 300,
    overlap: int = 50
) -> List[str]:

    tokens = tokenizer.encode(text)
    chunks = []

    start = 0
    total_tokens = len(tokens)

    while start < total_tokens:
        end = start + max_tokens
        chunk_tokens = tokens[start:end]

        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)

        start += max_tokens - overlap

    return chunks
