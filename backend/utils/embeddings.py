import os
import httpx
import dotenv

dotenv.load_dotenv('.env')

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_API_KEY = os.getenv('JINA_API_KEY')

headers = {
    "Authorization": f"Bearer {JINA_API_KEY}",
    "Content-Type": "application/json"
}

async def get_embedding(text: str):

    """Using JINA Embedding API to convert text to embedding vectors."""
    
    payload = {
        "model": "jina-embeddings-v3",
        "input": text
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(JINA_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]
