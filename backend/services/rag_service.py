from utils.file_loader import load_file
from utils.token_chunker import token_chunk_text
from utils.embeddings import JinaEmbedding
from services.vector_store import VectorStore
from config.settings import settings
from typing import List
from openai import OpenAI
import logging

logging.basicConfig(level=logging.INFO)

class RAGService:
    def __init__(self):
        self.embedder = JinaEmbedding()
        self.vector_store = VectorStore()

    def ingest_document(self, file_path: str):
        """
        Load → chunk → embed → store a document.
        """
        logging.info(f"[RAG] Loading file: {file_path}")
        text = load_file(file_path)

        logging.info("[RAG] Chunking text...")
        chunks = token_chunk_text(text)
        logging.info(f"[RAG] Total chunks: {len(chunks)}")

        logging.info("[RAG] Generating embeddings...")
        embeddings = self.embedder.embed_batch(chunks)

        logging.info("[RAG] Storing in vector DB...")
        self.vector_store.add_texts(chunks, embeddings)

        return {"status": "success", "chunks_added": len(chunks)}

    def retrieve(self, query: str, top_k: int = 4):
        """
        Vector search: embed query + find top chunks.
        """
        query_embedding = self.embedder.embed(query)
        return self.vector_store.search(query_embedding, top_k)

    def generate_answer(self, query: str, model="gpt-4o-mini"):
        
        retrieved = self.retrieve(query)
        
        if not retrieved:
            return "No relevant information found in the knowledge base."

        context = "\n\n".join([r["text"] for r in retrieved])

        prompt = f"""
You are an AI assistant. Use ONLY the provided context.

### CONTEXT:
{context}

### USER QUESTION:
{query}

### ANSWER:
"""

        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        logging.info("[RAG] Sending query to OpenAI API...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        logging.info("[RAG] Received response from OpenAI API.")
        return response.choices[0].message["content"]
