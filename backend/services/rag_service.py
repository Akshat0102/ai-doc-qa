import os
from utils.file_loader import load_file
from utils.chunker_token import token_chunk_text
from utils.embeddings import get_embedding
from utils.vector_store import VectorStore
from typing import List
from openai import OpenAI
import logging
import asyncio

logging.basicConfig(level=logging.INFO)

class RAGService:

    def __init__(self):
        self.vector_store = VectorStore()


    async def ingest_document(self, file_path: str):
        
        logging.info(f"[RAG] Loading file: {file_path}")
        text = load_file(file_path)

        logging.info("[RAG] Chunking text...")
        chunks = token_chunk_text(text)
        logging.info(f"[RAG] Total chunks: {len(chunks)}")

        logging.info("[RAG] Generating embeddings...")
        embeddings = await asyncio.gather(*(get_embedding(chunk) for chunk in chunks))

        logging.info("[RAG] Storing in vector DB...")
        
        self.vector_store.add_documents(
            embeddings=embeddings,
            chunks=chunks,
            metadatas=[{"source": file_path}] * len(chunks)
        )
        return {"status": "success", "chunks_added": len(chunks)}


    async def retrieve(self, query: str, top_k: int = 4):
        """
        Vector search: embed query + find top chunks.
        """
        query_embedding = await get_embedding(query)
        return self.vector_store.search(query_embedding, top_k)


    async def generate_answer(self, query: str, model="gpt-4o-mini"):
        
        retrieved = await self.retrieve(query)
        
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

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        logging.info("[RAG] Sending query to OpenAI API...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        logging.info("[RAG] Received response from OpenAI API.")
        return response.choices[0].message.content
