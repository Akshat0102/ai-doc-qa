import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any


class VectorStore:

    def __init__(self, persist_directory: str = "./backend/data/chroma_store"):
        """
        Function to initialize ChromaDb Client.
        """
        self.persist_directory = persist_directory

        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = chromadb.Client(
            Settings(
                persist_directory=self.persist_directory,
                chroma_db_impl="duckdb+parquet"
            )
        )

        self.collection = self.client.get_or_create_collection(
            name="docs",
            metadata={"hnsw:space": "cosine"}  # cosine similarity
        )

    def add_documents(
        self,
        embeddings: List[List[float]],
        chunks: List[str],
        metadatas: List[Dict[str, Any]]):
        """Store embeddings + documents + metadata."""
        ids = [f"doc_{i}_{len(chunks)}" for i in range(len(chunks))]

        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )

        self.client.persist()

    def search(self, query_embedding: List[float], top_k: int = 5):

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            {
                "chunk": docs[i],
                "metadata": metadatas[i],
                "score": 1 - distances[i] 
            }
            for i in range(len(docs))
        ]
