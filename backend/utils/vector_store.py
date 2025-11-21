import os
from chromadb import PersistentClient
from typing import List, Dict, Any


class VectorStore:
    
    def __init__(self, persist_directory: str = "/data/chroma_store"):

        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = PersistentClient(path=self.persist_directory)

        self.collection = self.client.get_or_create_collection(
            name="docs",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(
        self,
        embeddings: List[List[float]],
        chunks: List[str],
        metadatas: List[Dict[str, Any]]):

        ids = [f"doc_{i}" for i in range(len(chunks))]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )

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
                "text": docs[i],
                "metadata": metadatas[i],
                "score": 1 - distances[i]
            }
            for i in range(len(docs))
        ]
