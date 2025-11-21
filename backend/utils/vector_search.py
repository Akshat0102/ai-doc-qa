import json
import os
import numpy as np
from typing import List, Dict, Any

VECTOR_STORE_PATH = "backend/data/vector_store.json"

def load_vector_store() -> List[Dict[str, Any]]:
    """Function to load vector store from disk"""
    if not os.path.exists(VECTOR_STORE_PATH):
        return []

    with open(VECTOR_STORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_vector_store(store: List[Dict[str, Any]]):
    """Function to save vector store to disk"""
    with open(VECTOR_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)


def normalize(vec: List[float]) -> np.ndarray:
    """Function to L2-normalize embedding vector."""
    v = np.array(vec, dtype=float)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def dedupe_chunks(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Function to remove duplicate or near duplicate chunks"""
    seen = set()
    unique = []

    for r in results:
        chunk = r["chunk"].strip()

        if chunk not in seen:
            seen.add(chunk)
            unique.append(r)

    return unique


def apply_score_threshold(results: List[Dict[str, Any]], threshold: float = 0.4) -> List[Dict[str, Any]]:
    """Function to filter only relevant results"""
    return [r for r in results if r["score"] >= threshold]


def rerank_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for r in results:
        length_bonus = len(r["chunk"]) / 1000
        r["score"] = r["score"] + (0.02 * length_bonus)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def search_vector_store(
    query_embedding: List[float],
    top_k: int = 5,
    score_threshold: float = 0.4,
    rerank: bool = True
) -> List[Dict[str, Any]]:
    """Function to search the vector store for relevant chunks given a query embedding."""
    store = load_vector_store()
    if not store:
        return []

    query_vec = normalize(query_embedding)

    scored = []

    for item in store:
        doc_vec = normalize(item["embedding"])
        score = cosine_similarity(query_vec, doc_vec)

        scored.append({
            "chunk": item["chunk"],
            "metadata": item.get("metadata", {}),
            "score": score
        })

    filtered = apply_score_threshold(scored, threshold=score_threshold)

    unique = dedupe_chunks(filtered)

    if rerank:
        final_ranked = rerank_results(unique)
    else:
        final_ranked = sorted(unique, key=lambda x: x["score"], reverse=True)

    return final_ranked[:top_k]
