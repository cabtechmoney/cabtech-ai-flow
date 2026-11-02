
import math
from pathlib import Path

from app.rag.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    generate_embedding,
)
from app.rag.vector_store import get_stored_embeddings


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """Calculate cosine similarity between two vectors."""

    if len(vector_a) != len(vector_b):
        raise ValueError("Vector dimensions must match.")

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def search_knowledge(
    query: str,
    db_path: Path,
    top_k: int = 3,
) -> list[dict]:
    """Find the most semantically relevant knowledge chunks."""

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10.")

    query_embedding = generate_embedding(query)

    stored_chunks = get_stored_embeddings(db_path)

    results = []

    for chunk in stored_chunks:
        if chunk["embedding_model"] != EMBEDDING_MODEL:
            continue

        if chunk["embedding_dimensions"] != EMBEDDING_DIMENSIONS:
            continue

        score = cosine_similarity(
            query_embedding,
            chunk["embedding"],
        )

        results.append({
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "heading": chunk["heading"],
            "content": chunk["content"],
            "similarity": round(score, 4),
        })

    results.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return results[:top_k]
