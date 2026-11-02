
import json
import math
import sqlite3
from contextlib import closing

from pathlib import Path

from app.rag.documents import DocumentChunk


def initialize_vector_store(db_path: Path) -> None:
    """Create the knowledge-chunk storage table if needed."""

    with closing(sqlite3.connect(db_path)) as connection:
        with connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    heading TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_dimensions INTEGER NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)


def save_embedding(
    db_path: Path,
    chunk: DocumentChunk,
    embedding: list[float],
    model: str,
) -> None:
    """Save or update a document chunk and its embedding."""

    if not embedding:
        raise ValueError("Embedding cannot be empty.")

    if not all(math.isfinite(value) for value in embedding):
        raise ValueError("Embedding contains invalid values.")

    embedding_json = json.dumps(embedding)

    with closing(sqlite3.connect(db_path)) as connection:
        with connection:
            connection.execute("""
                INSERT INTO knowledge_chunks (
                    chunk_id,
                    source,
                    heading,
                    content,
                    embedding_json,
                    embedding_model,
                    embedding_dimensions
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)

                ON CONFLICT(chunk_id) DO UPDATE SET
                    source = excluded.source,
                    heading = excluded.heading,
                    content = excluded.content,
                    embedding_json = excluded.embedding_json,
                    embedding_model = excluded.embedding_model,
                    embedding_dimensions = excluded.embedding_dimensions,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                chunk.chunk_id,
                chunk.source,
                chunk.heading,
                chunk.text,
                embedding_json,
                model,
                len(embedding),
            ))


def get_stored_embeddings(db_path: Path) -> list[dict]:
    """Retrieve stored chunks and their embeddings."""

    with closing(sqlite3.connect(db_path)) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute("""
            SELECT
                chunk_id,
                source,
                heading,
                content,
                embedding_json,
                embedding_model,
                embedding_dimensions
            FROM knowledge_chunks
        """).fetchall()

    results = []

    for row in rows:
        embedding = json.loads(row["embedding_json"])

        if (
            not isinstance(embedding, list)
            or len(embedding) != row["embedding_dimensions"]
        ):
            raise ValueError(
                f"Invalid embedding stored for {row['chunk_id']}"
            )

        results.append({
            "chunk_id": row["chunk_id"],
            "source": row["source"],
            "heading": row["heading"],
            "content": row["content"],
            "embedding": embedding,
            "embedding_model": row["embedding_model"],
            "embedding_dimensions": row["embedding_dimensions"],
        })

    return results
