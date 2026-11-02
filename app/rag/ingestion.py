
from pathlib import Path

from app.rag.documents import build_chunks
from app.rag.embeddings import EMBEDDING_MODEL, generate_embedding
from app.rag.vector_store import (
    initialize_vector_store,
    save_embedding,
)


def ingest_knowledge_base(
    knowledge_base_dir: Path,
    db_path: Path,
) -> int:
    """Chunk, embed and store all knowledge-base documents."""

    chunks = build_chunks(knowledge_base_dir)

    if not chunks:
        raise ValueError("No knowledge-base chunks were found.")

    initialize_vector_store(db_path)

    successful_chunks = 0

    for index, chunk in enumerate(chunks, start=1):
        print(
            f"Embedding chunk {index}/{len(chunks)}: "
            f"{chunk.chunk_id}"
        )

        embedding = generate_embedding(chunk.text)

        save_embedding(
            db_path=db_path,
            chunk=chunk,
            embedding=embedding,
            model=EMBEDDING_MODEL,
        )

        successful_chunks += 1

    return successful_chunks
