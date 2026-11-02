
import tempfile
from pathlib import Path

from app.rag.documents import DocumentChunk
from app.rag.embeddings import (
    EMBEDDING_MODEL,
    generate_embedding,
)
from app.rag.vector_store import (
    get_stored_embeddings,
    initialize_vector_store,
    save_embedding,
)


def main() -> None:
    print("=" * 55)
    print("CABTECH AI FLOW - VECTOR STORAGE TEST")
    print("=" * 55)

    sample_text = (
        "CabTech Hub develops web applications "
        "and integrates backend APIs."
    )

    try:
        # Use a temporary database so your existing
        # customer database is never modified.
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_vectors.db"

            initialize_vector_store(db_path)

            chunk = DocumentChunk(
                chunk_id="test:1",
                source="test.md",
                heading="Web Development",
                text=sample_text,
            )

            embedding = generate_embedding(sample_text)

            save_embedding(
                db_path=db_path,
                chunk=chunk,
                embedding=embedding,
                model=EMBEDDING_MODEL,
            )

            stored = get_stored_embeddings(db_path)

            assert len(stored) == 1
            assert stored[0]["chunk_id"] == "test:1"
            assert stored[0]["embedding_dimensions"] == len(embedding)
            assert stored[0]["embedding"] == embedding

            # Save the same chunk again to test upsert.
            save_embedding(
                db_path=db_path,
                chunk=chunk,
                embedding=embedding,
                model=EMBEDDING_MODEL,
            )

            stored_again = get_stored_embeddings(db_path)

            assert len(stored_again) == 1

            print("\nVector storage successful!")
            print(f"Stored chunks: {len(stored_again)}")
            print(f"Vector dimensions: {len(embedding)}")
            print("Duplicate prevention: PASSED")
            print("\nVECTOR STORAGE TEST PASSED.")

    except Exception as error:
        print(f"\nVECTOR STORAGE TEST FAILED: {error}")
        raise


if __name__ == "__main__":
    main()
