
from pathlib import Path

from app.rag.ingestion import ingest_knowledge_base
from app.rag.vector_store import get_stored_embeddings


PROJECT_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
DATABASE_PATH = PROJECT_ROOT / "cabtech.db"


def main() -> None:
    print("=" * 55)
    print("CABTECH AI FLOW - KNOWLEDGE INGESTION TEST")
    print("=" * 55)

    try:
        count = ingest_knowledge_base(
            knowledge_base_dir=KNOWLEDGE_BASE_DIR,
            db_path=DATABASE_PATH,
        )

        stored_chunks = get_stored_embeddings(DATABASE_PATH)

        print(f"\nSuccessfully processed: {count}")
        print(f"Stored knowledge chunks: {len(stored_chunks)}")

        print("\nKNOWLEDGE INGESTION TEST PASSED.")

    except Exception as error:
        print(f"\nINGESTION TEST FAILED: {error}")
        raise


if __name__ == "__main__":
    main()
