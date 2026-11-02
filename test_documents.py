
from pathlib import Path

from app.rag.documents import build_chunks

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"


def main() -> None:
    print("=" * 60)
    print("CABTECH AI FLOW - DOCUMENT CHUNKING TEST")
    print("=" * 60)

    try:
        chunks = build_chunks(
            knowledge_base_dir=KNOWLEDGE_BASE_DIR,
            max_chars=1200,
        )

        if not chunks:
            print("\nNo chunks found.")
            print("Check your knowledge_base folder and Markdown files.")
            return

        print(f"\nSuccessfully created {len(chunks)} chunks.\n")

        for index, chunk in enumerate(chunks, start=1):
            print(f"CHUNK {index}")
            print("-" * 60)
            print(f"ID: {chunk.chunk_id}")
            print(f"Source: {chunk.source}")
            print(f"Heading: {chunk.heading}")
            print(f"Characters: {len(chunk.text)}")
            print("\nContent:")
            print(chunk.text)
            print("-" * 60)

        print("\nDOCUMENT CHUNKING TEST PASSED.")

    except FileNotFoundError as error:
        print(f"\nERROR: {error}")

    except Exception as error:
        print(f"\nUNEXPECTED ERROR: {error}")


if __name__ == "__main__":
    main()
