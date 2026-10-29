
from pathlib import Path

from app.rag.documents import build_chunks


PROJECT_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"


def main() -> None:
    chunks = build_chunks(KNOWLEDGE_BASE_DIR)

    if not chunks:
        print("No chunks found. Check your Markdown files.")
        return

    print(f"Successfully created {len(chunks)} chunks.\n")

    for chunk in chunks:
        print(f"ID: {chunk.chunk_id}")
        print(f"Source: {chunk.source}")
        print(f"Heading: {chunk.heading}")
        print(f"Content:\n{chunk.text}")
        print("-" * 50)


if __name__ == "__main__":
    main()
