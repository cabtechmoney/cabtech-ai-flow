
from pathlib import Path

from app.rag.retrieval import search_knowledge


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_ROOT / "cabtech.db"


def main() -> None:
    print("=" * 60)
    print("CABTECH AI FLOW - SEMANTIC RETRIEVAL TEST")
    print("=" * 60)

    queries = [
        "Can you build an online store for my clothing business?",
        "I need help connecting two applications through an API.",
        "My application is experiencing technical problems.",
    ]

    try:
        for query in queries:
            print(f"\nQUESTION: {query}")
            print("-" * 60)

            results = search_knowledge(
                query=query,
                db_path=DATABASE_PATH,
                top_k=3,
            )

            if not results:
                print("No relevant documents found.")
                continue

            for index, result in enumerate(results, start=1):
                print(f"\nResult {index}")
                print(f"Source: {result['source']}")
                print(f"Heading: {result['heading']}")
                print(f"Similarity: {result['similarity']}")
                print(f"Content: {result['content']}")

        print("\nSEMANTIC RETRIEVAL TEST COMPLETED.")

    except Exception as error:
        print(f"\nRETRIEVAL TEST FAILED: {error}")
        raise


if __name__ == "__main__":
    main()
