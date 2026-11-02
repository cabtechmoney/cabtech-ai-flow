
from pathlib import Path

from app.rag.generation import (
    GENERATION_MODEL,
    generate_customer_response,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_ROOT / "cabtech.db"


def main() -> None:
    print("=" * 60)
    print("CABTECH AI FLOW - RESPONSE GENERATION TEST")
    print("=" * 60)

    question = (
        "I want to create an online clothing store. "
        "Can CabTech Hub help me?"
    )

    try:
        result = generate_customer_response(
            question=question,
            db_path=DATABASE_PATH,
            top_k=3,
        )

        print(f"\nModel: {GENERATION_MODEL}")
        print(f"\nCustomer question: {question}")

        print("\nGENERATED ANSWER")
        print("-" * 60)
        print(result["answer"])

        print("\nSOURCES USED")
        print("-" * 60)

        for source in result["sources"]:
            print(f"File: {source['source']}")
            print(f"Section: {source['heading']}")
            print(f"Similarity: {source['similarity']}")
            print("-" * 40)

        print("\nRESPONSE GENERATION TEST PASSED.")

    except Exception as error:
        print(f"\nTEST FAILED: {error}")
        raise


if __name__ == "__main__":
    main()
