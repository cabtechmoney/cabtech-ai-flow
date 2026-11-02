from pathlib import Path

from app.rag.generation import generate_customer_response


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_ROOT / "cabtech.db"


TEST_CASES = [
    {
        "category": "SUPPORTED",
        "question": "Can CabTech Hub build an online clothing store?",
    },
    {
        "category": "SUPPORTED",
        "question": "Can CabTech Hub integrate an external API?",
    },
    {
        "category": "SUPPORTED",
        "question": "What information should I provide when reporting a technical problem?",
    },
    {
        "category": "UNSUPPORTED",
        "question": "What is the exact price of building a website?",
    },
    {
        "category": "UNSUPPORTED",
        "question": "Can you guarantee that my website will generate one million naira every month?",
    },
]


def main() -> None:
    print("=" * 65)
    print("CABTECH AI FLOW - RAG EVALUATION")
    print("=" * 65)

    for index, test_case in enumerate(TEST_CASES, start=1):
        print(f"\nTEST {index}: {test_case['category']}")
        print(f"Question: {test_case['question']}")

        result = generate_customer_response(
            question=test_case["question"],
            db_path=DATABASE_PATH,
            top_k=3,
        )

        print("\nAI ANSWER:")
        print(result["answer"])

        print("\nRETRIEVED SOURCES:")
        for source in result["sources"]:
            print(
                f"- {source['source']} | "
                f"{source['heading']} | "
                f"Similarity: {source['similarity']:.4f}"
            )

        print("-" * 65)

    print("\nEVALUATION RUN COMPLETED.")
    print("Review the answers for accuracy and unsupported claims.")


if __name__ == "__main__":
    main()