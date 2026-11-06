from pathlib import Path

from app.agents.graph import agent_graph


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_ROOT / "cabtech.db"


def run_test(question: str) -> None:
    print("\n" + "=" * 60)
    print(f"CUSTOMER QUESTION: {question}")
    print("=" * 60)

    result = agent_graph.invoke(
        {
            "question": question,
            "db_path": DATABASE_PATH,
            "top_k": 3,
        }
    )

    print("\nRETRIEVAL RESULTS:")
    for item in result.get("retrieved", []):
        print(
            f"- {item['heading']}: "
            f"{item['similarity']:.4f}"
        )

    print("\nCONTEXT SUFFICIENT:")
    print(result.get("sufficient_context"))

    print("\nFINAL ANSWER:")
    print(result.get("answer"))

    print("\nSOURCES:")
    for source in result.get("sources", []):
        print(f"- {source['source']} | {source['heading']}")


def main() -> None:
    run_test(
        "Can CabTech Hub build an online clothing store?"
    )

    run_test(
        "What is the exact price of building a website?"
    )


if __name__ == "__main__":
    main()