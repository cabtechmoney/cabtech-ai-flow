
import json

from app.agents.tools import search_cabtech_knowledge


def main() -> None:
    print("=" * 60)
    print("CABTECH AI FLOW - AGENT TOOL TEST")
    print("=" * 60)

    result = search_cabtech_knowledge.invoke({
        "query": "Does CabTech Hub offer web development?",
        "top_k": 3,
    })

    print("\nTOOL EXECUTION RESULT:")

    parsed_result = json.loads(result)

    print(json.dumps(
        parsed_result,
        indent=4,
        ensure_ascii=False,
    ))

    if not isinstance(parsed_result, list) or not parsed_result:
        raise RuntimeError("Knowledge search returned no results.")

    print("\nAGENT TOOL TEST PASSED.")


if __name__ == "__main__":
    main()
