
import json
from pathlib import Path

from langchain_core.tools import tool

from app.rag.retrieval import search_knowledge


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "cabtech.db"


@tool
def search_cabtech_knowledge(
    query: str,
    top_k: int = 3,
) -> str:
    """Search CabTech Hub's knowledge base for relevant information.

    Use this tool to find information about company services,
    policies, project capabilities, and customer support.
    """

    if not query.strip():
        return json.dumps({
            "error": "Search query cannot be empty."
        })

    if not 1 <= top_k <= 5:
        return json.dumps({
            "error": "top_k must be between 1 and 5."
        })

    results = search_knowledge(
        query=query,
        db_path=DATABASE_PATH,
        top_k=top_k,
    )

    if not results:
        return json.dumps({
            "message": "No relevant knowledge was found."
        })

    formatted_results = [
        {
            "source": item["source"],
            "heading": item["heading"],
            "content": item["content"],
            "similarity": item["similarity"],
        }
        for item in results
    ]

    return json.dumps(
        formatted_results,
        ensure_ascii=False,
    )
