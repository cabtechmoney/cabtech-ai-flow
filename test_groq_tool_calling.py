
import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing from .env")

client = Groq(api_key=api_key)

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Search the CabTech Hub knowledge base "
                "for information relevant to a customer question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The question or topic to search for."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def main() -> None:
    print("=" * 60)
    print("CABTECH AI FLOW - GROQ TOOL CALLING TEST")
    print("=" * 60)

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a CabTech Hub support agent. "
                    "Use the search_knowledge tool to find "
                    "information about CabTech Hub services."
                ),
            },
            {
                "role": "user",
                "content": (
                    "What web development services does "
                    "CabTech Hub offer?"
                ),
            },
        ],
        tools=tools,
        tool_choice="required",
        temperature=0.1,
    )

    message = response.choices[0].message

    if not message.tool_calls:
        raise RuntimeError("Groq did not return a tool call.")

    print("\nTOOL CALL SUCCESSFUL")

    for tool_call in message.tool_calls:
        print("\nFunction:", tool_call.function.name)
        print("Arguments:", tool_call.function.arguments)

    print("\nGROQ TOOL CALLING TEST PASSED.")


if __name__ == "__main__":
    main()
