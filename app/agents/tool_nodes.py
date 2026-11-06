
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.agents.state import AgentState
from app.agents.tools import search_cabtech_knowledge


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """
You are the AI customer-support agent for CabTech Hub.

Your responsibilities:
- Understand customer questions.
- Use search_cabtech_knowledge when company information is needed.
- Base factual answers on retrieved knowledge.
- Never invent prices, policies, guarantees or services.
- If information is unavailable, say so honestly.
- Use previous conversation messages when relevant.
- Never reveal API keys or confidential information.

Be professional, helpful and concise.
"""

tools = [search_cabtech_knowledge]

llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0.2,
)

llm_with_tools = llm.bind_tools(tools)


def prepare_agent_node(state: AgentState) -> dict:
    question = state["question"].strip()

    if not question:
        raise ValueError("Customer question cannot be empty.")

    return {
        "messages": [
            HumanMessage(content=question)
        ]
    }


def agent_node(state: AgentState) -> dict:
    messages = state.get("messages", [])

    response = llm_with_tools.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            *messages,
        ]
    )

    updates = {
        "messages": [response]
    }

    # Only save the answer when Groq has finished
    # and is not requesting another tool.
    if not response.tool_calls:
        if not isinstance(response.content, str):
            raise RuntimeError(
                "Groq returned an unexpected response format."
            )

        if not response.content.strip():
            raise RuntimeError("Groq returned an empty answer.")

        updates["answer"] = response.content.strip()

    return updates
