
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    question: str
    db_path: Path
    top_k: int

    messages: Annotated[list[AnyMessage], add_messages]

    retrieved: list[dict]
    sufficient_context: bool

    answer: str
    sources: list[dict]
