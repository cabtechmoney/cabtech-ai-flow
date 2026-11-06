
from langgraph.graph import END, START, StateGraph

from app.agents.state import AgentState
from app.agents.nodes import (
    evaluate_node,
    fallback_node,
    generate_node,
    retrieve_node,
)


def route_after_evaluation(state: AgentState) -> str:
    if state.get("sufficient_context", False):
        return "generate"

    return "fallback"


def build_agent_graph(checkpointer=None):
    builder = StateGraph(AgentState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("generate", generate_node)
    builder.add_node("fallback", fallback_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "evaluate")

    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "generate": "generate",
            "fallback": "fallback",
        },
    )

    builder.add_edge("generate", END)
    builder.add_edge("fallback", END)

    return builder.compile(checkpointer=checkpointer)


agent_graph = build_agent_graph()
