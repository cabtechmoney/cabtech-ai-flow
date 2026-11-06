
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agents.state import AgentState
from app.agents.tool_nodes import (
    agent_node,
    prepare_agent_node,
    tools,
)


def build_tool_agent_graph(checkpointer=None):
    builder = StateGraph(AgentState)

    builder.add_node("prepare", prepare_agent_node)
    builder.add_node("agent", agent_node)
    builder.add_node(
        "tools",
        ToolNode(tools, handle_tool_errors=True),
    )

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "agent")

    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    builder.add_edge("tools", "agent")

    return builder.compile(
        checkpointer=checkpointer
    )
