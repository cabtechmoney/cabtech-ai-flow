
from pathlib import Path

from langchain_core.messages import ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from app.agents.tool_graph import build_tool_agent_graph


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "agent_checkpoints.sqlite"

THREAD_ID = "cabtech-tool-agent-test-001"


def main() -> None:
    print("=" * 65)
    print("CABTECH AI FLOW - TOOL AGENT INTEGRATION TEST")
    print("=" * 65)

    config = {
        "configurable": {
            "thread_id": THREAD_ID
        }
    }

    with SqliteSaver.from_conn_string(
        str(CHECKPOINT_PATH)
    ) as checkpointer:

        agent_graph = build_tool_agent_graph(
            checkpointer=checkpointer
        )

        result = agent_graph.invoke(
            {
                "question": (
                    "What web development services "
                    "does CabTech Hub offer?"
                ),
            },
            config=config,
        )

        print("\nFIRST AI RESPONSE:")
        print(result["answer"])

        result_two = agent_graph.invoke(
            {
                "question": (
                    "Can those services help build "
                    "an online clothing store?"
                ),
            },
            config=config,
        )

        print("\nSECOND AI RESPONSE:")
        print(result_two["answer"])

        saved_state = agent_graph.get_state(config)
        messages = saved_state.values.get("messages", [])

        tool_messages = [
            message
            for message in messages
            if isinstance(message, ToolMessage)
        ]

        print("\nMEMORY AND TOOL VERIFICATION")
        print("Thread ID:", THREAD_ID)
        print("Total messages:", len(messages))
        print("Executed tool results:", len(tool_messages))

        if not tool_messages:
            raise RuntimeError(
                "No tool execution was recorded."
            )

        if not result_two.get("answer"):
            raise RuntimeError(
                "The agent did not produce a final answer."
            )

        print("\nTOOL AGENT INTEGRATION TEST PASSED.")


if __name__ == "__main__":
    main()
