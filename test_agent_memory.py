
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from app.agents.graph import build_agent_graph


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_ROOT / "cabtech.db"
CHECKPOINT_PATH = PROJECT_ROOT / "agent_checkpoints.sqlite"

# Use a fresh thread ID for this test.
THREAD_ID = "cabtech-multi-turn-test-001"


def main() -> None:
    print("=" * 60)
    print("CABTECH AI FLOW - MULTI-TURN MEMORY TEST")
    print("=" * 60)

    config = {
        "configurable": {
            "thread_id": THREAD_ID
        }
    }

    with SqliteSaver.from_conn_string(
        str(CHECKPOINT_PATH)
    ) as checkpointer:

        agent_graph = build_agent_graph(
            checkpointer=checkpointer
        )

        # FIRST MESSAGE
        question_one = (
            "Can CabTech Hub build an online clothing store?"
        )

        print("\nCUSTOMER - MESSAGE 1:")
        print(question_one)

        result_one = agent_graph.invoke(
            {
                "question": question_one,
                "db_path": DATABASE_PATH,
                "top_k": 3,
            },
            config=config,
        )

        print("\nAI RESPONSE - MESSAGE 1:")
        print(result_one["answer"])

        # SECOND MESSAGE - SAME THREAD
        question_two = (
            "For that online clothing store, "
            "what features should it include?"
        )

        print("\nCUSTOMER - MESSAGE 2:")
        print(question_two)

        result_two = agent_graph.invoke(
            {
                "question": question_two,
                "db_path": DATABASE_PATH,
                "top_k": 3,
            },
            config=config,
        )

        print("\nAI RESPONSE - MESSAGE 2:")
        print(result_two["answer"])

        # VERIFY CONVERSATION HISTORY
        saved_state = agent_graph.get_state(config)
        messages = saved_state.values.get("messages", [])

        print("\n" + "=" * 60)
        print("MEMORY VERIFICATION")
        print("=" * 60)

        print("Thread ID:", THREAD_ID)
        print("Total saved messages:", len(messages))

        for index, message in enumerate(messages, start=1):
            print(
                f"\nMessage {index}: "
                f"{type(message).__name__}"
            )
            print(message.content)

        if len(messages) == 4:
            print("\nMULTI-TURN MEMORY TEST PASSED.")
        else:
            print("\nMULTI-TURN MEMORY TEST FAILED.")


if __name__ == "__main__":
    main()
