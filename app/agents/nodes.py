
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.state import AgentState
from app.rag.generation import (
    generate_answer,
    retrieve_customer_knowledge,
)


def retrieve_node(state: AgentState) -> dict:
    retrieved = retrieve_customer_knowledge(
        question=state["question"],
        db_path=state["db_path"],
        top_k=state.get("top_k", 3),
    )

    return {
        "retrieved": retrieved,
        "messages": [
            HumanMessage(content=state["question"])
        ],
    }


def evaluate_node(state: AgentState) -> dict:
    retrieved = state.get("retrieved", [])
    threshold = 0.65

    sufficient = (
        bool(retrieved)
        and retrieved[0]["similarity"] >= threshold
    )

    return {"sufficient_context": sufficient}


def generate_node(state: AgentState) -> dict:
    history = state.get("messages", [])

    # Exclude the current question because it is
    # already supplied separately to generate_answer.
    previous_messages = history[:-1]

    answer = generate_answer(
        question=state["question"],
        retrieved=state["retrieved"],
        chat_history=previous_messages,
    )

    sources = [
        {
            "source": item["source"],
            "heading": item["heading"],
            "similarity": item["similarity"],
        }
        for item in state["retrieved"]
    ]

    return {
        "answer": answer,
        "sources": sources,
        "messages": [AIMessage(content=answer)],
    }


def fallback_node(state: AgentState) -> dict:
    answer = (
        "I couldn't find sufficiently relevant information "
        "in the CabTech Hub knowledge base to answer that "
        "question reliably. Please contact the CabTech Hub "
        "team for assistance."
    )

    return {
        "answer": answer,
        "sources": [],
        "messages": [AIMessage(content=answer)],
    }
