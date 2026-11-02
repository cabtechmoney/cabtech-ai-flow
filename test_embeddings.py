
from app.rag.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    generate_embedding,
)


def main() -> None:
    print("=" * 55)
    print("CABTECH AI FLOW - EMBEDDING TEST")
    print("=" * 55)

    sample_text = (
        "CabTech Hub builds web applications, "
        "integrates APIs and automates business workflows."
    )

    try:
        embedding = generate_embedding(sample_text)

        print("\nEmbedding generation successful!")
        print(f"Model: {EMBEDDING_MODEL}")
        print(f"Expected dimensions: {EMBEDDING_DIMENSIONS}")
        print(f"Actual dimensions: {len(embedding)}")
        print(f"First five values: {embedding[:5]}")
        print("\nEMBEDDING TEST PASSED.")

    except Exception as error:
        print(f"\nEMBEDDING TEST FAILED: {error}")


if __name__ == "__main__":
    main()
