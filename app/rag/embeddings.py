
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768


def get_gemini_client() -> genai.Client:
    """Create a Gemini API client."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from your environment."
        )

    return genai.Client(api_key=api_key)


def generate_embedding(text: str) -> list[float]:
    """Convert text into a Gemini embedding vector."""
    if not text.strip():
        raise ValueError("Cannot embed empty text.")

    client = get_gemini_client()

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSIONS
        ),
    )

    if not response.embeddings:
        raise RuntimeError("Gemini returned no embeddings.")

    values = response.embeddings[0].values

    if values is None:
        raise RuntimeError("Embedding values are missing.")

    embedding = [float(value) for value in values]

    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"Expected {EMBEDDING_DIMENSIONS} dimensions, "
            f"received {len(embedding)}."
        )

    if not all(math.isfinite(value) for value in embedding):
        raise RuntimeError("Embedding contains invalid numbers.")

    return embedding
