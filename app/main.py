
import json
import math
import os
import secrets
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.security import APIKeyHeader
from google import genai
from google.genai import types
from pydantic import BaseModel, EmailStr, Field

from app.rag.documents import build_chunks


# 1. Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "cabtech.db"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"

load_dotenv(PROJECT_ROOT / ".env")

CABTECH_API_KEY = os.getenv("CABTECH_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


# 2. Database connection
def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=10,
    )
    connection.row_factory = sqlite3.Row
    return connection


# 3. Authentication
def require_api_key(
    api_key: Annotated[str | None, Depends(api_key_header)],
) -> None:
    if not CABTECH_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key is not configured.",
        )

    if api_key is None or not secrets.compare_digest(
        api_key,
        CABTECH_API_KEY,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
        )


# 4. Gemini embedding client
def get_gemini_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini API key is not configured.",
        )

    return genai.Client(api_key=GEMINI_API_KEY)


def generate_embedding(text: str) -> list[float]:
    """Generate a hosted Gemini embedding for text."""
    try:
        client = get_gemini_client()

        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )

        if not result.embeddings or not result.embeddings[0].values:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Gemini returned no embedding.",
            )

        vector = [
            float(value)
            for value in result.embeddings[0].values
        ]

        if len(vector) != EMBEDDING_DIMENSIONS:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Gemini returned an unexpected embedding size.",
            )

        if not all(math.isfinite(value) for value in vector):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Gemini returned invalid embedding values.",
            )

        return vector

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not generate a Gemini embedding.",
        ) from exc


# 5. Initialize tables without deleting existing data
@asynccontextmanager
async def lifespan(app: FastAPI):
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                service TEXT NOT NULL,
                category TEXT,
                priority TEXT,
                summary TEXT,
                next_action TEXT
            )
            """
        )

        existing_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(customers)"
            ).fetchall()
        }

        customer_columns = {
            "category": "TEXT",
            "priority": "TEXT",
            "summary": "TEXT",
            "next_action": "TEXT",
        }

        for column, column_type in customer_columns.items():
            if column not in existing_columns:
                connection.execute(
                    f"ALTER TABLE customers "
                    f"ADD COLUMN {column} {column_type}"
                )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                chunk_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                heading TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_dimensions INTEGER NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    yield


# 6. FastAPI application
app = FastAPI(
    title="CabTech AI Flow",
    description=(
        "Customer workflow automation and retrieval-augmented generation."
    ),
    version="0.2.0",
    lifespan=lifespan,
)


# 7. Request and response models
class CustomerCreate(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    service: str = Field(min_length=2, max_length=100)

    category: Literal[
        "Web Development",
        "API Integration",
        "Workflow Automation",
        "Technical Support",
    ] | None = None

    priority: Literal["Low", "Medium", "High"] | None = None
    summary: str | None = Field(default=None, max_length=1000)
    next_action: str | None = Field(default=None, max_length=1000)


class CustomerResponse(BaseModel):
    id: int
    name: str
    email: EmailStr
    service: str
    category: str | None = None
    priority: str | None = None
    summary: str | None = None
    next_action: str | None = None


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


class KnowledgeResult(BaseModel):
    chunk_id: str
    source: str
    heading: str
    content: str
    similarity: float


class KnowledgeSearchResponse(BaseModel):
    query: str
    results: list[KnowledgeResult]


# 8. Health check
@app.get("/")
def home() -> dict[str, str]:
    return {
        "project": "CabTech AI Flow",
        "status": "running",
        "version": "0.2.0",
    }


# 9. Create a customer
@app.post(
    "/customers",
    response_model=CustomerResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
)
def create_customer(
    customer: CustomerCreate,
) -> CustomerResponse:
    try:
        with get_connection() as connection:
            existing = connection.execute(
                "SELECT id FROM customers WHERE email = ? LIMIT 1",
                (str(customer.email),),
            ).fetchone()

            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="A customer with this email already exists.",
                )

            cursor = connection.execute(
                """
                INSERT INTO customers (
                    name, email, service, category,
                    priority, summary, next_action
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    customer.name,
                    str(customer.email),
                    customer.service,
                    customer.category,
                    customer.priority,
                    customer.summary,
                    customer.next_action,
                ),
            )

            customer_id = cursor.lastrowid

            if customer_id is None:
                raise RuntimeError("Customer ID was not generated.")

            return CustomerResponse(
                id=customer_id,
                name=customer.name,
                email=customer.email,
                service=customer.service,
                category=customer.category,
                priority=customer.priority,
                summary=customer.summary,
                next_action=customer.next_action,
            )

    except HTTPException:
        raise
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save customer.",
        ) from exc


# 10. List customers
@app.get(
    "/customers",
    response_model=list[CustomerResponse],
    dependencies=[Depends(require_api_key)],
)
def list_customers() -> list[CustomerResponse]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id, name, email, service, category,
                priority, summary, next_action
            FROM customers
            ORDER BY id DESC
            """
        ).fetchall()

    return [
        CustomerResponse(**dict(row))
        for row in rows
    ]


# 11. Ingest Markdown documents into the separate knowledge table
@app.post(
    "/knowledge/ingest",
    dependencies=[Depends(require_api_key)],
)
def ingest_knowledge() -> dict[str, int | str]:
    if not KNOWLEDGE_BASE_DIR.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge-base directory was not found.",
        )

    try:
        chunks = build_chunks(KNOWLEDGE_BASE_DIR)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No non-empty Markdown documents were found.",
            )

        # Prepare embeddings before writing, so a failed API call
        # does not leave a partially updated knowledge index.
        indexed_chunks: list[tuple[object, list[float]]] = []

        for chunk in chunks:
            vector = generate_embedding(chunk.text)
            indexed_chunks.append((chunk, vector))

        with get_connection() as connection:
            for chunk, vector in indexed_chunks:
                connection.execute(
                    """
                    INSERT INTO knowledge_chunks (
                        chunk_id, source, heading, content,
                        embedding_json, embedding_model,
                        embedding_dimensions, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        source = excluded.source,
                        heading = excluded.heading,
                        content = excluded.content,
                        embedding_json = excluded.embedding_json,
                        embedding_model = excluded.embedding_model,
                        embedding_dimensions = excluded.embedding_dimensions,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        chunk.chunk_id,
                        chunk.source,
                        chunk.heading,
                        chunk.text,
                        json.dumps(vector),
                        EMBEDDING_MODEL,
                        EMBEDDING_DIMENSIONS,
                    ),
                )

        return {
            "status": "success",
            "chunks_processed": len(indexed_chunks),
        }

    except HTTPException:
        raise
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save knowledge chunks.",
        ) from exc


# 12. Semantic search across stored knowledge
@app.post(
    "/knowledge/search",
    response_model=KnowledgeSearchResponse,
    dependencies=[Depends(require_api_key)],
)
def search_knowledge(
    request: KnowledgeSearchRequest,
) -> KnowledgeSearchResponse:
    query_vector = generate_embedding(request.query)

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT chunk_id, source, heading, content,
                   embedding_json, embedding_model,
                   embedding_dimensions
            FROM knowledge_chunks
            """
        ).fetchall()

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base is empty. Ingest documents first.",
        )

    ranked: list[KnowledgeResult] = []

    for row in rows:
        if (
            row["embedding_model"] != EMBEDDING_MODEL
            or row["embedding_dimensions"] != EMBEDDING_DIMENSIONS
        ):
            continue

        try:
            stored_vector = json.loads(row["embedding_json"])
        except (json.JSONDecodeError, TypeError):
            continue

        if (
            not isinstance(stored_vector, list)
            or len(stored_vector) != len(query_vector)
        ):
            continue

        dot_product = sum(
            left * right
            for left, right in zip(query_vector, stored_vector)
        )
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        stored_norm = math.sqrt(sum(value * value for value in stored_vector))

        if query_norm == 0 or stored_norm == 0:
            continue

        similarity = dot_product / (query_norm * stored_norm)

        ranked.append(
            KnowledgeResult(
                chunk_id=row["chunk_id"],
                source=row["source"],
                heading=row["heading"],
                content=row["content"],
                similarity=round(similarity, 6),
            )
        )

    ranked.sort(
        key=lambda result: result.similarity,
        reverse=True,
    )

    return KnowledgeSearchResponse(
        query=request.query,
        results=ranked[:request.top_k],
    )
