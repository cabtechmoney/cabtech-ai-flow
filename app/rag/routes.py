from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.rag.generation import generate_customer_response


router = APIRouter(prefix="/ai", tags=["AI Assistant"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "cabtech.db"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    top_k: int = Field(default=3, ge=1, le=10)


@router.post("/ask")
def ask_ai(request: AskRequest) -> dict:
    try:
        return generate_customer_response(
            question=request.question,
            db_path=DATABASE_PATH,
            top_k=request.top_k,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail="The AI service is temporarily unavailable.",
        ) from error