from __future__ import annotations

from fastapi import APIRouter

from app.core.config import settings
from app.services.runtime import consultation_service


router = APIRouter()


@router.get("/status")
def agent_status() -> dict:
    return {
        "rag_top_k": settings.rag_top_k,
        "embedding_provider": settings.embedding_provider,
        "agent_llm_enabled": settings.agent_llm_enabled,
        "llm_model": settings.llm_model,
        "consultations": consultation_service.count(),
    }
