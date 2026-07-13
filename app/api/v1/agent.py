from __future__ import annotations

from fastapi import APIRouter

from app.core.config import settings
from app.services.runtime import consultation_service


router = APIRouter()


@router.get("/status")
def agent_status() -> dict:
    embedding = consultation_service.agent.rag_service.embedding_provider
    return {
        "rag_top_k": settings.rag_top_k,
        "embedding_provider": settings.embedding_provider,
        "embedding_backend": getattr(embedding, "backend", embedding.name),
        "embedding_dimension": getattr(embedding, "embedding_dimension", None),
        "embedding_fallback": bool(getattr(embedding, "is_fallback", False)),
        "agent_llm_enabled": settings.agent_llm_enabled,
        "llm_model": settings.llm_model,
        "consultations": consultation_service.count(),
    }
