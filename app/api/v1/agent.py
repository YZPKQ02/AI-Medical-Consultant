from __future__ import annotations

from fastapi import APIRouter

from app.core.config import settings
from app.services.runtime import consultation_service, runtime_readiness, storage_backend_name


router = APIRouter()


@router.get("/status")
def agent_status() -> dict:
    embedding = consultation_service.agent.rag_service.embedding_provider
    readiness = runtime_readiness()
    return {
        "environment": settings.app_environment,
        "storage_backend": storage_backend_name(),
        "database_connected": readiness["checks"]["database"]["ready"],
        "database_schema_version": readiness["checks"]["database"]["schema_version"],
        "rag_top_k": settings.rag_top_k,
        "embedding_provider": settings.embedding_provider,
        "embedding_backend": getattr(embedding, "backend", embedding.name),
        "embedding_dimension": getattr(embedding, "embedding_dimension", None),
        "embedding_fallback": bool(getattr(embedding, "is_fallback", False)),
        "embedding_device": getattr(embedding, "active_device", None),
        "agent_llm_enabled": settings.agent_llm_enabled,
        "llm_model": settings.llm_model,
        "consultations": consultation_service.count(),
    }
