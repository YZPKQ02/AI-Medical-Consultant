from __future__ import annotations

import atexit

from app.agents.medical_agent import MedicalAgent
from app.core.config import settings
from app.services.consultation_service import ConsultationService
from app.services.consultation_store import PostgreSQLConsultationStore, SQLiteConsultationStore


consultation_store = (
    PostgreSQLConsultationStore(settings.database_url)
    if settings.database_url
    else SQLiteConsultationStore(settings.consultation_store_path)
)
consultation_service = ConsultationService(
    agent=MedicalAgent(rag_top_k=settings.rag_top_k),
    store=consultation_store,
)


def storage_backend_name() -> str:
    if isinstance(consultation_store, PostgreSQLConsultationStore):
        return "postgresql"
    if isinstance(consultation_store, SQLiteConsultationStore):
        return "sqlite"
    return type(consultation_store).__name__.lower()


def database_schema_version() -> str:
    if not isinstance(consultation_store, PostgreSQLConsultationStore):
        return "not_applicable"
    try:
        with consultation_store._lock:
            with consultation_store._conn.cursor() as cur:
                cur.execute("SELECT version_num FROM alembic_version LIMIT 1")
                row = cur.fetchone()
        return str(row["version_num"]) if row else "unversioned"
    except Exception:
        consultation_store._conn.rollback()
        return "unversioned"


def runtime_readiness() -> dict:
    embedding = consultation_service.agent.rag_service.embedding_provider
    embedding_error = None
    if settings.require_qwen:
        try:
            ensure_ready = getattr(embedding, "ensure_ready", None)
            if ensure_ready is not None:
                ensure_ready()
        except Exception as exc:
            embedding_error = type(exc).__name__
    checks = {
        "database": {"ready": False, "backend": storage_backend_name()},
        "embedding": {
            "ready": not bool(getattr(embedding, "is_fallback", False)),
            "backend": getattr(embedding, "backend", embedding.name),
            "device": getattr(embedding, "active_device", None),
        },
    }
    errors = []

    try:
        consultation_store.count()
        checks["database"]["ready"] = True
    except Exception as exc:  # pragma: no cover - depends on external database state.
        errors.append(f"database: {type(exc).__name__}")

    checks["database"]["schema_version"] = database_schema_version()
    if checks["database"]["backend"] == "postgresql" and checks["database"]["schema_version"] == "unversioned":
        checks["database"]["ready"] = False
        errors.append("database: schema is not managed by Alembic")

    if settings.require_postgres and checks["database"]["backend"] != "postgresql":
        checks["database"]["ready"] = False
        errors.append("database: PostgreSQL is required")

    if settings.require_qwen and checks["embedding"]["backend"] != "qwen-local":
        checks["embedding"]["ready"] = False
        errors.append("embedding: local Qwen is required")
    if embedding_error:
        checks["embedding"]["ready"] = False
        errors.append(f"embedding: {embedding_error}")

    required_ready = checks["database"]["ready"] and (
        checks["embedding"]["ready"] or not settings.require_qwen
    )
    return {"ready": required_ready, "checks": checks, "errors": errors}


def validate_runtime_requirements() -> None:
    readiness = runtime_readiness()
    if not readiness["ready"]:
        raise RuntimeError("Runtime requirements are not satisfied: " + "; ".join(readiness["errors"]))

atexit.register(consultation_store.close)
