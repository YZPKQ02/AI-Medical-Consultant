from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.core.config import settings
from app.services.runtime import consultation_service, runtime_readiness


router = APIRouter()


@router.get("/health")
def health_check() -> dict:
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "consultations": consultation_service.count(),
    }


@router.get("/ready")
def readiness_check(response: Response) -> dict:
    readiness = runtime_readiness()
    if not readiness["ready"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ready" if readiness["ready"] else "not_ready",
        **readiness,
    }
