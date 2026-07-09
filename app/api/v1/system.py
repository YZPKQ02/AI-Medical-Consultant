from __future__ import annotations

from fastapi import APIRouter

from app.core.config import settings
from app.services.runtime import consultation_service


router = APIRouter()


@router.get("/health")
def health_check() -> dict:
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "consultations": consultation_service.count(),
    }
