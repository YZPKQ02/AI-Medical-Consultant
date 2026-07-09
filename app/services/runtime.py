from __future__ import annotations

from app.agents.medical_agent import MedicalAgent
from app.core.config import settings
from app.services.consultation_service import ConsultationService


consultation_service = ConsultationService(
    agent=MedicalAgent(rag_top_k=settings.rag_top_k),
)
