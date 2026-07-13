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

atexit.register(consultation_store.close)
