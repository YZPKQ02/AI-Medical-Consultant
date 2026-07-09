from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.runtime import consultation_service


class ConsultationCreate(BaseModel):
    chief_complaint: str = ""
    user_context: dict[str, Any] = Field(default_factory=dict)


class MessageCreate(BaseModel):
    content: str


router = APIRouter()


@router.get("")
def list_consultations() -> dict:
    return {"consultations": consultation_service.list_consultations()}


@router.post("", status_code=201)
def create_consultation(payload: ConsultationCreate) -> dict:
    return consultation_service.create_consultation(
        chief_complaint=payload.chief_complaint,
        user_context=payload.user_context,
    )


@router.get("/{consultation_id}")
def get_consultation(consultation_id: str) -> dict:
    consultation = consultation_service.get_consultation(consultation_id)
    if consultation is None:
        raise HTTPException(status_code=404, detail="Consultation not found")
    return consultation


@router.delete("/{consultation_id}")
def delete_consultation(consultation_id: str) -> dict:
    consultation = consultation_service.delete_consultation(consultation_id)
    if consultation is None:
        raise HTTPException(status_code=404, detail="Consultation not found")
    return {"deleted": True, "consultation_id": consultation_id}


@router.post("/{consultation_id}/messages", status_code=201)
def create_message(consultation_id: str, payload: MessageCreate) -> dict:
    try:
        result = consultation_service.add_user_message(consultation_id, payload.content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if result is None:
        raise HTTPException(status_code=404, detail="Consultation not found")
    return result
