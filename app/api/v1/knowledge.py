from __future__ import annotations

from fastapi import APIRouter, Query

from app.services.knowledge_base import MEDICAL_KNOWLEDGE_BASE, search_knowledge


router = APIRouter()


@router.get("")
def query_knowledge(q: str = Query(default=""), top_k: int = Query(default=5, ge=1, le=20)) -> dict:
    results = (
        search_knowledge(q, top_k=top_k)
        if q
        else [doc.to_dict() for doc in MEDICAL_KNOWLEDGE_BASE]
    )
    return {"query": q, "results": results}
