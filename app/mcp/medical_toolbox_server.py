from __future__ import annotations

from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - depends on optional runtime setup.
    raise RuntimeError(
        "MCP SDK is not installed. Run `.venv\\Scripts\\python.exe -m pip install -r requirements.txt`."
    ) from exc

from app.agents.toolbox import MedicalToolbox
from app.core.config import settings
from app.services.hospital_recommender import HospitalRecommendationService
from app.services.rag_service import RAGService


mcp = FastMCP("ai-medical-toolbox")


def create_toolbox() -> MedicalToolbox:
    return MedicalToolbox(
        rag_service=RAGService(top_k=settings.rag_top_k, index_path=settings.rag_index_path),
        hospital_recommendation_service=HospitalRecommendationService(),
    )


@mcp.tool()
def search_medical_knowledge(
    query: str,
    top_k: int = 3,
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """Search the local medical RAG knowledge base and return evidence chunks."""
    result = create_toolbox().search_medical_knowledge(
        message=query,
        top_k=max(1, min(int(top_k or 3), 10)),
        categories=categories,
        conversation_history=[],
        user_context={},
    )
    payload = result.payload
    return {
        "status": result.status,
        "query_expansion": payload.get("query_expansion", {}),
        "pipeline": payload.get("pipeline", []),
        "sources": [
            {
                "title": source.get("title"),
                "category": source.get("category"),
                "department": source.get("department"),
                "severity_hint": source.get("severity_hint"),
                "score": source.get("score"),
                "matched_keywords": source.get("matched_keywords"),
                "retrieval_reason": source.get("retrieval_reason"),
                "content": source.get("content"),
            }
            for source in payload.get("retrieved_docs", [])
        ],
    }


@mcp.tool()
def check_drug_safety(
    message: str,
    age: str = "",
    allergies: str = "",
    chronic_diseases: str = "",
    medications: str = "",
    intent: str = "medication",
) -> dict[str, Any]:
    """Run a local first-pass medication safety check without calling external services."""
    result = create_toolbox().check_drug_safety(
        message=message,
        intent=intent or "medication",
        slots={"medications": medications or None},
        user_context={
            "age": age,
            "allergies": allergies,
            "chronic_diseases": chronic_diseases,
        },
    )
    return result.payload


@mcp.tool()
def recommend_hospitals(
    city: str,
    department: str,
    urgency_level: int,
    symptoms: list[str] | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Recommend hospital department candidates by city and triage result."""
    service = HospitalRecommendationService()
    return service.recommend(
        city=city,
        department=department,
        urgency_level=int(urgency_level or 1),
        symptoms=symptoms or [],
        limit=limit,
    )


if __name__ == "__main__":
    mcp.run()
