from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.services.hospital_recommender import HospitalRecommendationService
from app.services.knowledge_base import normalize_text
from app.services.rag_service import RAGService


@dataclass
class ToolExecution:
    name: str
    status: str
    input_summary: dict[str, Any]
    output_summary: dict[str, Any]
    payload: dict[str, Any]


class MedicalToolbox:
    """Internal MCP-style toolbox used only by the Agent orchestration layer."""

    def __init__(
        self,
        *,
        rag_service: RAGService,
        hospital_recommendation_service: HospitalRecommendationService | None,
    ):
        self.rag_service = rag_service
        self.hospital_recommendation_service = hospital_recommendation_service

    def search_medical_knowledge(
        self,
        *,
        message: str,
        top_k: int,
        categories: list[str] | None,
        conversation_history: list[dict],
        user_context: dict,
    ) -> ToolExecution:
        rag_context = self.rag_service.build_context(
            query=message,
            top_k=top_k,
            categories=categories,
            conversation_history=conversation_history,
            user_context=user_context,
        )
        sources = rag_context["retrieved_docs"]
        return ToolExecution(
            name="search_medical_knowledge",
            status="available" if sources else "empty",
            input_summary={
                "query_length": len(str(message or "")),
                "top_k": top_k,
                "categories": categories or [],
            },
            output_summary={
                "source_count": len(sources),
                "top_titles": [source["title"] for source in sources[:3]],
            },
            payload=rag_context,
        )

    def check_drug_safety(
        self,
        *,
        message: str,
        intent: str,
        slots: dict[str, Any],
        user_context: dict[str, Any],
    ) -> ToolExecution:
        normalized = normalize_text(message)
        should_invoke = intent == "medication" or bool(slots.get("medications")) or any(
            term in normalized for term in DRUG_TERMS
        )

        if not should_invoke:
            return ToolExecution(
                name="check_drug_safety",
                status="skipped",
                input_summary={"intent": intent},
                output_summary={"reason": "not a medication-focused turn"},
                payload={
                    "status": "skipped",
                    "warnings": [],
                    "missing_context": [],
                    "recommendations": [],
                },
            )

        warnings: list[str] = []
        recommendations = [
            "不要自行叠加同类药物或超说明书剂量用药。",
            "处方药、儿童、孕期、老人或肝肾功能异常情况请优先咨询医生或药师。",
        ]
        missing_context = []
        allergies = normalize_text(str(user_context.get("allergies") or ""))
        chronic_diseases = normalize_text(str(user_context.get("chronic_diseases") or ""))
        age_text = str(user_context.get("age") or "")

        if not age_text:
            missing_context.append("年龄")
        if not allergies:
            missing_context.append("药物过敏史")
        if not chronic_diseases:
            missing_context.append("慢性病/肝肾功能情况")

        if "青霉素" in normalized and "青霉素" in allergies:
            warnings.append("已填写青霉素过敏史，不应自行使用青霉素相关药物。")
        if "头孢" in normalized and ("头孢" in allergies or "严重过敏" in allergies):
            warnings.append("已填写头孢或严重过敏史，用药前需由医生/药师确认安全性。")
        if any(term in normalized for term in ("布洛芬", "阿司匹林", "NSAID".lower())):
            if any(term in chronic_diseases for term in ("胃溃疡", "胃出血", "肾病", "肾功能")):
                warnings.append("非甾体抗炎药可能加重胃肠道出血或肾功能风险，请先咨询医生。")
            if is_child(age_text) and "阿司匹林" in normalized:
                warnings.append("儿童或青少年不应自行使用阿司匹林退热。")
        if any(term in normalized for term in ("一起吃", "同时吃", "混用", "叠加")):
            warnings.append("存在多药同服/叠加用药描述，需要核对通用名、成分、剂量和间隔。")

        return ToolExecution(
            name="check_drug_safety",
            status="available",
            input_summary={
                "intent": intent,
                "has_age": bool(age_text),
                "has_allergies": bool(allergies),
                "has_chronic_diseases": bool(chronic_diseases),
            },
            output_summary={
                "warning_count": len(warnings),
                "missing_context": missing_context,
            },
            payload={
                "status": "available",
                "warnings": warnings,
                "missing_context": missing_context,
                "recommendations": recommendations,
            },
        )

    def recommend_hospitals(
        self,
        *,
        city: str,
        department: str,
        risk: dict,
        symptoms: list[str],
    ) -> ToolExecution:
        normalized_city = str(city or "").strip()
        if not normalized_city:
            payload = {
                "status": "missing_city",
                "city": "",
                "department": department,
                "recommendations": [],
                "fallback_reason": "city is missing",
            }
            return ToolExecution(
                name="recommend_hospitals",
                status="missing_city",
                input_summary={
                    "city": "",
                    "department": department,
                    "urgency_level": risk.get("level"),
                    "symptom_count": len(symptoms),
                },
                output_summary={"recommendation_count": 0, "fallback_reason": "city is missing"},
                payload=payload,
            )

        if self.hospital_recommendation_service is None:
            payload = {
                "status": "unavailable",
                "city": normalized_city,
                "department": department,
                "recommendations": [],
                "fallback_reason": "hospital recommendation service is unavailable",
            }
        else:
            payload = self.hospital_recommendation_service.recommend(
                city=normalized_city,
                department=department,
                urgency_level=int(risk.get("level") or 1),
                symptoms=symptoms,
                limit=settings.hospital_recommender_limit,
            )

        return ToolExecution(
            name="recommend_hospitals",
            status=payload.get("status", "unknown"),
            input_summary={
                "city": normalized_city,
                "department": department,
                "urgency_level": risk.get("level"),
                "symptom_count": len(symptoms),
            },
            output_summary={
                "recommendation_count": len(payload.get("recommendations") or []),
                "fallback_reason": payload.get("fallback_reason"),
            },
            payload=payload,
        )


DRUG_TERMS = (
    "药",
    "用药",
    "布洛芬",
    "对乙酰氨基酚",
    "阿司匹林",
    "青霉素",
    "头孢",
    "抗生素",
    "感冒药",
    "退烧药",
)


def is_child(age_text: str) -> bool:
    digits = "".join(ch for ch in str(age_text or "") if ch.isdigit())
    if not digits:
        return False
    try:
        return int(digits) < 18
    except ValueError:
        return False
