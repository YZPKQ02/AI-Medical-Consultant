from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any
from urllib import error, request


try:  # LangChain is optional at runtime so local tests can stay offline.
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda

    LANGCHAIN_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when dependency is absent.
    ChatPromptTemplate = None
    RunnableLambda = None
    LANGCHAIN_AVAILABLE = False


SYSTEM_PROMPT = """You are a careful medical decision orchestration assistant.
You support, but never replace, a licensed clinician. Use retrieved medical
knowledge and the rule-based pre-assessment. Do not provide a definitive
diagnosis. Never downgrade urgent-care signals. Answer in Chinese.

Return only valid JSON with these keys:
- response: final patient-facing answer
- possible_causes: string array
- suggested_examinations: string array
- self_care_tips: string array
- follow_up_questions: string array
- safety_notes: string array
"""

HUMAN_PROMPT = """Patient profile:
{patient_profile}

Conversation history:
{conversation_history}

User question:
{question}

Rule-based pre-assessment:
{rule_assessment}

Retrieved RAG knowledge:
{retrieved_knowledge}

Draft decision:
{draft_decision}

Optimize the decision and final answer. Keep it concise, safe, and grounded in
the retrieved knowledge. If the rule-based risk level is 4, prioritize urgent
offline care and do not give home-management-only advice.
"""


@dataclass(frozen=True)
class LLMSettings:
    enabled: bool
    model: str
    api_key: str
    base_url: str
    temperature: float
    top_p: float
    max_tokens: int
    timeout_seconds: int


@dataclass(frozen=True)
class DecisionWorkflowResult:
    content: str | None
    decision: dict[str, Any] | None
    metadata: dict[str, Any]


class OpenAICompatibleChatClient:
    def __init__(self, config: LLMSettings):
        self.config = config

    def invoke_prompt(self, prompt_value: Any) -> str:
        messages = []
        for message in prompt_value.to_messages():
            role = message.type
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            messages.append({"role": role, "content": str(message.content)})
        return self.invoke_messages(messages)

    def invoke_messages(self, messages: list[dict[str, str]]) -> str:
        endpoint = chat_completions_endpoint(self.config.base_url)
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "authorization": f"Bearer {self.config.api_key}",
                "content-type": "application/json",
            },
        )

        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail[:300]}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"LLM network error: {exc.reason}") from exc

        data = json.loads(raw)
        return str(data["choices"][0]["message"]["content"])


class MedicalDecisionWorkflow:
    def __init__(self, config: LLMSettings):
        self.config = config
        self.client = OpenAICompatibleChatClient(config)
        self.chain = self._build_chain() if LANGCHAIN_AVAILABLE else None

    def run(
        self,
        *,
        message: str,
        conversation_history: list[dict],
        user_context: dict,
        patient_profile: dict,
        intent: str,
        slots: dict,
        analysis: dict,
        rag_context: dict,
        sources: list[dict],
        risk: dict,
        department: str,
        decision: dict,
        follow_up_questions: list[str],
    ) -> DecisionWorkflowResult:
        metadata = {
            "enabled": self.config.enabled,
            "provider": "openai-compatible",
            "model": self.config.model,
            "langchain_available": LANGCHAIN_AVAILABLE,
            "langchain_architecture": "ChatPromptTemplate | RunnableLambda(chat_completions)",
            "used": False,
            "fallback_reason": None,
            "pipeline": [
                "rule_precheck",
                "rag_retrieval",
                "prompt_assembly",
                "llm_generation",
                "json_parse",
                "safety_merge",
            ],
        }

        if not self.config.enabled:
            metadata["fallback_reason"] = "AGENT_LLM_ENABLED is not enabled"
            return DecisionWorkflowResult(content=None, decision=None, metadata=metadata)

        if not self.config.api_key or not self.config.base_url:
            metadata["fallback_reason"] = "LLM api key or base url is missing"
            return DecisionWorkflowResult(content=None, decision=None, metadata=metadata)

        prompt_variables = {
            "patient_profile": compact_json(patient_profile or user_context),
            "conversation_history": summarize_history(conversation_history),
            "question": message,
            "rule_assessment": compact_json(
                {
                    "intent": intent,
                    "slots": slots,
                    "symptom_analysis": analysis,
                    "risk": risk,
                    "department": department,
                }
            ),
            "retrieved_knowledge": summarize_sources(sources, rag_context.get("context_text", "")),
            "draft_decision": compact_json(
                {
                    **decision,
                    "follow_up_questions": follow_up_questions,
                }
            ),
        }

        try:
            raw_content = self._invoke(prompt_variables)
            parsed = parse_json_object(raw_content)
        except Exception as exc:  # pragma: no cover - network/provider dependent.
            metadata["fallback_reason"] = str(exc)
            return DecisionWorkflowResult(content=None, decision=None, metadata=metadata)

        metadata["used"] = True
        metadata["raw_response_preview"] = raw_content[:500]
        return DecisionWorkflowResult(
            content=clean_text(parsed.get("response")),
            decision=normalize_decision(parsed),
            metadata=metadata,
        )

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", HUMAN_PROMPT),
            ]
        )
        return prompt | RunnableLambda(self.client.invoke_prompt)

    def _invoke(self, prompt_variables: dict[str, str]) -> str:
        if self.chain is not None:
            return str(self.chain.invoke(prompt_variables))

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": HUMAN_PROMPT.format(**prompt_variables)},
        ]
        return self.client.invoke_messages(messages)


def chat_completions_endpoint(base_url: str) -> str:
    cleaned = str(base_url or "").rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    return f"{cleaned}/chat/completions"


def parse_json_object(value: str) -> dict[str, Any]:
    text = str(value or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError("LLM response JSON must be an object")
    return data


def normalize_decision(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "possible_causes": string_list(data.get("possible_causes")),
        "suggested_examinations": string_list(data.get("suggested_examinations")),
        "self_care_tips": string_list(data.get("self_care_tips")),
        "follow_up_questions": string_list(data.get("follow_up_questions")),
        "safety_notes": string_list(data.get("safety_notes")),
    }


def string_list(value: Any, limit: int = 6) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in (clean_text(item) for item in value) if item][:limit]


def clean_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def summarize_history(history: list[dict], limit: int = 6) -> str:
    items = []
    for message in history[-limit:]:
        role = message.get("role", "unknown")
        content = str(message.get("content", "")).strip()
        if content:
            items.append(f"{role}: {content[:300]}")
    return "\n".join(items) or "None"


def summarize_sources(sources: list[dict], context_text: str) -> str:
    if not sources:
        return context_text[:1200] if context_text else "None"

    lines = []
    for index, source in enumerate(sources[:5], start=1):
        lines.append(
            "\n".join(
                [
                    f"[Source {index}] {source.get('title', '')}",
                    f"department: {source.get('department', '')}",
                    f"reason: {source.get('retrieval_reason', '')}",
                    f"content: {str(source.get('content', ''))[:600]}",
                ]
            )
        )
    return "\n\n".join(lines)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
