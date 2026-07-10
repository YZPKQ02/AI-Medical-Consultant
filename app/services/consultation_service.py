from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
import time
from typing import Any

from app.agents.medical_agent import MedicalAgent, current_timestamp
from app.services.consultation_store import ConsultationStore, InMemoryConsultationStore


@dataclass
class ConsultationService:
    agent: MedicalAgent
    consultations: dict[str, dict[str, Any]] = field(default_factory=dict)
    store: ConsultationStore | None = None

    def __post_init__(self) -> None:
        self._lock = RLock()
        if self.store is None:
            self.store = InMemoryConsultationStore(self.consultations)

    def count(self) -> int:
        with self._lock:
            return self.store.count()

    def list_consultations(self, owner_user_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            return self.store.list(normalize_owner_user_id(owner_user_id))

    def get_consultation(
        self,
        consultation_id: str,
        owner_user_id: str | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            return self.store.get(consultation_id, normalize_owner_user_id(owner_user_id))

    def delete_consultation(
        self,
        consultation_id: str,
        owner_user_id: str | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            return self.store.delete(consultation_id, normalize_owner_user_id(owner_user_id))

    def create_consultation(
        self,
        chief_complaint: str = "",
        user_context: dict[str, Any] | None = None,
        owner_user_id: str | None = None,
    ) -> dict[str, Any]:
        now = current_timestamp()
        consultation_id = f"consult_{time.time_ns()}"
        consultation = {
            "id": consultation_id,
            "owner_user_id": normalize_owner_user_id(owner_user_id),
            "chief_complaint": chief_complaint,
            "user_context": user_context or {},
            "messages": [],
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            self.store.save(consultation)
        return consultation

    def append_message(self, consultation: dict[str, Any], role: str, content: str) -> dict[str, Any]:
        message = {
            "id": f"{role}_{time.time_ns()}",
            "role": role,
            "content": content,
            "created_at": current_timestamp(),
        }
        consultation["messages"].append(message)
        consultation["updated_at"] = message["created_at"]
        with self._lock:
            self.store.save(consultation)
        return message

    def add_user_message(
        self,
        consultation_id: str,
        content: str,
        owner_user_id: str | None = None,
    ) -> dict[str, Any] | None:
        normalized_content = str(content or "").strip()
        if not normalized_content:
            raise ValueError("Message content is required")

        consultation = self.get_consultation(consultation_id, owner_user_id=owner_user_id)
        if consultation is None:
            return None

        with self._lock:
            user_message = self.append_message(consultation, "user", normalized_content)
            assistant_message = self.agent.chat(
                normalized_content,
                conversation_history=consultation["messages"],
                user_context=consultation["user_context"],
            )
            consultation["messages"].append(assistant_message)
            consultation["updated_at"] = assistant_message["created_at"]
            self.store.save(consultation)

        return {
            "user_message": user_message,
            "assistant_message": assistant_message,
            "consultation": consultation,
        }


def normalize_owner_user_id(owner_user_id: str | None) -> str:
    normalized = str(owner_user_id or "").strip()
    if not normalized:
        return "anonymous"
    return normalized[:128]
