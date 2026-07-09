from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
import time
from typing import Any

from app.agents.medical_agent import MedicalAgent, current_timestamp


@dataclass
class ConsultationService:
    agent: MedicalAgent
    consultations: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._lock = RLock()

    def count(self) -> int:
        with self._lock:
            return len(self.consultations)

    def list_consultations(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "id": item["id"],
                    "chief_complaint": item["chief_complaint"],
                    "message_count": len(item["messages"]),
                    "created_at": item["created_at"],
                    "updated_at": item["updated_at"],
                }
                for item in self.consultations.values()
            ]

    def get_consultation(self, consultation_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self.consultations.get(consultation_id)

    def delete_consultation(self, consultation_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self.consultations.pop(consultation_id, None)

    def create_consultation(
        self,
        chief_complaint: str = "",
        user_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = current_timestamp()
        consultation_id = f"consult_{time.time_ns()}"
        consultation = {
            "id": consultation_id,
            "chief_complaint": chief_complaint,
            "user_context": user_context or {},
            "messages": [],
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            self.consultations[consultation_id] = consultation
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
        return message

    def add_user_message(
        self,
        consultation_id: str,
        content: str,
    ) -> dict[str, Any] | None:
        normalized_content = str(content or "").strip()
        if not normalized_content:
            raise ValueError("Message content is required")

        consultation = self.get_consultation(consultation_id)
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

        return {
            "user_message": user_message,
            "assistant_message": assistant_message,
            "consultation": consultation,
        }
