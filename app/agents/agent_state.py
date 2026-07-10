from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """Structured trace for one medical Agent turn."""

    user_context: dict[str, Any] = field(default_factory=dict)
    patient_profile: dict[str, Any] = field(default_factory=dict)
    intent: str = ""
    slots: dict[str, Any] = field(default_factory=dict)
    symptom_analysis: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    department: str = ""
    decision: dict[str, Any] = field(default_factory=dict)
    follow_up_questions: list[str] = field(default_factory=list)
    rag: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)

    def mark_step(self, name: str, status: str = "completed", **metadata: Any) -> None:
        self.steps.append(
            {
                "name": name,
                "status": status,
                "metadata": {key: value for key, value in metadata.items() if value is not None},
            }
        )

    def record_tool_result(
        self,
        *,
        name: str,
        status: str,
        input_summary: dict[str, Any],
        output_summary: dict[str, Any],
    ) -> None:
        self.tool_results.append(
            {
                "name": name,
                "status": status,
                "input_summary": input_summary,
                "output_summary": output_summary,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_context": self.user_context,
            "steps": self.steps,
            "patient_profile": self.patient_profile,
            "intent": self.intent,
            "slots": self.slots,
            "symptom_analysis": self.symptom_analysis,
            "risk": self.risk,
            "department": self.department,
            "decision": self.decision,
            "follow_up_questions": self.follow_up_questions,
            "rag": self.rag,
            "llm": self.llm,
            "tool_results": self.tool_results,
        }
