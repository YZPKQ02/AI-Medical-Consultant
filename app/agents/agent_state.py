from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Any
import uuid


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
    _last_step_ns: int = field(default_factory=time.perf_counter_ns, init=False, repr=False)
    _started_ns: int = field(default_factory=time.perf_counter_ns, init=False, repr=False)

    def mark_step(self, name: str, status: str = "completed", **metadata: Any) -> None:
        now_ns = time.perf_counter_ns()
        self.steps.append(
            {
                "name": name,
                "status": status,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round((now_ns - self._last_step_ns) / 1_000_000, 3),
                "metadata": {key: value for key, value in metadata.items() if value is not None},
            }
        )
        self._last_step_ns = now_ns

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
            "run_id": self.run_id,
            "workflow_version": self.workflow_version,
            "prompt_version": self.prompt_version,
            "started_at": self.started_at,
            "duration_ms": round((time.perf_counter_ns() - self._started_ns) / 1_000_000, 3),
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
    run_id: str = field(default_factory=lambda: f"run-{uuid.uuid4().hex}")
    workflow_version: str = "medical-agent-v1"
    prompt_version: str = "decision-prompt-v1"
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
