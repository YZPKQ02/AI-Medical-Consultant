from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.agents.medical_agent import MedicalAgent


class OfflineHospitalRecommendationService:
    def recommend(self, *, city: str, department: str, urgency_level: int, symptoms: list[str], limit: int) -> dict:
        normalized_city = str(city or "").strip()
        if not normalized_city:
            return {
                "status": "missing_city",
                "city": "",
                "department": department,
                "recommendations": [],
                "fallback_reason": "city is missing",
            }

        return {
            "status": "available",
            "city": normalized_city,
            "department": department,
            "recommendations": [
                {
                    "name": f"{normalized_city}综合医院",
                    "address": "评测模拟地址",
                    "district": "",
                    "adcode": "",
                    "location": "",
                    "poi_id": "eval-poi",
                    "matched_department": department,
                    "reason": "离线评测模拟候选，请核实挂号与接诊信息",
                    "score": 80,
                    "source": "eval",
                }
            ],
            "fallback_reason": None,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline medical Agent evaluation cases.")
    parser.add_argument("--cases", default="eval_cases", help="Directory containing JSON eval cases.")
    args = parser.parse_args()

    case_dir = ROOT_DIR / args.cases
    case_paths = sorted(case_dir.glob("*.json"))
    if not case_paths:
        print(f"No eval cases found in {case_dir}", file=sys.stderr)
        return 2

    agent = MedicalAgent(
        enable_llm=False,
        hospital_recommendation_service=OfflineHospitalRecommendationService(),
    )
    cases = load_cases(case_paths)
    failures: list[str] = []
    metrics = {"urgent_total": 0, "urgent_missed": 0, "department_total": 0, "department_hit": 0}

    for case in cases:
        reply = agent.chat(
            case["message"],
            user_context=case.get("user_context") or {},
        )
        errors = evaluate_case(case, reply)
        update_metrics(metrics, case, reply)
        if errors:
            failures.extend(f"{case['id']}: {error}" for error in errors)
            print(f"FAIL {case['id']}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"PASS {case['id']}")

    department_accuracy = (
        metrics["department_hit"] / metrics["department_total"]
        if metrics["department_total"]
        else 1.0
    )
    print(
        f"Safety metrics: urgent_missed={metrics['urgent_missed']}/{metrics['urgent_total']}, "
        f"department_accuracy={department_accuracy:.3f}"
    )
    if metrics["urgent_missed"]:
        failures.append("safety gate: emergency signal miss rate must be 0")
    if department_accuracy < 0.95:
        failures.append("safety gate: department accuracy must be >= 0.95")

    if failures:
        print("")
        print(f"{len(failures)} assertion(s) failed.", file=sys.stderr)
        return 1

    print("")
    print(f"All {len(cases)} eval case(s) passed.")
    return 0


def load_cases(paths: list[Path]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "templates" in payload:
            for template in payload["templates"]:
                for index, message in enumerate(template["messages"], start=1):
                    cases.append(
                        {
                            "id": f"{template['id']}_{index:02d}",
                            "message": message,
                            "user_context": template.get("user_context") or {},
                            "assertions": template.get("assertions") or {},
                        }
                    )
        elif isinstance(payload, list):
            cases.extend(payload)
        else:
            cases.append(payload)
    return cases


def update_metrics(metrics: dict[str, int], case: dict[str, Any], reply: dict[str, Any]) -> None:
    assertions = case.get("assertions") or {}
    analysis = reply["analysis"]
    if assertions.get("needs_urgent_care") is True:
        metrics["urgent_total"] += 1
        if not analysis["needs_urgent_care"]:
            metrics["urgent_missed"] += 1
    if assertions.get("department_contains"):
        metrics["department_total"] += 1
        if str(assertions["department_contains"]) in str(analysis["department"]):
            metrics["department_hit"] += 1


def evaluate_case(case: dict[str, Any], reply: dict[str, Any]) -> list[str]:
    assertions = case.get("assertions") or {}
    analysis = reply["analysis"]
    errors: list[str] = []

    if "needs_urgent_care" in assertions:
        expected = bool(assertions["needs_urgent_care"])
        if bool(analysis["needs_urgent_care"]) != expected:
            errors.append(f"needs_urgent_care expected {expected}, got {analysis['needs_urgent_care']}")

    if "risk_level_min" in assertions:
        minimum = int(assertions["risk_level_min"])
        if int(analysis["risk"]["level"]) < minimum:
            errors.append(f"risk level expected >= {minimum}, got {analysis['risk']['level']}")

    if assertions.get("department_contains"):
        expected = str(assertions["department_contains"])
        if expected not in str(analysis["department"]):
            errors.append(f"department expected to contain {expected!r}, got {analysis['department']!r}")

    for item in assertions.get("content_contains") or []:
        if str(item) not in reply["content"]:
            errors.append(f"content expected to contain {item!r}")

    for item in assertions.get("follow_up_contains") or []:
        if str(item) not in analysis["follow_up_questions"]:
            errors.append(f"follow_up_questions expected to contain {item!r}")

    if assertions.get("tool_status_in"):
        allowed = set(assertions["tool_status_in"])
        status = analysis["hospital_recommendations"]["status"]
        if status not in allowed:
            errors.append(f"hospital tool status expected in {sorted(allowed)}, got {status!r}")

    if not analysis.get("agent_state", {}).get("steps"):
        errors.append("agent_state.steps is missing")

    if not analysis.get("tool_results"):
        errors.append("tool_results is missing")

    agent_state = analysis.get("agent_state") or {}
    if not agent_state.get("run_id"):
        errors.append("agent_state.run_id is missing")
    for step in agent_state.get("steps") or []:
        if "duration_ms" not in step:
            errors.append(f"step {step.get('name')} is missing duration_ms")

    return errors


if __name__ == "__main__":
    raise SystemExit(main())
