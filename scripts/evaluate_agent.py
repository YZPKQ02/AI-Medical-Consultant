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
    cases = sorted(case_dir.glob("*.json"))
    if not cases:
        print(f"No eval cases found in {case_dir}", file=sys.stderr)
        return 2

    agent = MedicalAgent(
        enable_llm=False,
        hospital_recommendation_service=OfflineHospitalRecommendationService(),
    )
    failures: list[str] = []

    for case_path in cases:
        case = json.loads(case_path.read_text(encoding="utf-8"))
        reply = agent.chat(
            case["message"],
            user_context=case.get("user_context") or {},
        )
        errors = evaluate_case(case, reply)
        if errors:
            failures.extend(f"{case['id']}: {error}" for error in errors)
            print(f"FAIL {case['id']}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"PASS {case['id']}")

    if failures:
        print("")
        print(f"{len(failures)} assertion(s) failed.", file=sys.stderr)
        return 1

    print("")
    print(f"All {len(cases)} eval case(s) passed.")
    return 0


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

    return errors


if __name__ == "__main__":
    raise SystemExit(main())
