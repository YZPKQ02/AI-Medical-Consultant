import unittest

from scripts.evaluate_agent import evaluate_case


class AgentEvaluationTests(unittest.TestCase):
    def test_evaluate_case_accepts_expected_agent_shape(self):
        case = {
            "id": "sample",
            "assertions": {
                "needs_urgent_care": False,
                "risk_level_min": 1,
                "department_contains": "内科",
                "content_contains": ["建议科室"],
                "follow_up_contains": ["症状持续多久了？"],
                "tool_status_in": ["missing_city"],
            },
        }
        reply = {
            "content": "【建议科室】普通内科",
            "analysis": {
                "needs_urgent_care": False,
                "risk": {"level": 1},
                "department": "普通内科",
                "follow_up_questions": ["症状持续多久了？"],
                "hospital_recommendations": {"status": "missing_city"},
                "agent_state": {
                    "run_id": "run-test",
                    "steps": [{"name": "slot_extraction", "duration_ms": 1.0}],
                },
                "tool_results": [{"name": "recommend_hospitals"}],
            },
        }

        self.assertEqual(evaluate_case(case, reply), [])

    def test_evaluate_case_reports_failures(self):
        case = {
            "id": "sample",
            "assertions": {
                "needs_urgent_care": True,
                "department_contains": "急诊",
            },
        }
        reply = {
            "content": "普通建议",
            "analysis": {
                "needs_urgent_care": False,
                "risk": {"level": 1},
                "department": "普通内科",
                "follow_up_questions": [],
                "hospital_recommendations": {"status": "missing_city"},
                "agent_state": {"steps": []},
                "tool_results": [],
            },
        }

        errors = evaluate_case(case, reply)

        self.assertGreaterEqual(len(errors), 2)


if __name__ == "__main__":
    unittest.main()
