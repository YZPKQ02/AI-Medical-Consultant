import unittest

from app.agents.medical_agent import MedicalAgent
from app.services.knowledge_base import search_knowledge


class FakeHospitalRecommendationService:
    def __init__(self, status="available"):
        self.status = status
        self.calls = []

    def recommend(self, *, city, department, urgency_level, symptoms, limit):
        self.calls.append(
            {
                "city": city,
                "department": department,
                "urgency_level": urgency_level,
                "symptoms": symptoms,
                "limit": limit,
            }
        )
        if self.status == "unavailable":
            return {
                "status": "unavailable",
                "city": city,
                "department": department,
                "recommendations": [],
                "fallback_reason": "mock failure",
            }
        return {
            "status": "available",
            "city": city,
            "department": department,
            "recommendations": [
                {
                    "name": "城市三甲医院急诊中心",
                    "address": "健康路1号",
                    "district": "中心区",
                    "adcode": "000001",
                    "location": "120,30",
                    "poi_id": "poi-1",
                    "matched_department": department,
                    "reason": "急症风险下优先急诊/线下急诊相关机构",
                    "score": 100,
                    "source": "amap",
                }
            ],
            "fallback_reason": None,
        }


class KnowledgeBaseTests(unittest.TestCase):
    def test_search_knowledge_matches_synonym(self):
        results = search_knowledge("我发烧两天还咳嗽", top_k=2)

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "发热与感染初步处理")
        self.assertIn("retrieval_reason", results[0])

    def test_search_knowledge_supports_medication_intent(self):
        results = search_knowledge("布洛芬能不能吃，有什么副作用", categories=["medication"])

        self.assertEqual(results[0]["category"], "medication")


class MedicalAgentTests(unittest.TestCase):
    def test_agent_flags_emergency_chest_pain(self):
        agent = MedicalAgent()
        reply = agent.chat("胸痛伴呼吸困难和大汗")

        self.assertTrue(reply["analysis"]["needs_urgent_care"])
        self.assertEqual(reply["analysis"]["department"], "急诊科")
        self.assertEqual(reply["analysis"]["stage"], "urgent_guidance")
        self.assertIn("急症风险", reply["content"])

    def test_agent_handles_negated_emergency_symptom(self):
        agent = MedicalAgent()
        reply = agent.chat("头痛3天，没有胸痛", user_context={"age": "35"})

        self.assertFalse(reply["analysis"]["needs_urgent_care"])
        self.assertEqual(reply["analysis"]["duration"], "3天")
        self.assertIn("头痛", reply["analysis"]["symptoms"])
        self.assertIn("神经", reply["analysis"]["department"])

    def test_agent_classifies_medication_intent(self):
        agent = MedicalAgent()
        reply = agent.chat("感冒了能不能吃布洛芬，有没有副作用")

        self.assertEqual(reply["analysis"]["intent"], "medication")
        self.assertIn("用药", reply["content"])
        self.assertTrue(reply["source_knowledge"])
        self.assertEqual(reply["analysis"]["drug_safety"]["status"], "available")
        self.assertIn("check_drug_safety", [tool["name"] for tool in reply["analysis"]["tool_results"]])

    def test_agent_collects_history_when_age_missing(self):
        agent = MedicalAgent()
        reply = agent.chat("咳嗽2天，有点发热")

        self.assertEqual(reply["analysis"]["stage"], "history_collection")
        self.assertIn("患者年龄是多少？", reply["analysis"]["follow_up_questions"])

    def test_agent_moves_to_assessment_when_required_slots_exist(self):
        agent = MedicalAgent()
        reply = agent.chat("咳嗽2天，有点发热", user_context={"age": "28"})

        self.assertEqual(reply["analysis"]["stage"], "assessment")

    def test_agent_exposes_llm_orchestration_metadata_offline(self):
        agent = MedicalAgent(enable_llm=False)
        reply = agent.chat("咳嗽2天，有点发热")

        self.assertFalse(reply["analysis"]["llm"]["enabled"])
        self.assertFalse(reply["analysis"]["llm"]["used"])
        self.assertIn("prompt_assembly", reply["analysis"]["llm"]["pipeline"])

    def test_agent_exposes_structured_state_and_tool_results(self):
        service = FakeHospitalRecommendationService()
        agent = MedicalAgent(enable_llm=False, hospital_recommendation_service=service)

        reply = agent.chat(
            "咳嗽2天，有点发热",
            user_context={
                "age": "28",
                "city": "上海",
                "allergies": "青霉素过敏",
                "chronic_diseases": "哮喘",
            },
        )

        state = reply["analysis"]["agent_state"]
        self.assertTrue(state["steps"])
        self.assertEqual(state["patient_profile"]["city"], "上海")
        tool_names = [tool["name"] for tool in state["tool_results"]]
        self.assertEqual(
            tool_names,
            ["search_medical_knowledge", "check_drug_safety", "recommend_hospitals"],
        )
        self.assertEqual(reply["analysis"]["tool_results"][-1]["status"], "available")
        self.assertEqual(state["user_context"]["allergies"], "provided")
        self.assertNotIn("青霉素", str(state["user_context"]))

    def test_agent_uses_drug_safety_tool_for_medication_risk(self):
        agent = MedicalAgent(enable_llm=False)

        reply = agent.chat(
            "青霉素能不能吃？",
            user_context={"age": "30", "allergies": "青霉素过敏"},
        )

        self.assertEqual(reply["analysis"]["drug_safety"]["status"], "available")
        self.assertTrue(reply["analysis"]["drug_safety"]["warnings"])
        self.assertIn("青霉素过敏", reply["content"])

    def test_agent_requests_city_before_hospital_recommendation(self):
        service = FakeHospitalRecommendationService()
        agent = MedicalAgent(enable_llm=False, hospital_recommendation_service=service)

        reply = agent.chat("咳嗽2天，有点发热", user_context={"age": "28"})

        self.assertEqual(reply["analysis"]["hospital_recommendations"]["status"], "missing_city")
        self.assertIn("你当前所在城市是哪里？", reply["analysis"]["follow_up_questions"])
        self.assertEqual(service.calls, [])

    def test_agent_adds_hospital_recommendations_when_city_exists(self):
        service = FakeHospitalRecommendationService()
        agent = MedicalAgent(enable_llm=False, hospital_recommendation_service=service)

        reply = agent.chat("胸痛伴呼吸困难", user_context={"age": "45", "city": "杭州"})

        recommendations = reply["analysis"]["hospital_recommendations"]["recommendations"]
        self.assertEqual(service.calls[0]["city"], "杭州")
        self.assertEqual(service.calls[0]["urgency_level"], 4)
        self.assertTrue(recommendations)
        self.assertIn("医院推荐", reply["content"])
        self.assertIn("优先急诊", recommendations[0]["reason"])

    def test_agent_keeps_answer_when_hospital_service_fails(self):
        service = FakeHospitalRecommendationService(status="unavailable")
        agent = MedicalAgent(enable_llm=False, hospital_recommendation_service=service)

        reply = agent.chat("咳嗽2天，有点发热", user_context={"age": "28", "city": "上海"})

        self.assertEqual(reply["analysis"]["hospital_recommendations"]["status"], "unavailable")
        self.assertIn("医院推荐服务暂不可用", reply["content"])


if __name__ == "__main__":
    unittest.main()
