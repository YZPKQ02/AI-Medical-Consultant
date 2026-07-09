import unittest

from app.agents.medical_agent import MedicalAgent
from app.services.knowledge_base import search_knowledge


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


if __name__ == "__main__":
    unittest.main()
