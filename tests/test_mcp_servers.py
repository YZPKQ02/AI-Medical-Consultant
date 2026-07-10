import unittest

from app.mcp import medical_toolbox_server


class MedicalToolboxMCPServerTests(unittest.TestCase):
    def test_medical_toolbox_server_exports_expected_tools(self):
        self.assertEqual(medical_toolbox_server.mcp.name, "ai-medical-toolbox")
        self.assertTrue(callable(medical_toolbox_server.search_medical_knowledge))
        self.assertTrue(callable(medical_toolbox_server.check_drug_safety))
        self.assertTrue(callable(medical_toolbox_server.recommend_hospitals))

    def test_search_medical_knowledge_tool_returns_sources(self):
        result = medical_toolbox_server.search_medical_knowledge(
            query="胸痛伴呼吸困难",
            top_k=2,
            categories=["emergency"],
        )

        self.assertEqual(result["status"], "available")
        self.assertTrue(result["sources"])
        self.assertIn("胸痛", result["sources"][0]["title"])

    def test_check_drug_safety_tool_flags_allergy(self):
        result = medical_toolbox_server.check_drug_safety(
            message="青霉素能不能吃？",
            age="30",
            allergies="青霉素过敏",
            chronic_diseases="",
        )

        self.assertEqual(result["status"], "available")
        self.assertTrue(result["warnings"])
        self.assertIn("青霉素", result["warnings"][0])


if __name__ == "__main__":
    unittest.main()
