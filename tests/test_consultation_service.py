import unittest

from app.services.consultation_service import ConsultationService


class FakeAgent:
    def chat(self, message, conversation_history=None, user_context=None):
        return {
            "id": "assistant-test",
            "role": "assistant",
            "content": f"reply: {message}",
            "created_at": "2026-01-01T00:00:00Z",
            "analysis": {
                "history_count": len(conversation_history or []),
                "user_context": user_context or {},
            },
            "source_knowledge": [],
        }


class ConsultationServiceTests(unittest.TestCase):
    def test_create_and_list_consultations(self):
        service = ConsultationService(agent=FakeAgent())
        consultation = service.create_consultation(
            chief_complaint="cough",
            user_context={"age": "30"},
        )

        self.assertEqual(service.count(), 1)
        self.assertEqual(consultation["chief_complaint"], "cough")
        self.assertEqual(service.list_consultations()[0]["message_count"], 0)

    def test_add_user_message_invokes_agent_and_updates_session(self):
        service = ConsultationService(agent=FakeAgent())
        consultation = service.create_consultation(user_context={"age": "30"})

        result = service.add_user_message(consultation["id"], "hello")

        self.assertEqual(result["user_message"]["content"], "hello")
        self.assertEqual(result["assistant_message"]["content"], "reply: hello")
        self.assertEqual(len(result["consultation"]["messages"]), 2)
        self.assertEqual(result["assistant_message"]["analysis"]["history_count"], 1)

    def test_add_user_message_reports_missing_or_empty_message(self):
        service = ConsultationService(agent=FakeAgent())

        self.assertIsNone(service.add_user_message("missing", "hello"))
        with self.assertRaises(ValueError):
            service.add_user_message("missing", " ")

    def test_delete_consultation_removes_session(self):
        service = ConsultationService(agent=FakeAgent())
        consultation = service.create_consultation(chief_complaint="delete me")

        deleted = service.delete_consultation(consultation["id"])

        self.assertEqual(deleted["chief_complaint"], "delete me")
        self.assertIsNone(service.get_consultation(consultation["id"]))
        self.assertEqual(service.count(), 0)


if __name__ == "__main__":
    unittest.main()
