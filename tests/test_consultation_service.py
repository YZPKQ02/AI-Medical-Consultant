import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from app.services.consultation_service import ConsultationService
from app.services.consultation_store import SQLiteConsultationStore, postgresql_schema_sql


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

    def test_in_memory_store_filters_consultations_by_owner(self):
        service = ConsultationService(agent=FakeAgent())
        user_a = service.create_consultation(
            chief_complaint="user a",
            owner_user_id="user-a",
        )
        user_b = service.create_consultation(
            chief_complaint="user b",
            owner_user_id="user-b",
        )

        self.assertEqual(len(service.list_consultations(owner_user_id="user-a")), 1)
        self.assertEqual(service.get_consultation(user_a["id"], owner_user_id="user-a")["id"], user_a["id"])
        self.assertIsNone(service.get_consultation(user_b["id"], owner_user_id="user-a"))
        self.assertIsNone(service.delete_consultation(user_b["id"], owner_user_id="user-a"))
        self.assertIsNotNone(service.get_consultation(user_b["id"], owner_user_id="user-b"))

    def test_sqlite_store_persists_consultations_across_service_instances(self):
        with TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "consultations.sqlite3"
            store = SQLiteConsultationStore(store_path)
            service = ConsultationService(
                agent=FakeAgent(),
                store=store,
            )
            consultation = service.create_consultation(
                chief_complaint="persistent cough",
                user_context={"age": "30"},
            )
            service.add_user_message(consultation["id"], "hello")

            store.close()
            reloaded_store = SQLiteConsultationStore(store_path)
            reloaded = ConsultationService(
                agent=FakeAgent(),
                store=reloaded_store,
            )
            persisted = reloaded.get_consultation(consultation["id"])

            self.assertEqual(reloaded.count(), 1)
            self.assertEqual(persisted["chief_complaint"], "persistent cough")
            self.assertEqual(len(persisted["messages"]), 2)
            reloaded_store.close()

    def test_sqlite_store_soft_delete_hides_consultation(self):
        with TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "consultations.sqlite3"
            store = SQLiteConsultationStore(store_path)
            service = ConsultationService(
                agent=FakeAgent(),
                store=store,
            )
            consultation = service.create_consultation(chief_complaint="delete sqlite")

            deleted = service.delete_consultation(consultation["id"])

            self.assertEqual(deleted["chief_complaint"], "delete sqlite")
            self.assertEqual(service.count(), 0)
            self.assertEqual(service.list_consultations(), [])
            self.assertIsNone(service.get_consultation(consultation["id"]))
            store.close()

    def test_sqlite_store_filters_by_owner(self):
        with TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "consultations.sqlite3"
            store = SQLiteConsultationStore(store_path)
            service = ConsultationService(agent=FakeAgent(), store=store)
            user_a = service.create_consultation(
                chief_complaint="sqlite user a",
                owner_user_id="user-a",
            )
            user_b = service.create_consultation(
                chief_complaint="sqlite user b",
                owner_user_id="user-b",
            )

            self.assertEqual(len(service.list_consultations(owner_user_id="user-a")), 1)
            self.assertEqual(
                service.get_consultation(user_a["id"], owner_user_id="user-a")["chief_complaint"],
                "sqlite user a",
            )
            self.assertIsNone(service.get_consultation(user_b["id"], owner_user_id="user-a"))
            self.assertIsNone(service.delete_consultation(user_b["id"], owner_user_id="user-a"))
            self.assertIsNotNone(service.get_consultation(user_b["id"], owner_user_id="user-b"))
            store.close()

    def test_postgresql_schema_defines_normalized_agent_tables(self):
        schema = "\n".join(postgresql_schema_sql())

        for table_name in (
            "users",
            "consultations",
            "messages",
            "agent_runs",
            "tool_calls",
        ):
            self.assertIn(f"CREATE TABLE IF NOT EXISTS {table_name}", schema)

        self.assertIn("owner_user_id TEXT NOT NULL REFERENCES users(id)", schema)
        self.assertIn("payload_json JSONB NOT NULL", schema)
        self.assertIn("analysis_json JSONB NOT NULL", schema)
        self.assertIn("idx_consultations_owner_active_updated", schema)


if __name__ == "__main__":
    unittest.main()
