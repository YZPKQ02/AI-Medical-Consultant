import unittest

from fastapi.testclient import TestClient

from app.fastapi_main import create_app


class FastAPIEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app())

    def test_health_routes_expose_versioned_and_legacy_paths(self):
        versioned = self.client.get("/api/v1/health")
        legacy = self.client.get("/api/health")

        self.assertEqual(versioned.status_code, 200)
        self.assertEqual(legacy.status_code, 200)
        self.assertEqual(versioned.json()["status"], "healthy")
        self.assertEqual(legacy.json()["status"], "healthy")

    def test_consultation_routes_create_and_fetch_session(self):
        created = self.client.post(
            "/api/v1/consultations",
            json={"chief_complaint": "cough", "user_context": {"age": "30"}},
        )

        self.assertEqual(created.status_code, 201)
        consultation_id = created.json()["id"]

        fetched = self.client.get(f"/api/v1/consultations/{consultation_id}")
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.json()["chief_complaint"], "cough")

    def test_consultation_routes_delete_session(self):
        created = self.client.post(
            "/api/v1/consultations",
            json={"chief_complaint": "remove me", "user_context": {}},
        )
        consultation_id = created.json()["id"]

        deleted = self.client.delete(f"/api/v1/consultations/{consultation_id}")
        fetched = self.client.get(f"/api/v1/consultations/{consultation_id}")

        self.assertEqual(deleted.status_code, 200)
        self.assertTrue(deleted.json()["deleted"])
        self.assertEqual(fetched.status_code, 404)

    def test_legacy_knowledge_route_stays_frontend_compatible(self):
        response = self.client.get("/api/knowledge", params={"q": "cough"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("results", response.json())


if __name__ == "__main__":
    unittest.main()
