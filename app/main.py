from __future__ import annotations

from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from app.core.config import settings
from app.services.knowledge_base import MEDICAL_KNOWLEDGE_BASE, search_knowledge
from app.services.runtime import consultation_service

ROOT_DIR = Path(__file__).resolve().parent.parent
PUBLIC_DIR = ROOT_DIR / "public"

consultations = consultation_service.consultations
agent = consultation_service.agent


class MedicalConsultantHandler(SimpleHTTPRequestHandler):
    server_version = "AIMedicalConsultant/0.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PUBLIC_DIR), **kwargs)

    def log_message(self, format: str, *args) -> None:
        print(f"[server] {self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self.handle_api_get(parsed)
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self.handle_api_post(parsed)
            return
        self.send_json({"error": "Route not found"}, HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self.handle_api_delete(parsed)
            return
        self.send_json({"error": "Route not found"}, HTTPStatus.NOT_FOUND)

    def handle_api_get(self, parsed) -> None:
        if parsed.path == "/api/health":
            self.send_json(
                {
                    "status": "healthy",
                    "service": settings.app_name,
                    "version": settings.app_version,
                    "consultations": consultation_service.count(),
                }
            )
            return

        if parsed.path == "/api/knowledge":
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0]
            results = (
                search_knowledge(query, top_k=5)
                if query
                else [doc.to_dict() for doc in MEDICAL_KNOWLEDGE_BASE]
            )
            self.send_json({"query": query, "results": results})
            return

        if parsed.path == "/api/consultations":
            self.send_json(
                {"consultations": consultation_service.list_consultations(self.current_owner_user_id())}
            )
            return

        consultation_id = self.match_consultation_detail(parsed.path)
        if consultation_id:
            consultation = consultation_service.get_consultation(
                consultation_id,
                owner_user_id=self.current_owner_user_id(),
            )
            if not consultation:
                self.send_json({"error": "Consultation not found"}, HTTPStatus.NOT_FOUND)
                return
            self.send_json(consultation)
            return

        self.send_json({"error": "API route not found"}, HTTPStatus.NOT_FOUND)

    def handle_api_post(self, parsed) -> None:
        if parsed.path == "/api/consultations":
            body = self.read_json_body()
            consultation = consultation_service.create_consultation(
                chief_complaint=str(body.get("chief_complaint", "")),
                user_context=dict(body.get("user_context") or {}),
                owner_user_id=self.current_owner_user_id(),
            )
            self.send_json(consultation, HTTPStatus.CREATED)
            return

        consultation_id = self.match_message_create(parsed.path)
        if consultation_id:
            body = self.read_json_body()
            content = str(body.get("content", "")).strip()
            if not content:
                self.send_json({"error": "Message content is required"}, HTTPStatus.BAD_REQUEST)
                return

            result = consultation_service.add_user_message(
                consultation_id,
                content,
                owner_user_id=self.current_owner_user_id(),
            )
            if result is None:
                self.send_json({"error": "Consultation not found"}, HTTPStatus.NOT_FOUND)
                return

            self.send_json(result, HTTPStatus.CREATED)
            return

        self.send_json({"error": "API route not found"}, HTTPStatus.NOT_FOUND)

    def handle_api_delete(self, parsed) -> None:
        consultation_id = self.match_consultation_detail(parsed.path)
        if consultation_id:
            consultation = consultation_service.delete_consultation(
                consultation_id,
                owner_user_id=self.current_owner_user_id(),
            )
            if not consultation:
                self.send_json({"error": "Consultation not found"}, HTTPStatus.NOT_FOUND)
                return
            self.send_json({"deleted": True, "consultation_id": consultation_id})
            return

        self.send_json({"error": "API route not found"}, HTTPStatus.NOT_FOUND)

    def read_json_body(self) -> dict:
        content_length = int(self.headers.get("content-length", "0"))
        if content_length == 0:
            return {}

        raw_body = self.rfile.read(content_length)
        try:
            return json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_json({"error": "Invalid JSON request body"}, HTTPStatus.BAD_REQUEST)
            return {}

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("cache-control", "no-store")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def current_owner_user_id(self) -> str:
        return str(self.headers.get("X-User-Id") or "").strip()

    @staticmethod
    def match_consultation_detail(path: str) -> str | None:
        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[:2] == ["api", "consultations"]:
            return parts[2]
        return None

    @staticmethod
    def match_message_create(path: str) -> str | None:
        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["api", "consultations"] and parts[3] == "messages":
            return parts[2]
        return None


def create_consultation(chief_complaint: str = "", user_context: dict | None = None) -> dict:
    return consultation_service.create_consultation(chief_complaint, user_context)


def append_message(consultation: dict, role: str, content: str) -> dict:
    return consultation_service.append_message(consultation, role, content)


def run() -> None:
    server = ThreadingHTTPServer((settings.host, settings.port), MedicalConsultantHandler)
    print(f"{settings.app_name} running at http://127.0.0.1:{settings.port}", flush=True)
    print(f"Listening on {settings.host}:{settings.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    run()
