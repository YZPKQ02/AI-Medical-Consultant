from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any, Protocol

from app.agents.medical_agent import current_timestamp


class ConsultationStore(Protocol):
    def count(self) -> int: ...

    def list(self, owner_user_id: str | None = None) -> list[dict[str, Any]]: ...

    def get(self, consultation_id: str, owner_user_id: str | None = None) -> dict[str, Any] | None: ...

    def save(self, consultation: dict[str, Any]) -> None: ...

    def delete(self, consultation_id: str, owner_user_id: str | None = None) -> dict[str, Any] | None: ...


@dataclass
class InMemoryConsultationStore:
    consultations: dict[str, dict[str, Any]]

    def count(self) -> int:
        return len(self.consultations)

    def list(self, owner_user_id: str | None = None) -> list[dict[str, Any]]:
        return [
            consultation_summary(item)
            for item in self.consultations.values()
            if owner_user_id is None or item.get("owner_user_id") == owner_user_id
        ]

    def get(self, consultation_id: str, owner_user_id: str | None = None) -> dict[str, Any] | None:
        consultation = self.consultations.get(consultation_id)
        if consultation is None:
            return None
        if owner_user_id is not None and consultation.get("owner_user_id") != owner_user_id:
            return None
        return consultation

    def save(self, consultation: dict[str, Any]) -> None:
        self.consultations[consultation["id"]] = consultation

    def delete(self, consultation_id: str, owner_user_id: str | None = None) -> dict[str, Any] | None:
        consultation = self.get(consultation_id, owner_user_id=owner_user_id)
        if consultation is None:
            return None
        return self.consultations.pop(consultation_id, None)


class SQLiteConsultationStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._closed = False
        self._initialize()

    def _initialize(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS consultations (
                    id TEXT PRIMARY KEY,
                    owner_user_id TEXT NOT NULL DEFAULT 'anonymous',
                    chief_complaint TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT
                )
                """
            )
            self._ensure_owner_user_id_column()
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_consultations_active_updated
                ON consultations(deleted_at, updated_at)
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_consultations_owner_active_updated
                ON consultations(owner_user_id, deleted_at, updated_at)
                """
            )
            self._conn.commit()

    def _ensure_owner_user_id_column(self) -> None:
        columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(consultations)").fetchall()
        }
        if "owner_user_id" not in columns:
            self._conn.execute(
                "ALTER TABLE consultations ADD COLUMN owner_user_id TEXT NOT NULL DEFAULT 'anonymous'"
            )

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS total FROM consultations WHERE deleted_at IS NULL"
            ).fetchone()
            return int(row["total"])

    def list(self, owner_user_id: str | None = None) -> list[dict[str, Any]]:
        where = "deleted_at IS NULL"
        params: tuple[Any, ...] = ()
        if owner_user_id is not None:
            where += " AND owner_user_id = ?"
            params = (owner_user_id,)

        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT payload_json
                FROM consultations
                WHERE {where}
                ORDER BY updated_at DESC
                """,
                params,
            ).fetchall()
            return [consultation_summary(load_payload(row["payload_json"])) for row in rows]

    def get(self, consultation_id: str, owner_user_id: str | None = None) -> dict[str, Any] | None:
        where = "id = ? AND deleted_at IS NULL"
        params: tuple[Any, ...] = (consultation_id,)
        if owner_user_id is not None:
            where += " AND owner_user_id = ?"
            params = (consultation_id, owner_user_id)

        with self._lock:
            row = self._conn.execute(
                f"""
                SELECT payload_json
                FROM consultations
                WHERE {where}
                """,
                params,
            ).fetchone()
            return load_payload(row["payload_json"]) if row else None

    def save(self, consultation: dict[str, Any]) -> None:
        payload = dump_payload(consultation)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO consultations (
                    id,
                    owner_user_id,
                    chief_complaint,
                    payload_json,
                    created_at,
                    updated_at,
                    deleted_at
                )
                VALUES (?, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(id) DO UPDATE SET
                    owner_user_id = excluded.owner_user_id,
                    chief_complaint = excluded.chief_complaint,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at,
                    deleted_at = NULL
                """,
                (
                    consultation["id"],
                    str(consultation.get("owner_user_id") or "anonymous"),
                    str(consultation.get("chief_complaint") or ""),
                    payload,
                    consultation["created_at"],
                    consultation["updated_at"],
                ),
            )
            self._conn.commit()

    def delete(self, consultation_id: str, owner_user_id: str | None = None) -> dict[str, Any] | None:
        consultation = self.get(consultation_id, owner_user_id=owner_user_id)
        if consultation is None:
            return None

        where = "id = ?"
        params: tuple[Any, ...] = (consultation_id,)
        if owner_user_id is not None:
            where += " AND owner_user_id = ?"
            params = (consultation_id, owner_user_id)

        with self._lock:
            self._conn.execute(
                f"UPDATE consultations SET deleted_at = ? WHERE {where}",
                (current_timestamp(), *params),
            )
            self._conn.commit()
        return consultation

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.close()
            self._closed = True

    def __enter__(self) -> "SQLiteConsultationStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def consultation_summary(consultation: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": consultation["id"],
        "chief_complaint": consultation["chief_complaint"],
        "message_count": len(consultation["messages"]),
        "created_at": consultation["created_at"],
        "updated_at": consultation["updated_at"],
    }


def dump_payload(consultation: dict[str, Any]) -> str:
    return json.dumps(consultation, ensure_ascii=False, separators=(",", ":"))


def load_payload(value: str) -> dict[str, Any]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("Invalid consultation payload")
    return payload
