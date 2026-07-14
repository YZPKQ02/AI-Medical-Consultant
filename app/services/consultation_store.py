from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any, Protocol

from app.agents.medical_agent import current_timestamp

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
except ImportError:  # pragma: no cover - depends on production PostgreSQL dependency.
    psycopg = None
    dict_row = None
    Jsonb = None


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


class PostgreSQLConsultationStore:
    def __init__(self, database_url: str):
        if psycopg is None or dict_row is None:
            raise RuntimeError(
                "PostgreSQL support requires psycopg. Run `.venv\\Scripts\\python.exe -m pip install -r requirements.txt`."
            )

        self.database_url = database_url
        self._lock = RLock()
        self._conn = psycopg.connect(database_url, row_factory=dict_row)
        self._closed = False

    def count(self) -> int:
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS total FROM consultations WHERE deleted_at IS NULL")
                row = cur.fetchone()
                return int(row["total"])

    def list(self, owner_user_id: str | None = None) -> list[dict[str, Any]]:
        where = "deleted_at IS NULL"
        params: tuple[Any, ...] = ()
        if owner_user_id is not None:
            where += " AND owner_user_id = %s"
            params = (owner_user_id,)

        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, owner_user_id, chief_complaint, user_context_json,
                           created_at, updated_at
                    FROM consultations
                    WHERE {where}
                    ORDER BY updated_at DESC
                    """,
                    params,
                )
                rows = cur.fetchall()

                summaries = []
                for row in rows:
                    summaries.append(
                        {
                            "id": row["id"],
                            "chief_complaint": row["chief_complaint"],
                            "message_count": self._message_count(row["id"]),
                            "created_at": stringify_timestamp(row["created_at"]),
                            "updated_at": stringify_timestamp(row["updated_at"]),
                        }
                    )
                return summaries

    def _message_count(self, consultation_id: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS total
                FROM messages
                WHERE consultation_id = %s AND deleted_at IS NULL
                """,
                (consultation_id,),
            )
            row = cur.fetchone()
            return int(row["total"])

    def get(self, consultation_id: str, owner_user_id: str | None = None) -> dict[str, Any] | None:
        where = "id = %s AND deleted_at IS NULL"
        params: tuple[Any, ...] = (consultation_id,)
        if owner_user_id is not None:
            where += " AND owner_user_id = %s"
            params = (consultation_id, owner_user_id)

        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, owner_user_id, chief_complaint, user_context_json,
                           created_at, updated_at
                    FROM consultations
                    WHERE {where}
                    """,
                    params,
                )
                row = cur.fetchone()
                if row is None:
                    return None

                cur.execute(
                    """
                    SELECT payload_json
                    FROM messages
                    WHERE consultation_id = %s AND deleted_at IS NULL
                    ORDER BY sequence ASC, created_at ASC
                    """,
                    (consultation_id,),
                )
                messages = [ensure_dict(message_row["payload_json"]) for message_row in cur.fetchall()]

            return {
                "id": row["id"],
                "owner_user_id": row["owner_user_id"],
                "chief_complaint": row["chief_complaint"],
                "user_context": ensure_dict(row["user_context_json"]),
                "messages": messages,
                "created_at": stringify_timestamp(row["created_at"]),
                "updated_at": stringify_timestamp(row["updated_at"]),
            }

    def save(self, consultation: dict[str, Any]) -> None:
        owner_user_id = str(consultation.get("owner_user_id") or "anonymous")
        user_context = ensure_dict(consultation.get("user_context") or {})
        now = current_timestamp()

        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (id, display_name, created_at, updated_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET updated_at = excluded.updated_at
                    """,
                    (owner_user_id, owner_user_id, now, now),
                )
                cur.execute(
                    """
                    INSERT INTO consultations (
                        id, owner_user_id, chief_complaint, user_context_json,
                        created_at, updated_at, deleted_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, NULL)
                    ON CONFLICT (id) DO UPDATE SET
                        owner_user_id = excluded.owner_user_id,
                        chief_complaint = excluded.chief_complaint,
                        user_context_json = excluded.user_context_json,
                        updated_at = excluded.updated_at,
                        deleted_at = NULL
                    """,
                    (
                        consultation["id"],
                        owner_user_id,
                        str(consultation.get("chief_complaint") or ""),
                        to_jsonb(user_context),
                        consultation["created_at"],
                        consultation["updated_at"],
                    ),
                )
                self._replace_messages(cur, consultation)
            self._conn.commit()

    def _replace_messages(self, cur, consultation: dict[str, Any]) -> None:
        consultation_id = consultation["id"]
        cur.execute("DELETE FROM tool_calls WHERE consultation_id = %s", (consultation_id,))
        cur.execute("DELETE FROM agent_runs WHERE consultation_id = %s", (consultation_id,))
        cur.execute("DELETE FROM messages WHERE consultation_id = %s", (consultation_id,))

        for index, message in enumerate(consultation.get("messages") or []):
            message_payload = ensure_dict(message)
            message_id = str(message_payload.get("id") or f"message-{index}")
            cur.execute(
                """
                INSERT INTO messages (
                    id, consultation_id, sequence, role, content,
                    payload_json, created_at, deleted_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, NULL)
                """,
                (
                    message_id,
                    consultation_id,
                    index,
                    str(message_payload.get("role") or ""),
                    str(message_payload.get("content") or ""),
                    to_jsonb(message_payload),
                    message_payload.get("created_at") or consultation.get("updated_at"),
                ),
            )

            if message_payload.get("role") == "assistant":
                self._insert_agent_run(cur, consultation_id, message_id, message_payload)

    def _insert_agent_run(
        self,
        cur,
        consultation_id: str,
        message_id: str,
        message_payload: dict[str, Any],
    ) -> None:
        analysis = ensure_dict(message_payload.get("analysis") or {})
        agent_state = ensure_dict(analysis.get("agent_state") or {})
        run_id = str(agent_state.get("run_id") or f"run-{message_id}")
        risk = ensure_dict(analysis.get("risk") or {})
        cur.execute(
            """
            INSERT INTO agent_runs (
                id, consultation_id, assistant_message_id, intent, stage,
                risk_level, department, analysis_json, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                run_id,
                consultation_id,
                message_id,
                str(analysis.get("intent") or ""),
                str(analysis.get("stage") or ""),
                int(risk.get("level") or 0),
                str(analysis.get("department") or ""),
                to_jsonb(analysis),
                message_payload.get("created_at") or current_timestamp(),
            ),
        )

        for index, tool in enumerate(analysis.get("tool_results") or []):
            tool_payload = ensure_dict(tool)
            cur.execute(
                """
                INSERT INTO tool_calls (
                    agent_run_id, consultation_id, name, status,
                    input_summary_json, output_summary_json, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    consultation_id,
                    str(tool_payload.get("name") or f"tool-{index}"),
                    str(tool_payload.get("status") or ""),
                    to_jsonb(ensure_dict(tool_payload.get("input_summary") or {})),
                    to_jsonb(ensure_dict(tool_payload.get("output_summary") or {})),
                    message_payload.get("created_at") or current_timestamp(),
                ),
            )

    def delete(self, consultation_id: str, owner_user_id: str | None = None) -> dict[str, Any] | None:
        consultation = self.get(consultation_id, owner_user_id=owner_user_id)
        if consultation is None:
            return None

        where = "id = %s"
        params: tuple[Any, ...] = (consultation_id,)
        if owner_user_id is not None:
            where += " AND owner_user_id = %s"
            params = (consultation_id, owner_user_id)

        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"UPDATE consultations SET deleted_at = %s WHERE {where}",
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

    def __enter__(self) -> "PostgreSQLConsultationStore":
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


def ensure_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return load_payload(value)
    return {}


def stringify_timestamp(value: Any) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(value or "")


def to_jsonb(value: dict[str, Any]) -> Any:
    if Jsonb is not None:
        return Jsonb(value)
    return json.dumps(value, ensure_ascii=False)


def postgresql_schema_sql() -> tuple[str, ...]:
    return (
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            deleted_at TIMESTAMPTZ
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS consultations (
            id TEXT PRIMARY KEY,
            owner_user_id TEXT NOT NULL REFERENCES users(id),
            chief_complaint TEXT NOT NULL,
            user_context_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            deleted_at TIMESTAMPTZ
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            consultation_id TEXT NOT NULL REFERENCES consultations(id) ON DELETE CASCADE,
            sequence INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            payload_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            deleted_at TIMESTAMPTZ
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS agent_runs (
            id TEXT PRIMARY KEY,
            consultation_id TEXT NOT NULL REFERENCES consultations(id) ON DELETE CASCADE,
            assistant_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            intent TEXT NOT NULL,
            stage TEXT NOT NULL,
            risk_level INTEGER NOT NULL DEFAULT 0,
            department TEXT NOT NULL,
            analysis_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS tool_calls (
            id BIGSERIAL PRIMARY KEY,
            agent_run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
            consultation_id TEXT NOT NULL REFERENCES consultations(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            input_summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            output_summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_consultations_owner_active_updated
        ON consultations(owner_user_id, deleted_at, updated_at DESC)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_messages_consultation_sequence
        ON messages(consultation_id, sequence)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_agent_runs_consultation
        ON agent_runs(consultation_id, created_at DESC)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_tool_calls_consultation
        ON tool_calls(consultation_id, created_at DESC)
        """,
    )
