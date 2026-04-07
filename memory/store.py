"""
Memory Store

Three distinct memory types, all backed by the shared SQLite connection.
All other modules read/write through here.

  Episodic  — what happened in conversations
  Semantic  — facts about people and the world
  Self      — agent's own significant history

Plus: relationship state, pending intents, agent state.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone

log = logging.getLogger(__name__)

_NOW = lambda: datetime.now(timezone.utc).isoformat()


class MemoryStore:
    def __init__(self, conn: sqlite3.Connection):
        self._db = conn

    # ── Person Registry ────────────────────────────────────────────────────────

    def resolve_person_id(self, sender_id: int, name: str | None = None) -> int:
        """Map Telegram sender_id → internal person_id. Creates if new."""
        row = self._db.execute(
            "SELECT id FROM persons WHERE sender_id = ?", (sender_id,)
        ).fetchone()
        if row:
            return row["id"]
        self._db.execute(
            "INSERT INTO persons (sender_id, first_name, created_at) VALUES (?, ?, ?)",
            (sender_id, name, _NOW()),
        )
        self._db.commit()
        return self._db.execute(
            "SELECT id FROM persons WHERE sender_id = ?", (sender_id,)
        ).fetchone()["id"]

    def fetch_person_profile(self, person_id: int) -> dict:
        row = self._db.execute(
            "SELECT sender_id, first_name, created_at FROM persons WHERE id = ?",
            (person_id,)
        ).fetchone()
        if not row:
            return {}
        return dict(row)

    # ── Episodic Memory ────────────────────────────────────────────────────────

    def write_episode(
        self,
        person_id: int,
        chat_id: int,
        content: str,
        tags: list[str] | None = None,
    ) -> None:
        self._db.execute(
            "INSERT INTO episodes (person_id, chat_id, content, tags, created_at) VALUES (?, ?, ?, ?, ?)",
            (person_id, chat_id, content, json.dumps(tags or []), _NOW()),
        )
        self._db.commit()

    def read_recent_episodes(self, chat_id: int, n: int = 5) -> list[dict]:
        rows = self._db.execute(
            """SELECT person_id, content, tags, created_at FROM episodes
               WHERE chat_id = ? ORDER BY created_at DESC LIMIT ?""",
            (chat_id, n),
        ).fetchall()
        return [
            {**dict(r), "tags": json.loads(r["tags"])}
            for r in reversed(rows)
        ]

    def read_person_episodes(self, person_id: int, n: int = 10) -> list[dict]:
        rows = self._db.execute(
            """SELECT chat_id, content, tags, created_at FROM episodes
               WHERE person_id = ? ORDER BY created_at DESC LIMIT ?""",
            (person_id, n),
        ).fetchall()
        return [
            {**dict(r), "tags": json.loads(r["tags"])}
            for r in reversed(rows)
        ]

    def get_last_session_summary(self, chat_id: int) -> str | None:
        """Returns the most recent episode tagged 'session_summary', if any."""
        row = self._db.execute(
            """SELECT content FROM episodes
               WHERE chat_id = ? AND tags LIKE '%session_summary%'
               ORDER BY created_at DESC LIMIT 1""",
            (chat_id,),
        ).fetchone()
        return row["content"] if row else None

    # ── Semantic Memory ────────────────────────────────────────────────────────

    def write_fact(
        self,
        person_id: int,
        content: str,
        source: str | None = None,
        confidence: float = 1.0,
    ) -> None:
        now = _NOW()
        self._db.execute(
            """INSERT INTO facts (person_id, content, source, confidence, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (person_id, content, source, confidence, now, now),
        )
        self._db.commit()

    def read_person_facts(self, person_id: int) -> list[dict]:
        rows = self._db.execute(
            """SELECT id, content, source, confidence, updated_at FROM facts
               WHERE person_id = ? ORDER BY confidence DESC, updated_at DESC""",
            (person_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_fact(self, fact_id: int, content: str, confidence: float | None = None) -> None:
        if confidence is not None:
            self._db.execute(
                "UPDATE facts SET content = ?, confidence = ?, updated_at = ? WHERE id = ?",
                (content, confidence, _NOW(), fact_id),
            )
        else:
            self._db.execute(
                "UPDATE facts SET content = ?, updated_at = ? WHERE id = ?",
                (content, _NOW(), fact_id),
            )
        self._db.commit()

    # ── Self Memory ────────────────────────────────────────────────────────────

    def write_self_memory(self, content: str, event_type: str | None = None) -> None:
        self._db.execute(
            "INSERT INTO self_memory (content, event_type, created_at) VALUES (?, ?, ?)",
            (content, event_type, _NOW()),
        )
        self._db.commit()

    def read_self_history(self, n: int = 20) -> list[dict]:
        rows = self._db.execute(
            "SELECT content, event_type, created_at FROM self_memory ORDER BY created_at DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ── Relationship State ─────────────────────────────────────────────────────

    def get_relationship(self, person_id: int, chat_id: int) -> dict:
        row = self._db.execute(
            "SELECT tone, unresolved_threads, agent_commitments, updated_at FROM relationship WHERE person_id = ? AND chat_id = ?",
            (person_id, chat_id),
        ).fetchone()
        if not row:
            return {
                "tone": "neutral",
                "unresolved_threads": [],
                "agent_commitments": [],
                "updated_at": None,
            }
        return {
            "tone": row["tone"],
            "unresolved_threads": json.loads(row["unresolved_threads"]),
            "agent_commitments":  json.loads(row["agent_commitments"]),
            "updated_at":         row["updated_at"],
        }

    def update_relationship(self, person_id: int, chat_id: int, delta: dict) -> None:
        """
        delta can contain any subset of: tone, unresolved_threads, agent_commitments.
        Lists are replaced wholesale if provided.
        """
        existing = self.get_relationship(person_id, chat_id)
        tone       = delta.get("tone", existing["tone"])
        unresolved = json.dumps(delta.get("unresolved_threads", existing["unresolved_threads"]))
        commitments = json.dumps(delta.get("agent_commitments", existing["agent_commitments"]))
        now = _NOW()
        self._db.execute("""
            INSERT INTO relationship (person_id, chat_id, tone, unresolved_threads, agent_commitments, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(person_id, chat_id) DO UPDATE SET
                tone                = excluded.tone,
                unresolved_threads  = excluded.unresolved_threads,
                agent_commitments   = excluded.agent_commitments,
                updated_at          = excluded.updated_at
        """, (person_id, chat_id, tone, unresolved, commitments, now))
        self._db.commit()

    # ── Pending Intents ────────────────────────────────────────────────────────

    def get_pending_intents(self, chat_id: int) -> list[dict]:
        rows = self._db.execute(
            "SELECT id, intent, created_at FROM pending_intents"
            " WHERE chat_id = ? AND resolved = 0 ORDER BY created_at DESC LIMIT 3",
            (chat_id,),
        ).fetchall()
        return list(reversed([dict(r) for r in rows]))

    def set_pending_intent(self, chat_id: int, intent: str) -> None:
        # new commitment supersedes all previous ones
        self._db.execute(
            "UPDATE pending_intents SET resolved = 1 WHERE chat_id = ? AND resolved = 0",
            (chat_id,),
        )
        self._db.execute(
            "INSERT INTO pending_intents (chat_id, intent, created_at) VALUES (?, ?, ?)",
            (chat_id, intent, _NOW()),
        )
        self._db.commit()

    def resolve_pending_intent(self, intent_id: int) -> None:
        self._db.execute(
            "UPDATE pending_intents SET resolved = 1 WHERE id = ?", (intent_id,)
        )
        self._db.commit()

    # ── Agent State ────────────────────────────────────────────────────────────

    def get_agent_state(self) -> dict:
        row = self._db.execute(
            "SELECT mood, energy, updated_at FROM agent_state WHERE id = 1"
        ).fetchone()
        return dict(row) if row else {"mood": "neutral", "energy": 0.7, "updated_at": None}

    def set_agent_state(self, mood: str | None = None, energy: float | None = None) -> None:
        current = self.get_agent_state()
        self._db.execute("""
            UPDATE agent_state SET
                mood       = ?,
                energy     = ?,
                updated_at = ?
            WHERE id = 1
        """, (
            mood   if mood   is not None else current["mood"],
            energy if energy is not None else current["energy"],
            _NOW(),
        ))
        self._db.commit()

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def search(self, query: str, person_id: int | None = None, chat_id: int | None = None) -> dict:
        """Keyword search across episodes and facts."""
        like = f"%{query}%"
        results: dict = {"episodes": [], "facts": []}

        ep_query = "SELECT person_id, chat_id, content, tags, created_at FROM episodes WHERE content LIKE ?"
        ep_params: list = [like]
        if person_id:
            ep_query += " AND person_id = ?"
            ep_params.append(person_id)
        if chat_id:
            ep_query += " AND chat_id = ?"
            ep_params.append(chat_id)
        ep_query += " ORDER BY created_at DESC LIMIT 10"
        rows = self._db.execute(ep_query, ep_params).fetchall()
        results["episodes"] = [{**dict(r), "tags": json.loads(r["tags"])} for r in rows]

        fact_query = "SELECT id, person_id, content, source, confidence FROM facts WHERE content LIKE ?"
        fact_params: list = [like]
        if person_id:
            fact_query += " AND person_id = ?"
            fact_params.append(person_id)
        fact_query += " ORDER BY confidence DESC LIMIT 10"
        rows = self._db.execute(fact_query, fact_params).fetchall()
        results["facts"] = [dict(r) for r in rows]

        return results
