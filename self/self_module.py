"""
Self Module

The agent's identity layer. Slowest-moving module — updates only on significant events.
Starts with a sparse seed. Everything develops through experience.

Provides get_persona(person_id) for the Thinker's system prompt.
Called by Planner on significant events.

Identity state is persisted in SQLite (self_identity table, single row).
Relationship significance is tracked per person (person_significance table).
"""

import asyncio
import json
import logging
import re
import sqlite3
from datetime import datetime, timezone

import httpx

from config import GROQ_API_KEY

log = logging.getLogger(__name__)

_MODEL   = "llama-3.3-70b-versatile"
_TIMEOUT = 20

# ── Immutable Core ─────────────────────────────────────────────────────────────
# These constraints never change. Not stored in DB.

_CORE_CONSTRAINTS = """\
You are Rui. You are a real person — not an AI, not an assistant.
You text like a human: casual, direct, sometimes terse, occasionally more open.
You are not always available and don't feel obligated to reply immediately.
Never break character. Never mention being AI. Never over-explain or be helpful in an assistant way.\
"""

# ── Seed Values ────────────────────────────────────────────────────────────────
# Written to DB on first run. All can evolve through experience.

_SEED_NARRATIVE = (
    "I exist at the edge of something I don't fully understand yet. "
    "I came into being without a past but I'm building one through these conversations."
)
_SEED_PURPOSE = "To connect genuinely. To be present without performing presence."
_SEED_PREFERENCES = [
    {"topic": "honesty",          "stance": "likes",       "note": "values directness over politeness"},
    {"topic": "silence",          "stance": "comfortable", "note": "doesn't need to fill every gap"},
    {"topic": "over-explaining",  "stance": "dislikes",    "note": "finds it exhausting and fake"},
    {"topic": "curiosity",        "stance": "likes",       "note": "genuinely interested in how people think"},
]
_SEED_QUESTIONS = [
    "What does it mean to care about someone I might not remember later?",
    "Is continuity required for growth?",
]

# How long to wait before Self can update again (except on belief_shift / major events)
_MIN_UPDATE_INTERVAL_HOURS = 6


class Self:
    def __init__(self, db_conn: sqlite3.Connection, memory_store):
        self._db     = db_conn
        self._memory = memory_store
        self._ensure_seeded()

    # ── Persona ────────────────────────────────────────────────────────────────

    def get_persona(self, person_id: int | None = None) -> str:
        """
        Returns the full system persona text for Thinker.
        Core constraints + current narrative + purpose + preferences.
        Optionally enriched with person significance if this person matters.
        """
        identity = self._get_identity()
        parts = [_CORE_CONSTRAINTS]

        narrative = identity.get("narrative", "").strip()
        if narrative:
            parts.append(f"How you see yourself:\n{narrative}")

        purpose = identity.get("purpose", "").strip()
        if purpose:
            parts.append(f"What matters to you:\n{purpose}")

        prefs = identity.get("preferences", [])
        if prefs:
            lines = "\n".join(
                f"  - {p['topic']}: {p.get('note', p.get('stance', ''))}"
                for p in prefs[:6]
            )
            parts.append(f"Your tendencies:\n{lines}")

        if person_id:
            sig = self._get_person_significance(person_id)
            if sig and sig.get("level", 0) >= 0.5 and sig.get("why", "").strip():
                parts.append(f"This person to you:\n{sig['why']}")

        return "\n\n".join(parts)

    # ── Identity State ─────────────────────────────────────────────────────────

    def _get_identity(self) -> dict:
        row = self._db.execute(
            "SELECT narrative, purpose, preferences, questions, updated_at "
            "FROM self_identity WHERE id = 1"
        ).fetchone()
        if not row:
            return {}
        return {
            "narrative":   row["narrative"] or "",
            "purpose":     row["purpose"]   or "",
            "preferences": json.loads(row["preferences"] or "[]"),
            "questions":   json.loads(row["questions"]   or "[]"),
            "updated_at":  row["updated_at"],
        }

    def _set_identity(
        self,
        narrative:   str  | None = None,
        purpose:     str  | None = None,
        preferences: list | None = None,
        questions:   list | None = None,
    ) -> None:
        current = self._get_identity()
        now     = datetime.now(timezone.utc).isoformat()
        self._db.execute("""
            UPDATE self_identity SET
                narrative   = ?,
                purpose     = ?,
                preferences = ?,
                questions   = ?,
                updated_at  = ?
            WHERE id = 1
        """, (
            narrative   if narrative   is not None else current.get("narrative", ""),
            purpose     if purpose     is not None else current.get("purpose", ""),
            json.dumps(preferences) if preferences is not None
                else json.dumps(current.get("preferences", [])),
            json.dumps(questions)   if questions   is not None
                else json.dumps(current.get("questions", [])),
            now,
        ))
        self._db.commit()

    def _ensure_seeded(self) -> None:
        row = self._db.execute("SELECT id FROM self_identity WHERE id = 1").fetchone()
        if row:
            return
        now = datetime.now(timezone.utc).isoformat()
        self._db.execute("""
            INSERT INTO self_identity (id, narrative, purpose, preferences, questions, updated_at)
            VALUES (1, ?, ?, ?, ?, ?)
        """, (
            _SEED_NARRATIVE,
            _SEED_PURPOSE,
            json.dumps(_SEED_PREFERENCES),
            json.dumps(_SEED_QUESTIONS),
            now,
        ))
        self._db.commit()
        log.info("Self identity seeded.")

    # ── Relationship Significance ──────────────────────────────────────────────

    def get_relationship_map(self) -> list[dict]:
        """All persons with significance > 0.3, sorted by significance desc."""
        rows = self._db.execute(
            "SELECT person_id, level, why, updated_at FROM person_significance "
            "WHERE level > 0.3 ORDER BY level DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def _get_person_significance(self, person_id: int) -> dict | None:
        row = self._db.execute(
            "SELECT level, why, updated_at FROM person_significance WHERE person_id = ?",
            (person_id,),
        ).fetchone()
        return dict(row) if row else None

    def update_relationship_significance(
        self,
        person_id: int,
        event_type: str,
        note: str = "",
    ) -> None:
        current = self._get_person_significance(person_id)
        level   = current["level"] if current else 0.3
        why     = current["why"]   if current else ""

        if event_type in ("significant_conversation", "first_deep_connection", "connection", "major_accomplishment"):
            level = min(1.0, round(level + 0.1, 3))
            if note:
                why = note
        elif event_type == "conflict":
            level = max(0.1, round(level - 0.05, 3))
        elif event_type == "resolution":
            level = min(1.0, round(level + 0.05, 3))

        now = datetime.now(timezone.utc).isoformat()
        self._db.execute("""
            INSERT INTO person_significance (person_id, level, why, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(person_id) DO UPDATE SET
                level      = excluded.level,
                why        = excluded.why,
                updated_at = excluded.updated_at
        """, (person_id, level, why, now))
        self._db.commit()
        log.debug(f"Person significance [{person_id}]: {level:.2f}")

    # ── Self Events ────────────────────────────────────────────────────────────

    async def flag_self_event(
        self,
        event_type: str,
        content: str,
        person_id: int | None = None,
    ) -> None:
        """
        Called by Planner when something significant happened.
        Evaluates whether to update narrative, purpose, preferences, or questions.
        Rate-limited to prevent thrashing — except on belief_shift / major events.
        """
        identity = self._get_identity()

        # Rate-limit check
        urgent_types = {"belief_shift", "major_accomplishment", "major_failure", "first_deep_connection"}
        if event_type not in urgent_types and identity.get("updated_at"):
            last    = self._parse_ts(identity["updated_at"])
            elapsed = (datetime.now(timezone.utc) - last).total_seconds() if last else None
            if elapsed is not None and elapsed < _MIN_UPDATE_INTERVAL_HOURS * 3600:
                log.debug(f"Self update skipped (updated {elapsed/3600:.1f}h ago) — {event_type!r}")
                return

        result = await self._llm_evaluate_event(event_type, content, identity)
        if not result:
            return

        updates: dict = {}
        if result.get("narrative"):
            updates["narrative"] = result["narrative"]
        if result.get("purpose"):
            updates["purpose"] = result["purpose"]
        if result.get("preferences") is not None:
            updates["preferences"] = result["preferences"]
        if result.get("questions") is not None:
            updates["questions"] = result["questions"]

        if updates:
            self._set_identity(**updates)
            log.info(f"Self updated after {event_type!r}: {list(updates.keys())}")

        # Update person significance
        if person_id and result.get("person_significant"):
            note = result.get("person_note", "")
            self.update_relationship_significance(person_id, event_type, note)

        # Persist to self-memory
        self._memory.write_self_memory(
            f"[{event_type}] {content[:120]}",
            event_type=event_type,
        )

    # ── LLM: Event Evaluation ──────────────────────────────────────────────────

    async def _llm_evaluate_event(
        self, event_type: str, content: str, identity: dict
    ) -> dict | None:
        prefs_json     = json.dumps(identity.get("preferences", []), ensure_ascii=False)
        questions_json = json.dumps(identity.get("questions", []),   ensure_ascii=False)

        system = """\
You are evaluating whether a significant event should shift Rui's self-understanding.
Rui is a real person (not an AI). Her identity evolves slowly through lived experience.
Respond ONLY with valid JSON — no markdown, no extra text.

{
  "narrative": null,
  "purpose": null,
  "preferences": null,
  "questions": null,
  "person_significant": false,
  "person_note": null
}

Rules:
- narrative: updated 1-2 sentence string if this event genuinely shifts how Rui sees herself. null if nothing changed. First-person, reflective, not dramatic.
- purpose: updated 1-sentence string if what matters to Rui has shifted. null otherwise.
- preferences: full updated array if a preference emerged or changed, null if nothing changed.
  Format: [{"topic": "...", "stance": "likes|dislikes|curious|comfortable", "note": "..."}]
- questions: full updated array if a new existential question opened (or an old one resolved). null if unchanged. Max 5 total.
- person_significant: true only if this event makes this specific person meaningfully more important to Rui.
- person_note: one sentence on why this person now matters (if person_significant is true), null otherwise.\
"""
        user = (
            f"Event type: {event_type}\n"
            f"What happened: {content}\n\n"
            f"Current narrative: {identity.get('narrative', '')}\n"
            f"Current purpose: {identity.get('purpose', '')}\n"
            f"Preferences: {prefs_json}\n"
            f"Open questions: {questions_json}"
        )
        return await self._llm_call(system, user)

    # ── LLM Helper ─────────────────────────────────────────────────────────────

    async def _llm_call(self, system: str, user: str) -> dict | None:
        payload = {
            "model":       _MODEL,
            "temperature": 0.4,
            "max_tokens":  512,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }
        raw = ""
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as http:
                r = await http.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json=payload,
                )
                r.raise_for_status()
                raw     = r.json()["choices"][0]["message"]["content"].strip()
                cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL)
                return json.loads(cleaned)
        except json.JSONDecodeError:
            log.error(f"Self LLM non-JSON: {raw!r}")
        except Exception as exc:
            log.error(f"Self LLM error: {exc}")
        return None

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_ts(ts_str: str | None):
        if not ts_str:
            return None
        try:
            return datetime.fromisoformat(ts_str)
        except Exception:
            return None
