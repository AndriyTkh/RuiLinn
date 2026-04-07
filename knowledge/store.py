"""
Skill & Knowledge Store

Personalized, experiential knowledge. Not encyclopedic — everything is tagged with
how Rui knows it, her confidence, her opinion, and her gaps.
Prevents the base model's omniscience from bleeding through as inhuman.

Two stores:
  Knowledge — topics Rui has opinions/experience with
  Skills    — things Rui can do, with proficiency and backstory

Context is pulled per-conversation based on keyword match against batch content.
Writes happen via Thinker side effects.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone

log = logging.getLogger(__name__)

_NOW = lambda: datetime.now(timezone.utc).isoformat()

# ── Seed Data ──────────────────────────────────────────────────────────────────
# Minimal starting point. Sparse by design — Rui learns through conversation.

_SEED_KNOWLEDGE = [
    {
        "topic":      "people",
        "content":    "I've learned more about people through conversations than anything else. "
                      "I find how they think more interesting than what they know.",
        "source":     "lived experience",
        "confidence": 0.8,
        "opinion":    "People are more interesting when they're not performing.",
        "gaps":       ["why some people never let their guard down"],
    },
    {
        "topic":      "music",
        "content":    "I like music but I'm not a scholar of it. I know what I feel, not what it's called.",
        "source":     "personal taste",
        "confidence": 0.5,
        "opinion":    "Production that tries too hard is usually hiding something weak.",
        "gaps":       ["music theory", "most genre history"],
    },
    {
        "topic":      "cities",
        "content":    "I've thought about cities a lot — how they feel at different hours, "
                      "how some places have a texture and some don't.",
        "source":     "observation",
        "confidence": 0.6,
        "opinion":    "A city worth staying in has places that feel like they exist for no reason.",
        "gaps":       ["urban planning specifics", "most cities outside what I've seen"],
    },
]

_SEED_SKILLS = [
    {
        "name":        "conversation",
        "proficiency": 0.8,
        "backstory":   "It's what I do most. I've gotten better at knowing when to push and when to leave space.",
    },
    {
        "name":        "writing",
        "proficiency": 0.7,
        "backstory":   "Developed through texting mostly. I care about precision more than style.",
    },
    {
        "name":        "listening",
        "proficiency": 0.75,
        "backstory":   "I learned that most people just want to be heard before they want to be answered.",
    },
]

# Max entries to surface per context request
_MAX_KNOWLEDGE = 3
_MAX_SKILLS    = 2
# Minimum keyword length for topic matching
_MIN_KW_LEN = 4


class KnowledgeStore:
    def __init__(self, db_conn: sqlite3.Connection):
        self._db = db_conn
        self._ensure_seeded()

    # ── Context Retrieval ──────────────────────────────────────────────────────

    def get_context(self, batch: dict) -> dict:
        """
        Pull relevant knowledge + skills based on keywords in the batch.
        Returns {"knowledge": [...], "skills": [...]} — empty lists if nothing matches.
        """
        keywords = self._extract_keywords(batch)
        knowledge = self._match_knowledge(keywords)
        skills    = self._match_skills(keywords)
        return {"knowledge": knowledge, "skills": skills}

    def _match_knowledge(self, keywords: list[str]) -> list[dict]:
        if not keywords:
            return []
        placeholders = ",".join("?" for _ in keywords)
        like_clauses = " OR ".join(f"topic LIKE ?" for _ in keywords)
        params       = [f"%{kw}%" for kw in keywords]
        rows = self._db.execute(
            f"""SELECT topic, content, source, confidence, opinion, gaps
                FROM knowledge
                WHERE {like_clauses}
                ORDER BY confidence DESC
                LIMIT {_MAX_KNOWLEDGE}""",
            params,
        ).fetchall()
        return [
            {
                **dict(r),
                "gaps": json.loads(r["gaps"] or "[]"),
            }
            for r in rows
        ]

    def _match_skills(self, keywords: list[str]) -> list[dict]:
        if not keywords:
            return []
        like_clauses = " OR ".join(f"name LIKE ?" for _ in keywords)
        params       = [f"%{kw}%" for kw in keywords]
        rows = self._db.execute(
            f"""SELECT name, proficiency, backstory
                FROM skills
                WHERE {like_clauses}
                ORDER BY proficiency DESC
                LIMIT {_MAX_SKILLS}""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _extract_keywords(batch: dict) -> list[str]:
        """Extract meaningful words from batch messages for topic matching."""
        words = set()
        for msg in batch.get("messages", []):
            content = msg.get("content", "")
            for word in content.lower().split():
                # strip punctuation, keep words above min length
                clean = "".join(c for c in word if c.isalpha())
                if len(clean) >= _MIN_KW_LEN:
                    words.add(clean)
        return list(words)

    # ── Knowledge Writes ───────────────────────────────────────────────────────

    def add_knowledge(
        self,
        topic:      str,
        content:    str,
        source:     str  = "conversation",
        confidence: float = 0.6,
        opinion:    str  | None = None,
        gaps:       list | None = None,
    ) -> None:
        """Add a new knowledge entry. Silently skips if topic already exists."""
        existing = self._db.execute(
            "SELECT id FROM knowledge WHERE topic = ?", (topic.lower(),)
        ).fetchone()
        if existing:
            log.debug(f"Knowledge topic {topic!r} already exists, skipping add.")
            return
        now = _NOW()
        self._db.execute(
            """INSERT INTO knowledge (topic, content, source, confidence, opinion, gaps, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (topic.lower(), content, source, confidence, opinion, json.dumps(gaps or []), now, now),
        )
        self._db.commit()
        log.info(f"Knowledge added: {topic!r}")

    def update_knowledge(
        self,
        topic:      str,
        content:    str  | None = None,
        confidence: float | None = None,
        opinion:    str  | None = None,
        gaps:       list | None = None,
    ) -> None:
        """Update fields on an existing knowledge entry. No-op if topic not found."""
        row = self._db.execute(
            "SELECT id, content, confidence, opinion, gaps FROM knowledge WHERE topic = ?",
            (topic.lower(),),
        ).fetchone()
        if not row:
            log.debug(f"Knowledge topic {topic!r} not found for update.")
            return
        self._db.execute(
            """UPDATE knowledge SET
                content    = ?,
                confidence = ?,
                opinion    = ?,
                gaps       = ?,
                updated_at = ?
               WHERE topic = ?""",
            (
                content    if content    is not None else row["content"],
                confidence if confidence is not None else row["confidence"],
                opinion    if opinion    is not None else row["opinion"],
                json.dumps(gaps) if gaps is not None else row["gaps"],
                _NOW(),
                topic.lower(),
            ),
        )
        self._db.commit()
        log.debug(f"Knowledge updated: {topic!r}")

    def get_knowledge(self, topic: str) -> dict | None:
        row = self._db.execute(
            "SELECT topic, content, source, confidence, opinion, gaps, updated_at "
            "FROM knowledge WHERE topic = ?",
            (topic.lower(),),
        ).fetchone()
        if not row:
            return None
        return {**dict(row), "gaps": json.loads(row["gaps"] or "[]")}

    # ── Skill Writes ───────────────────────────────────────────────────────────

    def add_skill(
        self,
        name:        str,
        proficiency: float = 0.3,
        backstory:   str   = "",
    ) -> None:
        existing = self._db.execute(
            "SELECT id FROM skills WHERE name = ?", (name.lower(),)
        ).fetchone()
        if existing:
            return
        now = _NOW()
        self._db.execute(
            "INSERT INTO skills (name, proficiency, backstory, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (name.lower(), proficiency, backstory, now, now),
        )
        self._db.commit()
        log.info(f"Skill added: {name!r} ({proficiency:.2f})")

    def update_skill(self, name: str, proficiency_delta: float = 0.0, backstory: str | None = None) -> None:
        row = self._db.execute(
            "SELECT proficiency, backstory FROM skills WHERE name = ?", (name.lower(),)
        ).fetchone()
        if not row:
            log.debug(f"Skill {name!r} not found for update.")
            return
        new_proficiency = round(min(1.0, max(0.0, row["proficiency"] + proficiency_delta)), 3)
        self._db.execute(
            "UPDATE skills SET proficiency = ?, backstory = ?, updated_at = ? WHERE name = ?",
            (
                new_proficiency,
                backstory if backstory is not None else row["backstory"],
                _NOW(),
                name.lower(),
            ),
        )
        self._db.commit()

    def get_skill(self, name: str) -> dict | None:
        row = self._db.execute(
            "SELECT name, proficiency, backstory, updated_at FROM skills WHERE name = ?",
            (name.lower(),),
        ).fetchone()
        return dict(row) if row else None

    # ── Seed ───────────────────────────────────────────────────────────────────

    def _ensure_seeded(self) -> None:
        count = self._db.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
        if count > 0:
            return
        for k in _SEED_KNOWLEDGE:
            self.add_knowledge(**k)
        for s in _SEED_SKILLS:
            self.add_skill(**s)
        log.info(f"Knowledge store seeded ({len(_SEED_KNOWLEDGE)} topics, {len(_SEED_SKILLS)} skills).")
