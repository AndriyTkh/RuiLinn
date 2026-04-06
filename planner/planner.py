"""
Planner

Event-driven, async. Not called per-message.
Runs on conversation end (retrospective), significant events, and daily cycle.

Owns: goals, daily cycle, retrospective, emergent mood, and scheduling.
Emotions are emergent — derived from goal state and relationship outcomes, not hardcoded.

Wiring:
- run_retrospective(chat_id) — called as background task when gap_label == cold_open
- run_daily_cycle()          — background task, checks time every 5 min
- derive_mood()              — called after goal updates or interaction ends
- decay_mood()               — called at end of day and by daily cycle
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone, date as date_type, timedelta

import httpx

from config import GROQ_API_KEY
from db.store import (
    fetch_recent_batches,
    get_goals, add_goal, update_goal_status,
    get_daily_log, upsert_daily_log,
)
from memory.store import MemoryStore

log = logging.getLogger(__name__)

_MODEL        = "llama-3.3-70b-versatile"
_TIMEOUT      = 20
_MORNING_HOUR = 8   # UTC hour to run morning routine
_EVENING_HOUR = 22  # UTC hour to run evening routine


class Planner:
    def __init__(self, db_conn, memory: MemoryStore):
        self._db     = db_conn
        self._memory = memory

    # ── Retrospective ──────────────────────────────────────────────────────────

    async def run_retrospective(self, chat_id: int) -> None:
        """
        Triggered when a conversation goes cold (gap_label == cold_open).
        Summarize the last session, extract facts about the person, update mood.
        Runs as a background task — does not block the response pipeline.
        """
        # Skip if a session summary was written recently (< 1 hour)
        recent = self._memory.read_recent_episodes(chat_id, n=1)
        if recent and "session_summary" in recent[-1].get("tags", []):
            ts = self._parse_ts(recent[-1]["created_at"])
            if ts and (datetime.now(timezone.utc) - ts).total_seconds() < 3600:
                log.debug(f"Retrospective skipped (recent summary exists) [{chat_id}]")
                return

        person_id = self._resolve_person_id_for_chat(chat_id)
        if not person_id:
            log.debug(f"Retrospective skipped (no person resolved) [{chat_id}]")
            return

        batches = fetch_recent_batches(self._db, chat_id, n=30)
        if len(batches) < 2:
            return

        transcript = self._format_transcript(batches)
        result = await self._llm_retrospective(transcript)
        if not result:
            return

        # Session summary episode
        if result.get("summary"):
            self._memory.write_episode(
                person_id, chat_id, result["summary"],
                tags=["session_summary"],
            )

        # Person facts learned
        for fact in result.get("person_facts", []):
            content = fact.get("content", "").strip()
            if content:
                self._memory.write_fact(
                    person_id, content,
                    source="retrospective",
                    confidence=float(fact.get("confidence", 0.8)),
                )

        # Relationship tone
        if result.get("relationship_tone"):
            self._memory.update_relationship(
                person_id, chat_id,
                {"tone": result["relationship_tone"]},
            )

        # Write to self-memory if significant
        if result.get("significant"):
            summary_preview = result.get("summary", "")[:120]
            self._memory.write_self_memory(
                f"Notable conversation [{chat_id}]: {summary_preview}",
                event_type="significant_conversation",
            )

        # Update mood based on how the interaction went
        self._update_mood_from_interaction(result)

        n_facts = len(result.get("person_facts", []))
        log.info(f"── RETROSPECTIVE ── [{chat_id}] summary written, {n_facts} facts extracted")

    async def _llm_retrospective(self, transcript: str) -> dict | None:
        system = """\
Analyze this conversation transcript and produce a post-session analysis.

Respond ONLY with a valid JSON object — no markdown, no extra text:
{
  "summary": "2-3 sentence factual summary of what was discussed and the overall tone",
  "person_facts": [
    {"content": "...", "confidence": 0.8}
  ],
  "relationship_tone": "neutral",
  "significant": false
}

Rules:
- summary: neutral, factual. What happened and what was discussed.
- person_facts: concrete things learned about the other person (job, location, plans, feelings, opinions).
  Omit empty array if nothing new was learned.
- relationship_tone: warm | friendly | neutral | distant | tense — overall tone of this conversation.
- significant: true only if something genuinely notable happened (first deep exchange, conflict, major reveal, resolution).\
"""
        return await self._llm_call(system, transcript, max_tokens=512, temperature=0.2)

    # ── Daily Cycle ────────────────────────────────────────────────────────────

    async def run_morning(self) -> None:
        """Set today's intention based on goals + yesterday's summary."""
        today = date_type.today().isoformat()
        existing = get_daily_log(self._db, today)
        if existing and existing.get("intention"):
            log.debug("Morning routine already done today")
            return

        goals = get_goals(self._db)
        yesterday_str = (date_type.today() - timedelta(days=1)).isoformat()
        yesterday_log = get_daily_log(self._db, yesterday_str)

        parts = []
        if goals:
            g_text = "\n".join(f"  [{g['timescale']}] {g['content']}" for g in goals[:10])
            parts.append(f"Current goals:\n{g_text}")
        if yesterday_log and yesterday_log.get("summary"):
            parts.append(f"Yesterday:\n{yesterday_log['summary']}")

        if not parts:
            upsert_daily_log(self._db, today, intention="No specific intention set.")
            return

        system = """\
You are Rui, setting a personal intention for today.
Based on your goals and what happened yesterday, write a short, specific intention.

Respond ONLY with JSON: {"intention": "..."}

1-2 sentences. Personal and specific — not a to-do list.\
"""
        result = await self._llm_call(system, "\n\n".join(parts), max_tokens=120)
        intention = result.get("intention", "").strip() if result else ""
        upsert_daily_log(self._db, today, intention=intention or "No specific intention set.")
        if intention:
            log.info(f"── MORNING ── {intention[:80]!r}")

    async def run_evening(self) -> None:
        """Evaluate the day, write summary, update goal statuses, decay mood."""
        today = date_type.today().isoformat()
        existing = get_daily_log(self._db, today)
        if existing and existing.get("summary"):
            log.debug("Evening routine already done today")
            return

        goals     = get_goals(self._db)
        intention = existing.get("intention") if existing else None
        self_hist = self._memory.read_self_history(n=5)

        parts = []
        if intention:
            parts.append(f"Today's intention: {intention}")
        if goals:
            g_lines = "\n".join(
                f"  [id={g['id']} {g['timescale']}] {g['content']}" for g in goals[:10]
            )
            parts.append(f"Active goals:\n{g_lines}")
        if self_hist:
            events = "\n".join(f"  - {e['content'][:80]}" for e in self_hist[-3:])
            parts.append(f"Recent notable events:\n{events}")

        system = """\
You are Rui, reflecting at the end of the day.
Based on your intention, goals, and what happened, write a short personal summary.

Respond ONLY with JSON:
{
  "summary": "2-3 sentences. What happened, how it went, how you feel about it.",
  "goal_updates": [{"id": 1, "status": "achieved"}]
}

- goal_updates: only if a goal was clearly achieved or dropped today. Empty array if nothing changed.
  Valid statuses: achieved | dropped\
"""
        result = await self._llm_call(
            system, "\n\n".join(parts) or "Quiet day, nothing notable.", max_tokens=220
        )
        if not result:
            return

        summary = result.get("summary", "").strip()
        if summary:
            upsert_daily_log(self._db, today, summary=summary)

        for update in result.get("goal_updates", []):
            goal_id = update.get("id")
            status  = update.get("status")
            if goal_id and status in ("achieved", "dropped"):
                update_goal_status(self._db, goal_id, status)
                log.info(f"Goal id={goal_id} marked {status}")

        # Re-derive mood after goal state may have changed, then decay
        self.derive_mood()
        self.decay_mood()
        if summary:
            log.info(f"── EVENING ── {summary[:80]!r}")

    async def run_daily_cycle(self) -> None:
        """Background task. Wakes every 5 min to run morning/evening routines."""
        while True:
            try:
                hour = datetime.now(timezone.utc).hour
                if hour == _MORNING_HOUR:
                    await self.run_morning()
                elif hour == _EVENING_HOUR:
                    await self.run_evening()
            except Exception as exc:
                log.error(f"Daily cycle error: {exc}", exc_info=True)
            await asyncio.sleep(60 * 5)

    def get_daily_context(self) -> dict | None:
        """Today's intention + summary. Passed to thinker for grounding."""
        return get_daily_log(self._db, date_type.today().isoformat())

    # ── Goals ─────────────────────────────────────────────────────────────────

    def get_current_goals(self, timescale: str | None = None) -> list[dict]:
        return get_goals(self._db, status="active", timescale=timescale)

    def add_goal(self, content: str, timescale: str = "daily") -> None:
        add_goal(self._db, content, timescale)
        log.info(f"Goal added [{timescale}]: {content[:60]!r}")

    def achieve_goal(self, goal_id: int) -> None:
        update_goal_status(self._db, goal_id, "achieved")
        self.derive_mood()

    def drop_goal(self, goal_id: int) -> None:
        update_goal_status(self._db, goal_id, "dropped")
        self.derive_mood()

    # ── Mood (emergent) ───────────────────────────────────────────────────────

    def derive_mood(self) -> None:
        """
        Derive mood from goal achievement ratio. No LLM.
        More achieved goals relative to active ones → higher energy and warmer mood.
        """
        all_goals = get_goals(self._db, status=None)
        if not all_goals:
            return  # no goals → don't override whatever mood is set

        active   = sum(1 for g in all_goals if g["status"] == "active")
        achieved = sum(1 for g in all_goals if g["status"] == "achieved")
        total    = active + achieved

        ratio  = achieved / total if total > 0 else 0.5
        energy = round(0.4 + ratio * 0.5, 3)  # 0.4 (all active) → 0.9 (all achieved)

        if energy > 0.75:
            mood = "curious"
        elif energy > 0.55:
            mood = "neutral"
        else:
            mood = "withdrawn"

        self._memory.set_agent_state(mood=mood, energy=energy)
        log.debug(f"Mood derived: {mood} / energy={energy:.2f}")

    def get_current_mood(self) -> dict:
        return self._memory.get_agent_state()

    def decay_mood(self, rate: float = 0.05) -> None:
        """
        Gradually return energy toward 0.7 baseline.
        Called at end of day and periodically.
        """
        state  = self._memory.get_agent_state()
        e      = state["energy"]
        new_e  = round(e + (0.7 - e) * rate, 3)
        mood   = state["mood"] if abs(new_e - 0.7) > 0.15 else "neutral"
        self._memory.set_agent_state(mood=mood, energy=new_e)

    def _update_mood_from_interaction(self, retro_result: dict) -> None:
        """Shift mood based on how an interaction went (called after retrospective)."""
        tone        = retro_result.get("relationship_tone", "neutral")
        significant = retro_result.get("significant", False)

        state = self._memory.get_agent_state()
        e     = state["energy"]

        if tone in ("warm", "friendly"):
            e = min(0.95, e + 0.05)
        elif tone in ("distant", "tense"):
            e = max(0.20, e - 0.08)

        if significant:
            e = min(0.95, e + 0.05)

        mood_map = {
            "warm":    "curious",
            "friendly":"curious",
            "neutral": "neutral",
            "distant": "neutral",
            "tense":   "withdrawn",
        }
        mood = mood_map.get(tone, "neutral")
        self._memory.set_agent_state(mood=mood, energy=round(e, 3))

    # ── Scheduling ────────────────────────────────────────────────────────────

    async def schedule_unprompted(self, chat_id: int) -> None:
        """
        Decide if/when to reach out to this person unprompted.
        Currently logs the decision — actual send trigger is a future step.
        """
        person_id = self._resolve_person_id_for_chat(chat_id)
        if not person_id:
            return

        rel     = self._memory.get_relationship(person_id, chat_id)
        pending = self._memory.get_pending_intents(chat_id)
        goals   = self.get_current_goals()

        should_reach_out = rel["tone"] in ("warm", "friendly") and (
            bool(pending)
            or any("connection" in g["content"].lower() for g in goals)
        )

        if should_reach_out and pending:
            log.info(
                f"── SCHEDULE ── [{chat_id}] flagged for unprompted follow-up "
                f"(pending intents: {len(pending)})"
            )
            # Future: emit to scheduler queue → trigger actions.send_batch after delay

    def get_pending_followups(self) -> list[dict]:
        """All unresolved pending intents across all chats."""
        rows = self._db.execute(
            "SELECT chat_id, intent, created_at FROM pending_intents "
            "WHERE resolved = 0 ORDER BY created_at"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── LLM Helper ────────────────────────────────────────────────────────────

    async def _llm_call(
        self,
        system: str,
        user: str,
        max_tokens: int = 200,
        temperature: float = 0.3,
    ) -> dict | None:
        payload = {
            "model":           _MODEL,
            "temperature":     temperature,
            "max_tokens":      max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }
        raw = ""
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json=payload,
                )
                r.raise_for_status()
                raw = r.json()["choices"][0]["message"]["content"].strip()
                cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL)
                return json.loads(cleaned)
        except json.JSONDecodeError:
            log.error(f"Planner LLM non-JSON: {raw!r}")
        except Exception as exc:
            log.error(f"Planner LLM error: {exc}")
        return None

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _resolve_person_id_for_chat(self, chat_id: int) -> int | None:
        """Look up person_id from the most recent non-outgoing message in this chat."""
        row = self._db.execute(
            "SELECT sender_id FROM messages "
            "WHERE chat_id = ? AND is_outgoing = 0 "
            "ORDER BY logged_at DESC LIMIT 1",
            (chat_id,),
        ).fetchone()
        if not row:
            return None
        return self._memory.resolve_person_id(row["sender_id"])

    @staticmethod
    def _parse_ts(ts_str: str | None):
        if not ts_str:
            return None
        try:
            return datetime.fromisoformat(ts_str)
        except Exception:
            return None

    @staticmethod
    def _format_transcript(batches: list[dict]) -> str:
        lines = []
        for b in batches:
            direction = "You" if b["direction"] == "out" else "Them"
            for m in b["messages"]:
                content = m.get("content", "").strip()
                mtype   = m.get("media_type", "text")
                if content:
                    lines.append(f"{direction}: {content}")
                elif mtype not in ("text", "none"):
                    lines.append(f"{direction}: [{mtype}]")
        return "\n".join(lines) if lines else "(empty)"
