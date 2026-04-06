"""
Context Builder

Read-only assembly layer. No LLM calls, no writes.
Takes classifier output + batch, packages everything thinker needs
into one clean context_package dict.

Write interfaces (write_memory, update_relationship, set_pending_intent)
are defined here but executed by thinker — context builder never mutates state.

context_package schema:
{
    "chat_id":    int,
    "chat_name":  str,
    "person_id":  int,
    "batch":      dict,           # current flushed batch
    "classifier": dict,           # classifier result
    "history":    list[dict],     # last N batches both directions
    "person":     dict,           # profile + facts
    "memory": {
        "episodes":      list,
        "facts":         list,
        "last_summary":  str | None,
    },
    "relationship":  dict,        # tone, unresolved_threads, commitments
    "timing": {
        "gap_label":          str,    # fresh | resuming | cold_open
        "time_since_agent":   float | None,   # seconds
        "time_since_user":    float | None,
        "message_delays":     list[float],    # delay between msgs in batch
        "local_hour":         int | None,     # hour of day for person
    },
    "agent": {
        "last_output":    dict | None,
        "pending_intents": list,
        "state":          dict,       # mood, energy
    },
    "events": {
        "edits":     list,
        "deletions": list,
        "reactions": list,
    },
}
"""

import logging
import sqlite3
from datetime import datetime, timezone

from db.store import fetch_recent_batches
from memory.store import MemoryStore

log = logging.getLogger(__name__)

_GAP_FRESH    = 10 * 60          # < 10 min
_GAP_RESUMING = 24 * 60 * 60     # < 24 hrs


class ContextBuilder:
    def __init__(self, db_conn: sqlite3.Connection, memory: MemoryStore):
        self._db     = db_conn
        self._memory = memory

        # Event buffers — handlers write here, context builder reads + clears
        self._edits:     dict[int, list[dict]] = {}
        self._deletions: dict[int, list[dict]] = {}
        self._reactions: dict[int, list[dict]] = {}

    # ── Main Entry ─────────────────────────────────────────────────────────────

    def build_context(
        self,
        batch: dict,
        classifier_result: dict,
    ) -> dict:
        chat_id   = batch["chat_id"]
        chat_name = batch.get("chat_name", str(chat_id))

        # resolve person from the first incoming message's sender
        sender_id   = self._get_sender_id(batch)
        sender_name = self._get_sender_name(batch)
        person_id   = self._memory.resolve_person_id(sender_id, sender_name) if sender_id else 0

        history  = fetch_recent_batches(self._db, chat_id, n=20)
        timing   = self._get_timing_metadata(batch, history)
        person   = self._memory.fetch_person_profile(person_id)
        facts    = self._memory.read_person_facts(person_id)
        episodes = self._memory.read_recent_episodes(chat_id, n=5)
        summary  = self._memory.get_last_session_summary(chat_id) if timing["gap_label"] == "cold_open" else None

        return {
            "chat_id":   chat_id,
            "chat_name": chat_name,
            "person_id": person_id,
            "batch":     batch,
            "classifier": classifier_result,
            "history":   history,
            "person":    {**person, "facts": facts},
            "memory": {
                "episodes":     episodes,
                "facts":        facts,
                "last_summary": summary,
            },
            "relationship":  self._memory.get_relationship(person_id, chat_id),
            "timing":        timing,
            "agent": {
                "last_output":     self._get_agent_last_output(chat_id, history),
                "pending_intents": self._memory.get_pending_intents(chat_id),
                "state":           self._memory.get_agent_state(),
            },
            "events": {
                "edits":     self._edits.pop(chat_id, []),
                "deletions": self._deletions.pop(chat_id, []),
                "reactions": self._reactions.pop(chat_id, []),
            },
        }

    # ── Timing ─────────────────────────────────────────────────────────────────

    def _get_timing_metadata(self, batch: dict, history: list[dict]) -> dict:
        now = datetime.now(timezone.utc)

        # delays between messages within the batch
        timestamps = [self._parse_ts(m["ts"]) for m in batch["messages"] if m.get("ts")]
        delays = []
        for i in range(1, len(timestamps)):
            if timestamps[i] and timestamps[i-1]:
                delays.append((timestamps[i] - timestamps[i-1]).total_seconds())

        # time since last agent response
        time_since_agent = None
        time_since_user  = None
        for b in reversed(history):
            if b["direction"] == "out" and time_since_agent is None:
                ts = self._parse_ts(b["flushed_at"])
                if ts:
                    time_since_agent = (now - ts).total_seconds()
            if b["direction"] == "in" and time_since_user is None:
                ts = self._parse_ts(b["flushed_at"])
                if ts:
                    time_since_user = (now - ts).total_seconds()
            if time_since_agent is not None and time_since_user is not None:
                break

        # conversation gap from last agent response
        gap_label = self._detect_gap(time_since_agent)

        return {
            "gap_label":        gap_label,
            "time_since_agent": time_since_agent,
            "time_since_user":  time_since_user,
            "message_delays":   delays,
            "local_hour":       None,  # populated by Self when timezone is known
        }

    @staticmethod
    def _detect_gap(seconds: float | None) -> str:
        if seconds is None:
            return "cold_open"
        if seconds < _GAP_FRESH:
            return "fresh"
        if seconds < _GAP_RESUMING:
            return "resuming"
        return "cold_open"

    @staticmethod
    def _parse_ts(ts_str: str | None):
        if not ts_str:
            return None
        try:
            return datetime.fromisoformat(ts_str)
        except Exception:
            return None

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_sender_id(batch: dict) -> int | None:
        for m in batch["messages"]:
            if not m.get("is_outgoing") and m.get("sender_id"):
                return m["sender_id"]
        return None

    @staticmethod
    def _get_sender_name(batch: dict) -> str | None:
        for m in batch["messages"]:
            if not m.get("is_outgoing") and m.get("sender"):
                return m["sender"]
        return None

    @staticmethod
    def _get_agent_last_output(chat_id: int, history: list[dict]) -> dict | None:
        for b in reversed(history):
            if b["direction"] == "out":
                return b
        return None

    # ── Event Buffers (written by handlers, consumed here) ────────────────────

    def push_edit(self, chat_id: int, edit: dict) -> None:
        self._edits.setdefault(chat_id, []).append(edit)

    def push_deletion(self, chat_id: int, deletion: dict) -> None:
        self._deletions.setdefault(chat_id, []).append(deletion)

    def push_reaction(self, chat_id: int, reaction: dict) -> None:
        self._reactions.setdefault(chat_id, []).append(reaction)

    # ── Write Interfaces (executed by thinker, not context builder) ───────────

    def write_memory(self, person_id: int, chat_id: int, content: str, tags: list[str]) -> None:
        self._memory.write_episode(person_id, chat_id, content, tags)

    def update_relationship(self, person_id: int, chat_id: int, delta: dict) -> None:
        self._memory.update_relationship(person_id, chat_id, delta)

    def set_pending_intent(self, chat_id: int, intent: str) -> None:
        self._memory.set_pending_intent(chat_id, intent)
