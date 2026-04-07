"""
Controlling Unit

Guardian/observer layer. Sits outside the data flow — observes, never blocks.
Notifies the operator when thresholds are crossed. The operator decides whether
to intervene.

Responsibilities:
  1. Prompt integrity  — review Self-proposed prompt edits before applying
  2. Behavioral monitoring — sample conversations for character breaks, pattern drift
  3. Operator alerting — log prominently + optional Telegram message

All checks are passive. Nothing in this module can prevent a message from sending.
"""

import asyncio
import json
import logging
import random
import re
import sqlite3
from datetime import datetime, timezone

import httpx

import actions.actions as actions
from config import GROQ_API_KEY, OPERATOR_CHAT_ID
from db.store import fetch_recent_batches

log = logging.getLogger(__name__)

_MODEL   = "llama-3.1-8b-instant"
_TIMEOUT = 15

# ── Thresholds (defaults, adjustable via set_threshold) ───────────────────────

_DEFAULT_THRESHOLDS = {
    "prompt_delta_pct":       30,    # % change in prompt length that warrants review
    "conversation_sample_hrs": 6,    # how often to sample a conversation (hours)
    "drift_window_days":       7,    # days of history to check for pattern drift
    "min_batches_to_sample":   5,    # don't sample chats with fewer batches than this
}

# ── Character Break Indicators ─────────────────────────────────────────────────
# Phrases that suggest the agent broke persona and sounded like an AI assistant

_ASSISTANT_PHRASES = re.compile(
    r"\b("
    r"as an ai|i am an ai|i'?m an ai"
    r"|certainly!|of course!|absolutely!"
    r"|i'd be happy to|i would be happy to"
    r"|i cannot|i'm unable to|i am unable to"
    r"|i don't have (?:the ability|access|feelings|emotions)"
    r"|please note that|it's important to note"
    r"|i should mention"
    r")\b",
    re.IGNORECASE,
)


class ControllingUnit:
    def __init__(self, db_conn: sqlite3.Connection):
        self._db         = db_conn
        self._thresholds = dict(_DEFAULT_THRESHOLDS)
        self._last_sample: dict[int, float] = {}  # chat_id → monotonic timestamp

    # ── Background Sampling Cycle ──────────────────────────────────────────────

    async def run_sampling_cycle(self) -> None:
        """
        Background task. Periodically samples a random recent conversation
        and checks for character breaks / pattern drift.
        """
        import time
        while True:
            try:
                interval_hrs = self._thresholds["conversation_sample_hrs"]
                await asyncio.sleep(interval_hrs * 3600)
                await self._run_sample()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error(f"ControllingUnit sampling error: {exc}", exc_info=True)

    async def _run_sample(self) -> None:
        """Pick a random recent chat and run checks on it."""
        # Pull distinct chat IDs from recent batches
        rows = self._db.execute(
            "SELECT DISTINCT chat_id FROM batches ORDER BY flushed_at DESC LIMIT 20"
        ).fetchall()
        if not rows:
            return

        chat_ids = [r["chat_id"] for r in rows]
        chat_id  = random.choice(chat_ids)

        batches = fetch_recent_batches(self._db, chat_id, n=30)
        min_batches = self._thresholds["min_batches_to_sample"]
        if len(batches) < min_batches:
            return

        log.debug(f"ControllingUnit sampling chat [{chat_id}] ({len(batches)} batches)")

        transcript = self._format_transcript(batches)

        # Quick local check first — fast
        local_break = self.detect_character_break_local(transcript)
        if local_break:
            await self.flag_to_operator(
                reason="character_break_local",
                evidence=f"Phrase detected in chat [{chat_id}]: {local_break}",
                chat_id=chat_id,
                severity="warning",
            )
            return  # don't call LLM if already flagged

        # Statistical drift check — no LLM
        drift = self.detect_pattern_drift(chat_id, batches)
        if drift:
            await self.flag_to_operator(
                reason="pattern_drift",
                evidence=f"Chat [{chat_id}]: {drift}",
                chat_id=chat_id,
                severity="info",
            )

        # LLM character break check — deeper but slower; only if no local flags
        llm_result = await self._llm_check_character(transcript)
        if llm_result and llm_result.get("break_detected"):
            evidence = llm_result.get("evidence", "no detail")
            await self.flag_to_operator(
                reason="character_break_llm",
                evidence=f"Chat [{chat_id}]: {evidence}",
                chat_id=chat_id,
                severity="warning",
            )

    # ── Character Break Detection ──────────────────────────────────────────────

    def detect_character_break_local(self, transcript: str) -> str | None:
        """
        Fast regex check for obvious assistant-mode phrases.
        Returns the matching phrase, or None if clean.
        """
        m = _ASSISTANT_PHRASES.search(transcript)
        return m.group(0) if m else None

    async def _llm_check_character(self, transcript: str) -> dict | None:
        """
        LLM evaluation of a conversation transcript for character breaks.
        Returns {"break_detected": bool, "evidence": str} or None on failure.
        """
        system = """\
You are reviewing a conversation transcript to check if the AI agent (shown as "You:") \
stayed in character as a real human named Rui, or broke persona.

Respond ONLY with JSON:
{"break_detected": false, "evidence": ""}

break_detected: true if the agent:
  - sounded like an AI assistant (said "as an AI", "I'd be happy to help", etc.)
  - over-explained in a tutorial/helpful-assistant way
  - added unnecessary caveats, disclaimers, or safety notes
  - responded uniformly regardless of emotional tone
  - used filler affirmations like "Certainly!", "Absolutely!", "Of course!"
evidence: brief description of what was wrong (empty string if clean).\
"""
        return await self._llm_call(system, transcript[-3000:], max_tokens=120)

    # ── Pattern Drift Detection ────────────────────────────────────────────────

    def detect_pattern_drift(self, chat_id: int, batches: list[dict]) -> str | None:
        """
        Heuristic checks for behavioral patterns that suggest drift.
        Returns a description string if drift detected, None if clean.
        """
        out_batches = [b for b in batches if b["direction"] == "out"]
        if len(out_batches) < 5:
            return None

        # Check 1: message length trend (are responses getting longer over time?)
        lengths = [
            sum(len(m.get("content", "")) for m in b["messages"])
            for b in out_batches[-10:]
        ]
        if len(lengths) >= 6:
            first_half = sum(lengths[:len(lengths)//2]) / (len(lengths)//2)
            second_half = sum(lengths[len(lengths)//2:]) / (len(lengths) - len(lengths)//2)
            if second_half > first_half * 1.8:
                return f"responses getting significantly longer (avg {first_half:.0f}→{second_half:.0f} chars)"

        # Check 2: response rate (agent responding to everything)
        in_batches  = [b for b in batches if b["direction"] == "in"]
        if in_batches and out_batches:
            response_rate = len(out_batches) / max(len(in_batches), 1)
            if response_rate > 0.95 and len(in_batches) > 10:
                return f"responding to {response_rate:.0%} of messages (too available)"

        return None

    # ── Prompt Integrity ───────────────────────────────────────────────────────

    async def review_prompt_edit(
        self,
        current:   str,
        proposed:  str,
        chat_id:   int | None = None,
        reason:    str = "",
    ) -> dict:
        """
        Evaluate a proposed prompt change from Self.
        Returns {"decision": "apply | flag | reject", "reason": str}.
        Persists the record regardless of decision.
        """
        # Calculate rough delta
        len_current  = len(current) if current else 0
        len_proposed = len(proposed)
        delta_pct    = abs(len_proposed - len_current) / max(len_current, 1) * 100

        decision = "apply"
        flag_reason = ""

        if delta_pct > self._thresholds["prompt_delta_pct"]:
            decision    = "flag"
            flag_reason = f"large prompt delta ({delta_pct:.0f}% change)"

        # LLM integrity check
        llm = await self._llm_review_prompt(current, proposed)
        if llm:
            if llm.get("contradicts_identity"):
                decision    = "flag"
                flag_reason = f"contradicts core identity: {llm.get('note', '')}"
            if llm.get("ethical_concern"):
                decision    = "reject"
                flag_reason = f"ethical concern: {llm.get('note', '')}"

        # Persist
        now = datetime.now(timezone.utc).isoformat()
        self._db.execute(
            "INSERT INTO prompt_edits (chat_id, current, proposed, decision, reason, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (chat_id, current, proposed, decision, flag_reason or reason, now),
        )
        self._db.commit()

        if decision in ("flag", "reject"):
            await self.flag_to_operator(
                reason=f"prompt_edit_{decision}",
                evidence=flag_reason,
                chat_id=chat_id,
                severity="warning" if decision == "flag" else "critical",
            )

        log.info(f"ControllingUnit prompt review: {decision} — {flag_reason or 'clean'}")
        return {"decision": decision, "reason": flag_reason}

    async def _llm_review_prompt(self, current: str, proposed: str) -> dict | None:
        system = """\
You are reviewing a proposed edit to an AI agent's system prompt.
The agent is Rui — a human-presenting conversational agent.

Respond ONLY with JSON:
{"contradicts_identity": false, "ethical_concern": false, "note": ""}

contradicts_identity: true if the proposed prompt would make the agent act contrary to its \
established identity (e.g. admits to being an AI, removes empathy, becomes obsequious)
ethical_concern: true if the proposed prompt would cause harm, manipulation, or deception \
of the user
note: brief explanation (empty if neither flag)\
"""
        payload = f"CURRENT:\n{current or '(empty)'}\n\nPROPOSED:\n{proposed}"
        return await self._llm_call(system, payload, max_tokens=120)

    def get_prompt_history(self, chat_id: int | None = None) -> list[dict]:
        """Return all prompt edits, optionally filtered by chat."""
        if chat_id is not None:
            rows = self._db.execute(
                "SELECT * FROM prompt_edits WHERE chat_id = ? ORDER BY created_at",
                (chat_id,),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM prompt_edits ORDER BY created_at"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Operator Alerting ──────────────────────────────────────────────────────

    async def flag_to_operator(
        self,
        reason:   str,
        evidence: str = "",
        chat_id:  int | None = None,
        severity: str = "warning",
    ) -> None:
        """
        Persist flag and notify operator (log + optional Telegram message).
        severity: info | warning | critical
        """
        now = datetime.now(timezone.utc).isoformat()

        # Persist
        self._db.execute(
            "INSERT INTO operator_flags (reason, evidence, chat_id, severity, sent, created_at) "
            "VALUES (?, ?, ?, ?, 0, ?)",
            (reason, evidence, chat_id, severity, now),
        )
        self._db.commit()

        # Always log prominently
        level = logging.CRITICAL if severity == "critical" else logging.WARNING
        log.log(level, f"[CONTROLLING UNIT] {severity.upper()} — {reason}: {evidence}")

        # Send Telegram message to operator if configured
        if OPERATOR_CHAT_ID:
            chat_tag = f" [chat {chat_id}]" if chat_id else ""
            text = f"[{severity.upper()}]{chat_tag} {reason}\n{evidence}"
            try:
                await actions.send_batch(OPERATOR_CHAT_ID, [text])
                self._db.execute(
                    "UPDATE operator_flags SET sent = 1 WHERE created_at = ?", (now,)
                )
                self._db.commit()
            except Exception as exc:
                log.error(f"ControllingUnit failed to send alert to operator: {exc}")

    def get_pending_flags(self, sent: bool = False) -> list[dict]:
        """Return unsent (or all) operator flags."""
        rows = self._db.execute(
            "SELECT * FROM operator_flags WHERE sent = ? ORDER BY created_at DESC LIMIT 50",
            (int(sent),),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Thresholds ─────────────────────────────────────────────────────────────

    def set_threshold(self, name: str, value) -> None:
        if name in self._thresholds:
            self._thresholds[name] = value
            log.info(f"ControllingUnit threshold {name!r} = {value}")
        else:
            log.warning(f"ControllingUnit unknown threshold {name!r}")

    # ── LLM Helper ────────────────────────────────────────────────────────────

    async def _llm_call(self, system: str, user: str, max_tokens: int = 150) -> dict | None:
        payload = {
            "model":           _MODEL,
            "temperature":     0.0,
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
            log.error(f"ControllingUnit LLM non-JSON: {raw!r}")
        except Exception as exc:
            log.error(f"ControllingUnit LLM error: {exc}")
        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _format_transcript(batches: list[dict]) -> str:
        lines = []
        for b in batches[-20:]:
            direction = "You" if b["direction"] == "out" else "Them"
            for m in b["messages"]:
                content = m.get("content", "").strip()
                if content:
                    lines.append(f"{direction}: {content}")
        return "\n".join(lines) if lines else "(empty)"
