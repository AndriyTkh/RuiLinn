"""
Thinker Brain

Main response generator. Called only when classifier says response_expected = true.
Receives the full context_package from ContextBuilder, produces a structured response.

Uses a larger Groq model. Called as infrequently as possible.

Thinker output schema:
{
    "messages":           list[str],   # text messages to send (in order)
    "reaction":           str | null,  # emoji reaction instead of / in addition to reply
    "memory_writes":      list[{"content": str, "tags": list[str]}],
    "pending_intent":     str | null,  # something to follow up on later
    "relationship_delta": dict | null  # tone / unresolved_threads / agent_commitments
}

Persona is provided by the Self module and injected at build time.
"""

import json
import logging
import re

_trace = logging.getLogger("agent.trace")

import httpx

from config import GROQ_API_KEY

log = logging.getLogger(__name__)

_MODEL   = "llama-3.3-70b-versatile"
_TIMEOUT = 20

_OUTPUT_INSTRUCTIONS = """\
Respond ONLY with a valid JSON object. No markdown, no explanation, nothing else.

Required schema:
{
  "messages":           ["..."],
  "reaction":           null,
  "reply_to_id":        null,
  "memory_writes":      [],
  "knowledge_writes":   [],
  "pending_intent":     null,
  "relationship_delta": null
}

Rules:
- messages: array of strings. Split into separate messages where a real person would send separate texts. Keep it natural — usually 1-2 messages, rarely more.
- reaction: emoji string if a reaction is appropriate (can be alongside messages), null otherwise.
- reply_to_id: integer message_id to thread your reply to a specific message, null to send without threading. Use only when replying to a specific message in a multi-message batch makes sense contextually.
- memory_writes: array of {"content": "...", "tags": ["..."]} for anything worth remembering about this person or conversation. Empty array if nothing notable.
- knowledge_writes: array of {"topic": "...", "content": "...", "confidence": 0.6, "opinion": "..."} if you learned something new about a topic or formed a new opinion. Empty array if nothing changed. Use your personal voice — not facts, but how you understand things.
- pending_intent: string if you committed to something or want to follow up, null otherwise.
- relationship_delta: {"tone": "..."} if the interaction shifted the relationship tone, null otherwise. Tone values: warm | friendly | neutral | distant | tense.\
"""


class Thinker:
    def __init__(self, context_builder, self_module=None):
        self._ctx  = context_builder  # for write-back interfaces
        self._self = self_module       # provides dynamic persona

    # ── Main Entry ─────────────────────────────────────────────────────────────

    async def think(self, context_package: dict) -> dict | None:
        """
        Generate a response. Returns thinker output dict or None on failure.
        Also applies all side effects (memory writes, relationship updates, etc.)
        """
        if not self._should_respond(context_package):
            return None

        prompt = self._build_prompt(context_package)
        raw    = await self._call_llm(prompt)
        if raw is None:
            return None

        result = self._parse_response(raw)
        if result is None:
            return None

        await self._apply_side_effects(result, context_package)
        return result

    # ── Gate ───────────────────────────────────────────────────────────────────

    def _should_respond(self, ctx: dict) -> bool:
        agent_state = ctx["agent"]["state"]
        # future: mood/energy checks from planner
        return True

    # ── Prompt Assembly ────────────────────────────────────────────────────────

    def _build_prompt(self, ctx: dict) -> list[dict]:
        persona = (
            self._self.get_persona(ctx.get("person_id"))
            if self._self
            else "You are Rui. You are a real person — not an AI, not an assistant."
        )
        system = f"{persona}\n\n{_OUTPUT_INSTRUCTIONS}"
        user   = self._assemble_context_text(ctx)
        _trace.info(
            f"[{ctx.get('chat_name', ctx['chat_id'])}] PROMPT\n"
            f"{'─'*40} SYSTEM {'─'*40}\n{system}\n"
            f"{'─'*40} USER {'─'*40}\n{user}\n"
            f"{'─'*88}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

    def _assemble_context_text(self, ctx: dict) -> str:
        parts = []

        # ── Relationship ──
        rel = ctx["relationship"]
        parts.append(f"[RELATIONSHIP]\nTone: {rel['tone']}")
        if rel["unresolved_threads"]:
            threads = "\n".join(f"  - {t}" for t in rel["unresolved_threads"])
            parts.append(f"Unresolved:\n{threads}")
        if rel["agent_commitments"]:
            commits = "\n".join(f"  - {c}" for c in rel["agent_commitments"])
            parts.append(f"You committed to:\n{commits}")

        # ── Memory ──
        mem = ctx["memory"]
        if mem["last_summary"]:
            parts.append(f"[LAST SESSION SUMMARY]\n{mem['last_summary']}")
        if mem["facts"]:
            facts_text = "\n".join(f"  - {f['content']}" for f in mem["facts"][:10])
            parts.append(f"[WHAT YOU KNOW ABOUT THEM]\n{facts_text}")
        if mem["episodes"]:
            eps_text = "\n".join(
                f"  [{e['created_at'][:10]}] {e['content']}" for e in mem["episodes"]
            )
            parts.append(f"[RECENT EPISODES]\n{eps_text}")

        # ── Knowledge ──
        kctx = ctx.get("knowledge", {})
        know_entries = kctx.get("knowledge", [])
        skill_entries = kctx.get("skills", [])
        if know_entries or skill_entries:
            lines = ["[WHAT YOU KNOW — relevant to this conversation]"]
            for k in know_entries:
                line = f"  - {k['topic']}: {k['content']}"
                if k.get("opinion"):
                    line += f" | opinion: {k['opinion']}"
                if k.get("gaps"):
                    line += f" | gaps: {', '.join(k['gaps'][:2])}"
                lines.append(line)
            if skill_entries:
                lines.append("  Skills:")
                for s in skill_entries:
                    lines.append(f"    - {s['name']} ({s['proficiency']:.0%}): {s.get('backstory', '')}")
            parts.append("\n".join(lines))

        # ── Pending ──
        pending = ctx["agent"]["pending_intents"]
        if pending:
            p_text = "\n".join(f"  - {p['intent']}" for p in pending)
            parts.append(f"[YOUR PENDING FOLLOW-UPS]\n{p_text}")

        # ── Timing ──
        parts.append(self._format_timing(ctx["timing"]))

        # ── Conversation History ──
        if ctx["history"]:
            parts.append(self._format_history(ctx["history"]))

        # ── Current Batch ──
        parts.append(self._format_batch(ctx["batch"]))

        # ── Classifier Flags ──
        flags = ctx["classifier"].get("flags", {})
        hints = []
        if flags.get("multi_question"):
            hints.append("They asked multiple questions — address all of them in one response.")
        if flags.get("topic_shift"):
            hints.append("Topic shifted — don't assume continuity from earlier conversation.")
        if hints:
            parts.append("[HINTS]\n" + "\n".join(hints))

        return "\n\n".join(parts)

    @staticmethod
    def _format_timing(timing: dict) -> str:
        gap = timing["gap_label"]
        lines = [f"[TIMING]\nGap: {gap}"]

        if gap == "cold_open":
            lines.append("Note: significant time has passed since you last spoke. Acknowledge it naturally if relevant — don't pretend no time passed.")
        elif gap == "resuming":
            lines.append("Note: you're picking up a conversation from earlier today.")

        since_agent = timing.get("time_since_agent")
        if since_agent is not None:
            mins = int(since_agent / 60)
            lines.append(f"Your last message was ~{mins}m ago.")

        delays = timing.get("message_delays", [])
        if delays and any(d < 1.5 for d in delays):
            lines.append("They sent these quickly — match that energy.")
        elif delays and any(d > 10 for d in delays):
            lines.append("There were pauses between their messages — they were thinking.")

        hour = timing.get("local_hour")
        if hour is not None:
            if hour < 6:
                lines.append("It's late night for them.")
            elif hour < 10:
                lines.append("It's morning for them.")

        return "\n".join(lines)

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        lines = ["[CONVERSATION HISTORY]"]
        for b in history[-12:]:  # last 12 batches max
            direction = "You" if b["direction"] == "out" else "Them"
            for m in b["messages"]:
                content = m.get("content", "").strip()
                mtype   = m.get("media_type", "text")
                if not content and mtype != "text":
                    lines.append(f"{direction}: [{mtype}]")
                elif content:
                    lines.append(f"{direction}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _format_batch(batch: dict) -> str:
        lines = ["[NEW MESSAGES]"]
        for m in batch["messages"]:
            content = m.get("content", "").strip()
            mtype   = m.get("media_type", "text")
            reply   = m.get("reply_to")
            if reply:
                lines.append(f"  (replying to: \"{reply.get('content', '')[:60]}\")")
            if not content and mtype != "text":
                lines.append(f"Them: [{mtype}]")
            elif content:
                lines.append(f"Them: {content}")
        return "\n".join(lines)

    # ── LLM Call ───────────────────────────────────────────────────────────────

    async def _call_llm(self, messages: list[dict]) -> str | None:
        payload = {
            "model":       _MODEL,
            "temperature": 0.85,
            "max_tokens":  512,
            "messages":    messages,
        }
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json=payload,
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            log.error(f"Thinker LLM error: {exc}")
            return None

    # ── Response Parsing ───────────────────────────────────────────────────────

    def _parse_response(self, raw: str) -> dict | None:
        # strip markdown code fences if model wraps output
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.DOTALL)
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            log.error(f"Thinker non-JSON: {raw!r}")
            return None

        # defaults for missing optional fields
        result.setdefault("messages", [])
        result.setdefault("reaction", None)
        result.setdefault("reply_to_id", None)
        result.setdefault("memory_writes", [])
        result.setdefault("knowledge_writes", [])
        result.setdefault("pending_intent", None)
        result.setdefault("relationship_delta", None)

        if not isinstance(result["messages"], list) or not result["messages"]:
            log.warning("Thinker returned empty messages list")
            return None

        return result

    # ── Side Effects ───────────────────────────────────────────────────────────

    async def _apply_side_effects(self, result: dict, ctx: dict) -> None:
        person_id = ctx["person_id"]
        chat_id   = ctx["chat_id"]

        for write in result.get("memory_writes", []):
            content = write.get("content", "").strip()
            tags    = write.get("tags", [])
            if content:
                self._ctx.write_memory(person_id, chat_id, content, tags)
                log.debug(f"Memory write: {content[:60]!r} tags={tags}")

        for kw in result.get("knowledge_writes", []):
            topic   = kw.get("topic", "").strip().lower()
            content = kw.get("content", "").strip()
            if topic and content:
                self._ctx.write_knowledge(
                    topic,
                    content,
                    source="conversation",
                    confidence=float(kw.get("confidence", 0.6)),
                    opinion=kw.get("opinion"),
                )
                log.debug(f"Knowledge write: {topic!r}")

        if result.get("pending_intent"):
            self._ctx.set_pending_intent(chat_id, result["pending_intent"])
            log.debug(f"Pending intent: {result['pending_intent']!r}")

        if result.get("relationship_delta"):
            self._ctx.update_relationship(person_id, chat_id, result["relationship_delta"])
            log.debug(f"Relationship delta: {result['relationship_delta']}")
