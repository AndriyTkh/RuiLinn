"""
Classifier

Small, fast LLM (Groq) triage layer.
Decides whether thinker gets called at all and what kind of response is needed.

Output schema:
{
    "user_finished":      bool,   // has the person finished their thought?
    "response_expected":  bool,   // does this batch expect any response?
    "response_type":      str,    // "reply" | "react" | "silence"
    "confidence":         float,  // 0.0–1.0
    "flags": {
        "multi_question": bool,   // batch contains multiple questions → answer in one go
        "topic_shift":    bool,   // subject changed → context builder should pull fresh memory
        "media_only":     bool,   // sticker/photo with no text → force react, skip thinker
        "incomplete":     bool    // mid-thought, more likely coming
    }
}
"""

import json
import logging
import re

import httpx

from config import GROQ_API_KEY

log = logging.getLogger(__name__)

_MODEL   = "llama-3.1-8b-instant"
_TIMEOUT = 10  # seconds

_SYSTEM_PROMPT = """\
You are a triage classifier for a conversational AI. You receive a batch of messages \
from a single chat and decide what kind of response (if any) is needed.

Respond ONLY with a valid JSON object — no explanation, no markdown, no extra text.

Schema (all fields required):
{
  "user_finished":     true,
  "response_expected": true,
  "response_type":     "reply",
  "confidence":        0.85,
  "flags": {
    "multi_question": false,
    "topic_shift":    false,
    "media_only":     false,
    "incomplete":     false
  }
}

Field rules:
- user_finished: false if the batch reads like a mid-thought (no punctuation, very short, ellipsis)
- response_expected: false only if the person is clearly talking to themselves, forwarding without comment, or sending noise
- response_type:
    "reply"   → written response warranted
    "react"   → emoji/sticker reaction is enough (e.g. sticker, meme, one-word acknowledgement)
    "silence" → no response needed
- flags.media_only: true if batch is sticker/photo/video with zero text
- flags.multi_question: true if two or more distinct questions are asked
- flags.topic_shift: true if subject clearly changed from previous context
- flags.incomplete: true if sentence/thought appears cut off
"""


def _build_user_content(batch: dict) -> str:
    lines = []
    for m in batch["messages"]:
        sender  = m.get("sender", "?")
        content = m.get("content", "").strip()
        mtype   = m.get("media_type", "text")
        if mtype != "text" and not content:
            lines.append(f"{sender}: [{mtype}]")
        elif mtype != "text":
            lines.append(f"{sender}: [{mtype}] {content}")
        else:
            lines.append(f"{sender}: {content}")
    return "\n".join(lines)


async def classify_batch(batch: dict, verifier_flags: dict | None = None) -> dict | None:
    """
    Classify a flushed batch. Returns parsed dict or None on failure.
    Caller should check result["response_expected"] before forwarding to thinker.
    verifier_flags: optional output from Verifier.verdict() to inject as context.
    """
    # short-circuit: media_only if all messages are non-text with no content
    all_media = all(
        m.get("media_type") not in ("text", "none") and not m.get("content", "").strip()
        for m in batch["messages"]
    )
    if all_media:
        result = {
            "user_finished":     True,
            "response_expected": True,
            "response_type":     "react",
            "confidence":        1.0,
            "flags": {
                "verifier":       (verifier_flags or {}).get("verdict", "pass"),
                "multi_question": False,
                "topic_shift":    False,
                "media_only":     True,
                "incomplete":     False,
            },
        }
        return result

    user_content = _build_user_content(batch)

    # Prepend verifier context to user content when a flag was raised
    if verifier_flags and verifier_flags.get("verdict") != "pass":
        verdict      = verifier_flags["verdict"]
        tone         = verifier_flags.get("suggested_tone", "normal")
        user_content = (
            f"[VERIFIER FLAG: {verdict} — suggested tone: {tone}]\n\n"
            + user_content
        )
    payload = {
        "model":           _MODEL,
        "temperature":     0.0,
        "max_tokens":      180,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
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
            # strip markdown fences if model wraps output despite json_object mode
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL)
            result = json.loads(cleaned)
            return _fill_defaults(result, verifier_flags)

    except json.JSONDecodeError:
        log.error(f"Classifier non-JSON: {raw!r}")
    except Exception as exc:
        log.error(f"Classifier error: {exc}")

    return None


def _fill_defaults(result: dict, verifier_flags: dict | None = None) -> dict:
    """Fill missing fields with safe defaults instead of hard-failing."""
    result.setdefault("user_finished", True)
    result.setdefault("response_expected", True)
    result.setdefault("confidence", 0.5)

    # coerce response_type to a known value
    if result.get("response_type") not in ("reply", "react", "silence"):
        log.warning(f"Unknown response_type {result.get('response_type')!r}, defaulting to 'reply'")
        result["response_type"] = "reply"

    flags = result.setdefault("flags", {})
    flags["verifier"] = (verifier_flags or {}).get("verdict", "pass")
    flags.setdefault("multi_question", False)
    flags.setdefault("topic_shift", False)
    flags.setdefault("media_only", False)
    flags.setdefault("incomplete", False)

    return result
