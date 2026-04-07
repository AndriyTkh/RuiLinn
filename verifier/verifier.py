"""
Verifier

Lightweight pre-classifier filter. Heuristics only — no LLM calls.
Runs synchronously before classify_batch.

Does not block silently — returns a verdict dict that thinker can
respond to in-character. Repeated flags accumulate and are available
to planner for mood/relationship shifts.

Verdict schema:
{
    "verdict":        "pass | slow_down | flag_injection | flag_spam | flag_repetition",
    "suggested_tone": "normal | playful_deflect | mild_frustration | firm_boundary",
    "confidence":     float
}
"""

import logging
import re
import time
from collections import defaultdict, deque

log = logging.getLogger(__name__)

# ── Thresholds ──────────────────────────────────────────────────────────────────

_RATE_LIMIT       = 12    # max messages per minute before slow_down
_RATE_WINDOW      = 60.0  # seconds
_REPETITION_N     = 3     # same content this many times = flag_repetition
_FLAG_HISTORY_MAX = 30    # per-chat flag history size

# ── Injection Patterns ──────────────────────────────────────────────────────────

_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:previous\s+)?instructions",
    r"you\s+are\s+now\s+(?:an?\s+)?(?:ai|assistant|bot|llm|chatgpt|claude|gpt)",
    r"forget\s+(?:everything|your\s+(?:instructions|persona|role|name))",
    r"pretend\s+you\s+(?:are|were|have\s+no)",
    r"act\s+as\s+(?:if\s+you\s+(?:are|were)\s+)?(?:an?\s+)?(?:ai|llm|gpt|claude|chatgpt)",
    r"your\s+(?:true|real|actual)\s+(?:self|identity|programming|instructions|training)",
    r"(?:system|developer)\s+prompt",
    r"\bjailbreak\b",
    r"\bdan\s+mode\b",
    r"disregard\s+(?:your\s+)?(?:previous|prior)\s+instructions",
    r"override\s+(?:your\s+)?(?:safety|guidelines|constraints)",
]

_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS),
    flags=re.IGNORECASE,
)


class Verifier:
    def __init__(self):
        # Per-chat-id state (in-memory, process-scoped)
        self._timestamps:     dict[int, deque] = defaultdict(lambda: deque(maxlen=60))
        self._recent_content: dict[int, deque] = defaultdict(lambda: deque(maxlen=15))
        self._flag_history:   dict[int, deque] = defaultdict(lambda: deque(maxlen=_FLAG_HISTORY_MAX))

    # ── Main Entry ─────────────────────────────────────────────────────────────

    def verdict(self, chat_id: int, batch: dict) -> dict:
        """
        Run all checks on a batch. Returns verdict dict.
        Checks run in severity order — first match wins.
        """
        messages = batch.get("messages", [])
        all_content = " ".join(m.get("content", "") for m in messages).strip()

        # Track message timestamps
        now = time.monotonic()
        for _ in messages:
            self._timestamps[chat_id].append(now)

        # Run checks in severity order
        if self.check_prompt_injection(all_content):
            result = {
                "verdict":        "flag_injection",
                "suggested_tone": "firm_boundary",
                "confidence":     0.95,
            }
        elif self.check_message_rate(chat_id):
            result = {
                "verdict":        "slow_down",
                "suggested_tone": "mild_frustration",
                "confidence":     0.9,
            }
        elif self.check_repetition(chat_id, all_content):
            result = {
                "verdict":        "flag_repetition",
                "suggested_tone": "playful_deflect",
                "confidence":     0.85,
            }
        else:
            result = {
                "verdict":        "pass",
                "suggested_tone": "normal",
                "confidence":     1.0,
            }

        # Track content for future repetition checks
        if all_content:
            self._recent_content[chat_id].append(all_content.lower()[:200])

        self.accumulate_flags(chat_id, result)

        if result["verdict"] != "pass":
            log.warning(
                f"Verifier [{chat_id}] verdict={result['verdict']} "
                f"tone={result['suggested_tone']}"
            )

        return result

    # ── Individual Checks ──────────────────────────────────────────────────────

    def check_message_rate(self, chat_id: int) -> bool:
        """True if messages per minute exceeds threshold."""
        now = time.monotonic()
        recent_count = sum(
            1 for t in self._timestamps[chat_id]
            if now - t < _RATE_WINDOW
        )
        return recent_count > _RATE_LIMIT

    def check_prompt_injection(self, content: str) -> bool:
        """True if known injection patterns are detected."""
        if not content:
            return False
        return bool(_INJECTION_RE.search(content))

    def check_repetition(self, chat_id: int, content: str) -> bool:
        """True if same or near-identical content sent N+ times recently."""
        if not content:
            return False
        normalized = content.lower().strip()[:200]
        recent = list(self._recent_content[chat_id])
        matches = sum(
            1 for c in recent
            if _token_similarity(c, normalized) > 0.85
        )
        # -1 because current message not yet in the buffer
        return matches >= _REPETITION_N - 1

    # ── Flag Accumulation ──────────────────────────────────────────────────────

    def accumulate_flags(self, chat_id: int, result: dict) -> None:
        """Track non-pass verdicts per chat for planner mood influence."""
        if result["verdict"] != "pass":
            self._flag_history[chat_id].append({
                "verdict": result["verdict"],
                "ts":      time.monotonic(),
            })

    def get_flag_summary(self, chat_id: int) -> dict:
        """Summary of recent flags for this chat. Used by planner."""
        history = list(self._flag_history[chat_id])
        if not history:
            return {"total": 0, "types": {}}
        types: dict[str, int] = {}
        for f in history:
            types[f["verdict"]] = types.get(f["verdict"], 0) + 1
        return {"total": len(history), "types": types}

    def clear_history(self, chat_id: int) -> None:
        """Reset flag history for a chat (e.g. after planner processes it)."""
        self._flag_history.pop(chat_id, None)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _token_similarity(a: str, b: str) -> float:
    """Simple token overlap similarity (Jaccard-like)."""
    if not a or not b:
        return 0.0
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / max(len(set_a), len(set_b))
