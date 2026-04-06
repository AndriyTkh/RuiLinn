"""
Telethon Layer — Batcher

Pure-Python batching logic. No Telethon dependency — handlers.py
extracts everything from Telethon events and passes plain dicts here.

Batch object schema:
{
    "chat_id":   int,
    "chat_name": str,
    "reason":    "silence" | "typing" | "length",
    "flushed_at": ISO str,
    "messages": [
        {
            "sender_id":  int,
            "sender":     str,
            "content":    str,
            "ts":         ISO str,
            "message_id": int,
            "media_type": str,   # text|photo|sticker|voice|video|file|forward|none
            "is_forward": bool,
            "reply_to":   dict | None,
        },
        ...
    ]
}
"""

import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import Awaitable, Callable

from config import BATCH_LENGTH_LIMIT, BATCH_TIMEOUT_BASE, BATCH_TIMEOUT_MAX, BATCH_TIMEOUT_MIN

log = logging.getLogger(__name__)

_TYPING_RESET_MIN = 3.0  # timer reset range when typing action received
_TYPING_RESET_MAX = 5.0


class Batcher:
    def __init__(self, on_batch: Callable[[dict], Awaitable[None]]):
        self._on_batch = on_batch

        self._buffers:    dict[int, list[dict]]   = {}
        self._chat_names: dict[int, str]          = {}
        self._timers:     dict[int, asyncio.Task] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def add(self, chat_id: int, chat_name: str, message: dict) -> None:
        self._chat_names[chat_id] = chat_name
        self._buffers.setdefault(chat_id, []).append(message)
        n = len(self._buffers[chat_id])
        log.info(f"── BATCHER ── [{chat_name}] +1 msg (buf={n})")

        if n >= BATCH_LENGTH_LIMIT:
            self._cancel_timer(chat_id)
            asyncio.create_task(self._flush(chat_id, "length"))
            return

        content = message.get("content", "")
        timeout = self._calculate_timeout(content)
        self._reset_timer(chat_id, timeout)

    def on_typing(self, chat_id: int) -> None:
        """Called on every SendMessageTypingAction. Resets timer to 4-6s."""
        if chat_id not in self._buffers:
            return
        timeout = random.uniform(_TYPING_RESET_MIN, _TYPING_RESET_MAX)
        self._reset_timer(chat_id, timeout, reason="typing")

    # ── Timer ──────────────────────────────────────────────────────────────────

    def _reset_timer(self, chat_id: int, timeout: float, reason: str = "silence") -> None:
        self._cancel_timer(chat_id)
        task = asyncio.create_task(self._timer_task(chat_id, timeout, reason))
        self._timers[chat_id] = task
        log.info(f"── BATCHER ── [{chat_id}] timer set {timeout:.1f}s reason={reason}")

    def _cancel_timer(self, chat_id: int) -> None:
        task = self._timers.pop(chat_id, None)
        if task and not task.done():
            task.cancel()

    async def _timer_task(self, chat_id: int, timeout: float, reason: str) -> None:
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            log.info(f"── BATCHER ── [{chat_id}] timer cancelled")
            return
        log.info(f"── BATCHER ── [{chat_id}] timer fired → flush reason={reason}")
        try:
            await self._flush(chat_id, reason)
        except Exception as exc:
            log.error(f"── BATCHER ── flush error [{chat_id}]: {exc}", exc_info=True)

    async def _flush(self, chat_id: int, reason: str) -> None:
        messages = self._buffers.pop(chat_id, [])
        self._timers.pop(chat_id, None)

        if not messages:
            log.warning(f"── BATCHER ── flush called but buffer empty [{chat_id}]")
            return

        chat_name = self._chat_names.get(chat_id, str(chat_id))
        batch = {
            "chat_id":    chat_id,
            "chat_name":  chat_name,
            "reason":     reason,
            "flushed_at": datetime.now(timezone.utc).isoformat(),
            "messages":   messages,
        }

        preview = " / ".join(m["content"][:60] for m in messages if m.get("content"))
        log.info(
            f"── BATCH [{chat_name}] ({len(messages)} msg, reason={reason}) ──\n"
            f"  {preview}"
        )

        await self._on_batch(batch)

    # ── Timeout Calculation ────────────────────────────────────────────────────

    def _calculate_timeout(self, content: str) -> float:
        base  = BATCH_TIMEOUT_BASE
        delta = self._modifier_punctuation(content) + self._modifier_question(content)
        result = base * (1 + delta)
        return max(BATCH_TIMEOUT_MIN, min(BATCH_TIMEOUT_MAX, result))

    # ── Modifiers (return percentage delta, e.g. -0.4 = -40%) ────────────────

    @staticmethod
    def _modifier_punctuation(content: str) -> float:
        stripped = content.rstrip()
        if not stripped:
            return 0.0
        last = stripped[-1]
        if last in ".!?":
            if stripped.endswith("..."):
                return +0.30   # ellipsis → more coming
            return -0.40       # hard stop
        if last == ",":
            return +0.20       # comma → definitely more coming
        return 0.0

    @staticmethod
    def _modifier_question(content: str) -> float:
        stripped = content.rstrip()
        words = len(content.split())
        if stripped.endswith("?") and words >= 5:
            return -0.30       # complete question → respond sooner
        return 0.0
