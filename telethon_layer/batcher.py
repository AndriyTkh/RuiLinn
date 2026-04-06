"""
Telethon Layer — Batcher

Pure-Python batching logic. No Telethon dependency — handlers.py
extracts everything from Telethon events and passes plain dicts here.

Batch object schema:
{
    "chat_id":   int,
    "chat_name": str,
    "reason":    "silence" | "typing_stopped" | "length",
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
import time
from collections import deque
from datetime import datetime, timezone
from typing import Awaitable, Callable

from config import BATCH_LENGTH_LIMIT, BATCH_TIMEOUT_BASE, BATCH_TIMEOUT_MAX, BATCH_TIMEOUT_MIN

log = logging.getLogger(__name__)


class Batcher:
    def __init__(self, on_batch: Callable[[dict], Awaitable[None]]):
        self._on_batch = on_batch

        self._buffers:        dict[int, list[dict]]       = {}
        self._chat_names:     dict[int, str]              = {}
        self._timers:         dict[int, asyncio.TimerHandle] = {}
        self._timer_start:    dict[int, float]            = {}
        self._timer_timeout:  dict[int, float]            = {}
        self._paused_remaining: dict[int, float]          = {}
        self._typing:         dict[int, bool]             = {}
        self._typing_speeds:  dict[int, deque]            = {}  # deque(maxlen=5) of secs/word
        self._msg_timestamps: dict[int, list[float]]      = {}  # last few monotonic timestamps

        self._loop = asyncio.get_event_loop()

    # ── Public API ─────────────────────────────────────────────────────────────

    def add(self, chat_id: int, chat_name: str, message: dict) -> None:
        """
        Add an incoming message dict to the buffer and reset the silence timer.
        message must include at least: sender, content, ts, message_id, media_type, is_forward
        """
        content = message.get("content", "")

        self._chat_names[chat_id] = chat_name
        if chat_id not in self._buffers:
            self._buffers[chat_id] = []

        now_mono = time.monotonic()
        prev_ts = self._msg_timestamps.get(chat_id, [None])[-1]
        self._update_typing_speed(chat_id, content, prev_ts, now_mono)

        if chat_id not in self._msg_timestamps:
            self._msg_timestamps[chat_id] = []
        self._msg_timestamps[chat_id].append(now_mono)
        if len(self._msg_timestamps[chat_id]) > 5:
            self._msg_timestamps[chat_id].pop(0)

        self._buffers[chat_id].append(message)
        log.debug(f"Buffer [{chat_name}] +1 ({len(self._buffers[chat_id])} msgs)")

        # force-flush on length limit
        if len(self._buffers[chat_id]) >= BATCH_LENGTH_LIMIT:
            if chat_id in self._timers:
                self._timers[chat_id].cancel()
            asyncio.ensure_future(self._flush(chat_id, "length"))
            return

        timeout = self._calculate_timeout(chat_id, content)
        self._reset_timer(chat_id, timeout)

    def on_typing_started(self, chat_id: int) -> None:
        if chat_id not in self._buffers:
            return
        if chat_id in self._timers:
            self._timers.pop(chat_id).cancel()
            elapsed  = time.monotonic() - self._timer_start.get(chat_id, time.monotonic())
            remaining = max(0.0, self._timer_timeout.get(chat_id, BATCH_TIMEOUT_BASE) - elapsed)
            self._paused_remaining[chat_id] = remaining
        self._typing[chat_id] = True
        log.debug(f"Typing started [{chat_id}] — timer paused")

    def on_typing_stopped(self, chat_id: int) -> None:
        self._typing[chat_id] = False
        if chat_id not in self._buffers:
            return
        remaining = self._paused_remaining.pop(chat_id, BATCH_TIMEOUT_BASE)
        adjusted  = max(1.0, remaining * 0.80)  # −20%
        self._reset_timer(chat_id, adjusted, reason="typing_stopped")
        log.debug(f"Typing stopped [{chat_id}] — timer resumed {adjusted:.1f}s")

    # ── Timer ──────────────────────────────────────────────────────────────────

    def _reset_timer(self, chat_id: int, timeout: float, reason: str = "silence") -> None:
        if chat_id in self._timers:
            self._timers[chat_id].cancel()

        self._timer_start[chat_id]   = time.monotonic()
        self._timer_timeout[chat_id] = timeout

        self._timers[chat_id] = self._loop.call_later(
            timeout,
            lambda cid=chat_id, r=reason: asyncio.ensure_future(self._flush(cid, r)),
        )

    async def _flush(self, chat_id: int, reason: str) -> None:
        messages = self._buffers.pop(chat_id, [])
        self._timers.pop(chat_id, None)
        self._timer_start.pop(chat_id, None)
        self._timer_timeout.pop(chat_id, None)
        self._paused_remaining.pop(chat_id, None)
        self._msg_timestamps.pop(chat_id, None)

        if not messages:
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

    def _calculate_timeout(self, chat_id: int, content: str) -> float:
        speeds = self._typing_speeds.get(chat_id)
        if speeds:
            avg = sum(speeds) / len(speeds)
            base = max(BATCH_TIMEOUT_MIN, avg * 4)
        else:
            base = BATCH_TIMEOUT_BASE

        delta = (
            self._modifier_punctuation(content)
            + self._modifier_message_length(content)
            + self._modifier_velocity(chat_id)
            + self._modifier_question(content)
            + self._modifier_consecutive(chat_id)
        )

        result = base * (1 + delta)
        return max(BATCH_TIMEOUT_MIN, min(BATCH_TIMEOUT_MAX, result))

    def _update_typing_speed(
        self, chat_id: int, content: str, prev_ts: float | None, now_ts: float
    ) -> None:
        words = len(content.split())
        if words < 1 or prev_ts is None:
            return
        elapsed = now_ts - prev_ts
        if elapsed <= 0:
            return
        secs_per_word = elapsed / words
        if chat_id not in self._typing_speeds:
            self._typing_speeds[chat_id] = deque(maxlen=5)
        self._typing_speeds[chat_id].append(secs_per_word)

    # ── Modifiers (return percentage delta, e.g. -0.4 = -40%) ────────────────

    def _modifier_punctuation(self, content: str) -> float:
        stripped = content.rstrip()
        if not stripped:
            return 0.0
        last = stripped[-1]
        if last in ".!?":
            if stripped.endswith("..."):
                return +0.30  # ellipsis → more coming
            return -0.40      # hard stop
        if last == ",":
            return +0.20      # comma → definitely more coming
        return 0.0

    def _modifier_message_length(self, content: str) -> float:
        words = len(content.split())
        if 1 <= words <= 3:
            return +0.40      # short setup phrase, more likely coming
        return 0.0

    def _modifier_velocity(self, chat_id: int) -> float:
        timestamps = self._msg_timestamps.get(chat_id, [])
        if len(timestamps) < 2:
            return 0.0
        recent = timestamps[-3:]
        gaps = [recent[i+1] - recent[i] for i in range(len(recent) - 1)]
        if all(g < 2.0 for g in gaps):
            return +0.25      # rapid-fire messages → still going
        return 0.0

    def _modifier_question(self, content: str) -> float:
        stripped = content.rstrip()
        words = len(content.split())
        if stripped.endswith("?") and words >= 5:
            return -0.30      # complete question → respond sooner
        return 0.0

    def _modifier_consecutive(self, chat_id: int) -> float:
        count = len(self._buffers.get(chat_id, []))
        if count >= 3:
            return -0.15      # already 3+ in buffer → probably wrapping up
        return 0.0
