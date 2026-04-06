"""
Actions — Outgoing Telethon output

Single public entry point: send_batch().
All timing, typing indicators, and delivery are handled here.
Thinker never calls Telethon directly.

Stub: thinker not built yet. Structure is in place for when it is.
"""

import asyncio
import logging
import random

from telethon.tl.functions.messages import SendReactionRequest
from telethon.tl.types import ReactionEmoji

log = logging.getLogger(__name__)

_client = None  # set via init()


def init(client) -> None:
    global _client
    _client = client


async def send_batch(
    chat_id: int,
    messages: list[str],
    reply_to_id: int | None = None,
) -> None:
    """
    Main entry point. Takes a list of text strings from thinker,
    handles all timing and delivery internally.
    """
    if not messages:
        return
    if _client is None:
        log.error("actions.send_batch called before init()")
        return

    output_type = _classify_output(messages)

    for i, text in enumerate(messages):
        if i > 0:
            await asyncio.sleep(_inter_message_delay(output_type))

        typing_delay = _typing_delay(text)
        await _do_typing(chat_id, typing_delay)

        await _do_send(chat_id, text, reply_to_id=reply_to_id if i == 0 else None)


# ── Output Classification ──────────────────────────────────────────────────────

def _classify_output(messages: list[str]) -> str:
    """
    rapid_burst    — short, reactive messages with minimal delay
    separate_thoughts — distinct points, full typing per message
    """
    avg_words = sum(len(m.split()) for m in messages) / len(messages)
    if avg_words <= 5:
        return "rapid_burst"
    return "separate_thoughts"


# ── Timing ─────────────────────────────────────────────────────────────────────

def _typing_delay(text: str) -> float:
    chars = len(text)
    base  = chars * 0.045  # ~45ms per char
    variance = random.uniform(-0.3, 0.3)
    return max(1.0, min(8.0, base + variance))


def _inter_message_delay(output_type: str) -> float:
    if output_type == "rapid_burst":
        return random.uniform(0.3, 0.8)
    return random.uniform(2.0, 4.0)


# ── Delivery ───────────────────────────────────────────────────────────────────

async def _do_typing(chat_id: int, duration: float) -> None:
    try:
        async with _client.action(chat_id, "typing"):
            await asyncio.sleep(duration)
    except Exception as exc:
        log.warning(f"Typing indicator error [{chat_id}]: {exc}")


async def _do_send(chat_id: int, text: str, reply_to_id: int | None = None) -> None:
    try:
        await _client.send_message(chat_id, text, reply_to=reply_to_id)
        log.info(f"→ SENT [{chat_id}]: {text[:80]!r}")
    except Exception as exc:
        log.error(f"send_message error [{chat_id}]: {exc}")


async def do_react(chat_id: int, message_id: int, emoji: str) -> None:
    try:
        await _client(SendReactionRequest(
            peer=chat_id,
            msg_id=message_id,
            reaction=[ReactionEmoji(emoticon=emoji)],
        ))
        log.info(f"→ REACT [{chat_id}] id={message_id}: {emoji}")
    except Exception as exc:
        log.error(f"do_react error [{chat_id}]: {exc}")


async def do_edit(chat_id: int, message_id: int, text: str) -> None:
    try:
        await _client.edit_message(chat_id, message_id, text)
        log.info(f"→ EDIT [{chat_id}] id={message_id}: {text[:80]!r}")
    except Exception as exc:
        log.error(f"do_edit error [{chat_id}]: {exc}")
