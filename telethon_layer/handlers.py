"""
Telethon Layer — Event Handlers

Owns the Telethon-specific side: registers all event handlers,
extracts data from raw events, and passes clean dicts downstream.
"""

import logging
from datetime import datetime, timezone

from telethon import events
from telethon.errors import FloodWaitError
from telethon.tl.types import (
    MessageMediaDocument,
    MessageMediaPhoto,
    SendMessageTypingAction,
)

import asyncio
from db.store import log_message
from telethon_layer.batcher import Batcher

log = logging.getLogger(__name__)


class TelethonHandlers:
    def __init__(self, client, batcher: Batcher, db_conn, me):
        self._client  = client
        self._batcher = batcher
        self._db      = db_conn
        self._me      = me  # the logged-in account

    def register(self) -> None:
        self._client.on(events.NewMessage)(self._on_message)
        self._client.on(events.MessageEdited)(self._on_message_edited)
        self._client.on(events.MessageDeleted)(self._on_message_deleted)
        self._client.on(events.UserUpdate)(self._on_user_update)
        self._client.on(events.MessageRead)(self._on_read_receipt)
        log.info("Event handlers registered.")

    # ── Helpers ────────────────────────────────────────────────────────────────

    async def should_process(self, event) -> bool:
        # Use Telethon's built-in flags — more reliable than isinstance checks
        # on partially-resolved entities
        if getattr(event, "is_channel", False):
            return False
        if getattr(event, "is_private", False):
            return True
        # group: only if mentioned or the reply targets our message
        if event.message.mentioned:
            return True
        if event.message.reply_to:
            try:
                reply_msg = await event.message.get_reply_message()
                if reply_msg and reply_msg.sender_id == self._me.id:
                    return True
            except Exception:
                pass
        return False

    @staticmethod
    def classify_media_type(message) -> str:
        if message.sticker:
            return "sticker"
        if message.voice:
            return "voice"
        if message.video or message.video_note:
            return "video"
        if isinstance(message.media, MessageMediaPhoto):
            return "photo"
        if isinstance(message.media, MessageMediaDocument):
            return "file"
        if message.forward:
            return "forward"
        if message.text:
            return "text"
        return "none"

    @staticmethod
    async def resolve_reply_context(message) -> dict | None:
        if not message.reply_to:
            return None
        try:
            replied = await message.get_reply_message()
            if not replied:
                return None
            return {
                "message_id": replied.id,
                "sender_id":  replied.sender_id,
                "content":    replied.text or "",
                "media_type": TelethonHandlers.classify_media_type(replied),
            }
        except Exception:
            return None

    @staticmethod
    def _extract_name(entity) -> str | None:
        return (
            getattr(entity, "first_name", None)
            or getattr(entity, "title", None)
            or getattr(entity, "username", None)
        )

    # ── NewMessage ─────────────────────────────────────────────────────────────

    async def _on_message(self, event) -> None:
        try:
            await event.message.get_sender()
            await event.message.get_chat()
        except Exception as exc:
            log.warning(f"Failed to fetch sender/chat: {exc}")
            return

        try:
            msg        = event.message
            chat_id    = event.chat_id
            chat_name  = (
                self._extract_name(event.chat)
                or self._extract_name(msg.sender)
                or str(chat_id)
            )
            sender_id  = msg.sender_id
            sender     = self._extract_name(msg.sender) or str(sender_id)
            content    = msg.text or ""
            media_type = self.classify_media_type(msg)
            is_outgoing = bool(msg.out)
            is_forward  = bool(msg.forward)

            msg_dict = {
                "chat_id":    chat_id,
                "chat_name":  chat_name,
                "sender_id":  sender_id,
                "sender":     sender,
                "content":    content,
                "ts":         datetime.now(timezone.utc).isoformat(),
                "message_id": msg.id,
                "media_type": media_type,
                "is_outgoing": is_outgoing,
                "is_forward": is_forward,
                "reply_to":   None,  # resolved below for incoming only
            }

            log_message(self._db, msg_dict)

            direction = "→ OUT" if is_outgoing else "← IN "
            log.info(f"{direction} [{chat_name}] {sender}: {content[:80]!r} [{media_type}]")

            if is_outgoing:
                return

            if not await self.should_process(event):
                return

            # skip pure forwards — flag them but don't batch
            if is_forward:
                log.debug(f"Forward suppressed [{chat_name}]")
                return

            # resolve reply context (async, best-effort)
            msg_dict["reply_to"] = await self.resolve_reply_context(msg)

            self._batcher.add(chat_id, chat_name, msg_dict)

        except FloodWaitError as exc:
            log.warning(f"FloodWait: sleeping {exc.seconds}s")
            await asyncio.sleep(exc.seconds)
        except Exception as exc:
            log.error(f"on_message error: {exc}", exc_info=True)

    # ── MessageEdited ──────────────────────────────────────────────────────────

    async def _on_message_edited(self, event) -> None:
        try:
            msg       = event.message
            chat_id   = event.chat_id
            content   = msg.text or ""
            chat_name = self._extract_name(event.chat) or str(chat_id)

            # if still in buffer, update content in-place
            buf = self._batcher._buffers.get(chat_id, [])
            for m in buf:
                if m.get("message_id") == msg.id:
                    m["content"] = content
                    log.info(f"── EDIT (in buffer) [{chat_name}] id={msg.id}: {content[:80]!r}")
                    return

            # already flushed — emit edit signal upstream (placeholder for thinker)
            log.info(f"── EDIT (post-flush) [{chat_name}] id={msg.id}: {content[:80]!r}")

        except Exception as exc:
            log.error(f"on_message_edited error: {exc}", exc_info=True)

    # ── MessageDeleted ─────────────────────────────────────────────────────────

    async def _on_message_deleted(self, event) -> None:
        try:
            chat_id = event.chat_id
            deleted_ids = set(event.deleted_ids)

            buf = self._batcher._buffers.get(chat_id, [])
            removed = [m for m in buf if m.get("message_id") in deleted_ids]
            if removed:
                self._batcher._buffers[chat_id] = [
                    m for m in buf if m.get("message_id") not in deleted_ids
                ]
                log.info(f"── DELETE (from buffer) [{chat_id}] ids={deleted_ids}")
            else:
                log.info(f"── DELETE (post-flush) [{chat_id}] ids={deleted_ids}")

        except Exception as exc:
            log.error(f"on_message_deleted error: {exc}", exc_info=True)

    # ── UserUpdate (typing + presence) ────────────────────────────────────────

    async def _on_user_update(self, event) -> None:
        try:
            chat_id = event.chat_id or (event.user_id if hasattr(event, "user_id") else None)
            if chat_id is None:
                return

            action = getattr(event, "action", None)
            if isinstance(action, SendMessageTypingAction):
                self._batcher.on_typing(chat_id)
                return

            # presence (online / offline)
            status = getattr(event, "status", None)
            if status is not None:
                from telethon.tl.types import UserStatusOnline, UserStatusOffline
                if isinstance(status, UserStatusOnline):
                    log.debug(f"Presence: online [{chat_id}]")
                elif isinstance(status, UserStatusOffline):
                    log.debug(f"Presence: offline [{chat_id}]")

        except Exception as exc:
            log.error(f"on_user_update error: {exc}", exc_info=True)

    # ── MessageRead ────────────────────────────────────────────────────────────

    async def _on_read_receipt(self, event) -> None:
        try:
            chat_id = event.chat_id
            log.debug(f"Read receipt [{chat_id}] max_id={event.max_id}")
            # placeholder: thinker will consume this later
        except Exception as exc:
            log.error(f"on_read_receipt error: {exc}", exc_info=True)
