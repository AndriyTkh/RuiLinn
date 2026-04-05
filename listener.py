"""
Phase 1: Telegram Listener + Message Logger
Captures all incoming messages to SQLite. No processing, no responses.
"""

import asyncio
import logging
import signal
import sqlite3
from datetime import datetime
from pathlib import Path

from telethon import TelegramClient, events
from telethon.errors import (
    FloodWaitError,
    SessionPasswordNeededError,
    AuthKeyUnregisteredError,
)

# ── Config ────────────────────────────────────────────────────────────────────
API_ID = 30822732
API_HASH = "6b93513ce93fd0860587bed4cd52e41b"        # ← your api_hash from my.telegram.org
SESSION_NAME = "userbot"
DB_PATH = "messages.db"
LOG_PATH = "listener.log"

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),          # also prints to terminal
    ],
)
log = logging.getLogger(__name__)


# ── Database ──────────────────────────────────────────────────────────────────
def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at   TEXT    NOT NULL,
            chat_id     INTEGER NOT NULL,
            chat_name   TEXT,
            sender_id   INTEGER,
            sender_name TEXT,
            message_id  INTEGER,
            content     TEXT,
            is_outgoing INTEGER DEFAULT 0
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chat ON messages(chat_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_time ON messages(logged_at)")
    conn.commit()
    log.info(f"Database ready at {path}")
    return conn


def log_message(conn: sqlite3.Connection, event) -> None:
    msg = event.message
    now = datetime.utcnow().isoformat()

    chat_id = event.chat_id
    chat_name = getattr(event.chat, "title", None) or getattr(event.chat, "first_name", None)
    sender_id = msg.sender_id
    sender = getattr(msg.sender, "first_name", None) or getattr(msg.sender, "username", None)
    content = msg.text or ""
    is_outgoing = int(msg.out)

    conn.execute("""
        INSERT INTO messages
            (logged_at, chat_id, chat_name, sender_id, sender_name, message_id, content, is_outgoing)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (now, chat_id, chat_name, sender_id, sender, msg.id, content, is_outgoing))
    conn.commit()

    direction = "→ OUT" if is_outgoing else "← IN "
    log.info(f"{direction} [{chat_name or chat_id}] {sender or sender_id}: {content[:80]!r}")


# ── Client setup ──────────────────────────────────────────────────────────────
async def start_client(session: str) -> TelegramClient:
    client = TelegramClient(session, API_ID, API_HASH)

    await client.start()          # handles login / 2FA interactively on first run

    me = await client.get_me()
    log.info(f"Logged in as {me.first_name} (id={me.id})")
    return client


# ── Reconnect loop ────────────────────────────────────────────────────────────
async def run_with_reconnect(conn: sqlite3.Connection) -> None:
    backoff = 5  # seconds, doubles on each failure up to 5 min

    while True:
        try:
            client = await start_client(SESSION_NAME)

            @client.on(events.NewMessage)
            async def handler(event):
                try:
                    # Pre-fetch sender so we have their name
                    await event.message.get_sender()
                    await event.message.get_chat()
                    log_message(conn, event)
                except FloodWaitError as e:
                    log.warning(f"FloodWait: sleeping {e.seconds}s")
                    await asyncio.sleep(e.seconds)
                except Exception as exc:
                    log.error(f"Handler error: {exc}")

            log.info("Listener running. Ctrl-C to stop.")
            backoff = 5          # reset on successful connect
            await client.run_until_disconnected()

        except AuthKeyUnregisteredError:
            log.error("Session revoked by Telegram. Delete userbot.session and re-run.")
            break                # no point retrying — need fresh auth

        except SessionPasswordNeededError:
            log.error("2FA password required but not provided.")
            break

        except Exception as exc:
            log.error(f"Disconnected: {exc}. Reconnecting in {backoff}s…")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 300)


# ── Entry point ───────────────────────────────────────────────────────────────
async def main():
    conn = init_db(DB_PATH)

    # Graceful shutdown on Ctrl-C / SIGTERM
    loop = asyncio.get_running_loop()
    stop = loop.create_future()

    def _shutdown():
        log.info("Shutdown signal received.")
        stop.set_result(None)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    listener = asyncio.create_task(run_with_reconnect(conn))
    await asyncio.wait([listener, asyncio.ensure_future(stop)],
                       return_when=asyncio.FIRST_COMPLETED)
    listener.cancel()
    conn.close()
    log.info("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())