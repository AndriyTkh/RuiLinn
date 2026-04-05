"""
Phase 1: Telegram Listener + Message Logger
Captures all incoming messages to SQLite. No processing, no responses.
"""

import asyncio
import logging
import os
import sqlite3
from datetime import datetime

from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.errors import (
    AuthKeyUnregisteredError,
    FloodWaitError,
    SessionPasswordNeededError,
)

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

API_ID       = int(os.getenv("API_ID"))
API_HASH     = os.getenv("API_HASH")
SESSION_NAME = os.getenv("SESSION_NAME", "userbot")
DB_PATH      = os.getenv("DB_PATH", "messages.db")
LOG_PATH     = os.getenv("LOG_PATH", "listener.log")

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
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

    chat_id   = event.chat_id
    chat_name = getattr(event.chat, "title", None) or getattr(event.chat, "first_name", None)
    sender_id = msg.sender_id
    sender    = getattr(msg.sender, "first_name", None) or getattr(msg.sender, "username", None)
    content   = msg.text or ""
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
    await client.start()
    me = await client.get_me()
    log.info(f"Logged in as {me.first_name} (id={me.id})")
    return client


# ── Reconnect loop ────────────────────────────────────────────────────────────
async def run_with_reconnect(conn: sqlite3.Connection) -> None:
    backoff = 5

    while True:
        try:
            client = await start_client(SESSION_NAME)

            @client.on(events.NewMessage)
            async def handler(event):
                try:
                    await event.message.get_sender()
                    await event.message.get_chat()
                    log_message(conn, event)
                except FloodWaitError as e:
                    log.warning(f"FloodWait: sleeping {e.seconds}s")
                    await asyncio.sleep(e.seconds)
                except Exception as exc:
                    log.error(f"Handler error: {exc}")

            log.info("Listener running. Ctrl-C to stop.")
            backoff = 5
            await client.run_until_disconnected()

        except AuthKeyUnregisteredError:
            log.error("Session revoked. Delete userbot.session and re-run.")
            break

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
    try:
        await run_with_reconnect(conn)
    except KeyboardInterrupt:
        log.info("Shutdown signal received.")
    finally:
        conn.close()
        log.info("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())