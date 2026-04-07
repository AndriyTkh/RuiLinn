"""
Telethon Layer — Client

Owns connection setup and the reconnect loop.
Calls back into main via the on_ready(client, me) hook once connected.
"""

import asyncio
import logging
from typing import Awaitable, Callable

from telethon import TelegramClient
from telethon.errors import AuthKeyUnregisteredError, SessionPasswordNeededError

from config import API_HASH, API_ID, SESSION_NAME

log = logging.getLogger(__name__)


async def start_client(session: str = SESSION_NAME) -> tuple[TelegramClient, object]:
    client = TelegramClient(session, API_ID, API_HASH)
    await client.start()
    me = await client.get_me()
    log.info(f"Logged in as {me.first_name} (id={me.id})")
    return client, me


async def run_with_reconnect(
    on_ready: Callable[[TelegramClient, object], Awaitable[None]]
) -> None:
    """
    Connects and calls on_ready(client, me) after each successful connect.
    Handles exponential backoff on disconnect.
    on_ready is responsible for registering handlers before returning.
    """
    backoff = 5

    while True:
        try:
            client, me = await start_client()
            await on_ready(client, me)
            log.info("Listening. Ctrl-C to stop.")
            backoff = 5
            try:
                await client.run_until_disconnected()
            except asyncio.CancelledError:
                log.warning("Connection dropped (CancelledError).")

        except AuthKeyUnregisteredError:
            log.error("Session revoked. Delete the .session file and re-run.")
            break

        except SessionPasswordNeededError:
            log.error("2FA password required but not provided.")
            break

        except EOFError:
            log.error("No session file found and stdin is not interactive. Run once manually to authenticate.")
            break

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            log.error(f"Disconnected: {exc}. Reconnecting in {backoff}s…")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 300)
