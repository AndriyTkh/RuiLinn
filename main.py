"""
RuiLinn — Entry Point

Pipeline: Telethon → Batcher → Classifier → ContextBuilder → Thinker → Actions
"""

import asyncio
import json
import logging
import os
import sys
import time

import actions.actions as actions
from classifier.classifier import classify_batch
from config import DB_PATH, LOG_PATH
from context_builder.builder import ContextBuilder
from db.store import fetch_recent_batches, init_db, log_batch
from memory.store import MemoryStore
from planner.planner import Planner
from telethon_layer.batcher import Batcher
from telethon_layer.client import run_with_reconnect
from telethon_layer.handlers import TelethonHandlers
from thinker.thinker import Thinker


# ── Logging ────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True) if os.path.dirname(LOG_PATH) else None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(
            stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
        ),
    ],
)
log = logging.getLogger(__name__)


# ── Batch Pipeline ─────────────────────────────────────────────────────────────

async def handle_batch(batch: dict) -> None:
    chat_id   = batch["chat_id"]
    chat_name = batch.get("chat_name", str(chat_id))
    t0        = time.monotonic()

    def _step(n: int, label: str) -> None:
        elapsed = time.monotonic() - t0
        log.info(f"── STEP {n} [{chat_name}] {label} (+{elapsed:.2f}s)")

    # 1. Classify
    _step(1, "classify")
    classifier_result = await classify_batch(batch)
    if classifier_result is None:
        log.warning("── CLASSIFIER ── no result")
        return

    log.info(f"── CLASSIFIER ── {json.dumps(classifier_result, ensure_ascii=False)}")

    # 2. Persist batch + result for conversation history
    _step(2, "log batch")
    log_batch(db_conn, batch, classifier_result, direction="in")

    if not classifier_result["response_expected"]:
        log.info(f"── DONE [{chat_name}] no response needed (+{time.monotonic()-t0:.2f}s)")
        return

    response_type = classifier_result["response_type"]

    # react-only: skip thinker, go straight to actions
    if response_type == "react":
        last_msg_id = batch["messages"][-1].get("message_id")
        if last_msg_id:
            await actions.do_react(chat_id, last_msg_id, "❤️")
        return

    # 3. Build context
    _step(3, "build context")
    context_package = ctx_builder.build_context(batch, classifier_result)

    # 3a. If conversation was cold, run retrospective on the previous session in background
    if context_package["timing"]["gap_label"] == "cold_open":
        asyncio.create_task(planner.run_retrospective(chat_id))

    # 4. Think
    _step(4, "think")
    thinker_result = await thinker.think(context_package)
    if thinker_result is None:
        log.warning("── THINKER ── no result")
        return

    log.info(f"── THINKER ── messages={thinker_result['messages']}")

    # 5. React if thinker also wants one
    if thinker_result.get("reaction"):
        last_msg_id = batch["messages"][-1].get("message_id")
        if last_msg_id:
            await actions.do_react(chat_id, last_msg_id, thinker_result["reaction"])

    # 6. Send messages
    _step(5, "send")
    reply_to_id = batch["messages"][-1].get("message_id")
    await actions.send_batch(chat_id, thinker_result["messages"], reply_to_id=reply_to_id)

    # 7. Persist outgoing batch for history
    _step(6, "log outgoing")
    outgoing_batch = {
        "chat_id":    chat_id,
        "chat_name":  batch.get("chat_name"),
        "reason":     "reply",
        "flushed_at": None,  # log_batch fills this
        "messages":   [
            {"content": m, "media_type": "text", "sender": "agent"}
            for m in thinker_result["messages"]
        ],
    }
    log_batch(db_conn, outgoing_batch, direction="out")
    log.info(f"── DONE [{chat_name}] (+{time.monotonic()-t0:.2f}s)")


# ── Startup ────────────────────────────────────────────────────────────────────

async def on_ready(client, me) -> None:
    actions.init(client)
    handlers = TelethonHandlers(client, batcher, db_conn, me)
    handlers.register()
    asyncio.create_task(planner.run_daily_cycle())


async def main() -> None:
    global db_conn, batcher, memory, ctx_builder, thinker, planner

    db_conn     = init_db(DB_PATH)
    memory      = MemoryStore(db_conn)
    ctx_builder = ContextBuilder(db_conn, memory)
    thinker     = Thinker(ctx_builder)
    planner     = Planner(db_conn, memory)
    batcher     = Batcher(on_batch=handle_batch)

    try:
        await run_with_reconnect(on_ready)
    except KeyboardInterrupt:
        log.info("Shutdown signal received.")
    finally:
        db_conn.close()
        log.info("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
