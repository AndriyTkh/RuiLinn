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
from config import DB_PATH, LOG_PATH, TRACE_PATH
from context_builder.builder import ContextBuilder
from controlling_unit.unit import ControllingUnit
from db.store import fetch_recent_batches, init_db, log_batch
from knowledge.store import KnowledgeStore
from memory.store import MemoryStore
from planner.planner import Planner
from self.self_module import Self
from telethon_layer.batcher import Batcher
from telethon_layer.client import run_with_reconnect
from telethon_layer.handlers import TelethonHandlers
from thinker.thinker import Thinker
from verifier.verifier import Verifier


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


def _compact(obj):
    """Recursively strip null / False / empty-collection values for trace readability."""
    if isinstance(obj, dict):
        return {k: _compact(v) for k, v in obj.items()
                if v is not None and v is not False and v != [] and v != {}}
    if isinstance(obj, list):
        return [_compact(i) for i in obj]
    return obj


# ── Agent Trace Logger (file-only, no console) ─────────────────────────────────

_trace_handler = logging.FileHandler(TRACE_PATH, encoding="utf-8")
_trace_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
trace = logging.getLogger("agent.trace")
trace.setLevel(logging.DEBUG)
trace.addHandler(_trace_handler)
trace.propagate = False  # don't bubble up to root → stays out of console


# ── Batch Pipeline ─────────────────────────────────────────────────────────────

async def handle_batch(batch: dict) -> None:
    chat_id   = batch["chat_id"]
    chat_name = batch.get("chat_name", str(chat_id))
    t0        = time.monotonic()

    def _step(n: int, label: str) -> None:
        elapsed = time.monotonic() - t0
        log.info(f"── STEP {n} [{chat_name}] {label} (+{elapsed:.2f}s)")

    # 1. Verify (heuristics — no LLM)
    _step(1, "verify")
    verifier_result = verifier.verdict(chat_id, batch)
    if verifier_result["verdict"] != "pass":
        trace.info(f"[{chat_name}] VERIFIER {verifier_result['verdict']}")

    # 2. Classify
    _step(2, "classify")
    trace.info(f"[{chat_name}] CLASSIFY INPUT\n{json.dumps(_compact(batch), ensure_ascii=False, indent=2)}")
    classifier_result = await classify_batch(batch, verifier_flags=verifier_result)
    if classifier_result is None:
        log.warning("── CLASSIFIER ── no result")
        return

    log.info(f"── CLASSIFIER ── {json.dumps(classifier_result, ensure_ascii=False)}")
    trace.info(f"[{chat_name}] CLASSIFY OUTPUT\n{json.dumps(_compact(classifier_result), ensure_ascii=False, indent=2)}")

    # 3. Persist batch + result for conversation history
    _step(3, "log batch")
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

    # 4. Build context
    _step(4, "build context")
    context_package = ctx_builder.build_context(batch, classifier_result)

    # 4a. If conversation was cold, run retrospective on the previous session in background
    if context_package["timing"]["gap_label"] == "cold_open":
        asyncio.create_task(planner.run_retrospective(chat_id))

    # 5. Think
    _step(5, "think")
    thinker_result = await thinker.think(context_package)
    if thinker_result is None:
        log.warning("── THINKER ── no result")
        return

    log.info(f"── THINKER ── messages={thinker_result['messages']}")
    trace.info(f"[{chat_name}] THINK OUTPUT\n{json.dumps(_compact(thinker_result), ensure_ascii=False, indent=2)}")

    # 6. React if thinker also wants one
    if thinker_result.get("reaction"):
        last_msg_id = batch["messages"][-1].get("message_id")
        if last_msg_id:
            await actions.do_react(chat_id, last_msg_id, thinker_result["reaction"])

    # 7. Send messages (thinker decides reply_to_id; None = no threading)
    _step(6, "send")
    reply_to_id = thinker_result.get("reply_to_id")
    await actions.send_batch(chat_id, thinker_result["messages"], reply_to_id=reply_to_id)

    # 8. Persist outgoing batch for history
    _step(7, "log outgoing")
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
    asyncio.create_task(controlling_unit.run_sampling_cycle())


async def main() -> None:
    global db_conn, batcher, memory, ctx_builder, thinker, planner, verifier, controlling_unit

    db_conn          = init_db(DB_PATH)
    memory           = MemoryStore(db_conn)
    self_module      = Self(db_conn, memory)
    knowledge        = KnowledgeStore(db_conn)
    ctx_builder      = ContextBuilder(db_conn, memory, knowledge)
    thinker          = Thinker(ctx_builder, self_module)
    planner          = Planner(db_conn, memory, self_module)
    verifier         = Verifier()
    controlling_unit = ControllingUnit(db_conn)
    batcher          = Batcher(on_batch=handle_batch)

    try:
        await run_with_reconnect(on_ready)
    except KeyboardInterrupt:
        log.info("Shutdown signal received.")
    finally:
        db_conn.close()
        log.info("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
