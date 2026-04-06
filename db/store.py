import json
import logging
import os
import sqlite3
from datetime import datetime, timezone

log = logging.getLogger(__name__)


def init_db(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at   TEXT    NOT NULL,
            chat_id     INTEGER NOT NULL,
            chat_name   TEXT,
            sender_id   INTEGER,
            sender_name TEXT,
            message_id  INTEGER,
            content     TEXT,
            media_type  TEXT    DEFAULT 'text',
            is_outgoing INTEGER DEFAULT 0,
            is_forward  INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_msg_chat ON messages(chat_id);
        CREATE INDEX IF NOT EXISTS idx_msg_time ON messages(logged_at);

        -- Flushed batches (both incoming and outgoing) for conversation history
        CREATE TABLE IF NOT EXISTS batches (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id          INTEGER NOT NULL,
            chat_name        TEXT,
            direction        TEXT    NOT NULL DEFAULT 'in',  -- 'in' | 'out'
            reason           TEXT,
            flushed_at       TEXT    NOT NULL,
            messages_json    TEXT    NOT NULL,
            classifier_json  TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_batch_chat ON batches(chat_id);
        CREATE INDEX IF NOT EXISTS idx_batch_time ON batches(flushed_at);

        -- Person registry: maps Telegram sender_id → internal person_id
        CREATE TABLE IF NOT EXISTS persons (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id   INTEGER NOT NULL UNIQUE,
            first_name  TEXT,
            created_at  TEXT    NOT NULL
        );

        -- Episodic memory: what happened in conversations
        CREATE TABLE IF NOT EXISTS episodes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id   INTEGER NOT NULL,
            chat_id     INTEGER NOT NULL,
            content     TEXT    NOT NULL,
            tags        TEXT    DEFAULT '[]',  -- JSON array
            created_at  TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ep_person ON episodes(person_id);
        CREATE INDEX IF NOT EXISTS idx_ep_chat   ON episodes(chat_id);

        -- Semantic memory: facts about people
        CREATE TABLE IF NOT EXISTS facts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id   INTEGER NOT NULL,
            content     TEXT    NOT NULL,
            source      TEXT,
            confidence  REAL    DEFAULT 1.0,
            created_at  TEXT    NOT NULL,
            updated_at  TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_fact_person ON facts(person_id);

        -- Self memory: agent's own significant events
        CREATE TABLE IF NOT EXISTS self_memory (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            content     TEXT    NOT NULL,
            event_type  TEXT,
            created_at  TEXT    NOT NULL
        );

        -- Relationship state per person+chat
        CREATE TABLE IF NOT EXISTS relationship (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id           INTEGER NOT NULL,
            chat_id             INTEGER NOT NULL,
            tone                TEXT    DEFAULT 'neutral',
            unresolved_threads  TEXT    DEFAULT '[]',  -- JSON array
            agent_commitments   TEXT    DEFAULT '[]',  -- JSON array
            updated_at          TEXT    NOT NULL,
            UNIQUE(person_id, chat_id)
        );

        -- Unresolved follow-ups thinker flagged
        CREATE TABLE IF NOT EXISTS pending_intents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id     INTEGER NOT NULL,
            intent      TEXT    NOT NULL,
            resolved    INTEGER DEFAULT 0,
            created_at  TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_intent_chat ON pending_intents(chat_id);

        -- Agent state (single row, upserted)
        CREATE TABLE IF NOT EXISTS agent_state (
            id      INTEGER PRIMARY KEY CHECK (id = 1),
            mood    TEXT    DEFAULT 'neutral',
            energy  REAL    DEFAULT 0.7,
            updated_at TEXT NOT NULL
        );
        INSERT OR IGNORE INTO agent_state (id, mood, energy, updated_at)
        VALUES (1, 'neutral', 0.7, CURRENT_TIMESTAMP);

        -- Goals at all timescales
        CREATE TABLE IF NOT EXISTS goals (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timescale  TEXT    NOT NULL DEFAULT 'daily',  -- long_term | weekly | daily
            content    TEXT    NOT NULL,
            status     TEXT    NOT NULL DEFAULT 'active', -- active | achieved | dropped
            created_at TEXT    NOT NULL,
            updated_at TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_goal_status ON goals(status);

        -- Daily log: intention + end-of-day summary
        CREATE TABLE IF NOT EXISTS daily_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            date       TEXT    NOT NULL UNIQUE,  -- YYYY-MM-DD
            intention  TEXT,
            summary    TEXT,
            created_at TEXT    NOT NULL,
            updated_at TEXT    NOT NULL
        );
    """)
    conn.commit()
    log.info(f"Database ready at {path}")
    return conn


def log_message(conn: sqlite3.Connection, msg_dict: dict) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO messages
            (logged_at, chat_id, chat_name, sender_id, sender_name,
             message_id, content, media_type, is_outgoing, is_forward)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now,
        msg_dict["chat_id"],
        msg_dict.get("chat_name"),
        msg_dict.get("sender_id"),
        msg_dict.get("sender"),
        msg_dict.get("message_id"),
        msg_dict.get("content", ""),
        msg_dict.get("media_type", "text"),
        int(msg_dict.get("is_outgoing", False)),
        int(msg_dict.get("is_forward", False)),
    ))
    conn.commit()


def log_batch(
    conn: sqlite3.Connection,
    batch: dict,
    classifier_result: dict | None = None,
    direction: str = "in",
) -> int:
    """Persist a flushed batch. Returns the new row id."""
    conn.execute("""
        INSERT INTO batches (chat_id, chat_name, direction, reason, flushed_at, messages_json, classifier_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        batch["chat_id"],
        batch.get("chat_name"),
        direction,
        batch.get("reason"),
        batch.get("flushed_at") or datetime.now(timezone.utc).isoformat(),
        json.dumps(batch["messages"], ensure_ascii=False),
        json.dumps(classifier_result, ensure_ascii=False) if classifier_result else None,
    ))
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


# ── Goals ─────────────────────────────────────────────────────────────────────

def get_goals(
    conn: sqlite3.Connection,
    status: str | None = "active",
    timescale: str | None = None,
) -> list[dict]:
    query  = "SELECT id, timescale, content, status, created_at, updated_at FROM goals"
    params: list = []
    clauses: list[str] = []
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if timescale is not None:
        clauses.append("timescale = ?")
        params.append(timescale)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at"
    return [dict(r) for r in conn.execute(query, params).fetchall()]


def add_goal(conn: sqlite3.Connection, content: str, timescale: str = "daily") -> int:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO goals (timescale, content, status, created_at, updated_at) VALUES (?, ?, 'active', ?, ?)",
        (timescale, content, now, now),
    )
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def update_goal_status(conn: sqlite3.Connection, goal_id: int, status: str) -> None:
    conn.execute(
        "UPDATE goals SET status = ?, updated_at = ? WHERE id = ?",
        (status, datetime.now(timezone.utc).isoformat(), goal_id),
    )
    conn.commit()


# ── Daily Log ──────────────────────────────────────────────────────────────────

def get_daily_log(conn: sqlite3.Connection, date_str: str) -> dict | None:
    row = conn.execute(
        "SELECT date, intention, summary, created_at FROM daily_log WHERE date = ?",
        (date_str,),
    ).fetchone()
    return dict(row) if row else None


def upsert_daily_log(
    conn: sqlite3.Connection,
    date_str: str,
    intention: str | None = None,
    summary: str | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    existing = get_daily_log(conn, date_str)
    if existing:
        conn.execute("""
            UPDATE daily_log SET
                intention  = COALESCE(?, intention),
                summary    = COALESCE(?, summary),
                updated_at = ?
            WHERE date = ?
        """, (intention, summary, now, date_str))
    else:
        conn.execute(
            "INSERT INTO daily_log (date, intention, summary, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (date_str, intention, summary, now, now),
        )
    conn.commit()


# ── Batches ────────────────────────────────────────────────────────────────────

def fetch_recent_batches(conn: sqlite3.Connection, chat_id: int, n: int = 20) -> list[dict]:
    """Return last N batches for this chat (both directions), oldest first."""
    rows = conn.execute("""
        SELECT direction, reason, flushed_at, messages_json, classifier_json
        FROM batches
        WHERE chat_id = ?
        ORDER BY flushed_at DESC
        LIMIT ?
    """, (chat_id, n)).fetchall()

    result = []
    for row in reversed(rows):
        result.append({
            "direction":  row["direction"],
            "reason":     row["reason"],
            "flushed_at": row["flushed_at"],
            "messages":   json.loads(row["messages_json"]),
            "classifier": json.loads(row["classifier_json"]) if row["classifier_json"] else None,
        })
    return result
