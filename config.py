import os
from dotenv import load_dotenv

load_dotenv()

API_ID        = int(os.getenv("API_ID"))
API_HASH      = os.getenv("API_HASH")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
SESSION_NAME  = os.getenv("SESSION_NAME", "local_db/userbot")
DB_PATH       = os.getenv("DB_PATH", "local_db/messages.db")
LOG_PATH      = os.getenv("LOG_PATH",   "local_db/listener.log")
TRACE_PATH    = os.getenv("TRACE_PATH", "local_db/.log")

BATCH_TIMEOUT_BASE = 3.0   # seconds
BATCH_TIMEOUT_MIN  = 1.5
BATCH_TIMEOUT_MAX  = 6.0
BATCH_LENGTH_LIMIT = 10    # force-flush after this many buffered messages

# Operator chat ID for controlling unit flags (set in .env or leave 0 to disable Telegram alerts)
_op = os.getenv("OPERATOR_CHAT_ID", "0")
OPERATOR_CHAT_ID: int | None = int(_op) if _op and _op != "0" else None
