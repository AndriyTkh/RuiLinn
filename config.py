import os
from dotenv import load_dotenv

load_dotenv()

API_ID        = int(os.getenv("API_ID"))
API_HASH      = os.getenv("API_HASH")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
SESSION_NAME  = os.getenv("SESSION_NAME", "local_db/userbot")
DB_PATH       = os.getenv("DB_PATH", "local_db/messages.db")
LOG_PATH      = os.getenv("LOG_PATH", "local_db/listener.log")

BATCH_TIMEOUT_BASE = 5.0   # seconds
BATCH_TIMEOUT_MIN  = 3.0
BATCH_TIMEOUT_MAX  = 12.0
BATCH_LENGTH_LIMIT = 10    # force-flush after this many buffered messages
