"""`python -m ai_alarm.db` creates the database (default: data/ai_alarm.db, override with AI_ALARM_DB_URL)."""
from pathlib import Path
from sqlalchemy import Engine
from ai_alarm.db import make_engine, init_db


if __name__ == "__main__":
    engine = make_engine()
    init_db(engine)
    print(f"Database ready: {engine.url}")