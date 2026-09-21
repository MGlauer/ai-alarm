"""`python -m ai_alarm.db` creates the database (default: data/ai_alarm.db, override with AI_ALARM_DB_URL)."""
from pathlib import Path
from sqlalchemy import Engine
from ai_alarm.db import make_engine, Base


def init_db(engine: Engine) -> None:
    """Create the database file (and its directory) and all missing tables. Existing tables are left as they are."""
    path = engine.url.database
    if path and path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)

if __name__ == "__main__":
    engine = make_engine()
    init_db(engine)
    print(f"Database ready: {engine.url}")