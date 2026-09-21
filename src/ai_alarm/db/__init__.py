"""The SQLite database: engine setup and table creation. The tables are defined in `ai_alarm.db.models`."""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from ai_alarm.db.models import Base

DEFAULT_URL = os.environ.get("AI_ALARM_DB_URL", "sqlite:///data/ai_alarm.db")


def _sqlite_pragmas(dbapi_connection, _record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")  # SQLite ignores foreign keys unless asked to enforce them
    cursor.execute("PRAGMA journal_mode = WAL")  # readers (web interface) do not block the writers (workflows)
    cursor.close()


def make_engine(url: str = DEFAULT_URL) -> Engine:
    engine = create_engine(url)
    event.listen(engine, "connect", _sqlite_pragmas)
    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Sessions whose objects stay readable after a commit (no reload on access)."""
    return sessionmaker(engine, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Create the database file (and its directory) and all missing tables. Existing tables are left as they are."""
    path = engine.url.database
    if path and path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)


