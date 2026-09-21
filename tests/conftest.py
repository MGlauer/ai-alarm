"""Shared fixtures: a throw-away SQLite file (like the real database, with separate connections) with a small house."""
from __future__ import annotations

import pytest
from world import T0, World

from ai_alarm.controller import ControllerConfig
from ai_alarm.db import init_db, make_engine, make_session_factory
from ai_alarm.db.models import Person
from ai_alarm.kb import KnowledgeBase
from ai_alarm.seed import seed_demo_house


def seed_house(factory) -> None:
    """The demo house plus a "neighbour": in the knowledge base, but without images and roles, and not familiar."""
    seed_demo_house(factory)
    with factory() as s:
        s.add(Person(id="neighbour", name="Neighbour", familiarity=0.2))
        s.commit()


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    factory = make_session_factory(engine)
    seed_house(factory)
    yield factory
    engine.dispose()


@pytest.fixture
def kb(session_factory):
    return KnowledgeBase(session_factory)


@pytest.fixture
def make_world(session_factory, kb):
    """A controller world; `make_world(config, now)` for one with another configuration or start time."""
    return lambda config=ControllerConfig(), now=T0: World(session_factory, kb, config, now)


@pytest.fixture
def world(make_world):
    return make_world()
