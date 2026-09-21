"""Shared fixtures: a throw-away SQLite file (like the real database, with separate connections) with a small house."""
from __future__ import annotations

import pytest

from ai_alarm.db import init_db, make_engine, make_session_factory
from ai_alarm.db.models import (
    Area, AreaConnection, Contact, EntryPoint, PermissionRule, Person, PersonImage, Role, Sensor,
)
from ai_alarm.kb import KnowledgeBase


def seed_house(factory) -> None:
    """entry (the entryway) is connected to garden and house.
    Anna (family, familiar), Gustav (gardener, familiar); contacts: Bob (primary), Anna (when she is in the area)."""
    family, delivery, gardener = Role(name="family"), Role(name="delivery person"), Role(name="gardener")
    entry, garden, house = Area(id="entry", name="Entryway", is_entryway=True), Area(id="garden", name="Garden"), \
        Area(id="house", name="House")
    with factory() as s:
        s.add_all([
            family, delivery, gardener, entry, garden, house,
            AreaConnection(area_a_id="entry", area_b_id="garden"),
            AreaConnection(area_a_id="entry", area_b_id="house"),
            EntryPoint(id="door", name="Front door", area_id="entry", is_open=False),
            Sensor(id="cam_entry", kind="camera", area_id="entry", x=0, y=0, orientation_deg=0),
            Sensor(id="cam_garden", kind="camera", area_id="garden", x=5, y=0, orientation_deg=90),
            Sensor(id="mic_garden", kind="microphone", area_id="garden", x=5, y=1),
            Sensor(id="cam_house", kind="camera", area_id="house", x=0, y=5, orientation_deg=180),
            Person(id="anna", name="Anna", familiarity=1.0, roles=[family],
                   images=[PersonImage(uri="anna_1.png"), PersonImage(uri="anna_2.png")]),
            Person(id="gustav", name="Gustav", familiarity=0.8, roles=[gardener],
                   images=[PersonImage(uri="gustav_1.png", source="cctv")]),
            Person(id="neighbour", name="Neighbour", familiarity=0.2),  # no images, no roles
            Contact(id="c_bob", name="Bob", phone="+49 100", is_primary=True),
            Contact(id="c_anna", name="Anna", phone="+49 200", person_id="anna"),
            PermissionRule(description="family may enter all areas", applies_to="role", role_name="family",
                           all_areas=True),
            PermissionRule(description="delivery people: entryway", applies_to="role", role_name="delivery person",
                           areas=[entry]),
            PermissionRule(description="gardeners: garden and entryway", applies_to="role", role_name="gardener",
                           areas=[garden, entry]),
            PermissionRule(description="unfamiliar persons: entryway for two minutes, unlimited if invited",
                           applies_to="unfamiliar", areas=[entry], max_duration_s=120, invitation_lifts_limit=True),
        ])
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
