"""Data model: table creation, integrity rules of the database, and the typical queries of the design."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ai_alarm.db import make_engine, init_db

from ai_alarm.db.models import (
    AggregatedSummary, Area, AreaConnection, Base, Contact, IdempotencyKey, LogEntry, Notification,
    PermissionRule, Person, PersonImage, PersonPresence, Role, Scenario, Sensor, SensorEvent, Situation,
    SituationAlarm, SituationSummary, SituationWarning,
)

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def session():
    engine = make_engine("sqlite://")
    init_db(engine)
    with Session(engine) as s:
        s.add_all([Area(id="entry", name="Entryway", is_entryway=True), Area(id="garden", name="Garden"),
                   Scenario(id="scn")])
        s.commit()
        yield s
    engine.dispose()


def rejected(session, *objects):
    """Assert that the database refuses to store the objects."""
    session.add_all(objects)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


# ------------------------------------------------------------------ setup
def test_init_db_creates_file_and_all_tables(tmp_path):
    path = tmp_path / "nested" / "app.db"
    engine = make_engine(f"sqlite:///{path}")
    init_db(engine)
    init_db(engine)  # idempotent
    assert path.exists()
    assert set(inspect(engine).get_table_names()) == set(Base.metadata.tables)
    engine.dispose()


def test_foreign_keys_are_enforced(session):
    rejected(session, Sensor(id="cam1", kind="camera", area_id="nowhere", x=0, y=0))


# ------------------------------------------------------------------ values
def test_timestamps_roundtrip_as_utc_and_naive_ones_are_refused(session):
    plus_two_hours = timezone(timedelta(hours=2))
    session.add(Situation(id="s1", scenario_id="scn", area_id="garden", created_at=NOW.astimezone(plus_two_hours)))
    session.commit()
    session.expire_all()
    stored = session.get(Situation, "s1").created_at
    assert stored == NOW and stored.utcoffset() == timedelta(0)

    session.add(Situation(id="s2", scenario_id="scn", area_id="garden", created_at=datetime(2026, 1, 1)))
    with pytest.raises(Exception, match="naive datetime"):
        session.commit()


def test_scores_must_be_in_unit_interval(session):
    rejected(session, Person(id="p1", suspicion=1.5))
    rejected(session, Person(id="p2", familiarity=-0.1))


def test_enumerated_columns_are_checked(session):
    rejected(session, Sensor(id="cam1", kind="radar", area_id="garden", x=0, y=0))
    rejected(session, Situation(id="s1", scenario_id="scn", area_id="garden", status="sleeping"))


def test_json_columns_roundtrip(session):
    session.add(Situation(id="s1", scenario_id="scn", area_id="garden"))
    session.add(SituationSummary(situation_id="s1", threat_score=0.4, summary="a person",
                                 data={"objects": [{"object_id": "p1", "bbox": {"x": 0.1}}]}))
    session.commit()
    session.expire_all()
    assert session.scalars(select(SituationSummary)).one().data["objects"][0]["bbox"] == {"x": 0.1}


# ------------------------------------------------------------------ knowledge base
def test_person_has_roles_and_images(session):
    session.add_all([Role(name="family"), Role(name="gardener")])
    session.add(Person(id="anna", name="Anna", familiarity=1.0, roles=[Role(name="mother")]))
    session.commit()
    anna = session.get(Person, "anna")
    anna.roles.append(session.get(Role, "family"))
    anna.images.append(PersonImage(uri="anna.png"))
    session.commit()
    session.expire_all()
    anna = session.get(Person, "anna")
    assert {r.name for r in anna.roles} == {"mother", "family"}
    assert [i.uri for i in anna.images] == ["anna.png"]


def test_area_connections_are_stored_once_per_pair(session):
    session.add(AreaConnection(area_a_id="entry", area_b_id="garden"))
    session.commit()
    rejected(session, AreaConnection(area_a_id="garden", area_b_id="entry"))


def test_permission_rule_targets_role_or_unfamiliar_persons(session):
    session.add(Role(name="delivery person"))
    session.commit()
    entry = session.get(Area, "entry")
    session.add(PermissionRule(description="delivery", applies_to="role", role_name="delivery person",
                               areas=[entry]))
    session.add(PermissionRule(description="stranger", applies_to="unfamiliar", areas=[entry],
                               max_duration_s=120, invitation_lifts_limit=True))
    session.commit()
    assert [a.id for r in session.scalars(select(PermissionRule)) for a in r.areas] == ["entry", "entry"]

    rejected(session, PermissionRule(description="no role", applies_to="role"))
    rejected(session, PermissionRule(description="role for strangers", applies_to="unfamiliar",
                                     role_name="delivery person"))


def test_people_currently_in_an_area(session):
    session.add_all([Person(id="anna"), Person(id="bob")])
    session.add_all([PersonPresence(person_id="anna", area_id="garden"),
                     PersonPresence(person_id="bob", area_id="garden", left_at=NOW)])
    session.commit()
    present = session.scalars(select(PersonPresence.person_id).where(PersonPresence.left_at.is_(None))).all()
    assert present == ["anna"]


# ------------------------------------------------------------------ situations
def make_situation(session, sid="s1"):
    session.add_all([
        Sensor(id="cam1", kind="camera", area_id="garden", x=1, y=2, orientation_deg=90),
        Situation(id=sid, scenario_id="scn", area_id="garden"),
    ])
    session.commit()


def test_situation_collects_its_data(session):
    make_situation(session)
    session.add(SensorEvent(id="e1", situation_id="s1", sensor_id="cam1", kind="video", start_time=NOW,
                            evidence="clip.mp4", bbox={"x": 0, "y": 0, "w": 1, "h": 1}))
    session.add(SituationSummary(situation_id="s1", threat_score=0.5, summary="unclear"))
    session.add(SituationWarning(id="w1", situation_id="s1", area_id="garden", cause="unclear situation",
                                 suspicion=0.5, colour="yellow"))
    session.add(SituationAlarm(situation_id="s1", area_id="garden", cause="strong suspicion", origin="rule"))
    session.commit()
    s = session.get(Situation, "s1")
    assert [e.id for e in s.events] == ["e1"] and len(s.summaries) == len(s.alarms) == 1
    assert s.warnings[0].status == "open" and s.alarms[0].id.startswith("alm_")


def test_warning_and_alarm_values_are_checked(session):
    make_situation(session)
    ok = dict(situation_id="s1", area_id="garden", cause="x", suspicion=0.5)
    rejected(session, SituationWarning(**ok, colour="red"))
    rejected(session, SituationWarning(**ok, colour="yellow", resolved_by="nobody"))
    rejected(session, SituationAlarm(situation_id="s1", area_id="garden", cause="x", origin="whim"))


def test_find_similar_recent_alarm(session):
    make_situation(session)
    session.add_all([
        SituationAlarm(situation_id="s1", area_id="garden", cause="unpermitted entry", origin="rule",
                       created_at=NOW - timedelta(minutes=2)),
        SituationAlarm(situation_id="s1", area_id="garden", cause="unpermitted entry", origin="rule",
                       created_at=NOW - timedelta(hours=3)),
    ])
    session.commit()
    recent = session.scalars(select(SituationAlarm).where(
        SituationAlarm.cause == "unpermitted entry", SituationAlarm.area_id == "garden",
        SituationAlarm.created_at > NOW - timedelta(minutes=10))).all()
    assert len(recent) == 1


def test_notification_refers_to_exactly_one_of_warning_and_alarm(session):
    make_situation(session)
    session.add_all([
        Contact(id="c1", name="Anna", phone="+49 1", is_primary=True),
        SituationWarning(id="w1", situation_id="s1", area_id="garden", cause="x", suspicion=0.9, colour="orange"),
        SituationAlarm(id="a1", situation_id="s1", area_id="garden", cause="x", origin="rule"),
    ])
    session.commit()
    session.add(Notification(contact_id="c1", warning_id="w1", message="text", status="sent"))
    session.commit()
    rejected(session, Notification(contact_id="c1", message="text", status="sent"))  # neither
    rejected(session, Notification(contact_id="c1", warning_id="w1", alarm_id="a1", message="text", status="sent"))
    rejected(session, Notification(contact_id="c1", warning_id="w1", message="text", status="maybe"))


# ------------------------------------------------------------------ bookkeeping
def test_idempotency_key_can_only_be_used_once(session):
    session.add(IdempotencyKey(key="s1:strong suspicion", kind="alarm", result={"alarm_id": "a1"}))
    session.commit()
    rejected(session, IdempotencyKey(key="s1:strong suspicion", kind="alarm"))
    assert session.get(IdempotencyKey, "s1:strong suspicion").result == {"alarm_id": "a1"}


def test_log_entries_are_ordered_by_time(session):
    session.add_all([
        LogEntry(kind="alarm", message="b", created_at=NOW),
        LogEntry(kind="warning", message="a", created_at=NOW - timedelta(minutes=1)),
    ])
    session.commit()
    assert [e.message for e in session.scalars(select(LogEntry).order_by(LogEntry.created_at))] == ["a", "b"]


# ------------------------------------------------------------------ navigation by attribute
def test_related_rows_are_reached_by_attribute(session):
    make_situation(session)
    session.add_all([
        Contact(id="c1", name="Anna", phone="+49 1", is_primary=True),
        SituationWarning(id="w1", situation_id="s1", area_id="garden", cause="x", suspicion=0.9, colour="orange"),
        LogEntry(situation_id="s1", kind="warning", message="sent"),
    ])
    session.commit()
    session.add_all([
        SituationAlarm(id="a1", situation_id="s1", area_id="garden", cause="x", origin="user_elevation",
                       warning_id="w1"),
        Notification(contact_id="c1", warning_id="w1", message="text", status="sent"),
        SensorEvent(id="e1", situation_id="s1", sensor_id="cam1", kind="video", start_time=NOW, evidence="c.mp4"),
    ])
    session.commit()
    session.expire_all()

    assert session.scalars(select(LogEntry)).one().situation.status == "active"
    assert session.get(SituationAlarm, "a1").warning.situation.area.name == "Garden"
    assert session.scalars(select(Notification)).one().warning.notifications[0].contact.name == "Anna"
    assert session.get(SensorEvent, "e1").sensor.area.id == "garden"
    assert [e.id for e in session.get(Sensor, "cam1").events] == ["e1"]
    assert [s.id for s in session.get(Area, "garden").sensors] == ["cam1"]
    assert session.scalars(select(Notification)).one().alarm is None  # nullable side


def test_both_sides_of_a_relationship_stay_in_sync(session):
    make_situation(session)
    s = session.get(Situation, "s1")
    entry = LogEntry(kind="alarm", message="m")
    session.add(entry)
    entry.situation = s
    assert entry in s.log_entries  # before anything is flushed
    session.commit()
    assert entry.situation_id == "s1"

    entry.situation = None  # a log entry does not need a situation
    session.commit()
    assert entry.situation_id is None and s.log_entries == []


def test_area_connection_and_rule_role_by_attribute(session):
    session.add(Role(name="family"))
    session.add(AreaConnection(area_a_id="entry", area_b_id="garden"))
    session.add(PermissionRule(description="family", applies_to="role", role_name="family", all_areas=True))
    session.commit()
    session.expire_all()
    link = session.scalars(select(AreaConnection)).one()
    assert (link.area_a.name, link.area_b.name) == ("Entryway", "Garden")
    assert session.scalars(select(PermissionRule)).one().role.name == "family"
    assert session.scalars(select(Contact)).all() == []  # a contact needs no person


def test_scenario_groups_situations_and_aggregated_summaries(session):
    make_situation(session, "s1")
    session.add(Situation(id="s2", scenario_id="scn", area_id="entry"))
    session.add(AggregatedSummary(scenario_id="scn", threat_score=0.4, summary="two situations"))
    session.commit()
    session.expire_all()
    scenario = session.get(Scenario, "scn")
    assert scenario.status == "active" and [x.id for x in scenario.situations] == ["s1", "s2"]
    assert scenario.aggregated_summaries[0].scenario is scenario and session.get(Situation, "s2").scenario is scenario


def test_a_situation_needs_a_scenario_and_scenario_status_is_checked(session):
    rejected(session, Situation(id="s1", area_id="garden"))
    rejected(session, Situation(id="s2", scenario_id="nowhere", area_id="garden"))
    rejected(session, Scenario(id="scn2", status="sleeping"))
