"""SQLAlchemy data model (SQLite).

Four groups of tables:
* knowledge base (DESIGN_DOCUMENT "Knowledge Base"): Role, Person, PersonImage, Area, AreaConnection, EntryPoint,
  PermissionRule, Sensor, PersonPresence
* recipients of the communication unit: Contact
* what the controller and the situation interpreters produce: Scenario, Situation, SensorEvent, SituationSummary,
  AggregatedSummary, SituationWarning, SituationAlarm, Notification
* bookkeeping: IdempotencyKey, LogEntry

Conventions: ids that appear in signals are strings (readable for seeded data, `<prefix>_<uuid>` otherwise); scores are
floats in [0, 1]; timestamps are timezone-aware UTC; free text is for display only; agent output and other nested
data is stored as JSON. Enumerated columns are checked by the database (see `one_of`).
Every foreign key column `x_id` has a relationship `x` next to it, so related rows are reached by attribute
access (`log_entry.situation.status`); the parent side has the matching collection where one is useful.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON, CheckConstraint, Column, DateTime, ForeignKey, Index, Table, Text, TypeDecorator,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str):
    return lambda: f"{prefix}_{uuid.uuid4().hex}"


class UtcDateTime(TypeDecorator):
    """SQLite has no time zones: store UTC, refuse naive datetimes, return aware ones."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime: timestamps must be timezone-aware")
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value, dialect):
        return None if value is None else value.replace(tzinfo=timezone.utc)


def one_of(column: str, *values: str) -> CheckConstraint:
    return CheckConstraint(f"{column} IN ({', '.join(repr(v) for v in values)})")


def scores(*columns: str) -> tuple[CheckConstraint, ...]:
    return tuple(CheckConstraint(f"{c} BETWEEN 0 AND 1") for c in columns)


class Base(DeclarativeBase):
    type_annotation_map = {datetime: UtcDateTime, dict[str, Any]: JSON, list[Any]: JSON}


# ====================================================================== knowledge base
class Role(Base):
    __tablename__ = "role"

    name: Mapped[str] = mapped_column(primary_key=True)  # e.g. "family", "delivery person", "gardener"


person_roles = Table(
    "person_roles", Base.metadata,
    Column("person_id", ForeignKey("person.id"), primary_key=True),
    Column("role", ForeignKey("role.name"), primary_key=True),
)


class Person(Base):
    """A person the system has seen. Persons that are not known by name have no `name` and low familiarity."""

    __tablename__ = "person"
    __table_args__ = scores("familiarity", "suspicion")

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id("per"))
    name: Mapped[str | None]
    familiarity: Mapped[float] = mapped_column(default=0.0)
    suspicion: Mapped[float] = mapped_column(default=0.0)

    roles: Mapped[list[Role]] = relationship(secondary=person_roles)
    images: Mapped[list[PersonImage]] = relationship(back_populates="person", cascade="all, delete-orphan")
    presences: Mapped[list[PersonPresence]] = relationship(back_populates="person")


class PersonImage(Base):
    """Reference image of a person (uploaded photo or captured by a CCTV camera)."""

    __tablename__ = "person_image"
    __table_args__ = (one_of("source", "upload", "cctv"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[str] = mapped_column(ForeignKey("person.id"), index=True)
    uri: Mapped[str]
    source: Mapped[str] = mapped_column(default="upload")

    person: Mapped[Person] = relationship(back_populates="images")


class Area(Base):
    __tablename__ = "area"

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    is_entryway: Mapped[bool] = mapped_column(default=False)  # where unfamiliar persons may ring the bell
    is_drop_off: Mapped[bool] = mapped_column(default=False)  # dedicated drop-off point for deliveries

    sensors: Mapped[list[Sensor]] = relationship(back_populates="area")
    entry_points: Mapped[list[EntryPoint]] = relationship(back_populates="area")


class AreaConnection(Base):
    """Two areas that are directly connected. Undirected: stored once, with `area_a_id < area_b_id`."""

    __tablename__ = "area_connection"
    __table_args__ = (CheckConstraint("area_a_id < area_b_id"),)

    area_a_id: Mapped[str] = mapped_column(ForeignKey("area.id"), primary_key=True)
    area_b_id: Mapped[str] = mapped_column(ForeignKey("area.id"), primary_key=True)

    area_a: Mapped[Area] = relationship(foreign_keys=[area_a_id])
    area_b: Mapped[Area] = relationship(foreign_keys=[area_b_id])


class EntryPoint(Base):
    """A door or window of the house; an open one matters for the "dangerous animal" alarm."""

    __tablename__ = "entry_point"

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    area_id: Mapped[str] = mapped_column(ForeignKey("area.id"))
    is_open: Mapped[bool] = mapped_column(default=False)

    area: Mapped[Area] = relationship(back_populates="entry_points")


permission_rule_areas = Table(
    "permission_rule_areas", Base.metadata,
    Column("rule_id", ForeignKey("permission_rule.id"), primary_key=True),
    Column("area_id", ForeignKey("area.id"), primary_key=True),
)


class PermissionRule(Base):
    """A whitelist rule: persons it applies to may enter the listed areas (all areas, or those in `areas`).

    "Family members may enter all areas":   applies_to="role", role="family", all_areas=True
    "Delivery people: entryway and path to the drop-off point": applies_to="role", role="delivery person",
        areas=[entryway, ..., drop-off]
    "Unfamiliar persons: entryway for a limited time, unlimited when invited":
        applies_to="unfamiliar", areas=[entryway], max_duration_s=120, invitation_lifts_limit=True
    """

    __tablename__ = "permission_rule"
    __table_args__ = (
        one_of("applies_to", "role", "unfamiliar"),
        CheckConstraint(
            "(applies_to = 'role' AND role_name IS NOT NULL) OR (applies_to = 'unfamiliar' AND role_name IS NULL)"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    description: Mapped[str] = mapped_column(Text)  # display only
    applies_to: Mapped[str]  # "role", or "unfamiliar" (not familiar or not decidable)
    role_name: Mapped[str | None] = mapped_column(ForeignKey("role.name"))
    all_areas: Mapped[bool] = mapped_column(default=False)
    max_duration_s: Mapped[int | None]  # how long a person may stay; None = unlimited
    invitation_lifts_limit: Mapped[bool] = mapped_column(default=False)

    role: Mapped[Role | None] = relationship()
    areas: Mapped[list[Area]] = relationship(secondary=permission_rule_areas)


class Sensor(Base):
    __tablename__ = "sensor"
    __table_args__ = (one_of("kind", "camera", "microphone"),)

    id: Mapped[str] = mapped_column(primary_key=True)
    kind: Mapped[str]
    area_id: Mapped[str] = mapped_column(ForeignKey("area.id"), index=True)
    x: Mapped[float]  # position in the floor plan
    y: Mapped[float]
    orientation_deg: Mapped[float | None]  # viewing direction, clockwise from north

    area: Mapped[Area] = relationship(back_populates="sensors")
    events: Mapped[list[SensorEvent]] = relationship(back_populates="sensor")


class PersonPresence(Base):
    """A person in an area. Currently in the surveilled area = `left_at` is NULL."""

    __tablename__ = "person_presence"
    __table_args__ = (Index("ix_presence_person_left", "person_id", "left_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[str] = mapped_column(ForeignKey("person.id"))
    area_id: Mapped[str] = mapped_column(ForeignKey("area.id"), index=True)
    entered_at: Mapped[datetime] = mapped_column(default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(default=utcnow)
    left_at: Mapped[datetime | None]

    person: Mapped[Person] = relationship(back_populates="presences")
    area: Mapped[Area] = relationship()


# ====================================================================== communication unit
class Contact(Base):
    """A recipient of warnings and alarms: a primary contact, and/or a person who may be in the surveilled area."""

    __tablename__ = "contact"

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    phone: Mapped[str]  # where the text notification goes
    is_primary: Mapped[bool] = mapped_column(default=False)
    person_id: Mapped[str | None] = mapped_column(ForeignKey("person.id"))  # to find contacts within the area

    person: Mapped[Person | None] = relationship()
    notifications: Mapped[list[Notification]] = relationship(back_populates="contact")


# ====================================================================== situations
class Scenario(Base):
    """Everything that is going on together. It starts with the first event and lasts until all of its situations
    are resolved. Warnings and alarms are idempotent per scenario: several situations reaching the same conclusion
    trigger it once."""

    __tablename__ = "scenario"
    __table_args__ = (one_of("status", "active", "resolved"),)

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id("scn"))
    status: Mapped[str] = mapped_column(default="active")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    resolved_at: Mapped[datetime | None]

    situations: Mapped[list[Situation]] = relationship(back_populates="scenario", order_by="Situation.created_at")
    aggregated_summaries: Mapped[list[AggregatedSummary]] = relationship(
        back_populates="scenario", order_by="AggregatedSummary.created_at")


class Situation(Base):
    """One instance of the situation interpreter (one per area); its id is also the workflow's checkpoint thread id."""

    __tablename__ = "situation"
    __table_args__ = (
        one_of("status", "active", "resolved"),
        Index("ix_situation_area_status", "area_id", "status"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id("sit"))
    scenario_id: Mapped[str] = mapped_column(ForeignKey("scenario.id"), index=True)
    area_id: Mapped[str] = mapped_column(ForeignKey("area.id"))
    status: Mapped[str] = mapped_column(default="active")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)
    resolved_at: Mapped[datetime | None]

    scenario: Mapped[Scenario] = relationship(back_populates="situations")
    area: Mapped[Area] = relationship()
    events: Mapped[list[SensorEvent]] = relationship(
        back_populates="situation", order_by="SensorEvent.start_time")
    summaries: Mapped[list[SituationSummary]] = relationship(
        back_populates="situation", order_by="SituationSummary.created_at")
    warnings: Mapped[list[SituationWarning]] = relationship(
        back_populates="situation", order_by="SituationWarning.created_at")
    alarms: Mapped[list[SituationAlarm]] = relationship(
        back_populates="situation", order_by="SituationAlarm.created_at")
    log_entries: Mapped[list[LogEntry]] = relationship(
        back_populates="situation", order_by="LogEntry.created_at")


class SensorEvent(Base):
    """A {video_event/audio_event} of a processor, assigned to the situation it started or joined."""

    __tablename__ = "sensor_event"
    __table_args__ = (one_of("kind", "video", "audio"),)

    id: Mapped[str] = mapped_column(primary_key=True)  # event_id of the signal
    situation_id: Mapped[str] = mapped_column(ForeignKey("situation.id"), index=True)
    sensor_id: Mapped[str] = mapped_column(ForeignKey("sensor.id"))
    kind: Mapped[str]
    start_time: Mapped[datetime]
    evidence: Mapped[str]  # reference to the media file
    bbox: Mapped[dict[str, Any] | None]  # video only: {"x", "y", "w", "h"}

    situation: Mapped[Situation] = relationship(back_populates="events")
    sensor: Mapped[Sensor] = relationship(back_populates="events")


class SituationSummary(Base):
    """The {situation_summary} of one situation interpreter; a situation has one per interpretation."""

    __tablename__ = "situation_summary"
    __table_args__ = scores("threat_score")

    id: Mapped[int] = mapped_column(primary_key=True)
    situation_id: Mapped[str] = mapped_column(ForeignKey("situation.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    threat_score: Mapped[float]  # maximum over the individual objects/events
    summary: Mapped[str] = mapped_column(Text)  # display only
    data: Mapped[dict[str, Any]] = mapped_column(default=dict)  # the annotated agent results the score is based on

    situation: Mapped[Situation] = relationship(back_populates="summaries")


class AggregatedSummary(Base):
    """The controller's amalgamation of the latest summaries of the unresolved situations of a scenario."""

    __tablename__ = "aggregated_summary"
    __table_args__ = scores("threat_score")

    id: Mapped[int] = mapped_column(primary_key=True)
    scenario_id: Mapped[str] = mapped_column(ForeignKey("scenario.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    threat_score: Mapped[float]
    summary: Mapped[str] = mapped_column(Text)  # display only
    data: Mapped[dict[str, Any]] = mapped_column(default=dict)  # e.g. situation ids, combined suspicion per person

    scenario: Mapped[Scenario] = relationship(back_populates="aggregated_summaries")


class SituationWarning(Base):
    """A {warning}: sent to the contacts, answered by elevating or dismissing it, or by the fallback policy."""

    __tablename__ = "warning"
    __table_args__ = (
        *scores("suspicion"),
        one_of("colour", "yellow", "orange"),
        one_of("status", "open", "elevated", "dismissed"),
        one_of("resolved_by", "user", "fallback_policy"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id("wrn"))  # warning_id of the signals
    situation_id: Mapped[str] = mapped_column(ForeignKey("situation.id"), index=True)
    area_id: Mapped[str] = mapped_column(ForeignKey("area.id"))
    cause: Mapped[str]  # e.g. "suspicious person", "unpermitted entry"
    suspicion: Mapped[float]  # suspicion level of the situation summary it originates from
    colour: Mapped[str]  # decided when sent: the orange threshold is lower at night
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    answer_deadline: Mapped[datetime | None]  # after this, the fallback policy applies
    status: Mapped[str] = mapped_column(default="open")
    resolved_by: Mapped[str | None]
    resolved_at: Mapped[datetime | None]

    situation: Mapped[Situation] = relationship(back_populates="warnings")
    area: Mapped[Area] = relationship()
    notifications: Mapped[list[Notification]] = relationship(back_populates="warning")


class SituationAlarm(Base):
    """An {alarm}. Only the controller creates these: by rule, by a user elevating a warning, or by the fallback."""

    __tablename__ = "alarm"
    __table_args__ = (
        one_of("origin", "rule", "user_elevation", "fallback_policy"),
        Index("ix_alarm_cause_area_time", "cause", "area_id", "created_at"),  # "similar alarm in the immediate past"
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id("alm"))
    situation_id: Mapped[str] = mapped_column(ForeignKey("situation.id"), index=True)
    area_id: Mapped[str] = mapped_column(ForeignKey("area.id"))
    cause: Mapped[str]  # e.g. "strong suspicion", "dangerous animal"
    origin: Mapped[str]
    warning_id: Mapped[str | None] = mapped_column(ForeignKey("warning.id"))  # the warning that was elevated
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    situation: Mapped[Situation] = relationship(back_populates="alarms")
    area: Mapped[Area] = relationship()
    warning: Mapped[SituationWarning | None] = relationship()
    notifications: Mapped[list[Notification]] = relationship(back_populates="alarm")


class Notification(Base):
    """A text message for one contact about a warning or an alarm (built from a template, never generated)."""

    __tablename__ = "notification"
    __table_args__ = (
        one_of("status", "sent", "failed"),
        CheckConstraint("(warning_id IS NULL) <> (alarm_id IS NULL)"),  # exactly one of the two
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    contact_id: Mapped[str] = mapped_column(ForeignKey("contact.id"))
    warning_id: Mapped[str | None] = mapped_column(ForeignKey("warning.id"), index=True)
    alarm_id: Mapped[str | None] = mapped_column(ForeignKey("alarm.id"), index=True)
    message: Mapped[str] = mapped_column(Text)
    status: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    contact: Mapped[Contact] = relationship(back_populates="notifications")
    warning: Mapped[SituationWarning | None] = relationship(back_populates="notifications")
    alarm: Mapped[SituationAlarm | None] = relationship(back_populates="notifications")


# ====================================================================== bookkeeping
class IdempotencyKey(Base):
    """Keys of warnings, alarms and speaker signals that were already sent, e.g. `alarm:<scenario_id>:<cause>`."""

    __tablename__ = "idempotency_key"
    __table_args__ = (one_of("kind", "warning", "alarm", "speak"),)

    key: Mapped[str] = mapped_column(primary_key=True)
    kind: Mapped[str]
    result: Mapped[dict[str, Any]] = mapped_column(default=dict)  # returned instead of sending again
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class LogEntry(Base):
    """Append-only event log: what happened, with the data it was based on."""

    __tablename__ = "log_entry"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow, index=True)
    situation_id: Mapped[str | None] = mapped_column(ForeignKey("situation.id"), index=True)
    kind: Mapped[str]  # e.g. "warning", "alarm", "situation_summary", "speak", "agent_failure"
    message: Mapped[str] = mapped_column(Text)  # display only
    data: Mapped[dict[str, Any]] = mapped_column(default=dict)

    situation: Mapped[Situation | None] = relationship(back_populates="log_entries")
