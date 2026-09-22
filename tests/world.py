"""Test harness: a controller with the real database, knowledge base and communication unit, and fakes for
everything outside of it (situation interpreters, speaker, text transport, weather, audio, two agents' models)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from ai_alarm.agents import BehaviouralInterpreter, NoiseInterpreter
from ai_alarm.comm import CommunicationUnit
from ai_alarm.controller import (
    Controller, ObscuredSignal, PersonAssessment, SensorEventSignal, SituationResolvedSignal,
    SituationSummarySignal,
)
from ai_alarm.db.models import LogEntry, Situation, SituationAlarm, SituationWarning

T0 = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
SENSORS = {"garden": "cam_garden", "entry": "cam_entry", "house": "cam_house"}


# ------------------------------------------------------------------ fakes for everything outside the controller
class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


class FakeInterpreters:
    def __init__(self):
        self.started, self.forwarded = [], []

    def start(self, situation_id, event):
        self.started.append((situation_id, event))

    def forward(self, situation_id, event):
        self.forwarded.append((situation_id, event))


class FakeSpeaker:
    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)


class FakeTransport:
    def __init__(self):
        self.sent, self.reachable = [], True

    def send(self, phone, message):
        if self.reachable:
            self.sent.append((phone, message))
        return self.reachable


class FakeWeather:
    def __init__(self):
        self.conditions, self.fail = set(), False

    def conditions_at(self, at):
        if self.fail:
            raise ConnectionError("weather service down")
        return self.conditions


class FakeAudio:
    def __init__(self):
        self.reference = "audio.wav"

    def latest_audio(self, area_id, at):
        return self.reference


def person(identity="unknown", obj="p1", suspicion=0.1, **kw):
    return PersonAssessment(object_id=obj, identity=identity, suspicion=suspicion, **kw)


def known(person_id="anna", obj="p1", suspicion=0.0, **kw):
    return person("known", obj, suspicion, person_id=person_id, **kw)


class World:
    def __init__(self, session_factory, kb, config, now):
        self.sf, self.kb = session_factory, kb
        self.clock = Clock(now)
        self._events = 0
        self.interpreters, self.speaker, self.transport = FakeInterpreters(), FakeSpeaker(), FakeTransport()
        self.weather, self.audio = FakeWeather(), FakeAudio()
        # what the models of the behavioural and the noise interpreter answer
        self.behaviour = {"persons": [{"object_id": "p1", "role_mismatch": 0.0,
                                       "intents": [{"intent": "move_to_area", "confidence": 0.9}]}],
                          "relations": [], "health_emergency": False}
        self.noise = {"category": "human_activity", "confidence": 0.9, "is_sensor_artefact": False}
        self.controller = Controller(
            session_factory=session_factory, kb=kb, comm=CommunicationUnit(kb, self.transport),
            interpreters=self.interpreters, speaker=self.speaker, weather=self.weather, audio=self.audio,
            behavioural_interpreter=BehaviouralInterpreter(lambda **kw: self.behaviour, kb),
            noise_interpreter=NoiseInterpreter(lambda **kw: self.noise, kb),
            config=config, clock=self.clock)

    # ---- actions
    def event(self, event_id, area="garden", sensor=None):
        sensor = sensor or SENSORS[area]
        return self.controller.handle_sensor_event(SensorEventSignal(
            type="audio_event" if sensor.startswith("mic") else "video_event", event_id=event_id, sensor_id=sensor,
            area_id=area, start_time=self.clock(), evidence=f"{event_id}.mp4"))

    def situation(self, area="garden"):
        """Send a new event for the area; returns its situation (the running one, or a new one)."""
        self._events += 1
        return self.event(f"auto_{self._events}", area)

    def summary(self, sid, area="garden", score=0.1, persons=(), **kw):
        self.controller.handle_situation_summary(SituationSummarySignal(
            situation_id=sid, area_id=area, event_id=kw.pop("event_id", "evt_test"),
            summary=kw.pop("text", "something"), threat_score=score, persons=list(persons), **kw))

    def obscured(self, sid, area="garden", persons=()):
        self.controller.handle_obscured(ObscuredSignal(
            situation_id=sid, area_id=area, evidence=["frame.png"], persons=list(persons)))

    def resolve(self, sid):
        self.controller.handle_situation_resolved(SituationResolvedSignal(situation_id=sid))

    # ---- what is in the database
    def rows(self, model, **where):
        with self.sf() as s:
            return list(s.scalars(select(model).filter_by(**where)))

    def warnings(self, sid=None, **where):
        return self.rows(SituationWarning, **({"situation_id": sid} if sid else {}), **where)

    def alarms(self, sid=None, **where):
        return self.rows(SituationAlarm, **({"situation_id": sid} if sid else {}), **where)

    def log_kinds(self, sid):
        return [e.kind for e in self.rows(LogEntry, situation_id=sid)]

    def status(self, sid):
        return self.rows(Situation, id=sid)[0].status
