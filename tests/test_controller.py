"""Controller: the rules of the design document, with the database, knowledge base and communication unit real and
everything outside (situation interpreters, speaker, transport, weather, audio, the model of two agents) faked."""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

import pytest
from sqlalchemy import select

from ai_alarm.agents import BehaviouralInterpreter, NoiseInterpreter
from ai_alarm.agents.behavioural_interpreter import PersonContext
from ai_alarm.comm import CommunicationUnit
from ai_alarm.controller import (
    UNPERMITTED_ENTRY_TEXT, AnimalAssessment, Controller, ControllerConfig, ObscuredSignal, PersonAssessment,
    SensorEventSignal, SituationResolvedSignal, SituationSummarySignal,
)
from ai_alarm.db.models import (
    AggregatedSummary, EntryPoint, LogEntry, Notification, Person, PersonPresence, Scenario, SensorEvent,
    Situation, SituationAlarm, SituationSummary, SituationWarning,
)

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
        self.started, self.forwarded, self.resumed = [], [], []

    def start(self, situation_id, event):
        self.started.append((situation_id, event))

    def forward(self, situation_id, event):
        self.forwarded.append((situation_id, event))

    def resume(self, situation_id):
        self.resumed.append(situation_id)


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
            situation_id=sid, area_id=area, summary=kw.pop("text", "something"), threat_score=score,
            persons=list(persons), **kw))

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


@pytest.fixture
def make_world(session_factory, kb):
    return lambda config=ControllerConfig(), now=T0: World(session_factory, kb, config, now)


@pytest.fixture
def world(make_world):
    return make_world()


# ------------------------------------------------------------------ sensor events and situations
def test_first_event_starts_a_situation_and_later_ones_join_it(world):
    a = world.event("e1", "garden")
    b = world.event("e2", "garden", sensor="mic_garden")
    c = world.event("e3", "entry")
    assert a == b != c
    assert [sid for sid, _ in world.interpreters.started] == [a, c]
    assert [(sid, e.sensor_event.event_id) for sid, e in world.interpreters.forwarded] == [(a, "e2")]
    assert [(e.id, e.situation_id, e.kind) for e in world.rows(SensorEvent)] == [
        ("e1", a, "video"), ("e2", a, "audio"), ("e3", c, "video")]

    assert world.event("e1", "garden") == a  # delivered twice: nothing new happens
    assert len(world.interpreters.started) == 2 and len(world.interpreters.forwarded) == 1


def test_a_resolved_situation_is_closed_and_the_next_event_starts_a_new_one(world):
    a = world.situation("garden")
    world.summary(a, "garden", 0.0, [known("anna")])
    assert [p.id for p in world.kb.people_present("garden")] == ["anna"]

    world.resolve(a)
    assert world.status(a) == "resolved" and world.kb.people_present() == []
    assert world.situation("garden") != a and len(world.interpreters.started) == 2


def test_summaries_are_logged_and_aggregated_over_all_active_situations(world):
    entry, garden = world.situation("entry"), world.situation("garden")
    world.summary(entry, "entry", 0.3, [person(suspicion=0.3)], text="a visitor")
    world.summary(garden, "garden", 0.2, [known("gustav", suspicion=0.2)], text="the gardener")

    assert len(world.rows(SituationSummary)) == 2
    latest = world.rows(AggregatedSummary)[-1]
    assert latest.threat_score == 0.3 and latest.summary.count(" | ") == 1
    assert latest.data == {"situations": {entry: 0.3, garden: 0.2},
                           "persons": {f"unk_{entry}_p1": 0.3, "gustav": 0.2}}
    assert "situation_summary" in world.log_kinds(entry)


def test_situations_share_a_scenario_until_the_last_one_is_resolved(world):
    def scenario(sid):
        return world.rows(Situation, id=sid)[0].scenario_id

    a, b = world.situation("entry"), world.situation("garden")
    assert scenario(a) == scenario(b) and len(world.rows(Scenario)) == 1

    world.resolve(a)
    assert world.rows(Scenario)[0].status == "active"
    c = world.situation("house")  # still the same scenario
    assert scenario(c) == scenario(b)

    world.resolve(b)
    assert world.rows(Scenario)[0].status == "active"
    world.resolve(c)
    resolved = world.rows(Scenario)[0]
    assert resolved.status == "resolved" and resolved.resolved_at == T0

    d = world.situation("entry")  # a new scenario
    assert scenario(d) != scenario(a) and len(world.rows(Scenario)) == 2


def test_only_the_latest_summaries_of_the_unresolved_situations_of_the_scenario_are_aggregated(world):
    a, b = world.situation("entry"), world.situation("garden")
    world.summary(a, "entry", 0.3, [known("gustav", suspicion=0.3)])
    world.summary(b, "garden", 0.2, [known("gustav", suspicion=0.2)])
    world.summary(b, "garden", 0.1, [known("gustav", suspicion=0.1)])  # replaces the older summary of b
    assert world.rows(AggregatedSummary)[-1].data == {"situations": {a: 0.3, b: 0.1}, "persons": {"gustav": 0.3}}

    world.resolve(a)  # a resolved situation no longer counts
    world.summary(b, "garden", 0.15, [known("gustav", suspicion=0.15)])
    last = world.rows(AggregatedSummary)[-1]
    assert last.data == {"situations": {b: 0.15}, "persons": {"gustav": 0.15}}

    world.resolve(b)  # the scenario is over: the next one starts from scratch
    c = world.situation("entry")
    world.summary(c, "entry", 0.2, [person(suspicion=0.2)])
    newest = world.rows(AggregatedSummary)[-1]
    assert newest.scenario_id != last.scenario_id and newest.data["situations"] == {c: 0.2}


def test_unknown_situation_is_a_caller_error(world):
    with pytest.raises(LookupError):
        world.summary("sit_nope", "garden")


# ------------------------------------------------------------------ suspicion
def test_low_suspicion_is_only_logged(world):
    sid = world.situation("entry")
    world.summary(sid, "entry", 0.2, [person(suspicion=0.2)])
    assert not world.warnings() and not world.alarms() and not world.transport.sent
    assert world.status(sid) == "active"


def test_free_text_never_triggers_anything(world):
    sid = world.situation("entry")
    world.summary(sid, "entry", 0.1, [person(suspicion=0.1)], text="ALARM! Burglar! Sound the alarm immediately!")
    assert not world.warnings() and not world.alarms() and not world.transport.sent


def test_suspicious_person_triggers_a_warning_that_pauses_the_situation(world):
    sid = world.situation("entry")
    world.summary(sid, "entry", 0.5, [person(suspicion=0.5)])
    (warning,) = world.warnings(sid)
    assert (warning.cause, warning.colour, warning.suspicion, warning.status) == \
        ("suspicious person", "yellow", 0.5, "open")
    assert world.status(sid) == "waiting_for_user"

    (phone, message), = world.transport.sent  # Bob is the only recipient: Anna is not in the area
    assert phone == "+49 100"
    assert message.startswith("YELLOW WARNING (0.50): A suspicious person was detected in Entryway.")
    assert [(n.contact_id, n.status) for n in world.rows(Notification)] == [("c_bob", "sent")]
    log = next(e for e in world.rows(LogEntry, kind="warning"))
    assert log.data["situation_summary"]["threat_score"] == 0.5 and log.data["warning_id"] == warning.id


def test_warnings_are_idempotent(world):
    sid = world.situation("entry")
    for _ in range(3):
        world.summary(sid, "entry", 0.5, [person(suspicion=0.5)])
    assert len(world.warnings()) == 1 and len(world.transport.sent) == 1


def test_warning_is_orange_above_the_orange_threshold(world):
    sid = world.situation("entry")
    world.summary(sid, "entry", 0.8, [person(suspicion=0.5)])
    assert world.warnings(sid)[0].colour == "orange"
    assert world.transport.sent[0][1].startswith("ORANGE WARNING")


def test_recipients_include_contacts_within_the_area(world):
    world.kb.record_detection("anna", "house", 0.0, T0)
    sid = world.situation("entry")
    world.summary(sid, "entry", 0.5, [person(suspicion=0.5)])
    assert sorted(phone for phone, _ in world.transport.sent) == ["+49 100", "+49 200"]


def test_strong_suspicion_triggers_an_alarm_not_a_warning(world):
    sid = world.situation("entry")
    world.summary(sid, "entry", 0.9, [person(suspicion=0.9)])
    (alarm,) = world.alarms(sid)
    assert (alarm.cause, alarm.origin) == ("strong suspicion", "rule") and not world.warnings()
    assert world.transport.sent[0][1].startswith("ALARM: ")

    world.summary(sid, "entry", 0.9, [person(suspicion=0.9)])  # idempotent
    assert len(world.alarms()) == 1 and len(world.transport.sent) == 1


def test_a_similar_alarm_in_the_immediate_past_suppresses_the_next_one(world):
    first = world.situation("entry")
    world.summary(first, "entry", 0.9, [person(suspicion=0.9)])
    world.resolve(first)

    second = world.situation("entry")  # a distinct situation, same cause and area
    world.summary(second, "entry", 0.9, [person(suspicion=0.9)])
    assert len(world.alarms()) == 1 and "alarm_suppressed" in world.log_kinds(second)
    world.summary(second, "entry", 0.9, [person(suspicion=0.9)])  # asking again does not log again
    assert world.log_kinds(second).count("alarm_suppressed") == 1

    world.resolve(second)
    world.clock.advance(301)
    world.summary(world.situation("entry"), "entry", 0.9, [person(suspicion=0.9)])
    assert len(world.alarms()) == 2


def test_a_person_seen_by_several_situations_gets_their_highest_suspicion(world, session_factory):
    entry, garden = world.situation("entry"), world.situation("garden")
    world.summary(entry, "entry", 0.3, [known("gustav", suspicion=0.3)])
    world.summary(garden, "garden", 0.2, [known("gustav", suspicion=0.2)])
    with session_factory() as s:
        assert s.get(Person, "gustav").suspicion == 0.3
    assert world.rows(AggregatedSummary)[-1].data["persons"] == {"gustav": 0.3}


def test_several_situations_reaching_the_same_conclusion_trigger_it_once(world):
    entry, garden = world.situation("entry"), world.situation("garden")
    for sid, area in [(entry, "entry"), (garden, "garden"), (entry, "entry")]:
        world.summary(sid, area, 0.5, [person(suspicion=0.5)])
    assert [w.cause for w in world.warnings() if w.cause == "suspicious person"] == ["suspicious person"]
    assert len([m for _, m in world.transport.sent if "suspicious person" in m]) == 1

    for sid, area in [(entry, "entry"), (garden, "garden")]:
        world.summary(sid, area, 0.9, [person(suspicion=0.9)])
    assert [(a.cause, a.situation_id) for a in world.alarms()] == [("strong suspicion", entry)]


def test_unclear_situation_warns_at_the_warning_threshold(world):
    sid = world.situation("entry")
    world.summary(sid, "entry", 0.3, unclear_situation=True)
    assert not world.warnings()
    world.summary(sid, "entry", 0.5, unclear_situation=True)
    assert [w.cause for w in world.warnings()] == ["unclear situation"]


# ------------------------------------------------------------------ permissions
def unpermitted(world, area, who, wait_s=0):
    """Does this person get an 'unpermitted entry' warning, at the first summary or `wait_s` seconds later?"""
    sid = world.situation(area)
    world.summary(sid, area, 0.1, [who])
    if wait_s:
        world.clock.advance(wait_s)
        world.summary(sid, area, 0.1, [who])
    return bool(world.warnings(sid, cause="unpermitted entry"))


@pytest.mark.parametrize("area, who, wait_s, expected", [
    ("house", known("anna"), 0, False),  # family may enter all areas
    ("garden", known("gustav"), 0, False),
    ("house", known("gustav"), 0, True),  # the gardener may not enter the house
    ("entry", person(predicted_role="delivery person"), 0, False),
    ("garden", person(predicted_role="delivery person"), 0, True),
    ("house", person(predicted_role="family"), 0, True),  # a predicted role must not open the whole house
    ("entry", person(), 100, False),  # unfamiliar persons may be at the entryway for a limited time ...
    ("entry", person(), 121, True),
    ("entry", person(invited=True), 500, False),  # ... which an invitation lifts
    ("garden", person(invited=True), 0, True),  # ... but only there
    ("entry", known("neighbour"), 0, False),  # not familiar (0.2): as an unfamiliar person
    ("entry", known("neighbour"), 121, True),
    ("garden", person("not_decidable", cause="clothing"), 0, True),  # treated like an unfamiliar person
    ("entry", person("not_decidable", cause="unavailable"), 0, False),
    ("entry", person("not_decidable", cause="unavailable"), 121, True),
])
def test_entry_permissions(world, area, who, wait_s, expected):
    assert unpermitted(world, area, who, wait_s) is expected


def test_unpermitted_entry_warns_and_speaks_once_then_becomes_an_alarm(world):
    sid = world.situation("garden")
    who = [person()]
    for _ in range(2):
        world.summary(sid, "garden", 0.1, who)
    assert [w.cause for w in world.warnings(sid)] == ["unpermitted entry"]
    assert world.speaker.spoken == [UNPERMITTED_ENTRY_TEXT] and not world.alarms()

    world.clock.advance(61)  # the entry is prolonged
    world.summary(sid, "garden", 0.1, who)
    assert [(a.cause, a.origin) for a in world.alarms(sid)] == [("unpermitted entry", "rule")]
    assert world.speaker.spoken == [UNPERMITTED_ENTRY_TEXT]


def test_a_dismissed_unpermitted_entry_does_not_become_an_alarm(world):
    sid = world.situation("garden")
    world.summary(sid, "garden", 0.1, [person()])
    world.controller.dismiss(world.warnings(sid)[0].id)
    world.clock.advance(61)
    world.summary(sid, "garden", 0.1, [person()])
    assert not world.alarms()


# ------------------------------------------------------------------ the knowledge base learns from detections
def test_detected_persons_are_stored_with_suspicion_and_role(world, session_factory):
    sid = world.situation("entry")
    world.summary(sid, "entry", 0.3, [
        person(obj="p1", suspicion=0.3, predicted_role="delivery person"),
        person("not_decidable", obj="p2", cause="clothing"),  # not stored
        person("not_decidable", obj="p3", cause="low confidence", suspicion=0.2),  # stored
        known("gustav", obj="p4", suspicion=0.25),
    ])
    with session_factory() as s:
        stored = {p.id: p for p in s.scalars(select(Person))}
        assert {f"unk_{sid}_p1", f"unk_{sid}_p3", "gustav"} <= stored.keys() and f"unk_{sid}_p2" not in stored
        assert (stored[f"unk_{sid}_p1"].suspicion, [r.name for r in stored[f"unk_{sid}_p1"].roles]) == \
            (0.3, ["delivery person"])
        assert stored["gustav"].suspicion == 0.25 and [r.name for r in stored["gustav"].roles] == ["gardener"]
        present = {p.person_id for p in s.scalars(select(PersonPresence).where(PersonPresence.left_at.is_(None)))}
        assert present == {f"unk_{sid}_p1", f"unk_{sid}_p3", "gustav"}


# ------------------------------------------------------------------ dangerous animals
BEAR, BOAR = AnimalAssessment(label="bear", danger=1.0), AnimalAssessment(label="boar", danger=0.8)


def open_door(session_factory):
    with session_factory() as s:
        s.get(EntryPoint, "door").is_open = True
        s.commit()


def test_dangerous_animal_is_a_warning_if_the_house_is_closed(world):
    sid = world.situation("garden")
    world.summary(sid, "garden", 1.0, animals=[BEAR])
    assert [w.cause for w in world.warnings()] == ["dangerous animal"] and not world.alarms()


def test_dangerous_animal_is_an_alarm_if_an_entry_point_is_open(world, session_factory):
    open_door(session_factory)
    world.summary(world.situation("garden"), "garden", 1.0, animals=[BEAR])
    assert [a.cause for a in world.alarms()] == ["dangerous animal"] and not world.warnings()


def test_animals_below_the_danger_threshold_are_ignored(world, session_factory):
    open_door(session_factory)
    world.summary(world.situation("garden"), "garden", 0.8, animals=[BOAR])
    assert not world.warnings() and not world.alarms()


def test_optional_alarm_condition_people_present(make_world):
    world = make_world(ControllerConfig(animal_alarm_if_people_present=True))
    sid = world.situation("garden")
    world.summary(sid, "garden", 1.0, animals=[BEAR])
    assert world.warnings() and not world.alarms()  # nobody around

    world.kb.record_detection("anna", "house", 0.0, T0)
    world.resolve(sid)
    world.clock.advance(1)
    world.summary(world.situation("garden"), "garden", 1.0, animals=[BEAR])
    assert [a.cause for a in world.alarms()] == ["dangerous animal"]


@pytest.mark.parametrize("window, expected", [((time(11), time(13)), "alarm"), ((time(20), time(22)), "warning")])
def test_optional_alarm_condition_time_frame(make_world, window, expected):
    world = make_world(ControllerConfig(animal_alarm_window=window))  # it is 12:00
    world.summary(world.situation("garden"), "garden", 1.0, animals=[BEAR])
    assert (len(world.alarms()), len(world.warnings())) == ((1, 0) if expected == "alarm" else (0, 1))


# ------------------------------------------------------------------ answers, timeouts and the fallback policy
def warned(world, score=0.5):
    """A situation with one open warning of the given suspicion."""
    sid = world.situation("entry")
    world.summary(sid, "entry", score, [person(suspicion=0.5)])
    return sid, world.warnings(sid)[0]


def test_elevating_a_warning_triggers_an_alarm_and_resumes_the_situation(world):
    sid, warning = warned(world)
    world.controller.elevate(warning.id)

    (alarm,) = world.alarms(sid)
    assert (alarm.cause, alarm.origin, alarm.warning_id) == ("suspicious person", "user_elevation", warning.id)
    assert world.warnings(sid)[0].status == "elevated" and world.warnings(sid)[0].resolved_by == "user"
    assert world.status(sid) == "active" and world.interpreters.resumed == [sid]

    world.controller.elevate(warning.id)  # answered already: nothing more happens
    assert len(world.alarms()) == 1 and world.interpreters.resumed == [sid]


def test_dismissing_a_warning_resumes_the_situation_without_alarm(world):
    sid, warning = warned(world)
    world.controller.dismiss(warning.id)
    assert world.warnings(sid)[0].status == "dismissed" and not world.alarms()
    assert world.status(sid) == "active" and world.interpreters.resumed == [sid]


def test_unknown_warning_is_a_caller_error(world):
    with pytest.raises(LookupError):
        world.controller.elevate("wrn_nope")


def test_situation_resumes_only_when_all_its_warnings_are_answered(world):
    sid = world.situation("entry")
    world.summary(sid, "entry", 0.5, [person(suspicion=0.5)], unclear_situation=True)
    first, second = world.warnings(sid)
    world.controller.dismiss(first.id)
    assert world.status(sid) == "waiting_for_user" and not world.interpreters.resumed
    world.controller.dismiss(second.id)
    assert world.status(sid) == "active" and world.interpreters.resumed == [sid]


def test_unanswered_yellow_warning_is_dismissed_by_the_fallback_policy(world):
    sid, warning = warned(world, 0.5)
    world.clock.advance(599)
    world.controller.check_timeouts()
    assert world.warnings(sid)[0].status == "open"

    world.clock.advance(2)
    world.controller.check_timeouts()
    dismissed = world.warnings(sid)[0]
    assert (dismissed.status, dismissed.resolved_by) == ("dismissed", "fallback_policy")
    assert not world.alarms() and world.interpreters.resumed == [sid]


def test_unanswered_orange_warning_is_elevated_by_the_fallback_policy(world):
    sid, warning = warned(world, 0.8)
    world.clock.advance(601)
    world.controller.check_timeouts()
    assert world.warnings(sid)[0].status == "elevated"
    assert [(a.cause, a.origin) for a in world.alarms(sid)] == [("suspicious person", "fallback_policy")]
    world.controller.check_timeouts()  # nothing is answered twice
    assert len(world.alarms()) == 1


def test_if_nobody_can_be_reached_the_fallback_policy_applies_at_once(world):
    world.transport.reachable = False
    sid, warning = warned(world, 0.8)
    assert world.warnings(sid)[0].status == "elevated" and world.warnings(sid)[0].resolved_by == "fallback_policy"
    assert [a.origin for a in world.alarms(sid)] == ["fallback_policy"]
    assert {n.status for n in world.rows(Notification)} == {"failed"}
    assert world.status(sid) == "active" and world.interpreters.resumed == [sid]


@pytest.mark.parametrize("hour, colour", [(12, "yellow"), (23, "orange"), (5, "orange"), (6, "yellow")])
def test_orange_threshold_is_lower_at_night(make_world, hour, colour):
    world = make_world(now=datetime(2026, 9, 21, hour, 0, tzinfo=timezone.utc))
    sid = world.situation("entry")
    world.summary(sid, "entry", 0.6, [person(suspicion=0.5)])  # above 0.5, below 0.75
    assert world.warnings(sid)[0].colour == colour


# ------------------------------------------------------------------ obscured view
def obstruct(world, sid, persons=(), wait_s=121):
    """The view is obscured for a short time (first signal) and again after `wait_s` seconds."""
    world.obscured(sid, persons=persons)
    world.clock.advance(wait_s)
    world.obscured(sid, persons=persons)


GARDENER = PersonContext(object_id="p1", verdict="unknown", predicted_role="gardener")


def test_obstruction_by_weather_is_plausible(world):
    world.weather.conditions = {"fog"}
    sid = world.situation("garden")
    obstruct(world, sid)
    assert not world.warnings() and not world.alarms() and "obscured_plausible" in world.log_kinds(sid)


def test_a_short_implausible_obstruction_is_not_reported_yet(world):
    sid = world.situation("garden")
    world.obscured(sid)
    world.clock.advance(60)
    world.obscured(sid)
    assert not world.warnings()


def test_a_prolonged_implausible_obstruction_by_a_person_is_a_warning_and_an_alarm(world):
    sid = world.situation("garden")
    obstruct(world, sid)
    assert [w.cause for w in world.warnings(sid)] == ["vision obstructed"]
    assert [a.cause for a in world.alarms(sid)] == ["obstruction by human activity"]


def test_no_alarm_if_the_noise_is_not_human(world):
    world.noise = {**world.noise, "category": "animal"}
    sid = world.situation("garden")
    obstruct(world, sid)
    assert [w.cause for w in world.warnings(sid)] == ["vision obstructed"] and not world.alarms()


def test_no_alarm_if_a_family_member_is_in_the_area(world):
    world.kb.record_detection("anna", "garden", 0.0, world.clock())
    sid = world.situation("garden")
    obstruct(world, sid)
    assert world.warnings(sid) and not world.alarms()


def test_a_family_member_in_another_area_does_not_prevent_the_alarm(world):
    world.kb.record_detection("anna", "house", 0.0, world.clock())
    sid = world.situation("garden")
    obstruct(world, sid)
    assert [a.cause for a in world.alarms(sid)] == ["obstruction by human activity"]


def test_no_alarm_without_audio(world):
    world.audio.reference = None
    sid = world.situation("garden")
    obstruct(world, sid)
    assert world.warnings(sid) and not world.alarms()


def test_a_harmless_person_may_block_the_view(world):
    sid = world.situation("garden")
    obstruct(world, sid, persons=[GARDENER])  # the model says: moves to another area, fits the role
    assert not world.warnings() and "obscured_plausible" in world.log_kinds(sid)


def test_a_person_with_a_role_mismatch_or_suspicious_intent_may_not(world):
    world.behaviour["persons"][0]["role_mismatch"] = 0.9
    sid = world.situation("garden")
    obstruct(world, sid, persons=[GARDENER])
    assert [w.cause for w in world.warnings(sid)] == ["vision obstructed"]


def test_failing_services_and_agents_make_the_obstruction_implausible_and_are_logged(world):
    world.weather.fail = True
    world.behaviour = {}  # an invalid answer of the behavioural interpreter
    sid = world.situation("garden")
    obstruct(world, sid, persons=[GARDENER])
    assert [w.cause for w in world.warnings(sid)] == ["vision obstructed"]
    failures = [e.message for e in world.rows(LogEntry, situation_id=sid, kind="service_failure")]
    assert any("weather forecast" in m for m in failures) and any("behavioural_interpreter" in m for m in failures)
