"""Controller (Deterministic component): plain code that turns sensor events into situations and situation summaries
into warnings and alarms, following the rules of the design document.

No agent output, and in particular no free text, can trigger an alarm: the controller only reads scores and enums.
Situations (one interpreter each) are grouped in a scenario, which is resolved when all of its situations are. Every
warning, alarm and speaker signal is idempotent per scenario (persistent key `<kind>:<scenario_id>:<cause>`), so
several situations reaching the same conclusion trigger it once.

Everything the controller talks to is passed in: the situation interpreters, speaker, weather service and audio source
are small protocols (those components are separate), the agents and the communication unit are the real classes.
Calls are serialised by a lock; the database is only used in short, non-nested transactions.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Any, Callable, Literal, Protocol

from pydantic import model_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ai_alarm.agents.behavioural_interpreter import BehaviouralInterpreter, BehaviouralInterpreterRequest, PersonContext
from ai_alarm.agents.noise_interpreter import NoiseInterpreter, NoiseInterpreterRequest
from ai_alarm.agents.person_identifier import Verdict
from ai_alarm.comm import CommunicationUnit, Delivery, in_window
from ai_alarm.db.models import (
    AggregatedSummary, IdempotencyKey, LogEntry, Notification, Scenario, SensorEvent, Situation, SituationAlarm,
    SituationSummary, SituationWarning, new_id, utcnow,
)
from ai_alarm.kb import KnowledgeBase
from ai_alarm.signals import BBox, Media, Part, Score, SignalBase, SituationSignal

UNPERMITTED_ENTRY_TEXT = "You are entering without permission. Please leave the area immediately."


# ====================================================================== signals received by the controller
class SensorEventSignal(SignalBase):
    """{video_event/audio_event}: an anomaly found by a CCTV/audio processor. Exists before its situation does."""

    type: Literal["video_event", "audio_event"]
    event_id: str
    sensor_id: str
    area_id: str
    start_time: datetime
    evidence: Media
    bbox: BBox | None = None  # video only


class SituationEventSignal(SituationSignal):
    """{event}: sent to a situation interpreter, to start a situation or to add to a running one."""

    type: Literal["event"] = "event"
    sensor_event: SensorEventSignal


class PersonAssessment(Part):
    """What a situation interpreter found out about one detected person."""

    object_id: str
    identity: Verdict
    person_id: str | None = None  # the knowledge base id, for identity "known" only
    name: str | None = None  # display name from the knowledge base, for identity "known" only (may still be unset)
    cause: str | None = None  # why not decidable, incl. "unavailable"
    predicted_role: str | None = None
    suspicion: Score  # of this person (role and behaviour mismatch etc.)
    invited: bool = False  # invited or accompanied by a family member in the immediate past
    bbox: BBox | None = None  # where they are in the evidence of the event this assessment is based on

    @model_validator(mode="after")
    def _known_persons_have_an_id(self):
        if (self.identity == "known") != (self.person_id is not None):
            raise ValueError("person_id is required for, and only for, the identity 'known'")
        return self


class AnimalAssessment(Part):
    label: str
    danger: Score  # the pre-defined danger score of this kind of animal
    bbox: BBox | None = None  # where it is in the evidence of the event this assessment is based on


class SituationSummarySignal(SituationSignal):
    """{situation_summary}: the result of an interpretation. `summary` is display text; only the rest is evaluated."""

    type: Literal["situation_summary"] = "situation_summary"
    area_id: str
    event_id: str  # the sensor event this interpretation is based on, so its evidence can be shown with it
    summary: str
    threat_score: Score  # the maximum of the scores of the individual objects/events
    persons: list[PersonAssessment] = []
    animals: list[AnimalAssessment] = []
    weather_suspicion: Score = 0.0  # of a weather event that does not match the forecast
    unclear_situation: bool = False


class ObscuredSignal(SituationSignal):
    """{obscured}: the object detector reports that the view of a camera is blocked."""

    type: Literal["obscured"] = "obscured"
    area_id: str
    evidence: list[Media]  # frames
    persons: list[PersonContext] = []  # persons in view, for the plausibility check


class SituationResolvedSignal(SituationSignal):
    type: Literal["situation_resolved"] = "situation_resolved"


# ====================================================================== what the controller talks to
class SituationInterpreters(Protocol):
    def start(self, situation_id: str, event: SituationEventSignal) -> None:
        """Create the interpreter of a new situation and give it its first event."""

    def forward(self, situation_id: str, event: SituationEventSignal) -> None:
        """Give a further event to the running interpreter of the situation."""


class Speaker(Protocol):
    def speak(self, text: str) -> None: ...


class WeatherService(Protocol):
    def conditions_at(self, at: datetime) -> set[str]:
        """The forecast conditions valid at that time, e.g. {"fog"}."""


class AudioSource(Protocol):
    def latest_audio(self, area_id: str, at: datetime) -> str | None:
        """Reference to the most recent audio of the area's audio processor, if there is any."""


@dataclass(frozen=True)
class ControllerConfig:
    warning_threshold: float = 0.4  # a person that exceeds this suspicion triggers a warning ...
    alarm_threshold: float = 0.85  # ... and one that exceeds this an alarm
    familiar_threshold: float = 0.5  # a known person with at least this familiarity is familiar
    answer_timeout_s: float = 600  # after this, the fallback policy answers an open warning
    similar_alarm_window_s: float = 300  # no second alarm with the same cause and area within this time
    unpermitted_grace_s: float = 60  # an unpermitted entry that lasts this long becomes an alarm
    obstruction_prolonged_s: float = 120  # an implausible obstruction that lasts this long is reported
    default_suspicion: float = 0.5  # suspicion of a warning that does not come from a situation summary
    plausible_obstruction_weather: frozenset[str] = frozenset({"fog", "snowfall", "rain"})
    harmless_role_mismatch: float = 0.3  # a person that blocks the view is harmless up to this role mismatch ...
    suspicious_intents: frozenset[str] = frozenset({"suspicious_activity", "unknown_activity"})  # ... and unless these
    family_role: str = "family"
    dangerous_animal_threshold: float = 0.9  # danger score from which an animal is high-danger (1 = bear)
    animal_alarm_if_people_present: bool = False  # optional condition for the "dangerous animal" alarm
    animal_alarm_window: tuple[time, time] | None = None  # optional daily time frame for the same


# ====================================================================== controller
class Controller:
    def __init__(
        self, *,
        session_factory: Callable[[], Session],
        kb: KnowledgeBase,
        comm: CommunicationUnit,
        interpreters: SituationInterpreters,
        speaker: Speaker,
        weather: WeatherService,
        audio: AudioSource,
        behavioural_interpreter: BehaviouralInterpreter,
        noise_interpreter: NoiseInterpreter,
        config: ControllerConfig = ControllerConfig(),
        clock: Callable[[], datetime] = utcnow,
    ):
        self.session_factory = session_factory
        self.kb = kb
        self.comm = comm
        self.interpreters = interpreters
        self.speaker = speaker
        self.weather = weather
        self.audio = audio
        self.behavioural_interpreter = behavioural_interpreter
        self.noise_interpreter = noise_interpreter
        self.config = config
        self.clock = clock
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ sensor events
    def handle_sensor_event(self, signal: SensorEventSignal) -> str:
        """Assign the event to the running situation of its area, or start a new one in the open scenario (which is
        started if there is none). Returns the situation id."""
        with self._lock, self.session_factory() as s:
            stored = s.get(SensorEvent, signal.event_id)
            if stored is not None:  # delivered twice
                return stored.situation_id
            situation = s.scalars(select(Situation).where(
                Situation.area_id == signal.area_id, Situation.status != "resolved")).first()
            is_new = situation is None
            if is_new:
                scenario = s.scalars(select(Scenario).where(Scenario.status == "active")).first()
                if scenario is None:
                    scenario = Scenario(created_at=self.clock())
                    s.add(scenario)
                    s.flush()
                situation = Situation(scenario_id=scenario.id, area_id=signal.area_id, created_at=self.clock())
                s.add(situation)
                s.flush()
            situation_id = situation.id
            s.add(SensorEvent(
                id=signal.event_id, situation_id=situation_id, sensor_id=signal.sensor_id,
                kind=signal.type.removesuffix("_event"), start_time=signal.start_time, evidence=signal.evidence,
                bbox=signal.bbox.model_dump() if signal.bbox else None))
            s.commit()
        event = SituationEventSignal(situation_id=situation_id, sensor_event=signal)
        (self.interpreters.start if is_new else self.interpreters.forward)(situation_id, event)
        return situation_id

    def handle_situation_resolved(self, signal: SituationResolvedSignal) -> None:
        with self._lock:
            now = self.clock()
            with self.session_factory() as s:
                situation = self._situation(s, signal.situation_id)
                situation.status, situation.resolved_at = "resolved", now
                s.add(LogEntry(created_at=now, situation_id=situation.id, kind="situation_resolved",
                               message="situation resolved"))
                s.flush()
                unresolved = s.scalars(select(Situation.id).where(
                    Situation.scenario_id == situation.scenario_id, Situation.status != "resolved")).first()
                if unresolved is None:  # the scenario is resolved when its last situation is
                    scenario = s.get(Scenario, situation.scenario_id)
                    scenario.status, scenario.resolved_at = "resolved", now
                area_id = situation.area_id
                s.commit()
            self.kb.close_presences(area_id, now)

    # ------------------------------------------------------------------ situation summaries
    def handle_situation_summary(self, signal: SituationSummarySignal) -> None:
        with self._lock:
            now = self.clock()
            combined = self._store_summary(signal, now)
            stays = self._record_persons(signal, combined, now)
            self._check_suspicion(signal, combined)
            self._check_permissions(signal, combined, stays, now)
            self._check_animals(signal, now)
            if signal.unclear_situation and signal.threat_score >= self.config.warning_threshold:
                self._warn(signal.situation_id, signal.area_id, "unclear situation", signal.threat_score)

    def _store_summary(self, signal: SituationSummarySignal, now: datetime) -> dict[str, float]:
        """Log the summary and the aggregated summary of the scenario: the latest summaries of its unresolved
        situations. Returns the combined suspicion of each detected person: the highest one in those summaries."""
        with self.session_factory() as s:
            scenario_id = self._situation(s, signal.situation_id).scenario_id
            data = signal.model_dump(mode="json")
            s.add(SituationSummary(situation_id=signal.situation_id, threat_score=signal.threat_score,
                                   summary=signal.summary, data=data, created_at=now))
            s.add(LogEntry(created_at=now, situation_id=signal.situation_id, kind="situation_summary",
                           message=signal.summary, data=data))
            s.flush()
            latest = self._latest_summaries(s, scenario_id)
            combined: dict[str, float] = {}
            for summary in latest:
                for p in summary.persons:
                    key = self._person_key(summary.situation_id, p)
                    combined[key] = max(combined.get(key, 0.0), p.suspicion)
            s.add(AggregatedSummary(
                scenario_id=scenario_id, created_at=now,
                threat_score=max((x.threat_score for x in latest), default=0.0),
                summary=" | ".join(x.summary for x in latest),
                data={"situations": {x.situation_id: x.threat_score for x in latest}, "persons": combined}))
            s.commit()
            return combined

    @staticmethod
    def _latest_summaries(s: Session, scenario_id: str) -> list[SituationSummarySignal]:
        """The most recent summary of each unresolved situation of the scenario."""
        newest = select(func.max(SituationSummary.id)).group_by(SituationSummary.situation_id)
        rows = s.scalars(select(SituationSummary).join(Situation).where(
            SituationSummary.id.in_(newest), Situation.scenario_id == scenario_id, Situation.status != "resolved"))
        return [SituationSummarySignal.model_validate(r.data) for r in rows]

    @staticmethod
    def _person_key(situation_id: str, person: PersonAssessment) -> str:
        """The knowledge base id of a known person; persons that are not known only exist within their situation."""
        return person.person_id or f"unk_{situation_id}_{person.object_id}"

    def _record_persons(
        self, signal: SituationSummarySignal, combined: dict[str, float], now: datetime
    ) -> dict[str, float]:
        """Store the detected persons in the knowledge base, unless they are not decidable (other than for "low
        confidence"). Returns for each person, by object id, for how long they have been in the area."""
        with self.session_factory() as s:
            started = self._situation(s, signal.situation_id).created_at
        stays = {}
        for p in signal.persons:
            key = self._person_key(signal.situation_id, p)
            if p.identity != "not_decidable" or p.cause == "low confidence":
                predicted = None if p.person_id else p.predicted_role
                entered = self.kb.record_detection(key, signal.area_id, combined[key], now, predicted)
            else:
                entered = started
            stays[p.object_id] = (now - entered).total_seconds()
        return stays

    # ------------------------------------------------------------------ rules for a summary
    def _check_suspicion(self, signal: SituationSummarySignal, combined: dict[str, float]) -> None:
        """A person, or the weather, that exceeds the suspicion thresholds triggers a warning or an alarm."""
        top_person = max((combined[self._person_key(signal.situation_id, p)] for p in signal.persons), default=0.0)
        for suspicion, cause in ((top_person, "suspicious person"), (signal.weather_suspicion, "suspicious weather")):
            if suspicion > self.config.alarm_threshold:
                self._alarm(signal.situation_id, signal.area_id, "strong suspicion", "rule")
            elif suspicion > self.config.warning_threshold:
                self._warn(signal.situation_id, signal.area_id, cause, signal.threat_score)

    def _check_permissions(
        self, signal: SituationSummarySignal, combined: dict[str, float], stays: dict[str, float], now: datetime
    ) -> None:
        sid, area_id = signal.situation_id, signal.area_id
        for p in signal.persons:
            info = self.kb.person(p.person_id) if p.person_id else None
            roles = list(info.roles) if info else [p.predicted_role] if p.predicted_role else []
            familiar = info is not None and info.familiarity >= self.config.familiar_threshold
            if self.kb.entry_permitted(area_id, roles, info is not None, familiar, p.invited, stays[p.object_id]):
                continue
            if combined[self._person_key(sid, p)] > self.config.alarm_threshold:
                # already alarmed by _check_suspicion: a redundant "please leave or we'll call the police" warning
                # asking a human to decide doesn't add up when that decision has already been made for them. The
                # audible warning still makes sense on its own, though, so it is not skipped.
                self._speak(sid, "unpermitted entry", UNPERMITTED_ENTRY_TEXT)
                continue
            self._warn(sid, area_id, "unpermitted entry", signal.threat_score, speak=UNPERMITTED_ENTRY_TEXT)
            if self._entry_is_prolonged(sid, now):
                self._alarm(sid, area_id, "unpermitted entry", "rule")

    def _entry_is_prolonged(self, situation_id: str, now: datetime) -> bool:
        """The unpermitted entry was warned about long ago, and the warning was not dismissed as harmless."""
        with self.session_factory() as s:
            scenario_id = self._situation(s, situation_id).scenario_id
            warning = s.scalars(select(SituationWarning).join(Situation).where(
                Situation.scenario_id == scenario_id, SituationWarning.cause == "unpermitted entry")).first()
            return (warning is not None and warning.status != "dismissed"
                    and (now - warning.created_at).total_seconds() >= self.config.unpermitted_grace_s)

    def _check_animals(self, signal: SituationSummarySignal, now: datetime) -> None:
        if not any(a.danger >= self.config.dangerous_animal_threshold for a in signal.animals):
            return
        # an animal at the very top of the danger scale (danger 1.0, e.g. a bear) is too dangerous to sit as a
        # warning under any circumstances -- asking a human to decide would be moot. Anything merely dangerous
        # (at or above dangerous_animal_threshold but below the maximum) still goes through the door/people/time
        # context below, same as before.
        if any(a.danger >= 1.0 for a in signal.animals):
            self._alarm(signal.situation_id, signal.area_id, "dangerous animal", "rule")
            return
        window = self.config.animal_alarm_window
        if (self.kb.has_open_entry_point()
                or (self.config.animal_alarm_if_people_present and self.kb.people_present())
                or (window and in_window(now.astimezone(self.comm.config.tz).time(), *window))):
            self._alarm(signal.situation_id, signal.area_id, "dangerous animal", "rule")
        else:
            self._warn(signal.situation_id, signal.area_id, "dangerous animal", signal.threat_score)

    # ------------------------------------------------------------------ obscured view
    def handle_obscured(self, signal: ObscuredSignal) -> None:
        """A camera view is blocked. Plausible causes (weather, a harmless person) are only logged; an implausible
        obstruction that lasts is a warning, and if a person seems to be the cause and no family member is around,
        an alarm."""
        with self._lock:
            now = self.clock()
            sid, area_id = signal.situation_id, signal.area_id
            first = self._note_obscured(sid, now)
            if self._plausible_by_weather(sid, now):
                self._log(sid, "obscured_plausible", "the weather explains the obstruction")
                return
            if self._plausible_by_person(signal):
                self._log(sid, "obscured_plausible", "a harmless person explains the obstruction")
                return
            if (now - first).total_seconds() < self.config.obstruction_prolonged_s:
                return
            self._warn(sid, area_id, "vision obstructed", self._threat(sid))

            audio = self._call(sid, "audio source", self.audio.latest_audio, area_id, now)
            if audio is None:
                return
            noise = self._ask(sid, self.noise_interpreter, NoiseInterpreterRequest(
                situation_id=sid, evidence=audio, area_id=area_id))
            family_around = any(self.config.family_role in p.roles for p in self.kb.people_present(area_id))
            if noise is not None and noise.category == "human_activity" and not family_around:
                self._alarm(sid, area_id, "obstruction by human activity", "rule")

    def _note_obscured(self, situation_id: str, now: datetime) -> datetime:
        """Log the signal. Returns when the situation was first obscured."""
        with self.session_factory() as s:
            s.add(LogEntry(created_at=now, situation_id=situation_id, kind="obscured", message="view obscured"))
            s.commit()
            return s.scalars(select(func.min(LogEntry.created_at)).where(
                LogEntry.situation_id == situation_id, LogEntry.kind == "obscured")).one()

    def _plausible_by_weather(self, situation_id: str, now: datetime) -> bool:
        conditions = self._call(situation_id, "weather forecast", self.weather.conditions_at, now)
        return bool(conditions and set(conditions) & self.config.plausible_obstruction_weather)

    def _plausible_by_person(self, signal: ObscuredSignal) -> bool:
        if not signal.persons:
            return False
        response = self._ask(signal.situation_id, self.behavioural_interpreter, BehaviouralInterpreterRequest(
            situation_id=signal.situation_id, evidence=signal.evidence, persons=signal.persons, area_id=signal.area_id))
        return response is not None and all(
            p.role_mismatch <= self.config.harmless_role_mismatch
            and p.intents[0].intent not in self.config.suspicious_intents
            for p in response.persons)

    # ------------------------------------------------------------------ answers to warnings
    def elevate(self, warning_id: str) -> None:
        """The user elevates a warning to an alarm."""
        self._answer(warning_id, "elevated", "user")

    def dismiss(self, warning_id: str) -> None:
        self._answer(warning_id, "dismissed", "user")

    def check_timeouts(self) -> None:
        """Apply the fallback policy to warnings that were not answered in time. Call this periodically."""
        with self.session_factory() as s:
            due = list(s.scalars(select(SituationWarning.id).where(
                SituationWarning.status == "open", SituationWarning.answer_deadline <= self.clock())))
        for warning_id in due:
            self._fallback(warning_id)

    def _fallback(self, warning_id: str) -> None:
        """Yellow warnings are dismissed, orange ones elevated."""
        with self.session_factory() as s:
            colour = s.get(SituationWarning, warning_id).colour
        self._answer(warning_id, "elevated" if colour == "orange" else "dismissed", "fallback_policy")

    def _answer(self, warning_id: str, status: str, resolved_by: str) -> None:
        """Answer an open warning. Answering a warning that is answered already changes nothing."""
        with self._lock:
            now = self.clock()
            with self.session_factory() as s:
                warning = s.get(SituationWarning, warning_id)
                if warning is None:
                    raise LookupError(f"unknown warning {warning_id}")
                if warning.status != "open":  # answered already, e.g. by the timeout
                    return
                warning.status, warning.resolved_by, warning.resolved_at = status, resolved_by, now
                s.add(LogEntry(created_at=now, situation_id=warning.situation_id, kind=f"warning_{status}",
                               message=f"{warning.cause}: {status} ({resolved_by})", data={"warning_id": warning.id}))
                situation_id, area_id, cause = warning.situation_id, warning.area_id, warning.cause
                s.commit()
            if status == "elevated":
                origin = "user_elevation" if resolved_by == "user" else "fallback_policy"
                self._alarm(situation_id, area_id, cause, origin, warning_id)

    # ------------------------------------------------------------------ warnings, alarms, speaker (idempotent)
    def _warn(self, situation_id: str, area_id: str, cause: str, suspicion: float, speak: str | None = None) -> str:
        """Warn the contacts once per scenario and cause. Returns the id of the warning."""
        if speak:
            self._speak(situation_id, cause, speak)
        key = self._key("warning", situation_id, cause)
        now = self.clock()
        with self.session_factory() as s:
            prior = s.get(IdempotencyKey, key)
            if prior is not None:
                return prior.result["warning_id"]

        warning_id, colour = new_id("wrn")(), self.comm.colour(suspicion, now)
        deliveries = self.comm.send_warning(warning_id, cause, area_id, suspicion, colour)
        with self.session_factory() as s:
            s.add(SituationWarning(
                id=warning_id, situation_id=situation_id, area_id=area_id, cause=cause, suspicion=suspicion,
                colour=colour, created_at=now, answer_deadline=now + timedelta(seconds=self.config.answer_timeout_s)))
            s.flush()
            s.add_all(self._notifications(deliveries, now, warning_id=warning_id))
            s.add(IdempotencyKey(key=key, kind="warning", result={"warning_id": warning_id}, created_at=now))
            s.add(self._log_entry(s, now, situation_id, "warning", f"{cause} ({colour})", {
                "warning_id": warning_id, "area_id": area_id, "cause": cause, "suspicion": suspicion,
                "colour": colour, "delivered_to": [d.contact_id for d in deliveries if d.sent]}))
            s.commit()
        if not any(d.sent for d in deliveries):  # nobody could be reached: the fallback policy applies right away
            self._fallback(warning_id)
        return warning_id

    def _alarm(self, situation_id: str, area_id: str, cause: str, origin: str, warning_id: str | None = None) -> str:
        """Trigger the alarm once per scenario and cause, and not if a similar one was raised just before."""
        key = self._key("alarm", situation_id, cause)
        now = self.clock()
        with self.session_factory() as s:
            prior = s.get(IdempotencyKey, key)
            if prior is not None:
                return prior.result["alarm_id"]
            similar = s.scalars(select(SituationAlarm).where(
                SituationAlarm.cause == cause, SituationAlarm.area_id == area_id,
                SituationAlarm.created_at >= now - timedelta(seconds=self.config.similar_alarm_window_s))).first()
            if similar is not None:  # no flooding for distinct situations
                s.add(IdempotencyKey(key=key, kind="alarm", result={"alarm_id": similar.id, "suppressed": True},
                                     created_at=now))
                s.add(self._log_entry(s, now, situation_id, "alarm_suppressed",
                                      f"{cause}: similar alarm {similar.id} was raised recently"))
                s.commit()
                return similar.id

        alarm_id = new_id("alm")()
        deliveries = self.comm.send_alarm(alarm_id, cause, area_id)
        with self.session_factory() as s:
            s.add(SituationAlarm(id=alarm_id, situation_id=situation_id, area_id=area_id, cause=cause, origin=origin,
                                 warning_id=warning_id, created_at=now))
            s.flush()
            s.add_all(self._notifications(deliveries, now, alarm_id=alarm_id))
            s.add(IdempotencyKey(key=key, kind="alarm", result={"alarm_id": alarm_id}, created_at=now))
            s.add(self._log_entry(s, now, situation_id, "alarm", f"{cause} ({origin})", {
                "alarm_id": alarm_id, "area_id": area_id, "cause": cause, "origin": origin,
                "delivered_to": [d.contact_id for d in deliveries if d.sent]}))
            s.commit()
        return alarm_id

    def _speak(self, situation_id: str, cause: str, text: str) -> None:
        key = self._key("speak", situation_id, cause)
        with self.session_factory() as s:
            if s.get(IdempotencyKey, key) is not None:
                return
        self.speaker.speak(text)
        now = self.clock()
        with self.session_factory() as s:
            s.add(IdempotencyKey(key=key, kind="speak", result={"text": text}, created_at=now))
            s.add(self._log_entry(s, now, situation_id, "speak", text))
            s.commit()

    @staticmethod
    def _notifications(deliveries: list[Delivery], now: datetime, **subject: str) -> list[Notification]:
        return [Notification(contact_id=d.contact_id, message=d.message, status="sent" if d.sent else "failed",
                             created_at=now, **subject) for d in deliveries]

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _situation(s: Session, situation_id: str) -> Situation:
        situation = s.get(Situation, situation_id)
        if situation is None:
            raise LookupError(f"unknown situation {situation_id}")
        return situation

    def _key(self, kind: str, situation_id: str, cause: str) -> str:
        """The idempotency key: one warning/alarm/speaker signal per cause and scenario."""
        with self.session_factory() as s:
            return f"{kind}:{self._situation(s, situation_id).scenario_id}:{cause}"

    def _threat(self, situation_id: str) -> float:
        """The threat score of the latest summary of the situation."""
        with self.session_factory() as s:
            score = s.scalars(select(SituationSummary.threat_score).where(
                SituationSummary.situation_id == situation_id).order_by(SituationSummary.id.desc())).first()
        return self.config.default_suspicion if score is None else score

    @staticmethod
    def _log_entry(s: Session, now: datetime, situation_id: str, kind: str, message: str,
                   data: dict[str, Any] | None = None) -> LogEntry:
        """A log entry with the data it is based on: the details given and the situation's latest summary."""
        latest = s.scalars(select(SituationSummary.data).where(
            SituationSummary.situation_id == situation_id).order_by(SituationSummary.id.desc())).first()
        return LogEntry(created_at=now, situation_id=situation_id, kind=kind, message=message,
                        data={**(data or {}), "situation_summary": latest})

    def _log(self, situation_id: str, kind: str, message: str) -> None:
        with self.session_factory() as s:
            s.add(LogEntry(created_at=self.clock(), situation_id=situation_id, kind=kind, message=message))
            s.commit()

    def _call(self, situation_id: str, what: str, fn: Callable[..., Any], *args: Any) -> Any:
        """Call an external service; a failure is logged and treated as "no answer"."""
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001
            self._log(situation_id, "service_failure", f"{what}: {e}")
            return None

    def _ask(self, situation_id: str, agent, request):
        """Ask an agent; an unusable or missing answer is logged and treated as "no answer"."""
        return self._call(situation_id, agent.name, agent.handle, request)
