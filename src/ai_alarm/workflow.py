"""Situation Interpreter (Workflow): the LangGraph graph, with one checkpointed thread per situation.

Between two runs a situation rests in `idle` or `observe`; the checkpoint keeps its state. A run is triggered by
`start`/`forward` (a sensor event) or `tick` (time passed), both of which lead to `_advance`:

    intake -> detect -> people | animals | weather (in parallel, as the detection requires) -> merge -> report
           -> observe
    intake -> resolve   (in `observe`, when nothing happened for a while)

Events are never held back: a warning that waits for the user does not pause anything, and every event is
interpreted and reported to the controller, which does not repeat a warning, alarm or speaker signal for the same
cause. New events are read from the database, so none is lost. After a crash the run is continued from its last
checkpoint; what is repeated only re-runs analyses, and the side effects are protected by the controller's
idempotency keys. Agents and services that do not answer in time, or answer unusably, are "not available": the
situation is then unclear, not silently fine.
"""
from __future__ import annotations

import logging
import queue
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from sqlalchemy import select
from sqlalchemy.orm import Session

from ai_alarm.agents import (
    BehaviouralInterpreter, NoiseInterpreter, ObjectDetector, PersonIdentifier, WeatherInterpreter,
)
from ai_alarm.agents.behavioural_interpreter import BehaviouralInterpreterRequest, PersonContext
from ai_alarm.agents.noise_interpreter import NoiseInterpreterRequest
from ai_alarm.agents.object_detector import ObjectDetectorRequest
from ai_alarm.agents.person_identifier import PersonIdentifierRequest
from ai_alarm.agents.weather_interpreter import WeatherInterpreterRequest
from ai_alarm.controller import (
    AnimalAssessment, Controller, ObscuredSignal, PersonAssessment, SituationEventSignal, SituationResolvedSignal,
    SituationSummarySignal,
)
from ai_alarm.db.models import LogEntry, SensorEvent, Situation, utcnow
from ai_alarm.kb import KnowledgeBase
from ai_alarm.weather import WeatherForecastFetcher

log = logging.getLogger(__name__)

ANIMAL_DANGER = {"bear": 1.0, "boar": 0.8, "large dog": 0.6, "small dog": 0.4, "fox": 0.2}
SIGNIFICANT_WEATHER = {"high_wind", "snowfall", "fog", "rain"}


def animal_danger(label: str) -> float:
    """The pre-defined danger score of a kind of animal. Animals the system does not know are not dangerous."""
    return next((score for kind, score in ANIMAL_DANGER.items() if kind in label.lower()), 0.0)


@dataclass(frozen=True)
class WorkflowConfig:
    agent_timeout_s: float = 30  # an agent or service that does not answer within this time is not available
    unclear_score: float = 0.5  # threat score of an unclear situation
    resolve_after_s: float = 60  # a situation in `observe` without news for this long is resolved
    role_mismatch_score: float = 0.6  # suspicion of a known person whose predicted role is not one of their roles
    weather_mismatch_score: float = 0.5  # suspicion of weather that does not match the forecast
    wind_tolerance_kmh: float = 30
    min_condition_confidence: float = 0.5
    family_role: str = "family"


class State(TypedDict, total=False):
    situation_id: str
    area_id: str
    phase: str  # "idle" | "interpret" | "observe"
    seen: list[str]  # ids of the sensor events that were interpreted
    event: dict[str, Any]  # the event that is interpreted: id, kind, evidence, bbox
    detection: dict[str, Any]  # class ("person" ...), objects, obscured
    persons: list[dict[str, Any]]  # PersonAssessment
    contexts: list[dict[str, Any]]  # PersonContext, for the obscured check
    animals: list[dict[str, Any]]  # AnimalAssessment
    weather: dict[str, Any]  # score and text
    health_emergency: bool
    summary: dict[str, Any]  # text, threat_score, unclear
    last_activity: str


def make_checkpointer(path: str) -> SqliteSaver:
    """The checkpoint store: a SQLite file of its own (":memory:" for tests)."""
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


class SituationWorkflows:
    """All situation interpreters. Implements the `SituationInterpreters` protocol of the controller.

    The controller and this class need each other: create this first and set `controller` afterwards.
    """

    controller: Controller

    def __init__(
        self, *,
        session_factory: Callable[[], Session],
        kb: KnowledgeBase,
        object_detector: ObjectDetector,
        person_identifier: PersonIdentifier,
        behavioural_interpreter: BehaviouralInterpreter,
        noise_interpreter: NoiseInterpreter,
        weather_interpreter: WeatherInterpreter,
        forecast: WeatherForecastFetcher,
        checkpointer: SqliteSaver,
        config: WorkflowConfig = WorkflowConfig(),
        clock: Callable[[], datetime] = utcnow,
    ):
        self.session_factory = session_factory
        self.kb = kb
        self.object_detector = object_detector
        self.person_identifier = person_identifier
        self.behavioural_interpreter = behavioural_interpreter
        self.noise_interpreter = noise_interpreter
        self.weather_interpreter = weather_interpreter
        self.forecast = forecast
        self.config = config
        self.clock = clock
        self.graph = self._build().compile(checkpointer=checkpointer)
        self._executor = ThreadPoolExecutor(max_workers=8)  # to put a time limit on agents and services
        self._jobs: queue.Queue[str] = queue.Queue()
        self._queued: set[str] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ triggers (SituationInterpreters)
    def start(self, situation_id: str, event: SituationEventSignal) -> None:
        self._enqueue(situation_id)

    def forward(self, situation_id: str, event: SituationEventSignal) -> None:
        self._enqueue(situation_id)

    def tick(self) -> None:
        """Time passed: look at every situation that is not resolved. Call this periodically."""
        with self.session_factory() as s:
            unresolved = list(s.scalars(select(Situation.id).where(Situation.status != "resolved")))
        for situation_id in unresolved:
            self._enqueue(situation_id)

    # ------------------------------------------------------------------ running
    def run_worker(self) -> None:
        """Process triggers one after the other, forever. Run this in a thread."""
        while True:
            self._process_next(block=True)

    def drain(self) -> None:
        """Process the queued triggers now, in the calling thread."""
        while self._process_next(block=False):
            pass

    def wait_idle(self) -> None:
        """Wait until the worker has processed all triggers."""
        self._jobs.join()

    def _enqueue(self, situation_id: str) -> None:
        with self._lock:
            if situation_id in self._queued:  # a run is due anyway
                return
            self._queued.add(situation_id)
        self._jobs.put(situation_id)

    def _process_next(self, block: bool) -> bool:
        try:
            situation_id = self._jobs.get(block=block)
        except queue.Empty:
            return False
        with self._lock:
            self._queued.discard(situation_id)
        try:
            self._advance(situation_id)
        except Exception as e:  # noqa: BLE001 - the next trigger continues from the last checkpoint
            log.exception("situation %s: workflow failed", situation_id)
            self._log(situation_id, "workflow_failure", repr(e))
        finally:
            self._jobs.task_done()
        return True

    def _advance(self, situation_id: str) -> None:
        with self.session_factory() as s:
            situation = s.get(Situation, situation_id)
            if situation is None or situation.status == "resolved":
                return
            area_id = situation.area_id
        config = {"configurable": {"thread_id": situation_id}}
        if self.graph.get_state(config).next:
            self.graph.invoke(None, config)  # a run that was cut short: continue it
        else:
            self.graph.invoke({"situation_id": situation_id, "area_id": area_id}, config)

    # ------------------------------------------------------------------ the graph
    def _build(self) -> StateGraph:
        g = StateGraph(State)
        for node in ("intake", "detect", "people", "animals", "weather", "merge", "report", "observe", "resolve"):
            g.add_node(node, getattr(self, f"_{node}"))
        g.add_edge(START, "intake")
        g.add_conditional_edges("intake", self._after_intake, {"detect": "detect", "resolve": "resolve", "end": END})
        g.add_conditional_edges("detect", self._after_detect, ["people", "animals", "weather", "merge"])
        for branch in ("people", "animals", "weather"):
            g.add_edge(branch, "merge")
        g.add_edge("merge", "report")
        g.add_edge("report", "observe")
        g.add_conditional_edges("observe", self._after_observe, {"intake": "intake", "end": END})
        g.add_edge("resolve", END)
        return g

    def _intake(self, state: State) -> dict[str, Any]:
        seen = state.get("seen", [])
        unseen = self._unseen_events(state["situation_id"], seen)
        if not unseen:
            return {"phase": state.get("phase", "idle")}
        # One event per run, oldest first: `observe` comes back here until all of them are interpreted.
        return {"phase": "interpret", "seen": seen + [unseen[0]["id"]], "event": unseen[0],
                "detection": None, "persons": [], "contexts": [], "animals": [], "weather": None,
                "health_emergency": False}

    def _after_intake(self, state: State) -> str:
        if state["phase"] == "interpret":
            return "detect"
        if state["phase"] == "observe" and self._quiet_for(state) >= self.config.resolve_after_s:
            return "resolve"
        return "end"

    def _detect(self, state: State) -> dict[str, Any]:
        sid, area_id, event = state["situation_id"], state["area_id"], state["event"]
        if event["kind"] == "video":
            found = self._ask(sid, self.object_detector, ObjectDetectorRequest(
                situation_id=sid, evidence=event["evidence"], area_id=area_id, bbox_hint=event["bbox"]))
            if found is None:
                return {"detection": {"class": "unclear", "objects": [], "obscured": False}}
            anomaly = {"weather_environment": "weather", "sensor_artefact": "artefact"}.get(
                found.anomaly_class, found.anomaly_class)
            return {"detection": {"class": anomaly, "obscured": found.obscured,
                                  "objects": [o.model_dump(mode="json") for o in found.objects]}}

        noise = self._ask(sid, self.noise_interpreter, NoiseInterpreterRequest(
            situation_id=sid, evidence=event["evidence"], area_id=area_id))
        anomaly = "unclear" if noise is None else {
            "weather": "weather", "animal": "animal", "technical_noise": "artefact"}.get(noise.category, "unclear")
        # Audio alone cannot say who is there: human activity stays unclear. An animal is one that is not known.
        objects = [{"object_id": "a1", "kind": "animal", "label": "unknown animal"}] if anomaly == "animal" else []
        return {"detection": {"class": anomaly, "objects": objects, "obscured": False}}

    @staticmethod
    def _after_detect(state: State) -> list[str]:
        detection = state["detection"]
        kinds = {o["kind"] for o in detection["objects"]}
        branches = [b for b, wanted in (("people", "person" in kinds), ("animals", "animal" in kinds),
                                        ("weather", detection["class"] == "weather")) if wanted]
        return branches or ["merge"]

    def _people(self, state: State) -> dict[str, Any]:
        """Identify each person, then interpret their behaviour with the identity and the roles as context."""
        sid, area_id, event = state["situation_id"], state["area_id"], state["event"]
        found = []
        for o in (o for o in state["detection"]["objects"] if o["kind"] == "person"):
            answer = self._ask(sid, self.person_identifier, PersonIdentifierRequest(
                situation_id=sid, video=event["evidence"], object_id=o["object_id"], bbox=o["bbox"]))
            if answer is None:  # not decidable, and not suspicious by itself; treated as not familiar
                identity, person_id, cause = "not_decidable", None, "unavailable"
            else:
                identity, person_id, cause = answer.verdict, answer.person, answer.cause
            info = self.kb.person(person_id) if person_id else None
            found.append({"object": o, "identity": identity, "person_id": person_id, "cause": cause,
                          "kb_roles": list(info.roles) if info else []})

        contexts = [PersonContext(object_id=f["object"]["object_id"], verdict=f["identity"], kb_roles=f["kb_roles"],
                                  predicted_role=f["object"].get("role")) for f in found]
        behaviour = self._ask(sid, self.behavioural_interpreter, BehaviouralInterpreterRequest(
            situation_id=sid, evidence=[event["evidence"]], persons=contexts, area_id=area_id))
        by_object = {p.object_id: p for p in behaviour.persons} if behaviour else {}  # unavailable: unknown behaviour
        family = {f["object"]["object_id"] for f in found if self.config.family_role in f["kb_roles"]}
        invited = {r.object for r in behaviour.relations if r.subject in family} if behaviour else set()

        assessments = []
        for f in found:
            object_id, predicted = f["object"]["object_id"], f["object"].get("role")
            # A notable difference between the roles on file and the predicted role is suspicious ...
            role_part = self.config.role_mismatch_score if (
                f["kb_roles"] and predicted and predicted not in f["kb_roles"]) else 0.0
            # ... and so is behaviour that does not fit the role, in proportion to the mismatch.
            behaviour_part = 0.0
            if object_id in by_object:
                assessed = by_object[object_id]
                behaviour_part = max([assessed.role_mismatch] + [
                    i.confidence for i in assessed.intents if i.intent == "suspicious_activity"])
            assessments.append(PersonAssessment(
                object_id=object_id, identity=f["identity"], person_id=f["person_id"], cause=f["cause"],
                predicted_role=predicted, suspicion=max(role_part, behaviour_part), invited=object_id in invited))
        return {"persons": [a.model_dump(mode="json") for a in assessments],
                "contexts": [c.model_dump(mode="json") for c in contexts],
                "health_emergency": bool(behaviour and behaviour.health_emergency)}

    @staticmethod
    def _animals(state: State) -> dict[str, Any]:
        animals = [AnimalAssessment(label=o["label"], danger=animal_danger(o["label"]))
                   for o in state["detection"]["objects"] if o["kind"] == "animal"]
        return {"animals": [a.model_dump(mode="json") for a in animals]}

    def _weather(self, state: State) -> dict[str, Any]:
        """Compare the observed weather with the forecast. If either is not available: unknown, not suspicious."""
        sid, area_id, event = state["situation_id"], state["area_id"], state["event"]
        observed = self._ask(sid, self.weather_interpreter, WeatherInterpreterRequest(
            situation_id=sid, evidence=event["evidence"], area_id=area_id))
        forecast = self._call(sid, "weather forecast", self.forecast.get_forecast, self.clock())
        if observed is None or forecast is None or not forecast.entries:
            return {"weather": {"score": 0.0, "text": "weather event (unknown: no forecast comparison possible)"}}

        forecast_conditions = {c for e in forecast.entries for c in e.conditions}
        forecast_wind = max(e.wind_speed_kmh for e in forecast.entries)
        for o in observed.conditions:
            if o.confidence < self.config.min_condition_confidence or o.condition not in SIGNIFICANT_WEATHER:
                continue
            if o.condition not in forecast_conditions:
                return self._weather_mismatch(f"{o.condition} that was not forecast")
            if o.wind_speed_kmh is not None and abs(o.wind_speed_kmh - forecast_wind) > self.config.wind_tolerance_kmh:
                return self._weather_mismatch(f"wind of {o.wind_speed_kmh:.0f} km/h, {forecast_wind:.0f} forecast")
        return {"weather": {"score": 0.0, "text": "weather event as forecast"}}

    def _weather_mismatch(self, what: str) -> dict[str, Any]:
        return {"weather": {"score": self.config.weather_mismatch_score, "text": f"weather event: {what}"}}

    def _merge(self, state: State) -> dict[str, Any]:
        """The threat of the situation is the highest score of its objects and events."""
        detection, persons, animals = state["detection"], state["persons"], state["animals"]
        weather = state.get("weather") or {}
        unclear = detection["class"] == "unclear"  # includes: the object detector is not available
        threat = max([p["suspicion"] for p in persons] + [a["danger"] for a in animals]
                     + [weather.get("score", 0.0)] + [self.config.unclear_score if unclear else 0.0])

        parts = [self._describe_person(p) for p in persons]
        parts += [f"{a['label']} (danger {a['danger']:.2f})" for a in animals]
        parts += [weather["text"]] if weather else []
        parts += ["sensor artefact, ignored"] if detection["class"] == "artefact" else []
        parts += ["unclear anomaly"] if unclear else []
        parts += ["camera view obscured"] if detection["obscured"] else []
        parts += ["possible health emergency"] if state.get("health_emergency") else []
        area = self.kb.area_info(state["area_id"])["name"]
        return {"summary": {"text": f"{area}: {'; '.join(parts) or 'nothing found'}.", "threat_score": threat,
                            "unclear": unclear}}

    def _describe_person(self, p: dict[str, Any]) -> str:
        if p["identity"] == "known":
            who = self.kb.person(p["person_id"]).name or p["person_id"]
        elif p["identity"] == "unknown":
            who = "unknown person"
        else:
            who = f"person who cannot be identified ({p['cause']})"
        return f"{who} (suspicion {p['suspicion']:.2f})"

    def _report(self, state: State) -> dict[str, Any]:
        """Tell the controller. (Repeated after a crash: the controller's idempotency keys protect the side effects.)"""
        sid, area_id, summary = state["situation_id"], state["area_id"], state["summary"]
        self.controller.handle_situation_summary(SituationSummarySignal(
            situation_id=sid, area_id=area_id, summary=summary["text"], threat_score=summary["threat_score"],
            persons=[PersonAssessment.model_validate(p) for p in state["persons"]],
            animals=[AnimalAssessment.model_validate(a) for a in state["animals"]],
            weather_suspicion=(state.get("weather") or {}).get("score", 0.0), unclear_situation=summary["unclear"]))
        if state["detection"]["obscured"]:
            self.controller.handle_obscured(ObscuredSignal(
                situation_id=sid, area_id=area_id, evidence=[state["event"]["evidence"]],
                persons=[PersonContext.model_validate(c) for c in state["contexts"]]))
        return {}

    def _observe(self, state: State) -> dict[str, Any]:
        return {"phase": "observe", "last_activity": self.clock().isoformat()}

    def _after_observe(self, state: State) -> str:
        return "intake" if self._unseen_events(state["situation_id"], state["seen"]) else "end"

    def _resolve(self, state: State) -> dict[str, Any]:
        self.controller.handle_situation_resolved(SituationResolvedSignal(situation_id=state["situation_id"]))
        return {"phase": "idle"}

    # ------------------------------------------------------------------ helpers
    def _quiet_for(self, state: State) -> float:
        last = state.get("last_activity")
        return (self.clock() - datetime.fromisoformat(last)).total_seconds() if last else 0.0

    def _unseen_events(self, situation_id: str, seen: list[str]) -> list[dict[str, Any]]:
        with self.session_factory() as s:
            rows = s.scalars(select(SensorEvent).where(SensorEvent.situation_id == situation_id).order_by(
                SensorEvent.start_time, SensorEvent.id))
            return [{"id": e.id, "kind": e.kind, "evidence": e.evidence, "bbox": e.bbox}
                    for e in rows if e.id not in seen]

    def _call(self, situation_id: str, what: str, fn: Callable[..., Any], *args: Any) -> Any:
        """Call an agent or service with a time limit. Failing, timing out or answering unusably = not available."""
        try:
            return self._executor.submit(fn, *args).result(timeout=self.config.agent_timeout_s)
        except Exception as e:  # noqa: BLE001
            self._log(situation_id, "agent_failure", f"{what}: {e!r}")
            return None

    def _ask(self, situation_id: str, agent, request):
        return self._call(situation_id, agent.name, agent.handle, request)

    def _log(self, situation_id: str, kind: str, message: str) -> None:
        with self.session_factory() as s:
            s.add(LogEntry(created_at=self.clock(), situation_id=situation_id, kind=kind, message=message))
            s.commit()
