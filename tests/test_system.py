"""The whole system (simulated sensors, mock agents, workflow, controller, database) and the situation workflow."""
from __future__ import annotations

import time
from dataclasses import replace
from datetime import timedelta

import pytest
from sqlalchemy import select
from world import T0, Clock

from ai_alarm.controller import SensorEventSignal, SituationSummarySignal
from ai_alarm.db.models import (
    Area, LogEntry, Scenario, SensorEvent, Situation, SituationAlarm, SituationSummary, SituationWarning,
)
from ai_alarm.demos import BOX, DEMOS, Demo, ScriptedEvent, _behaviour, _detected, _obj, _person
from ai_alarm.system import build_system
from ai_alarm.workflow import WorkflowConfig, animal_danger


class Sim:
    """A complete system in a temporary directory, with a clock the test controls."""

    def __init__(self, tmp_path, **kwargs):
        self.clock = Clock(T0)
        self.system = build_system(f"sqlite:///{tmp_path / 'system.db'}", str(tmp_path / "checkpoints.db"),
                                   clock=self.clock, **kwargs)
        self.workflows, self.controller = self.system.workflows, self.system.controller

    def play(self, demo: str | Demo) -> None:
        """Let the sensors report the demo and let the workflows settle."""
        if isinstance(demo, Demo):
            self.system.processor.demos[demo.name] = demo
            demo = demo.name
        self.system.play(demo, time_scale=0, wait=True)
        self.settle()

    def settle(self) -> None:
        self.workflows.drain()

    def tick(self) -> None:
        self.workflows.tick()
        self.settle()

    def rows(self, model, **where):
        with self.system.session_factory() as s:
            return list(s.scalars(select(model).filter_by(**where)))

    def situation(self) -> str:
        (situation,) = self.rows(Situation)
        return situation.id

    def summaries(self) -> list[dict]:
        return [s.data for s in self.rows(SituationSummary)]

    def warning_causes(self) -> list[str]:
        return sorted(w.cause for w in self.rows(SituationWarning))

    def alarm_causes(self) -> list[str]:
        return sorted(a.cause for a in self.rows(SituationAlarm))

    def failures(self) -> list[str]:
        return [f"{e.kind}: {e.message}" for e in self.rows(LogEntry) if e.kind.endswith("_failure")]

    def state(self):
        return self.workflows.graph.get_state({"configurable": {"thread_id": self.situation()}})


@pytest.fixture
def make_sim(tmp_path):
    sims = []

    def make(**kwargs):
        sims.append(Sim(tmp_path, **kwargs))
        return sims[-1]

    yield make
    for sim in sims:
        sim.system.close()


@pytest.fixture
def sim(make_sim):
    return make_sim()


def scripted(name: str, sensor: str = "cam_garden", **answers) -> tuple[Demo, str]:
    """A demo with one event; `answers` are the answers per agent for its clip. Returns the demo and the clip."""
    clip = f"{name}/clip.wav" if sensor.startswith("mic") else f"{name}/clip.mp4"
    demo = Demo(name, "test", (ScriptedEvent(0, sensor, clip, None if sensor.startswith("mic") else BOX),),
                {agent: {clip: answer} for agent, answer in answers.items()})
    return demo, clip


# ------------------------------------------------------------------ the shipped demos, end to end
@pytest.mark.parametrize("name, warnings, alarms, threat", [
    ("wind", [], [], 0.0),  # low risk: as forecast
    ("unforecast_wind", ["suspicious weather"], [], 0.5),  # a weather mismatch is suspicious, like a role mismatch
    ("lens_flare", [], [], 0.0),  # low risk: sensor artefact
    ("gardener", [], [], 0.0),  # plausible person
    ("delivery", [], [], 0.05),  # plausible person, unknown but allowed at the entryway
    ("hooded_person", ["suspicious person", "unpermitted entry"], [], 0.6),  # unclear: a human decides
    ("intruder", ["unpermitted entry"], ["strong suspicion"], 0.9),  # critical
    ("bear", ["dangerous animal"], [], 1.0),  # critical: the house is closed, so a warning
])
def test_demo_outcome(sim, name, warnings, alarms, threat):
    sim.play(name)
    assert (sim.warning_causes(), sim.alarm_causes()) == (warnings, alarms)
    assert max(s["threat_score"] for s in sim.summaries()) == threat
    assert sim.failures() == []


def test_the_intruder_alarm_and_the_speaker(sim):
    sim.play("intruder")
    assert [text for _, text in sim.system.speaker.spoken] == [
        "You are entering without permission. Please leave the area immediately."]
    assert any(m.startswith("ALARM: ") for _, _, m in sim.system.gateway.sent)


def test_the_demo_house_is_seeded_once(make_sim, tmp_path):
    make_sim()
    make_sim()  # a restart on the same database
    with make_sim().system.session_factory() as s:
        assert len(list(s.scalars(select(Area)))) == 3


# ------------------------------------------------------------------ events and the graph
def test_every_event_is_interpreted_so_an_alarm_is_not_masked(sim):
    # An intruder in the garden (an alarm, and a warning that waits for the user) and a harmless second clip.
    demo, first = scripted("burglar", "cam_garden", object_detector=_detected("person", _person()),
                           person_identifier={"verdict": "unknown"},
                           behavioural_interpreter=_behaviour(("suspicious_activity", 0.9), role_mismatch=0.9))
    second = "burglar/flare.mp4"
    demo = replace(demo, events=demo.events + (ScriptedEvent(0, "cam_garden", second, BOX),))
    demo.answers["object_detector"][second] = _detected("sensor_artefact")
    sim.play(demo)
    # Both events have the same time here, so which is first does not matter: each one is interpreted.
    assert sorted(s["threat_score"] for s in sim.summaries()) == [0.0, 0.9]
    assert sim.alarm_causes() == ["strong suspicion"] and sim.warning_causes() == ["unpermitted entry"]


def test_state_is_checkpointed_and_the_situation_observes(sim):
    sim.play("gardener")
    state = sim.state()
    assert state.next == () and state.values["phase"] == "observe"
    assert len(state.values["seen"]) == 1 and state.values["summary"]["threat_score"] == 0.0


def test_people_and_animals_are_analysed_together(sim):
    demo, _ = scripted("dog_walker", object_detector=_detected("person", _person(), _obj("a1", "animal", "boar")),
                       person_identifier={"verdict": "unknown"},
                       behavioural_interpreter=_behaviour(("move_to_area", 0.9)))
    sim.play(demo)
    (summary,) = sim.summaries()
    assert [p["identity"] for p in summary["persons"]] == ["unknown"]
    assert [(a["label"], a["danger"]) for a in summary["animals"]] == [("boar", 0.8)]
    assert summary["threat_score"] == 0.8 and "boar (danger 0.80)" in summary["summary"]


def test_a_person_invited_by_a_family_member_is_marked(sim):
    demo, clip = scripted("visit", "cam_entry",
                          object_detector=_detected("person", _person(), {**_person(), "object_id": "p2"}),
                          behavioural_interpreter={
                              "persons": [{"object_id": oid, "role_mismatch": 0, "intents": [
                                  {"intent": "accompany_person" if oid == "p1" else "ring_the_bell",
                                   "confidence": 0.8}]} for oid in ("p1", "p2")],
                              "relations": [{"subject": "p1", "predicate": "accompanies", "object": "p2"}],
                              "health_emergency": False})
    demo.answers["person_identifier"] = {f"{clip}#p1": {"verdict": "known", "person": "anna", "confidence": 0.9},
                                         f"{clip}#p2": {"verdict": "unknown"}}
    sim.play(demo)
    persons = {p["object_id"]: p for p in sim.summaries()[0]["persons"]}
    assert (persons["p1"]["person_id"], persons["p1"]["invited"]) == ("anna", False)
    assert (persons["p2"]["identity"], persons["p2"]["invited"]) == ("unknown", True)


# ------------------------------------------------------------------ agents that are not available
def test_without_object_detector_the_situation_is_unclear_not_fine(sim):
    demo, _ = scripted("no_answer")  # the detector has nothing to say about the clip
    sim.play(demo)
    (summary,) = sim.summaries()
    assert (summary["unclear_situation"], summary["threat_score"]) == (True, 0.5)
    assert sim.warning_causes() == ["unclear situation"]
    assert any("object_detector" in f for f in sim.failures())


def test_an_unusable_answer_counts_as_not_available(sim):
    demo, _ = scripted("garbage", object_detector=None)
    sim.play(demo)
    assert sim.summaries()[0]["unclear_situation"] and sim.warning_causes() == ["unclear situation"]
    assert any(f.startswith("agent_failure: object_detector") for f in sim.failures())


def test_a_person_that_cannot_be_identified_because_the_agent_is_down_is_treated_as_not_familiar(sim):
    demo, _ = scripted("no_identifier", object_detector=_detected("person", _person()),
                       behavioural_interpreter=_behaviour(("move_to_area", 0.9)))
    sim.play(demo)
    (person,) = sim.summaries()[0]["persons"]
    assert (person["identity"], person["cause"], person["suspicion"]) == ("not_decidable", "unavailable", 0.0)
    assert sim.warning_causes() == ["unpermitted entry"]  # the garden is not open to persons that are not familiar


def test_without_behavioural_interpreter_the_behaviour_is_unknown_and_not_suspicious(sim):
    demo, _ = scripted("no_behaviour", object_detector=_detected("person", _person("gardener")),
                       person_identifier={"verdict": "known", "person": "gustav", "confidence": 0.9})
    sim.play(demo)
    assert sim.summaries()[0]["persons"][0]["suspicion"] == 0.0 and sim.warning_causes() == []
    assert any("behavioural_interpreter" in f for f in sim.failures())


def test_a_role_that_differs_from_the_roles_on_file_is_suspicious(sim):
    demo, _ = scripted("wrong_role", object_detector=_detected("person", _person("delivery person")),
                       person_identifier={"verdict": "known", "person": "gustav", "confidence": 0.9},
                       behavioural_interpreter=_behaviour(("move_to_area", 0.9)))
    sim.play(demo)
    assert sim.summaries()[0]["persons"][0]["suspicion"] == 0.6
    assert "suspicious person" in sim.warning_causes()


def test_an_agent_that_does_not_answer_in_time_is_not_available(make_sim):
    sim = make_sim(workflow_config=WorkflowConfig(agent_timeout_s=0.2))
    sim.workflows.object_detector.model = lambda **kw: time.sleep(1) or _detected("sensor_artefact")
    demo, _ = scripted("slow")
    sim.play(demo)
    assert sim.summaries()[0]["unclear_situation"] and any("TimeoutError" in f for f in sim.failures())


# ------------------------------------------------------------------ weather and audio
@pytest.mark.parametrize("conditions, wind, threat, text, warnings", [
    (("high_wind", "cloudy"), 50, 0.0, "as forecast", []),
    (("sunny",), 10, 0.5, "high_wind that was not forecast", ["suspicious weather"]),
    (("high_wind",), 10, 0.5, "wind of 55 km/h, 10 forecast", ["suspicious weather"]),
])
def test_weather_is_compared_with_the_forecast(sim, conditions, wind, threat, text, warnings):
    sim.play(replace(DEMOS["wind"], name="wind_x", forecast_conditions=conditions, forecast_wind_kmh=wind))
    assert {s["threat_score"] for s in sim.summaries()} == {threat}
    assert all(text in s["summary"] for s in sim.summaries())
    assert sim.warning_causes() == warnings  # a mismatch is suspicious, like a role mismatch; warned about once


def test_weather_without_forecast_is_unknown_and_not_suspicious(sim):
    def down():
        raise ConnectionError("weather service down")

    sim.system.forecast.fetch = down
    sim.play("wind")
    assert {s["threat_score"] for s in sim.summaries()} == {0.0}
    assert all("unknown" in s["summary"] for s in sim.summaries())
    assert any("weather forecast" in f for f in sim.failures())


@pytest.mark.parametrize("category, threat, warnings", [
    ("animal", 0.0, []),  # an animal that is not known is not dangerous
    ("technical_noise", 0.0, []),  # a sensor artefact
    ("human_activity", 0.5, ["unclear situation"]),  # audio alone cannot say who is there
    ("unknown", 0.5, ["unclear situation"]),
])
def test_audio_events(sim, category, threat, warnings):
    demo, _ = scripted("noise", "mic_garden", noise_interpreter={
        "category": category, "confidence": 0.8, "is_sensor_artefact": category == "technical_noise"})
    sim.play(demo)
    assert [s["threat_score"] for s in sim.summaries()] == [threat] and sim.warning_causes() == warnings


# ------------------------------------------------------------------ events while a warning waits, and resolving
def hooded_again_demo():
    """A hooded person, again (the same causes), and a harmless clip afterwards."""
    first, again, quiet = "waiting/first.mp4", "waiting/again.mp4", "waiting/quiet.mp4"
    hooded = {"object_detector": _detected("person", _person()),
              "person_identifier": {"verdict": "not_decidable", "cause": "clothing"},
              "behavioural_interpreter": _behaviour(("unknown_activity", 0.7), role_mismatch=0.6)}
    answers = {agent: {first: answer, again: answer} for agent, answer in hooded.items()}
    answers["object_detector"][quiet] = _detected("sensor_artefact")
    return Demo("waiting", "test", (ScriptedEvent(0, "cam_garden", first, BOX),), answers), again, quiet


def test_events_are_processed_while_a_warning_waits_but_do_not_repeat_its_side_effects(sim):
    demo, again, quiet = hooded_again_demo()
    sim.play(demo)
    assert sim.warning_causes() == ["suspicious person", "unpermitted entry"]
    texts, spoken = len(sim.system.gateway.sent), len(sim.system.speaker.spoken)
    assert (texts, spoken) == (2, 1) and {w.status for w in sim.rows(SituationWarning)} == {"open"}

    for n, (event_id, clip) in enumerate([("b_again", again), ("c_quiet", quiet)], start=1):
        sim.controller.handle_sensor_event(SensorEventSignal(
            type="video_event", event_id=event_id, sensor_id="cam_garden", area_id="garden",
            start_time=T0 + timedelta(seconds=n), evidence=clip))
    sim.settle()

    assert [s["threat_score"] for s in sim.summaries()] == [0.6, 0.6, 0.0]  # nothing was held back ...
    assert sim.warning_causes() == ["suspicious person", "unpermitted entry"]  # ... and nothing was triggered twice
    assert (len(sim.system.gateway.sent), len(sim.system.speaker.spoken)) == (texts, spoken)
    assert sim.rows(Situation)[0].status == "active" and sim.state().next == ()


def test_a_situation_without_news_is_resolved_and_the_scenario_ends(sim):
    sim.play("gardener")
    assert sim.rows(Situation)[0].status == "active"

    sim.clock.advance(59)
    sim.tick()
    assert sim.rows(Situation)[0].status == "active"

    sim.clock.advance(2)
    sim.tick()
    assert sim.rows(Situation)[0].status == "resolved" and sim.rows(Scenario)[0].status == "resolved"
    assert sim.state().values["phase"] == "idle" and sim.system.kb.people_present() == []
    sim.tick()  # nothing left to do, and nothing breaks
    assert len(sim.summaries()) == 1


def test_a_new_event_starts_a_new_situation_after_resolution(sim):
    sim.play("gardener")
    sim.clock.advance(61)
    sim.tick()
    sim.system.play("gardener", time_scale=0, wait=True)
    sim.settle()
    assert [s.status for s in sim.rows(Situation)] == ["resolved", "active"] and len(sim.rows(Scenario)) == 2


# ------------------------------------------------------------------ crash and restart
def test_a_crashed_run_continues_from_its_checkpoint_after_a_restart(make_sim):
    first = make_sim()

    def crash(signal):
        raise RuntimeError("controller crashed")

    first.controller.handle_situation_summary = crash
    first.play("intruder")
    assert first.summaries() == [] and any(f.startswith("workflow_failure") for f in first.failures())
    assert first.state().next == ("report",)  # the analyses are checkpointed; only the report is missing
    first.system.close()

    second = make_sim()  # a restart on the same files
    assert second.state().next == ("report",)
    second.tick()
    assert len(second.summaries()) == 1 and second.alarm_causes() == ["strong suspicion"]
    assert second.state().next == () and second.state().values["phase"] == "observe"
    assert second.warning_causes() == ["unpermitted entry"]


def test_repeating_a_report_does_not_repeat_warnings_and_alarms(sim):
    sim.play("intruder")
    again = SituationSummarySignal.model_validate({**sim.summaries()[0], "situation_id": sim.situation()})
    sim.controller.handle_situation_summary(again)
    assert sim.alarm_causes() == ["strong suspicion"] and sim.warning_causes() == ["unpermitted entry"]
    assert len(sim.system.gateway.sent) == 2


# ------------------------------------------------------------------ helpers
@pytest.mark.parametrize("label, danger", [
    ("bear", 1.0), ("Wild Boar", 0.8), ("large dog", 0.6), ("small dog", 0.4), ("fox", 0.2), ("unicorn", 0.0)])
def test_animal_danger_scores(label, danger):
    assert animal_danger(label) == danger  # an animal the system does not know is not dangerous


# ------------------------------------------------------------------ the API on top of the whole system
def test_the_api_lists_and_starts_demos_and_answers_warnings(sim):
    client = sim.system.app.test_client()
    assert {d["name"] for d in client.get("/api/demos").json} == set(DEMOS)
    assert client.post("/api/demos/nothing").status_code == 404

    assert client.post("/api/demos/hooded_person").status_code == 202
    for _ in range(100):  # the demo is played by a background thread
        if sim.rows(SensorEvent):
            break
        time.sleep(0.02)
    sim.settle()
    sid = sim.situation()
    warnings = client.get("/api/warnings?status=open").json
    assert len(warnings) == 2 and client.get(f"/api/situations/{sid}").json["status"] == "active"

    for w in warnings:
        assert client.post(f"/api/warnings/{w['id']}/dismiss").json["status"] == "dismissed"
    assert client.get("/api/warnings?status=open").json == []
    assert client.get(f"/api/situations/{sid}").json["status"] == "active" and sim.state().next == ()
