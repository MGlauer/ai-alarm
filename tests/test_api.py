"""HTTP API: what the frontend sees and does, on top of a controller world (see world.py)."""
from __future__ import annotations

from datetime import datetime

import pytest
from world import T0, person

from ai_alarm.api import create_app

BBOX = {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}


@pytest.fixture
def client(world):
    return create_app(world.sf, world.controller).test_client()


def event(event_id="e1", sensor="cam_garden", area="garden", **overrides):
    return {"type": "video_event", "event_id": event_id, "sensor_id": sensor, "area_id": area,
            "start_time": "2026-09-21T12:00:00Z", "evidence": "clip.mp4", "bbox": BBOX, **overrides}


def warned(world, client):
    """A situation with one open warning ('suspicious person', yellow). Returns (situation id, warning)."""
    sid = client.post("/api/events", json=event("e1", "cam_entry", "entry")).json["situation_id"]
    world.summary(sid, "entry", 0.5, [person(suspicion=0.5)], text="a visitor")
    (warning,) = client.get("/api/warnings").json
    return sid, warning


# ------------------------------------------------------------------ events
def test_posting_an_event_starts_a_situation_and_later_events_join_it(world, client):
    first = client.post("/api/events", json=event("e1"))
    assert first.status_code == 201
    second = client.post("/api/events", json=event("e2", "mic_garden", type="audio_event", bbox=None))
    assert second.status_code == 201 and second.json == first.json
    assert len(world.interpreters.started) == 1 and len(world.interpreters.forwarded) == 1


@pytest.mark.parametrize("body, status", [
    (event(type="thermal_event"), 400),
    (event(bbox={"x": 5, "y": 0, "w": 0, "h": 0}), 400),
    ({"event_id": "e1"}, 400),
    ([], 400),
    (event(sensor="cam_moon"), 400),  # no such sensor
    (event(area="moon"), 400),  # no such area
])
def test_invalid_events_are_rejected_with_json(client, body, status):
    response = client.post("/api/events", json=body)
    assert response.status_code == status and "error" in response.json


def test_an_event_that_is_not_json_is_rejected_with_json(client):
    response = client.post("/api/events", data="not json", content_type="text/plain")
    assert response.status_code in (400, 415) and "error" in response.json


# ------------------------------------------------------------------ scenarios and situations
def test_scenarios_and_their_situations(world, client):
    a = client.post("/api/events", json=event("e1", "cam_garden", "garden")).json["situation_id"]
    b = client.post("/api/events", json=event("e2", "cam_entry", "entry")).json["situation_id"]
    world.summary(a, "garden", 0.2)

    (scenario,) = client.get("/api/scenarios").json
    assert (scenario["status"], scenario["situation_ids"]) == ("active", [a, b])
    assert datetime.fromisoformat(scenario["created_at"]) == T0  # ISO 8601, UTC

    detail = client.get(f"/api/scenarios/{scenario['id']}").json
    assert [s["id"] for s in detail["situations"]] == [a, b]
    assert [x["threat_score"] for x in detail["aggregated_summaries"]] == [0.2]
    assert detail["aggregated_summaries"][0]["data"]["situations"] == {a: 0.2}


def test_scenarios_are_listed_newest_first(world, client):
    old = client.post("/api/events", json=event("e1")).json["situation_id"]
    world.resolve(old)
    world.clock.advance(60)
    client.post("/api/events", json=event("e2"))
    assert [s["status"] for s in client.get("/api/scenarios").json] == ["active", "resolved"]


def test_a_situation_with_everything_that_belongs_to_it(world, client):
    sid, warning = warned(world, client)
    detail = client.get(f"/api/situations/{sid}").json

    assert (detail["id"], detail["area_id"], detail["status"]) == (sid, "entry", "active")
    assert [(e["id"], e["kind"], e["bbox"]) for e in detail["events"]] == [("e1", "video", BBOX)]
    (summary,) = detail["summaries"]
    assert (summary["summary"], summary["threat_score"]) == ("a visitor", 0.5)
    assert summary["data"]["persons"][0]["suspicion"] == 0.5  # what the summary was based on
    assert [w["id"] for w in detail["warnings"]] == [warning["id"]] and detail["alarms"] == []
    assert {"situation_summary", "warning"} <= {e["kind"] for e in detail["log_entries"]}


def test_unknown_things_are_404_with_json(client):
    for url in ["/api/situations/sit_nope", "/api/scenarios/scn_nope", "/api/nothing"]:
        response = client.get(url)
        assert response.status_code == 404 and "error" in response.json
    assert client.get("/api/situations/sit_nope").json["error"] == "unknown situation sit_nope"
    assert client.delete("/api/events").status_code == 405 and "error" in client.delete("/api/events").json


# ------------------------------------------------------------------ warnings and alarms
def test_warnings_can_be_filtered_by_status(world, client):
    sid, warning = warned(world, client)
    assert [w["status"] for w in client.get("/api/warnings?status=open").json] == ["open"]
    assert client.get("/api/warnings?status=dismissed").json == []
    assert (warning["cause"], warning["colour"], warning["suspicion"]) == ("suspicious person", "yellow", 0.5)


def test_elevating_a_warning_raises_an_alarm(world, client):
    sid, warning = warned(world, client)
    response = client.post(f"/api/warnings/{warning['id']}/elevate")
    assert response.status_code == 200
    assert (response.json["status"], response.json["resolved_by"]) == ("elevated", "user")

    (alarm,) = client.get("/api/alarms").json
    assert (alarm["cause"], alarm["origin"], alarm["warning_id"]) == (
        "suspicious person", "user_elevation", warning["id"])

    client.post(f"/api/warnings/{warning['id']}/elevate")  # answering twice changes nothing
    assert len(client.get("/api/alarms").json) == 1


def test_dismissing_a_warning_raises_no_alarm(world, client):
    sid, warning = warned(world, client)
    assert client.post(f"/api/warnings/{warning['id']}/dismiss").json["status"] == "dismissed"
    assert client.get("/api/alarms").json == [] and client.get("/api/warnings?status=open").json == []
    assert client.get(f"/api/situations/{sid}").json["status"] == "active"


def test_answering_an_unknown_warning_is_404(client):
    response = client.post("/api/warnings/wrn_nope/elevate")
    assert response.status_code == 404 and response.json == {"error": "unknown warning wrn_nope"}


# ------------------------------------------------------------------ log and sensors
def test_log_is_newest_first_and_can_be_filtered_and_limited(world, client):
    a = client.post("/api/events", json=event("e1", "cam_garden", "garden")).json["situation_id"]
    b = client.post("/api/events", json=event("e2", "cam_entry", "entry")).json["situation_id"]
    world.summary(a, "garden", 0.1, text="first")
    world.clock.advance(60)
    world.summary(b, "entry", 0.1, text="second")

    log = client.get("/api/log").json
    assert [e["message"] for e in log] == ["second", "first"] and log[0]["kind"] == "situation_summary"
    assert [e["message"] for e in client.get(f"/api/log?situation_id={a}").json] == ["first"]
    assert [e["message"] for e in client.get("/api/log?limit=1").json] == ["second"]


def test_sensors_come_with_their_latest_event(world, client):
    sensors = {s["id"]: s for s in client.get("/api/sensors").json}
    assert set(sensors) == {"cam_entry", "cam_garden", "cam_house", "mic_garden"}
    assert (sensors["cam_garden"]["kind"], sensors["cam_garden"]["area_id"]) == ("camera", "garden")
    assert all(s["latest_event"] is None for s in sensors.values())

    client.post("/api/events", json=event("e1", "cam_garden", evidence="old.mp4", start_time="2026-09-21T11:00:00Z"))
    client.post("/api/events", json=event("e2", "cam_garden", evidence="new.mp4"))
    latest = {s["id"]: s["latest_event"] for s in client.get("/api/sensors").json}
    assert latest["cam_garden"]["evidence"] == "new.mp4" and latest["cam_entry"] is None


# ------------------------------------------------------------------ demos
class FakeSimulation:
    demos = {"wind": "a storm", "bear": "a bear"}

    def __init__(self):
        self.played = []

    def play(self, name):
        if name not in self.demos:
            raise LookupError(f"unknown demo {name}")
        self.played.append(name)


def test_demos_can_be_listed_and_started_if_there_is_a_simulation(world):
    simulation = FakeSimulation()
    client = create_app(world.sf, world.controller, simulation).test_client()
    assert client.get("/api/demos").json == [{"name": "wind", "description": "a storm"},
                                             {"name": "bear", "description": "a bear"}]
    response = client.post("/api/demos/bear")
    assert (response.status_code, response.json) == (202, {"demo": "bear"}) and simulation.played == ["bear"]
    assert client.post("/api/demos/nothing").status_code == 404


def test_without_a_simulation_there_are_no_demo_endpoints(client):
    assert client.get("/api/demos").status_code == 404 and client.post("/api/demos/wind").status_code == 404
