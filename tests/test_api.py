"""HTTP API: what the frontend sees and does, on top of a controller world (see world.py)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from world import T0, person

from ai_alarm.api import create_app
from ai_alarm.db.models import SensorEvent

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


# ------------------------------------------------------------------ persons
def test_persons_come_with_their_roles_and_a_sprite_picture(world, client):
    persons = {p["id"]: p for p in client.get("/api/persons").json}
    assert set(persons) == {"anna", "gustav", "neighbour"}
    assert (persons["anna"]["name"], persons["anna"]["roles"]) == ("Anna", ["family"])
    # anna and gustav are drawn as themselves (their own identity sprite); an unidentified person with no
    # sprite-worthy role, like the seeded neighbour, still gets a generic picture -- never no picture at all
    assert persons["anna"]["picture_url"] == "/api/media/sprites/family_anna.png"
    assert persons["gustav"]["picture_url"] == "/api/media/sprites/gardener.png"
    assert persons["neighbour"]["picture_url"] == "/api/media/sprites/unknown_person.png"


def test_sprite_image_serves_the_file_and_404s_when_missing(world, tmp_path):
    sprites_dir = tmp_path / "sprites"
    sprites_dir.mkdir()
    (sprites_dir / "gardener.png").write_bytes(b"PNG-BYTES")
    client = create_app(world.sf, world.controller, assets_dir=tmp_path).test_client()

    with client.get("/api/media/sprites/gardener.png") as response:
        assert response.status_code == 200 and response.data == b"PNG-BYTES"
    assert client.get("/api/media/sprites/nope.png").status_code == 404


def test_sprite_image_resolves_a_relative_assets_dir_against_the_working_directory(world, tmp_path, monkeypatch):
    """A relative `assets_dir` must be resolved against the process's cwd, not against ai_alarm's own package
    directory -- which is what Flask's send_from_directory does by default for a relative path."""
    monkeypatch.chdir(tmp_path)
    sprites_dir = tmp_path / "media" / "sprites"
    sprites_dir.mkdir(parents=True)
    (sprites_dir / "gardener.png").write_bytes(b"PNG-BYTES")
    client = create_app(world.sf, world.controller, assets_dir=Path("media")).test_client()

    with client.get("/api/media/sprites/gardener.png") as response:
        assert response.status_code == 200 and response.data == b"PNG-BYTES"


# ------------------------------------------------------------------ areas and the frontend
def test_areas_come_with_the_names_of_the_people_in_them(world, client):
    areas = {a["id"]: a for a in client.get("/api/areas").json}
    assert areas["garden"]["name"] == "Garden" and all(a["people"] == [] for a in areas.values())

    world.kb.record_detection("anna", "house", 0.0, T0)
    world.kb.record_detection("unk_1", "garden", 0.0, T0)  # a person that is not known by name
    areas = {a["id"]: a for a in client.get("/api/areas").json}
    assert (areas["house"]["people"], areas["garden"]["people"]) == (["Anna"], ["Unknown person"])


def test_the_built_frontend_is_served_next_to_the_api(world, tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<div id=root></div>")
    (tmp_path / "assets" / "app.js").write_text("console.log(1)")
    client = create_app(world.sf, world.controller, static_dir=tmp_path).test_client()

    for path, content in [("/", b"<div id=root></div>"), ("/assets/app.js", b"console.log(1)")]:
        with client.get(path) as response:  # closes the file that is being served
            assert response.data == content
    assert client.get("/api/warnings").json == []  # the API is not shadowed
    missing = client.get("/api/nothing")
    assert missing.status_code == 404 and "error" in missing.json


def test_the_frontend_index_is_never_cached_so_a_rebuild_is_never_served_stale(world, tmp_path):
    """index.html is the only built file without a content hash in its name; Vite deletes the previous build's
    hashed JS/CSS on every rebuild, so a cached copy of index.html would reference files that no longer exist."""
    (tmp_path / "index.html").write_text("<div id=root></div>")
    client = create_app(world.sf, world.controller, static_dir=tmp_path).test_client()
    with client.get("/") as response:
        assert response.headers["Cache-Control"] == "no-cache"


def test_without_a_built_frontend_there_is_only_the_api(world, tmp_path, client):
    assert client.get("/").status_code == 404
    empty = create_app(world.sf, world.controller, static_dir=tmp_path).test_client()  # a directory without index.html
    assert empty.get("/").status_code == 404 and empty.get("/api/warnings").status_code == 200


# ------------------------------------------------------------------ media
class FrameSimulation(FakeSimulation):
    def __init__(self):
        super().__init__()
        self.frame_calls = []
        self.idle_frame_calls = []
        self.audio_calls = []
        self.idle_audio_calls = []

    def frame(self, **kwargs):
        self.frame_calls.append(kwargs)
        return b"PNG-BYTES", "image/png"

    def idle_frame(self, **kwargs):
        self.idle_frame_calls.append(kwargs)
        return b"IDLE-PNG-BYTES", "image/png"

    def audio(self, **kwargs):
        self.audio_calls.append(kwargs)
        return b"WAV-BYTES"

    def idle_audio(self, **kwargs):
        self.idle_audio_calls.append(kwargs)
        return b"IDLE-WAV-BYTES"


def test_media_serves_a_synthetic_image_for_a_sensor_event(world):
    sid = world.situation("garden")
    simulation = FrameSimulation()
    client = create_app(world.sf, world.controller, simulation).test_client()
    (event,) = world.rows(SensorEvent, situation_id=sid)
    with client.get(f"/api/media/{event.id}") as response:
        assert response.status_code == 200 and response.data == b"PNG-BYTES"
        assert response.mimetype == "image/png"
    assert simulation.frame_calls == [{
        "evidence": event.evidence, "kind": "video", "sensor_id": "cam_garden", "area_id": "garden",
        "area_name": "Garden", "at": event.start_time,
    }]


def test_media_serves_whatever_mimetype_the_simulation_reports(world):
    """A pre-rendered situation is an animated GIF, not a PNG -- the route must not hard-code the mimetype."""
    sid = world.situation("garden")
    simulation = FrameSimulation()
    simulation.frame = lambda **kw: (b"GIF-BYTES", "image/gif")  # type: ignore[method-assign]
    client = create_app(world.sf, world.controller, simulation).test_client()
    (event,) = world.rows(SensorEvent, situation_id=sid)
    with client.get(f"/api/media/{event.id}") as response:
        assert (response.data, response.mimetype) == (b"GIF-BYTES", "image/gif")


def test_media_404s_for_an_unknown_event_and_without_a_simulation(world, client):
    sid = world.situation("garden")
    (event,) = world.rows(SensorEvent, situation_id=sid)
    assert client.get(f"/api/media/{event.id}").status_code == 404  # `client` has no simulation
    simulation = FrameSimulation()
    with_sim = create_app(world.sf, world.controller, simulation).test_client()
    assert with_sim.get("/api/media/evt_nope").status_code == 404


def test_sensor_frame_serves_an_ambient_image_before_any_event(world):
    simulation = FrameSimulation()
    client = create_app(world.sf, world.controller, simulation).test_client()
    with client.get("/api/sensors/cam_garden/frame") as response:
        assert response.status_code == 200 and response.data == b"IDLE-PNG-BYTES"
        assert response.mimetype == "image/png"
    assert simulation.idle_frame_calls == [
        {"sensor_id": "cam_garden", "kind": "camera", "area_id": "garden", "area_name": "Garden",
         "at": world.controller.clock()},
    ]
    assert simulation.frame_calls == []


def test_sensor_frame_serves_its_latest_capture_once_it_has_one(world):
    sid = world.situation("garden")
    (event,) = world.rows(SensorEvent, situation_id=sid)
    simulation = FrameSimulation()
    client = create_app(world.sf, world.controller, simulation).test_client()

    with client.get("/api/sensors/cam_garden/frame") as response:
        assert response.status_code == 200 and response.data == b"PNG-BYTES"
    assert simulation.frame_calls == [{
        "evidence": event.evidence, "kind": "video", "sensor_id": "cam_garden", "area_id": "garden",
        "area_name": "Garden", "at": event.start_time,
    }]
    assert simulation.idle_frame_calls == []  # the URL is the same either way; the choice is entirely the backend's


def test_sensor_frame_reverts_to_idle_once_its_situation_is_resolved(world):
    """The bug this guards against: a camera that showed one demo's evidence must not keep showing it forever
    once that situation is over -- it should go back to its idle view, the same as before anything ever happened."""
    sid = world.situation("garden")
    world.resolve(sid)
    simulation = FrameSimulation()
    client = create_app(world.sf, world.controller, simulation).test_client()

    with client.get("/api/sensors/cam_garden/frame") as response:
        assert response.status_code == 200 and response.data == b"IDLE-PNG-BYTES"
    assert simulation.frame_calls == []
    assert simulation.idle_frame_calls == [
        {"sensor_id": "cam_garden", "kind": "camera", "area_id": "garden", "area_name": "Garden",
         "at": world.controller.clock()},
    ]


def test_sensor_frame_404s_for_an_unknown_sensor_and_without_a_simulation(world, client):
    assert client.get("/api/sensors/cam_garden/frame").status_code == 404  # `client` has no simulation
    simulation = FrameSimulation()
    with_sim = create_app(world.sf, world.controller, simulation).test_client()
    assert with_sim.get("/api/sensors/cam_moon/frame").status_code == 404


def test_media_audio_serves_the_sound_of_an_audio_event(world):
    sid = world.situation("garden")
    world.event("e2", "garden", sensor="mic_garden")
    simulation = FrameSimulation()
    client = create_app(world.sf, world.controller, simulation).test_client()
    (event,) = world.rows(SensorEvent, id="e2")

    with client.get(f"/api/media/{event.id}/audio") as response:
        assert response.status_code == 200 and response.data == b"WAV-BYTES"
        assert response.mimetype == "audio/wav"
    assert simulation.audio_calls == [{"evidence": event.evidence}]


def test_sensor_audio_serves_an_ambient_sound_before_any_audio_event(world):
    simulation = FrameSimulation()
    client = create_app(world.sf, world.controller, simulation).test_client()
    with client.get("/api/sensors/mic_garden/audio") as response:
        assert response.status_code == 200 and response.data == b"IDLE-WAV-BYTES"
        assert response.mimetype == "audio/wav"
    assert simulation.idle_audio_calls == [{"sensor_id": "mic_garden"}]
    assert simulation.audio_calls == []


def test_sensor_audio_serves_its_latest_audio_event_ignoring_video_events(world):
    world.situation("garden")  # a video event on cam_garden, not mic_garden
    world.event("e2", "garden", sensor="mic_garden")
    simulation = FrameSimulation()
    client = create_app(world.sf, world.controller, simulation).test_client()
    (event,) = world.rows(SensorEvent, id="e2")

    with client.get("/api/sensors/mic_garden/audio") as response:
        assert response.status_code == 200 and response.data == b"WAV-BYTES"
    assert simulation.audio_calls == [{"evidence": event.evidence}]
    assert simulation.idle_audio_calls == []


def test_sensor_audio_reverts_to_ambient_once_its_situation_is_resolved(world):
    sid = world.event("e2", "garden", sensor="mic_garden")
    world.resolve(sid)
    simulation = FrameSimulation()
    client = create_app(world.sf, world.controller, simulation).test_client()

    with client.get("/api/sensors/mic_garden/audio") as response:
        assert response.status_code == 200 and response.data == b"IDLE-WAV-BYTES"
    assert simulation.audio_calls == []


def test_sensor_audio_404s_for_an_unknown_sensor_and_without_a_simulation(world, client):
    assert client.get("/api/sensors/mic_garden/audio").status_code == 404  # `client` has no simulation
    simulation = FrameSimulation()
    with_sim = create_app(world.sf, world.controller, simulation).test_client()
    assert with_sim.get("/api/sensors/mic_moon/audio").status_code == 404


def test_starting_a_demo_overwrites_whatever_scenario_is_currently_playing(world):
    """Not blocked by a scenario in progress: a new demo resolves it first and always takes over immediately."""
    sid = world.situation("garden")
    simulation = FrameSimulation()
    client = create_app(world.sf, world.controller, simulation).test_client()

    response = client.post("/api/demos/wind")
    assert response.status_code == 202 and simulation.played == ["wind"]
    assert world.status(sid) == "resolved"
    assert client.get(f"/api/situations/{sid}").json["status"] == "resolved"


def test_starting_a_demo_with_no_scenario_playing_just_plays_it(world, client):
    simulation = FrameSimulation()
    client = create_app(world.sf, world.controller, simulation).test_client()
    assert client.post("/api/demos/wind").status_code == 202 and simulation.played == ["wind"]
