"""HTTP API (Flask) for the frontend: shows what the controller has stored and forwards the few user actions.

    GET  /api/scenarios                       scenarios, newest first
    GET  /api/scenarios/<id>                  a scenario with its situations and aggregated summaries
    GET  /api/situations/<id>                 a situation with its events, summaries, warnings, alarms and log entries
    GET  /api/warnings[?status=open]          warnings, newest first
    POST /api/warnings/<id>/elevate           the user's answer to a warning: elevate it to an alarm ...
    POST /api/warnings/<id>/dismiss           ... or dismiss it
    GET  /api/alarms                          alarms, newest first
    GET  /api/log[?situation_id=&limit=]      the event log, newest first (default limit: 100)
    GET  /api/sensors                         the sensors with their latest event
    GET  /api/areas                           the areas, with the names of the people who are in them now
    GET  /api/persons                         the people in the knowledge base, with their roles and picture_url
    POST /api/events                          a video/audio event of a sensor: starts or joins a situation
    GET  /api/demos                           the demos of the simulation (only if there is a simulation)
    POST /api/demos/<name>                    let the simulated sensors report a demo's anomalies (202); resolves
                                               whatever scenario is currently playing first, so a new one always
                                               takes over immediately instead of being blocked by it
    GET  /api/media/<event_id>                an animated clip for a sensor event's evidence, built fresh from its
                                               real area/weather/objects, or a placeholder (only with a simulation)
    GET  /api/media/<event_id>/audio          the sound of an audio event's evidence (only if there is a simulation)
    GET  /api/media/sprites/<name>.png        a character/animal sprite (media/sprites/), e.g. for `picture_url`
    GET  /api/sensors/<id>/frame              what the sensor currently shows (only if there is a simulation): its
                                               latest capture, or an ambient view if it has none yet
    GET  /api/sensors/<id>/audio              on demand: what the microphone currently hears, likewise (only if
                                               there is a simulation)

Rows are returned as they are stored (column names as keys, timestamps as ISO 8601 UTC). Errors are JSON:
`{"error": "..."}`. There is no authentication (out of the scope of the challenge). If the frontend is built
(`static_dir`), it is served at `/`.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol

from flask import Flask, Response, abort, jsonify, request, send_from_directory
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from werkzeug.exceptions import HTTPException

from ai_alarm.controller import Controller, SensorEventSignal, SituationResolvedSignal
from ai_alarm.db.models import (
    Area, LogEntry, Person, Scenario, Sensor, SensorEvent, Situation, SituationAlarm, SituationWarning,
)
from ai_alarm.media import sprite_for_person

DEFAULT_LOG_LIMIT = 100


class Simulation(Protocol):
    @property
    def demos(self) -> dict[str, str]:
        """The demos that can be played: name -> description."""

    def play(self, name: str) -> None:
        """Raises `LookupError` for an unknown demo."""

    def frame(
        self, *, evidence: str, kind: str, sensor_id: str, area_id: str, area_name: str, at: datetime
    ) -> tuple[bytes, str]:
        """An image for a piece of sensor evidence, and its mimetype (a still PNG, or an animated GIF for evidence
        a pre-rendered situation exists for)."""

    def idle_frame(self, *, sensor_id: str, kind: str, area_id: str, area_name: str, at: datetime) -> tuple[bytes, str]:
        """An image of what a sensor sees when it has not reported anything, and its mimetype."""

    def audio(self, *, evidence: str) -> bytes:
        """The WAV sound of a piece of audio evidence."""

    def idle_audio(self, *, sensor_id: str) -> bytes:
        """The WAV ambient sound of a microphone that has not reported anything."""


def _dict(row) -> dict[str, Any]:
    """A row as a JSON-able dict: its columns, with timestamps as ISO 8601 strings."""
    values = {c.name: getattr(row, c.name) for c in row.__table__.columns}
    return {k: v.isoformat() if isinstance(v, datetime) else v for k, v in values.items()}


def create_app(
    session_factory: Callable[[], Session], controller: Controller, simulation: Simulation | None = None,
    static_dir: Path | None = None, assets_dir: Path = Path("media"),
) -> Flask:
    # relative to the current working directory -- not to this package's own directory, which is what
    # Flask's send_from_directory would otherwise resolve a relative `directory` against
    assets_dir = assets_dir.resolve()
    frontend = static_dir if static_dir is not None and (static_dir / "index.html").exists() else None
    app = Flask(__name__, static_folder=str(frontend) if frontend else None, static_url_path="")

    def get_or_404(s: Session, model, row_id: str):
        row = s.get(model, row_id)
        if row is None:
            abort(404, f"unknown {model.__name__.lower()} {row_id}")
        return row

    # ------------------------------------------------------------------ situations
    @app.get("/api/scenarios")
    def scenarios():
        with session_factory() as s:
            rows = s.scalars(select(Scenario).order_by(Scenario.created_at.desc()))
            return jsonify([{**_dict(x), "situation_ids": [y.id for y in x.situations]} for x in rows])

    @app.get("/api/scenarios/<scenario_id>")
    def scenario(scenario_id: str):
        with session_factory() as s:
            x = get_or_404(s, Scenario, scenario_id)
            return jsonify({
                **_dict(x),
                "situations": [_dict(y) for y in x.situations],
                "aggregated_summaries": [_dict(y) for y in x.aggregated_summaries],
            })

    @app.get("/api/situations/<situation_id>")
    def situation(situation_id: str):
        with session_factory() as s:
            x = get_or_404(s, Situation, situation_id)
            return jsonify({
                **_dict(x),
                "events": [_dict(y) for y in x.events],
                "summaries": [_dict(y) for y in x.summaries],
                "warnings": [_dict(y) for y in x.warnings],
                "alarms": [_dict(y) for y in x.alarms],
                "log_entries": [_dict(y) for y in x.log_entries],
            })

    # ------------------------------------------------------------------ warnings and alarms
    @app.get("/api/warnings")
    def warnings():
        query = select(SituationWarning).order_by(SituationWarning.created_at.desc())
        if "status" in request.args:
            query = query.where(SituationWarning.status == request.args["status"])
        with session_factory() as s:
            return jsonify([_dict(x) for x in s.scalars(query)])

    def answer(warning_id: str, action: Callable[[str], None]):
        action(warning_id)  # raises LookupError for an unknown warning; answering twice changes nothing
        with session_factory() as s:
            return jsonify(_dict(get_or_404(s, SituationWarning, warning_id)))

    @app.post("/api/warnings/<warning_id>/elevate")
    def elevate(warning_id: str):
        return answer(warning_id, controller.elevate)

    @app.post("/api/warnings/<warning_id>/dismiss")
    def dismiss(warning_id: str):
        return answer(warning_id, controller.dismiss)

    @app.get("/api/alarms")
    def alarms():
        with session_factory() as s:
            rows = s.scalars(select(SituationAlarm).order_by(SituationAlarm.created_at.desc()))
            return jsonify([_dict(x) for x in rows])

    # ------------------------------------------------------------------ log and sensors
    @app.get("/api/log")
    def log():
        query = select(LogEntry).order_by(LogEntry.created_at.desc(), LogEntry.id.desc())
        if "situation_id" in request.args:
            query = query.where(LogEntry.situation_id == request.args["situation_id"])
        with session_factory() as s:
            limit = request.args.get("limit", DEFAULT_LOG_LIMIT, type=int)
            return jsonify([_dict(x) for x in s.scalars(query.limit(limit))])

    @app.get("/api/sensors")
    def sensors():
        with session_factory() as s:
            result = []
            for sensor in s.scalars(select(Sensor).order_by(Sensor.id)):
                latest = s.scalars(select(SensorEvent).where(SensorEvent.sensor_id == sensor.id).order_by(
                    SensorEvent.start_time.desc())).first()
                result.append({**_dict(sensor), "latest_event": _dict(latest) if latest else None})
            return jsonify(result)

    @app.get("/api/areas")
    def areas():
        with session_factory() as s:
            return jsonify([
                {**_dict(a), "people": [p.name or "Unknown person" for p in controller.kb.people_present(a.id)]}
                for a in s.scalars(select(Area).order_by(Area.name))
            ])

    @app.get("/api/persons")
    def persons():
        with session_factory() as s:
            rows = s.scalars(select(Person).order_by(Person.id))
            result = []
            for x in rows:
                role_names = [r.name for r in x.roles]
                sprite = sprite_for_person(x.id, role_names)
                result.append({**_dict(x), "roles": role_names, "picture_url": f"/api/media/sprites/{sprite}.png"})
            return jsonify(result)

    @app.get("/api/media/sprites/<path:filename>")
    def sprite_image(filename: str):
        sprites_dir = assets_dir / "sprites"
        if not (sprites_dir / filename).is_file():
            abort(404, "no such sprite")
        return send_from_directory(sprites_dir, filename)

    @app.post("/api/events")
    def post_event():
        signal = SensorEventSignal.model_validate(request.get_json())
        return jsonify(situation_id=controller.handle_sensor_event(signal)), 201

    if simulation is not None:
        @app.get("/api/demos")
        def demos():
            return jsonify([{"name": name, "description": text} for name, text in simulation.demos.items()])

        @app.post("/api/demos/<name>")
        def play_demo(name: str):
            if name not in simulation.demos:
                abort(404, f"unknown demo {name}")
            # a new scenario overwrites whatever is currently playing, rather than being blocked by it: resolve
            # every active situation first, so its sensors fall back to their idle view (see sensor_frame) instead
            # of getting stuck showing the old scenario's evidence forever
            with session_factory() as s:
                active_situation_ids = list(s.scalars(select(Situation.id).where(Situation.status == "active")))
            for situation_id in active_situation_ids:
                controller.handle_situation_resolved(SituationResolvedSignal(situation_id=situation_id))
            simulation.play(name)
            return jsonify(demo=name), 202

        @app.get("/api/media/<event_id>")
        def media(event_id: str):
            with session_factory() as s:
                event = get_or_404(s, SensorEvent, event_id)
                sensor = s.get(Sensor, event.sensor_id)
                area = s.get(Area, sensor.area_id) if sensor else None
            data, mimetype = simulation.frame(
                evidence=event.evidence, kind=event.kind, sensor_id=event.sensor_id,
                area_id=area.id if area else "", area_name=area.name if area else event.sensor_id,
                at=event.start_time)
            return Response(data, mimetype=mimetype, headers={"Cache-Control": "public, max-age=300"})

        @app.get("/api/media/<event_id>/audio")
        def media_audio(event_id: str):
            with session_factory() as s:
                event = get_or_404(s, SensorEvent, event_id)
            wav = simulation.audio(evidence=event.evidence)
            return Response(wav, mimetype="audio/wav", headers={"Cache-Control": "public, max-age=300"})

        @app.get("/api/sensors/<sensor_id>/frame")
        def sensor_frame(sensor_id: str):
            """What this sensor currently shows: its latest capture, or an ambient view before it has any -- or
            once the situation that capture belongs to is resolved, since a live view reverts to idle once
            whatever it saw is over, rather than sitting on stale evidence forever. Which of the two applies is a
            backend decision; the URL and the response are the same either way."""
            with session_factory() as s:
                sensor = get_or_404(s, Sensor, sensor_id)
                area = s.get(Area, sensor.area_id)
                latest = s.scalars(select(SensorEvent).join(Situation).where(
                    SensorEvent.sensor_id == sensor_id, Situation.status == "active",
                ).order_by(SensorEvent.start_time.desc())).first()
            area_id, area_name = (area.id, area.name) if area else ("", sensor.id)
            if latest is not None:
                data, mimetype = simulation.frame(evidence=latest.evidence, kind=latest.kind, sensor_id=sensor.id,
                                                  area_id=area_id, area_name=area_name, at=latest.start_time)
            else:
                data, mimetype = simulation.idle_frame(sensor_id=sensor.id, kind=sensor.kind, area_id=area_id,
                                                        area_name=area_name, at=controller.clock())
            return Response(data, mimetype=mimetype, headers={"Cache-Control": "public, max-age=5"})

        @app.get("/api/sensors/<sensor_id>/audio")
        def sensor_audio(sensor_id: str):
            """On demand: the sound of this microphone's latest audio event, or its ambient sound before it has
            any (or once the situation it belongs to is resolved) -- the design document's "audio of each
            [sensor] can be streamed on-demand"."""
            with session_factory() as s:
                sensor = get_or_404(s, Sensor, sensor_id)
                latest = s.scalars(select(SensorEvent).join(Situation).where(
                    SensorEvent.sensor_id == sensor_id, SensorEvent.kind == "audio", Situation.status == "active",
                ).order_by(SensorEvent.start_time.desc())).first()
            wav = simulation.audio(evidence=latest.evidence) if latest else simulation.idle_audio(
                sensor_id=sensor.id)
            return Response(wav, mimetype="audio/wav", headers={"Cache-Control": "public, max-age=5"})

    if frontend is not None:
        @app.get("/")
        def index():
            # Vite fingerprints every JS/CSS file with a content hash and deletes the previous build's files on
            # each rebuild -- so a stale cached copy of *this* file (the only one without a hash in its name) would
            # reference filenames that no longer exist. The fingerprinted assets themselves are safe to cache
            # (Flask's default static handler does, via ETag/Last-Modified); this one response must never be.
            response = send_from_directory(frontend, "index.html")
            response.headers["Cache-Control"] = "no-cache"
            return response

    # ------------------------------------------------------------------ errors
    @app.errorhandler(HTTPException)
    def http_error(e: HTTPException):
        return jsonify(error=e.description), e.code

    @app.errorhandler(ValidationError)
    def invalid_request(e: ValidationError):
        details = json.loads(e.json(include_url=False, include_context=False))
        return jsonify(error="invalid request", details=details), 400

    @app.errorhandler(LookupError)
    def not_found(e: LookupError):
        return jsonify(error=str(e.args[0]) if e.args else "not found"), 404

    @app.errorhandler(IntegrityError)
    def unknown_reference(e: IntegrityError):
        return jsonify(error="the event refers to a sensor or area that does not exist"), 400

    return app
