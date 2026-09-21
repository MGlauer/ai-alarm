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
    POST /api/events                          a video/audio event of a sensor: starts or joins a situation
    GET  /api/demos                           the demos of the simulation (only if there is a simulation)
    POST /api/demos/<name>                    let the simulated sensors report a demo's anomalies (202)

Rows are returned as they are stored (column names as keys, timestamps as ISO 8601 UTC). Errors are JSON:
`{"error": "..."}`. There is no authentication (out of the scope of the challenge).
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable, Protocol

from flask import Flask, abort, jsonify, request
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from werkzeug.exceptions import HTTPException

from ai_alarm.controller import Controller, SensorEventSignal
from ai_alarm.db.models import (
    LogEntry, Scenario, Sensor, SensorEvent, Situation, SituationAlarm, SituationWarning,
)

DEFAULT_LOG_LIMIT = 100


class Simulation(Protocol):
    @property
    def demos(self) -> dict[str, str]:
        """The demos that can be played: name -> description."""

    def play(self, name: str) -> None:
        """Raises `LookupError` for an unknown demo."""


def _dict(row) -> dict[str, Any]:
    """A row as a JSON-able dict: its columns, with timestamps as ISO 8601 strings."""
    values = {c.name: getattr(row, c.name) for c in row.__table__.columns}
    return {k: v.isoformat() if isinstance(v, datetime) else v for k, v in values.items()}


def create_app(
    session_factory: Callable[[], Session], controller: Controller, simulation: Simulation | None = None
) -> Flask:
    app = Flask(__name__)

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
            simulation.play(name)  # raises LookupError for an unknown demo
            return jsonify(demo=name), 202

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
