"""The demos: predetermined anomalies for the simulated CCTV/audio processors, together with what the (mock) agents
say about them and what the (simulated) weather service forecasts.

They cover the demo scenarios of the challenge brief: low risk without an alarm (wind, sensor artefact), a plausible
person (the gardener, a delivery), and critical or unclear cases (a hooded person, an intruder, a bear). The persons
and the house they refer to are in seed.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from ai_alarm.weather import ForecastEntry


@dataclass(frozen=True)
class ScriptedEvent:
    delay_s: float  # after the start of the demo
    sensor_id: str
    evidence: str  # reference of the recorded clip
    bbox: dict[str, float] | None = None  # rough area of the anomaly (video only)


@dataclass(frozen=True)
class Demo:
    name: str
    description: str
    events: tuple[ScriptedEvent, ...]
    answers: dict[str, dict[str, Any]]  # agent name -> evidence reference -> the answer of its model
    forecast_conditions: tuple[str, ...] = ("sunny",)
    forecast_wind_kmh: float = 10
    audio: dict[str, str] = field(default_factory=dict)  # area id -> reference of the latest audio

    def forecast(self, now: datetime) -> list[ForecastEntry]:
        return [ForecastEntry(valid_from=now - timedelta(hours=12), valid_to=now + timedelta(hours=12),
                              conditions=list(self.forecast_conditions), wind_speed_kmh=self.forecast_wind_kmh)]


BOX = {"x": 0.4, "y": 0.3, "w": 0.2, "h": 0.5}


def _obj(object_id: str, kind: str, label: str, **extra: Any) -> dict[str, Any]:
    return {"object_id": object_id, "kind": kind, "label": label, "confidence": 0.9, "bbox": BOX, **extra}


def _person(role: str | None = None) -> dict[str, Any]:
    return _obj("p1", "person", "person", **({"role": role, "role_confidence": 0.8} if role else {}))


def _detected(anomaly_class: str, *objects: dict[str, Any], relations: tuple = ()) -> dict[str, Any]:
    return {"anomaly_class": anomaly_class, "objects": list(objects), "relations": list(relations),
            "sensor_artefact": anomaly_class == "sensor_artefact", "obscured": False}


def _behaviour(*intents: tuple[str, float], role_mismatch: float = 0.0) -> dict[str, Any]:
    return {"persons": [{"object_id": "p1", "role_mismatch": role_mismatch,
                         "intents": [{"intent": i, "confidence": c} for i, c in intents]}],
            "relations": [], "health_emergency": False}


def _clip(demo: str, n: int = 1) -> str:
    return f"demo/{demo}/clip_{n}.mp4"


def _single(name: str, description: str, sensor_id: str, **answers: dict[str, Any]) -> Demo:
    """A demo with one event; `answers` are the answers per agent for its clip."""
    clip = _clip(name)
    return Demo(name, description, (ScriptedEvent(0, sensor_id, clip, BOX),),
                {agent: {clip: answer} for agent, answer in answers.items()})


_WIND_VIDEO, _WIND_AUDIO = _clip("wind"), "demo/wind/audio_1.wav"
_WIND = {"conditions": [{"condition": "high_wind", "confidence": 0.9, "wind_speed_kmh": 55}]}

DEMOS: dict[str, Demo] = {d.name: d for d in [
    Demo(
        "wind", "Low risk: a storm gust moves the garden, as forecast. Video and audio agree; no warning, no alarm.",
        (ScriptedEvent(0, "cam_garden", _WIND_VIDEO, BOX), ScriptedEvent(2, "mic_garden", _WIND_AUDIO)),
        {"object_detector": {_WIND_VIDEO: _detected("weather_environment")},
         "noise_interpreter": {_WIND_AUDIO: {"category": "weather", "confidence": 0.8, "is_sensor_artefact": False}},
         "weather_interpreter": {_WIND_VIDEO: _WIND, _WIND_AUDIO: _WIND}},
        forecast_conditions=("high_wind", "cloudy"), forecast_wind_kmh=50),
    Demo(
        "unforecast_wind", "Suspicious: the same gust, but the forecast says calm and sunny. A weather mismatch is "
                           "suspicious, like a person whose role does not fit: a warning.",
        (ScriptedEvent(0, "cam_garden", _clip("unforecast_wind"), BOX),),
        {"object_detector": {_clip("unforecast_wind"): _detected("weather_environment")},
         "weather_interpreter": {_clip("unforecast_wind"): _WIND}},
        forecast_conditions=("sunny",), forecast_wind_kmh=5),
    _single("lens_flare", "Low risk: the low sun causes a lens flare, a sensor artefact. Logged only.",
            "cam_garden", object_detector=_detected("sensor_artefact")),
    _single("gardener", "Plausible person: the gardener works in the garden. Known, allowed, harmless.",
            "cam_garden",
            object_detector=_detected("person", _person("gardener")),
            person_identifier={"verdict": "known", "person": "gustav", "confidence": 0.95},
            behavioural_interpreter=_behaviour(("move_to_area", 0.9))),
    _single("delivery", "Plausible person: a delivery person rings the bell. Unknown, but allowed at the entryway.",
            "cam_entry",
            object_detector=_detected("person", _person("delivery person")),
            person_identifier={"verdict": "unknown"},
            behavioural_interpreter=_behaviour(("delivery", 0.8), ("ring_the_bell", 0.6), role_mismatch=0.05)),
    _single("hooded_person", "Unclear: a hooded person in the garden cannot be identified. Unpermitted entry and an "
                             "odd behaviour: a warning that a human has to answer.",
            "cam_garden",
            object_detector=_detected("person", _person()),
            person_identifier={"verdict": "not_decidable", "cause": "clothing"},
            behavioural_interpreter=_behaviour(("unknown_activity", 0.7), role_mismatch=0.6)),
    _single("intruder", "Critical: an unknown person with a crowbar tries the garden door. Alarm.",
            "cam_garden",
            object_detector=_detected(
                "person", _person(), _obj("t1", "object", "crowbar"),
                relations=({"subject": "p1", "predicate": "carries", "object": "t1"},)),
            person_identifier={"verdict": "unknown"},
            behavioural_interpreter=_behaviour(("suspicious_activity", 0.9), role_mismatch=0.9)),
    _single("bear", "Critical: a bear in the garden. A warning, or an alarm if an entry point of the house is open.",
            "cam_garden", object_detector=_detected("animal", _obj("a1", "animal", "bear"))),
]}
