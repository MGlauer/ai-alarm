"""CCTV/Audio Processor and Filter (Service, simulated).

The real processors watch a region, store the last minutes and report anomalies. This stub returns a predetermined
set of anomalous sequences: it plays a demo, event by event, as `{video_event/audio_event}` signals. The streaming
mode of the design is not simulated.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping

from ai_alarm.controller import SensorEventSignal
from ai_alarm.db.models import Sensor, utcnow
from ai_alarm.demos import DEMOS, Demo, known_person_for_object, noise_category_for_evidence, scene_for_evidence
from ai_alarm.media import (
    ANIMAL_SPRITES, IDENTITY_SPRITES, ROLE_SPRITES, render_audio_icon, render_camera_clip, render_camera_frame,
    render_noise_clip, render_placeholder,
)
from ai_alarm.signals import BBox

log = logging.getLogger(__name__)

# The repo's top-level media/ (see scripts/generate_media.py), *not* the gitignored data/media/ that
# media.py used to write to.
DEFAULT_ASSETS_DIR = Path(__file__).resolve().parents[3] / "media"


def _is_night(at: datetime) -> bool:
    """Whether a rendered scene should be dark, decided once, the same way everywhere it is needed, from the real
    time the picture is of -- never from a demo's own scripting (`ScriptedEvent.night` is cosmetic-only and is not
    read for this), so every camera agrees on day vs night at any given moment."""
    return at.hour < 6 or at.hour >= 20


class CctvAudioProcessor:
    def __init__(
        self, session_factory: Callable, demos: Mapping[str, Demo] = DEMOS, clock: Callable[[], datetime] = utcnow,
        assets_dir: Path = DEFAULT_ASSETS_DIR,
    ):
        self.session_factory = session_factory
        self.demos = demos
        self.clock = clock
        self.assets_dir = assets_dir
        self.demo: Demo | None = None  # the demo that was played last: the agents' mock models read its script

    def play(self, name: str, emit: Callable[[SensorEventSignal], object], time_scale: float = 1.0,
             wait: bool = False) -> None:
        """Play a demo: `emit` each of its events after its delay (times `time_scale`; 0 = at once). Returns at once
        unless `wait` is set. Raises `LookupError` for an unknown demo."""
        if name not in self.demos:
            raise LookupError(f"unknown demo {name}")
        self.demo = demo = self.demos[name]
        thread = threading.Thread(target=self._run, args=(demo, emit, time_scale), daemon=True)
        thread.start()
        if wait:
            thread.join()

    def latest_audio(self, area_id: str, at: datetime) -> str | None:
        """The most recent audio of the area (the controller asks for it when a camera view is obscured)."""
        return self.demo.audio.get(area_id) if self.demo else None

    def idle_frame(
        self, *, sensor_id: str, kind: str, area_id: str, area_name: str, at: datetime
    ) -> tuple[bytes, str]:
        """What a sensor shows before it has ever reported anything, and its mimetype: an ambient view of the area
        it watches (or an ambient audio icon), never a blank tile — a real camera always has a picture, anomaly or
        not. During the day, prefers the pre-rendered default scene for the sensor
        (media/scenes/sensors/<id>.png) over drawing one; at night it always draws one fresh instead, so it goes
        dark like every other camera does (see `frame`) rather than sitting on a permanently daylit picture."""
        if kind != "camera":
            return render_audio_icon(sensor_id=sensor_id, area_name=area_name, at=at), "image/png"
        night = _is_night(at)
        prerendered = self.assets_dir / "scenes" / "sensors" / f"{sensor_id}.png"
        if not night and prerendered.is_file():
            return prerendered.read_bytes(), "image/png"
        return render_camera_frame(sensor_id=sensor_id, area_id=area_id, area_name=area_name, at=at, objects=[],
                                   weather=None, night=night), "image/png"

    def frame(
        self, *, evidence: str, kind: str, sensor_id: str, area_id: str, area_name: str, at: datetime
    ) -> tuple[bytes, str]:
        """An animated clip (or audio icon) for a piece of evidence, and its mimetype: built fresh for every
        request from the real area/weather this evidence belongs to plus the character/animal sprites
        (media/sprites/) for whatever the demo's mock object detector (and person identifier) reports -- never a
        pre-baked scene, which could (and did) drift out of sync with the area a piece of evidence actually shows.
        Night is decided from `at` (when the evidence was captured), the same way `idle_frame` decides it from the
        current time, rather than each demo scripting its own -- otherwise different cameras could each show a
        different time of day at once. Falls back to a plain placeholder for evidence no known demo produced."""
        if kind != "video":
            return render_audio_icon(sensor_id=sensor_id, area_name=area_name, at=at), "image/png"
        spec = scene_for_evidence(evidence, self.demos)
        if spec is None:
            return render_placeholder(sensor_id=sensor_id, area_name=area_name, at=at), "image/png"
        objects, weather, _ = spec
        objects = [self._with_sprite(o, evidence) for o in objects]
        return render_camera_clip(sensor_id=sensor_id, area_id=area_id, area_name=area_name, at=at,
                                  objects=objects, weather=weather, night=_is_night(at)), "image/gif"

    def _with_sprite(self, obj: dict, evidence: str) -> dict:
        """`obj`, plus the name of the sprite (see IDENTITY_SPRITES etc.) it should be drawn as, if any."""
        if obj.get("kind") == "person":
            person_id = known_person_for_object(evidence, obj.get("object_id", ""), self.demos)
            sprite = IDENTITY_SPRITES.get(person_id) or ROLE_SPRITES.get(obj.get("role")) or "unknown_person"
        elif obj.get("kind") == "animal":
            label = obj.get("label", "").lower()
            sprite = next((name for key, name in ANIMAL_SPRITES.items() if key in label), None)
        else:
            sprite = None
        return {**obj, "sprite": sprite} if sprite else obj

    def audio(self, evidence: str) -> bytes:
        """The actual sound of a piece of audio evidence, for on-demand playback (the design document's "audio of
        each [sensor] can be streamed on-demand"). No demo has pre-rendered audio of its own (only the shared
        ambient sound below), so this is always synthesised per the noise interpreter's scripted category."""
        category = noise_category_for_evidence(evidence, self.demos)
        return render_noise_clip(category or "unknown", seed=evidence)

    def idle_audio(self, sensor_id: str) -> bytes:
        """The ambient sound of a microphone before it has reported anything -- the one sound shared by every
        microphone's default stream (media/scenes/audio/ambient.wav), falling back to a synthesised one if it is
        missing."""
        prerendered = self.assets_dir / "scenes" / "audio" / "ambient.wav"
        if prerendered.is_file():
            return prerendered.read_bytes()
        return render_noise_clip("ambient", seed=sensor_id)

    def _run(self, demo: Demo, emit: Callable[[SensorEventSignal], object], time_scale: float) -> None:
        started = time.monotonic()
        for event in sorted(demo.events, key=lambda e: e.delay_s):
            time.sleep(max(0.0, event.delay_s * time_scale - (time.monotonic() - started)))
            try:
                emit(self._signal(event))
            except Exception:  # noqa: BLE001 - one failing event must not stop the demo
                log.exception("demo %s: event on %s failed", demo.name, event.sensor_id)

    def _signal(self, event) -> SensorEventSignal:
        with self.session_factory() as s:
            sensor = s.get(Sensor, event.sensor_id)
            if sensor is None:
                raise LookupError(f"unknown sensor {event.sensor_id}")
            is_video = sensor.kind == "camera"
            return SensorEventSignal(
                type="video_event" if is_video else "audio_event", event_id=f"evt_{uuid.uuid4().hex}",
                sensor_id=sensor.id, area_id=sensor.area_id, start_time=self.clock(), evidence=event.evidence,
                bbox=BBox(**event.bbox) if is_video and event.bbox else None)
