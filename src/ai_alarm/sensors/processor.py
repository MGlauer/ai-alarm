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
from typing import Callable, Mapping

from ai_alarm.controller import SensorEventSignal
from ai_alarm.db.models import Sensor, utcnow
from ai_alarm.demos import DEMOS, Demo
from ai_alarm.signals import BBox

log = logging.getLogger(__name__)


class CctvAudioProcessor:
    def __init__(
        self, session_factory: Callable, demos: Mapping[str, Demo] = DEMOS, clock: Callable[[], datetime] = utcnow
    ):
        self.session_factory = session_factory
        self.demos = demos
        self.clock = clock
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
