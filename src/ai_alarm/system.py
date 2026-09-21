"""The complete system, wired together with simulated sensors, speaker, text gateway and weather service, and
scripted (mock) agent models. `System.play(demo)` lets the simulated processors report a demo's anomalies."""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Callable, Mapping

from flask import Flask

from ai_alarm.agents import (
    BehaviouralInterpreter, NoiseInterpreter, ObjectDetector, PersonIdentifier, WeatherInterpreter,
)
from ai_alarm.agents.mock import ScriptedModel
from ai_alarm.api import create_app
from ai_alarm.comm import CommConfig, CommunicationUnit
from ai_alarm.controller import Controller, ControllerConfig
from ai_alarm.db import DEFAULT_URL, init_db, make_engine, make_session_factory
from ai_alarm.db.models import Area, utcnow
from ai_alarm.demos import DEMOS, Demo
from ai_alarm.hardware import SimulatedSpeaker, SimulatedTextGateway
from ai_alarm.kb import KnowledgeBase
from ai_alarm.seed import seed_demo_house
from ai_alarm.sensors import CctvAudioProcessor
from ai_alarm.weather import WeatherForecastFetcher
from ai_alarm.workflow import SituationWorkflows, WorkflowConfig, make_checkpointer

log = logging.getLogger(__name__)


class System:
    def __init__(
        self, db_url: str = DEFAULT_URL, checkpoint_path: str = "data/checkpoints/checkpoints.db", *,
        seed: bool = True, demos: Mapping[str, Demo] = DEMOS, clock: Callable[[], datetime] = utcnow,
        tick_interval_s: float = 5,
        controller_config: ControllerConfig = ControllerConfig(), workflow_config: WorkflowConfig = WorkflowConfig(),
        comm_config: CommConfig = CommConfig(),
    ):
        self.engine = make_engine(db_url)
        init_db(self.engine)
        self.session_factory = make_session_factory(self.engine)
        if seed:
            self._seed_if_empty()
        self.tick_interval_s = tick_interval_s
        self._stop = threading.Event()
        self.kb = KnowledgeBase(self.session_factory)

        # simulated outside world
        self.processor = CctvAudioProcessor(self.session_factory, dict(demos), clock=clock)
        self.speaker, self.gateway = SimulatedSpeaker(clock), SimulatedTextGateway(clock)
        self.forecast = WeatherForecastFetcher(
            lambda: self.processor.demo.forecast(clock()) if self.processor.demo else [], clock)

        # the agents, with mock models that answer what the running demo says
        def agent(agent_class, name):
            script = lambda: (self.processor.demo.answers if self.processor.demo else {}).get(name, {})  # noqa: E731
            return agent_class(ScriptedModel(script), self.kb)

        agents = {
            "object_detector": agent(ObjectDetector, "object_detector"),
            "person_identifier": agent(PersonIdentifier, "person_identifier"),
            "behavioural_interpreter": agent(BehaviouralInterpreter, "behavioural_interpreter"),
            "noise_interpreter": agent(NoiseInterpreter, "noise_interpreter"),
            "weather_interpreter": agent(WeatherInterpreter, "weather_interpreter"),
        }
        self.checkpointer = make_checkpointer(checkpoint_path)
        self.workflows = SituationWorkflows(
            session_factory=self.session_factory, kb=self.kb, forecast=self.forecast,
            checkpointer=self.checkpointer, config=workflow_config, clock=clock, **agents)
        self.controller = Controller(
            session_factory=self.session_factory, kb=self.kb,
            comm=CommunicationUnit(self.kb, self.gateway, comm_config), interpreters=self.workflows,
            speaker=self.speaker, weather=self.forecast, audio=self.processor,
            behavioural_interpreter=agents["behavioural_interpreter"], noise_interpreter=agents["noise_interpreter"],
            config=controller_config, clock=clock)
        self.workflows.controller = self.controller
        self.app: Flask = create_app(self.session_factory, self.controller, simulation=self)

    def _seed_if_empty(self) -> None:
        with self.session_factory() as s:
            empty = s.query(Area).first() is None
        if empty:
            seed_demo_house(self.session_factory)

    # ------------------------------------------------------------------ simulation
    @property
    def demos(self) -> dict[str, str]:
        """The demos that can be played: name -> description."""
        return {name: demo.description for name, demo in self.processor.demos.items()}

    def play(self, name: str, time_scale: float = 1.0, wait: bool = False) -> None:
        """Let the simulated processors report the anomalies of a demo. Raises `LookupError` for an unknown demo."""
        self.forecast.clear_cache()  # the simulated weather service has a new forecast for each demo
        self.processor.play(name, self.controller.handle_sensor_event, time_scale, wait)

    # ------------------------------------------------------------------ running
    def start(self) -> None:
        """Start the workflow worker and the timer that ticks the workflows and applies the fallback policy."""
        threading.Thread(target=self.workflows.run_worker, name="workflows", daemon=True).start()
        threading.Thread(target=self._tick_loop, name="ticker", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        """Release the database and checkpoint connections."""
        self.stop()
        self.checkpointer.conn.close()
        self.engine.dispose()

    def _tick_loop(self) -> None:
        while not self._stop.wait(self.tick_interval_s):
            try:
                self.workflows.tick()
                self.controller.check_timeouts()
            except Exception:  # noqa: BLE001 - keep ticking
                log.exception("tick failed")


def build_system(*args, **kwargs) -> System:
    return System(*args, **kwargs)
