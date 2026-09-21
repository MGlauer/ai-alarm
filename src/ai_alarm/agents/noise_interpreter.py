"""Noise Interpreter: classifies an audio clip."""
from __future__ import annotations

from typing import Literal

from ai_alarm.agents.base import Agent, InvalidResponse
from ai_alarm.signals import Media, Score, SignalBase

NoiseCategory = Literal["human_activity", "animal", "weather", "technical_noise", "unknown"]


class NoiseInterpreterRequest(SignalBase):
    type: Literal["interpret_noise"] = "interpret_noise"
    evidence: Media
    area_id: str


class NoiseInterpreterResponse(SignalBase):
    type: Literal["noise_interpreted"] = "noise_interpreted"
    category: NoiseCategory
    confidence: Score
    is_sensor_artefact: bool


class NoiseInterpreter(Agent[NoiseInterpreterRequest, NoiseInterpreterResponse]):
    name = "noise_interpreter"
    request_type = NoiseInterpreterRequest
    response_type = NoiseInterpreterResponse
    prompt = """
You classify an audio clip as human_activity, animal, weather (e.g. wind, rain), technical_noise (e.g. an electrical
hum or a sensor artefact) or unknown, and give your confidence (0..1). Set is_sensor_artefact to true only if the
sound is caused by the sensor itself (then the category is technical_noise).
"""

    def postprocess(
        self, request: NoiseInterpreterRequest, response: NoiseInterpreterResponse
    ) -> NoiseInterpreterResponse:
        if response.is_sensor_artefact and response.category != "technical_noise":
            raise InvalidResponse(f"{self.name}: a sensor artefact must have the category technical_noise")
        return response
