"""Weather Interpreter: describes the weather conditions visible/audible in a video or audio clip."""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ai_alarm.agents.base import Agent
from ai_alarm.signals import Media, Part, Score, SituationSignal

WeatherCondition = Literal["high_wind", "snowfall", "fog", "rain", "cloudy", "sunny", "unknown"]


class WeatherInterpreterRequest(SituationSignal):
    type: Literal["interpret_weather"] = "interpret_weather"
    evidence: Media
    area_id: str


class ObservedCondition(Part):
    condition: WeatherCondition
    confidence: Score
    wind_speed_kmh: float | None = Field(default=None, ge=0)  # approximate, only for high wind

    @model_validator(mode="after")
    def _wind_speed_only_for_high_wind(self):
        if self.wind_speed_kmh is not None and self.condition != "high_wind":
            raise ValueError("wind_speed_kmh is only allowed for high_wind")
        return self


class WeatherInterpreterResponse(SituationSignal):
    type: Literal["weather_interpreted"] = "weather_interpreted"
    conditions: list[ObservedCondition] = Field(min_length=1)


class WeatherInterpreter(Agent[WeatherInterpreterRequest, WeatherInterpreterResponse]):
    name = "weather_interpreter"
    request_type = WeatherInterpreterRequest
    response_type = WeatherInterpreterResponse
    prompt = """
You analyse the weather conditions in a video or audio clip. Report every condition you observe with your
confidence (0..1), choosing from: high_wind (also give the approximate wind speed in km/h), snowfall, fog, rain,
cloudy, sunny, unknown. If you cannot tell, report unknown.
"""
