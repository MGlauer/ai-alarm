"""Weather Forecast Fetcher (Service): the local forecast from a pre-configured weather service.

To limit the traffic to outside systems it asks the service at most once an hour and otherwise returns cached data.
The service itself is a function `fetch() -> list[ForecastEntry]`; for the challenge it is simulated (see demos.py).
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Callable

from ai_alarm.agents.weather_interpreter import WeatherCondition
from ai_alarm.db.models import utcnow
from ai_alarm.signals import Part


class ForecastEntry(Part):
    valid_from: datetime
    valid_to: datetime
    conditions: list[WeatherCondition]
    wind_speed_kmh: float


class Forecast(Part):
    entries: list[ForecastEntry]  # those valid at the requested time
    fetched_at: datetime
    from_cache: bool


class WeatherForecastFetcher:
    def __init__(
        self,
        fetch: Callable[[], list[ForecastEntry]],
        clock: Callable[[], datetime] = utcnow,
        max_age_s: float = 3600,
    ):
        self.fetch = fetch
        self.clock = clock
        self.max_age_s = max_age_s
        self._lock = threading.Lock()
        self._entries: list[ForecastEntry] | None = None
        self._fetched_at: datetime | None = None
        self._asked_at: datetime | None = (
            None  # when the service was asked last, successfully or not
        )

    def get_forecast(self, at: datetime) -> Forecast:
        """The forecast entries valid at `at`. Raises if the service fails and there is nothing cached."""
        with self._lock:
            now = self.clock()
            from_cache = True
            if (
                self._asked_at is None
                or (now - self._asked_at).total_seconds() >= self.max_age_s
            ):
                self._asked_at = now
                try:
                    self._entries, self._fetched_at, from_cache = (
                        list(self.fetch()),
                        now,
                        False,
                    )
                except Exception:
                    if self._entries is None:
                        self._asked_at = (
                            None  # nothing to fall back on: ask again next time
                        )
                        raise
            entries = [e for e in self._entries if e.valid_from <= at < e.valid_to]
            return Forecast(
                entries=entries, fetched_at=self._fetched_at, from_cache=from_cache
            )

    def conditions_at(self, at: datetime) -> set[str]:
        """The conditions forecast for that time. This is what the controller asks for."""
        return {c for entry in self.get_forecast(at).entries for c in entry.conditions}

    def clear_cache(self) -> None:
        """Forget the cached forecast (a simulated service changes its mind when a new demo starts)."""
        with self._lock:
            self._entries = self._fetched_at = self._asked_at = None
