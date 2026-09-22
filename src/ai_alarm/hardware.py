"""Speaker (Hardware) and the text message gateway, simulated: they log what they would do and remember it."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable

from ai_alarm.db.models import utcnow

log = logging.getLogger(__name__)


class SimulatedSpeaker:
    def __init__(self, clock: Callable[[], datetime] = utcnow):
        self.clock = clock
        self.spoken: list[tuple[datetime, str]] = []

    def speak(self, text: str) -> None:
        log.info("SPEAKER: %s", text)
        self.spoken.append((self.clock(), text))


class SimulatedTextGateway:
    """Delivers every message. `reachable = False` simulates that nobody can be reached."""

    def __init__(self, clock: Callable[[], datetime] = utcnow):
        self.clock = clock
        self.reachable = True
        self.sent: list[tuple[datetime, str, str]] = []

    def send(self, phone: str, message: str) -> bool:
        if not self.reachable:
            return False
        log.info("TEXT to %s: %s", phone, message)
        self.sent.append((self.clock(), phone, message))
        return True
