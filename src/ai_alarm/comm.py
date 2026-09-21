"""Communication Unit (Deterministic component): decides the colour of warnings and sends the text messages.

Messages are completed from fixed templates with data from the signal; nothing is generated. Warnings and alarms go
to the same recipients: all primary contacts and all contacts who are currently in the surveilled area.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone, tzinfo
from typing import Protocol

from ai_alarm.kb import KnowledgeBase

TEMPLATES = {
    "suspicious person": "A suspicious person was detected in {area}.",
    "strong suspicion": "A person in {area} is strongly suspected of a break-in.",
    "unpermitted entry": "A person entered {area} without permission.",
    "dangerous animal": "A dangerous animal was detected in {area}.",
    "vision obstructed": "The view of the camera in {area} is obstructed.",
    "obstruction by human activity": "The view of the camera in {area} is obstructed by a person.",
    "unclear situation": "Something unclear was detected in {area}.",
}


def in_window(t: time, start: time, end: time) -> bool:
    """Is `t` in the daily window [start, end)? The window may span midnight (22:00-06:00)."""
    return t >= start or t < end if start > end else start <= t < end


class TextTransport(Protocol):
    def send(self, phone: str, message: str) -> bool:
        """Send a text message. True if it was delivered."""


@dataclass(frozen=True)
class CommConfig:
    orange_threshold: float = 0.75  # a warning above this suspicion is orange, otherwise yellow
    night_orange_threshold: float = 0.5  # lowered at night: users answer less often, so unanswered ones are elevated
    night_start: time = time(22, 0)
    night_end: time = time(6, 0)
    tz: tzinfo = timezone.utc  # the time zone in which night is evaluated


@dataclass(frozen=True)
class Delivery:
    contact_id: str
    message: str
    sent: bool


class CommunicationUnit:
    def __init__(self, kb: KnowledgeBase, transport: TextTransport, config: CommConfig = CommConfig()):
        self.kb = kb
        self.transport = transport
        self.config = config

    def is_night(self, now: datetime) -> bool:
        return in_window(now.astimezone(self.config.tz).time(), self.config.night_start, self.config.night_end)

    def colour(self, suspicion: float, now: datetime) -> str:
        threshold = self.config.night_orange_threshold if self.is_night(now) else self.config.orange_threshold
        return "orange" if suspicion > threshold else "yellow"

    def send_warning(self, warning_id: str, cause: str, area_id: str, suspicion: float, colour: str) -> list[Delivery]:
        text = TEMPLATES[cause].format(area=self.kb.area_info(area_id)["name"])
        return self._send(f"{colour.upper()} WARNING ({suspicion:.2f}): {text} Elevate or dismiss it in the app "
                          f"(warning {warning_id}).")

    def send_alarm(self, alarm_id: str, cause: str, area_id: str) -> list[Delivery]:
        text = TEMPLATES[cause].format(area=self.kb.area_info(area_id)["name"])
        return self._send(f"ALARM: {text} (alarm {alarm_id})")

    def _send(self, message: str) -> list[Delivery]:
        deliveries = []
        for recipient in self.kb.recipients():
            try:
                sent = self.transport.send(recipient.phone, message)
            except Exception:  # noqa: BLE001 - one unreachable contact must not stop the others
                sent = False
            deliveries.append(Delivery(recipient.contact_id, message, sent))
        return deliveries
