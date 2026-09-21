"""Signal envelope and value types shared by several signals."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Score = Annotated[float, Field(ge=0, le=1)]  # 0..1; higher = more suspicious/dangerous or more certain (confidence)
Media = Annotated[str, Field(min_length=1)]  # reference (path or URL) to an image, video or audio file


class SignalBase(BaseModel):
    """Envelope of every signal. `type` is the lower-case snake_case signal name and is set by each subclass."""

    model_config = ConfigDict(extra="forbid")

    type: str
    signal_id: str = Field(default_factory=lambda: f"sig_{uuid.uuid4().hex}")
    sent_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    in_reply_to: str | None = None  # signal_id of the request; only set on responses


class SituationSignal(SignalBase):
    """A signal that belongs to a situation. (Raw sensor events do not: they exist before their situation.)"""

    situation_id: str


class Part(BaseModel):
    """Base of the nested value objects (no envelope)."""

    model_config = ConfigDict(extra="forbid")


class BBox(Part):
    """Bounding box in relative image coordinates: top-left corner, width and height, all in [0, 1]."""

    x: Score
    y: Score
    w: Score
    h: Score
