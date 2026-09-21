"""Person Identifier: matches a person in an image/video against the reference images of the knowledge base."""
from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from ai_alarm.agents.base import Agent, Model
from ai_alarm.signals import BBox, Media, Score, SignalBase

Verdict = Literal["known", "unknown", "not_decidable"]
# "unavailable" is not in this list: only the SI sets it, when the agent does not answer.
NotDecidableCause = Literal["clothing", "object on person", "external object", "low confidence", "unknown"]


class PersonIdentifierRequest(SignalBase):
    type: Literal["identify_person"] = "identify_person"
    video: Media | None = None
    image: Media | None = None
    object_id: str
    bbox: BBox

    @model_validator(mode="after")
    def _one_media(self):
        if (self.video is None) == (self.image is None):
            raise ValueError("exactly one of `video` and `image` is required")
        return self


class PersonIdentifierResponse(SignalBase):
    type: Literal["identification"] = "identification"
    verdict: Verdict
    person: str | None = None  # person id, only for "known"; name, roles etc. are looked up in the knowledge base
    confidence: Score | None = None  # only for "known"
    cause: NotDecidableCause | None = None  # only for "not_decidable"

    @model_validator(mode="after")
    def _fields_match_verdict(self):
        present = {k for k in ("person", "confidence", "cause") if getattr(self, k) is not None}
        required = {"known": {"person", "confidence"}, "unknown": set(), "not_decidable": {"cause"}}[self.verdict]
        if present != required:
            raise ValueError(
                f"verdict '{self.verdict}' needs exactly the fields {sorted(required)}, got {sorted(present)}"
            )
        return self


class PersonIdentifier(Agent[PersonIdentifierRequest, PersonIdentifierResponse]):
    name = "person_identifier"
    request_type = PersonIdentifierRequest
    response_type = PersonIdentifierResponse
    prompt = """
You identify the person at the given bounding box in the image or video by comparing them to the reference
images of the known persons.
- verdict "known": the person matches a known person. Give that person's id and your confidence (0..1).
- verdict "unknown": a person is visible but matches nobody.
- verdict "not_decidable": you cannot tell, e.g. because of a hood, an umbrella or another obstruction.
  Give the cause: "clothing", "object on person", "external object", "low confidence" or "unknown".
Never guess a person id.
"""

    def __init__(self, model: Model, threshold: float = 0.7):
        super().__init__(model)
        self.threshold = threshold  # a match below this confidence is reported as not decidable

    def postprocess(
        self, request: PersonIdentifierRequest, response: PersonIdentifierResponse
    ) -> PersonIdentifierResponse:
        if response.verdict == "known" and response.confidence < self.threshold:
            return response.model_copy(
                update={"verdict": "not_decidable", "person": None, "confidence": None, "cause": "low confidence"}
            )
        return response
