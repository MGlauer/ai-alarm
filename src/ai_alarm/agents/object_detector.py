"""Object Detector: detects and classifies the objects of an anomaly in a video/image."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import model_validator

from ai_alarm.agents.base import Agent, InvalidResponse
from ai_alarm.signals import BBox, Media, Part, Score, SituationSignal

AnomalyClass = Literal[
    "person", "animal", "weather_environment", "sensor_artefact", "unclear"
]


class ObjectDetectorRequest(SituationSignal):
    type: Literal["detect_objects"] = "detect_objects"
    evidence: Media
    area_id: str
    bbox_hint: BBox | None = None  # rough area of the anomaly, only for video events


class DetectedObject(Part):
    object_id: (
        str  # unique within the response; other signals refer to the object by this id
    )
    kind: Literal["person", "animal", "object"]
    label: str  # e.g. "woman", "bear", "crowbar"
    confidence: Score
    bbox: BBox
    role: str | None = None  # persons only, e.g. "gardener"
    role_confidence: Score | None = None

    @model_validator(mode="after")
    def _role_only_for_persons(self):
        if self.kind != "person" and (
            self.role is not None or self.role_confidence is not None
        ):
            raise ValueError("role and role_confidence are only allowed for persons")
        if self.role is None and self.role_confidence is not None:
            raise ValueError("role_confidence requires a role")
        return self


class ObjectRelation(Part):
    subject: str  # object_id
    predicate: str  # display only, e.g. "carries", "leaves"
    object: str  # object_id


class ObjectDetectorResponse(SituationSignal):
    type: Literal["objects_detected"] = "objects_detected"
    anomaly_class: AnomalyClass
    objects: list[DetectedObject]
    relations: list[ObjectRelation] = []
    sensor_artefact: bool
    obscured: bool  # the view of the camera is blocked


class ObjectDetector(Agent[ObjectDetectorRequest, ObjectDetectorResponse]):
    name = "object_detector"
    request_type = ObjectDetectorRequest
    response_type = ObjectDetectorResponse
    prompt = """
You detect and classify the objects of an anomaly in a CCTV image or video. Report all objects with kind
(person, animal, object), label, confidence and bounding box, and the relations between them (a man leaving a car,
a woman carrying a crowbar). For persons also give their likely role (e.g. delivery person, gardener) with a
confidence. Classify the anomaly as person, animal, weather_environment, sensor_artefact or unclear.
Use sensor_artefact (e.g. lens flare, compression glitch, insect on the lens) only if the anomaly shows no object;
then also set sensor_artefact to true. Set obscured to true if the view of the camera is blocked.
You also get the `known_roles` of the house. Use one of them for a person if it fits; otherwise name the role in
your own words.
"""

    def context(self, request: ObjectDetectorRequest) -> dict[str, Any]:
        return {"known_roles": self.kb.role_names()}

    def postprocess(
        self, request: ObjectDetectorRequest, response: ObjectDetectorResponse
    ) -> ObjectDetectorResponse:
        ids = [o.object_id for o in response.objects]
        kinds = {o.kind for o in response.objects}
        if len(set(ids)) != len(ids):
            raise InvalidResponse(f"{self.name}: object ids are not unique")
        if response.sensor_artefact != (response.anomaly_class == "sensor_artefact"):
            raise InvalidResponse(
                f"{self.name}: sensor_artefact contradicts anomaly_class"
            )
        if response.anomaly_class == "sensor_artefact" and response.objects:
            raise InvalidResponse(
                f"{self.name}: a sensor artefact must not contain objects"
            )
        if (
            response.anomaly_class in ("person", "animal")
            and response.anomaly_class not in kinds
        ):
            raise InvalidResponse(
                f"{self.name}: anomaly_class '{response.anomaly_class}' but no such object"
            )
        if any(r.subject not in ids or r.object not in ids for r in response.relations):
            raise InvalidResponse(
                f"{self.name}: a relation refers to an unknown object"
            )
        return response
