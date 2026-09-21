"""Behavioural Interpreter: predicts the intentions and interrelations of the persons in a sequence of frames."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from ai_alarm.agents.base import Agent, InvalidResponse
from ai_alarm.agents.person_identifier import Verdict
from ai_alarm.signals import Media, Part, Score, SituationSignal

Intent = Literal[
    "delivery", "pickup", "ring_the_bell", "move_to_area", "invite_person", "accompany_person",
    "health_emergency", "unknown_activity", "suspicious_activity",
]


class PersonContext(Part):
    object_id: str  # as given by the object detector
    verdict: Verdict  # identity verdict of the person identifier
    kb_roles: list[str] = []  # roles from the knowledge base (known persons only)
    predicted_role: str | None = None  # role predicted by the object detector


class BehaviouralInterpreterRequest(SituationSignal):
    type: Literal["interpret_behaviour"] = "interpret_behaviour"
    evidence: list[Media] = Field(min_length=1)  # frames
    persons: list[PersonContext] = Field(min_length=1)
    area_id: str


class RankedIntent(Part):
    intent: Intent
    confidence: Score


class IntentAssessment(Part):
    object_id: str
    intents: list[RankedIntent] = Field(min_length=1)  # most likely first
    role_mismatch: Score  # degree of mismatch between the predicted intent and the person's role


class PersonRelation(Part):
    subject: str  # object_id
    predicate: Literal["accompanies", "invites"]
    object: str  # object_id


class BehaviouralInterpreterResponse(SituationSignal):
    type: Literal["behaviour_interpreted"] = "behaviour_interpreted"
    persons: list[IntentAssessment]
    relations: list[PersonRelation] = []
    health_emergency: bool


class BehaviouralInterpreter(Agent[BehaviouralInterpreterRequest, BehaviouralInterpreterResponse]):
    name = "behavioural_interpreter"
    request_type = BehaviouralInterpreterRequest
    response_type = BehaviouralInterpreterResponse
    prompt = """
You analyse a sequence of frames showing one or more persons and predict their intentions and interrelations.
Each person comes with an identity verdict and their roles (from the knowledge base and as predicted from the image).
For every given person return their intents ranked by confidence, choosing from: delivery, pickup, ring_the_bell,
move_to_area, invite_person, accompany_person, health_emergency, unknown_activity, suspicious_activity.
Also return role_mismatch (0..1): how badly the most likely intent fits the person's role (0 = fits, 1 = does not
fit at all). Report relations between persons (accompanies, invites) and set health_emergency to true if you
suspect one. You also get the `area` where this happens, with the areas it is connected to and whether it is the
entryway or the drop-off point for deliveries.
"""

    def context(self, request: BehaviouralInterpreterRequest) -> dict[str, Any]:
        return {"area": self.kb.area_info(request.area_id)}

    def postprocess(
        self, request: BehaviouralInterpreterRequest, response: BehaviouralInterpreterResponse
    ) -> BehaviouralInterpreterResponse:
        expected = sorted(p.object_id for p in request.persons)
        if sorted(p.object_id for p in response.persons) != expected:
            raise InvalidResponse(f"{self.name}: expected exactly one entry for each of {expected}")
        if any(r.subject not in expected or r.object not in expected for r in response.relations):
            raise InvalidResponse(f"{self.name}: a relation refers to an unknown person")
        for p in response.persons:
            p.intents.sort(key=lambda i: i.confidence, reverse=True)
        if any(p.intents[0].intent == "health_emergency" for p in response.persons):
            response.health_emergency = True  # never let the flag contradict the ranked intents
        return response
