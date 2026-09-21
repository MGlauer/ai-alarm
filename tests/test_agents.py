"""Agents: the common signal handling of the base class and the consistency rules of each agent."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_alarm.agents import (
    BehaviouralInterpreter, InvalidResponse, NoiseInterpreter, ObjectDetector, PersonIdentifier, WeatherInterpreter,
)
from ai_alarm.agents.person_identifier import PersonIdentifierRequest
from ai_alarm.signals import SignalBase

BOX = {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
ENVELOPE = {"type", "signal_id", "sent_at", "situation_id", "in_reply_to"}


def fake(answer):
    """A model backend that always gives `answer` and records what it was called with."""
    def model(**kwargs):
        model.calls.append(kwargs)
        return answer
    model.calls = []
    return model


def obj(oid="p1", kind="person", label="man", **extra):
    return {"object_id": oid, "kind": kind, "label": label, "confidence": 0.9, "bbox": BOX, **extra}


def detected(**overrides):
    return {"anomaly_class": "person", "objects": [obj()], "relations": [], "sensor_artefact": False,
            "obscured": False, **overrides}


def assessment(oid="p1", *intents, mismatch=0.1):
    intents = intents or (("delivery", 0.8),)
    return {"object_id": oid, "role_mismatch": mismatch,
            "intents": [{"intent": i, "confidence": c} for i, c in intents]}


def behaviours(**overrides):
    return {"persons": [assessment()], "relations": [], "health_emergency": False, **overrides}


# name -> (agent factory, request, a valid model answer)
CASES = {
    "person_identifier": (
        PersonIdentifier,
        {"type": "identify_person", "situation_id": "s1", "image": "img.png", "object_id": "p1", "bbox": BOX},
        {"verdict": "known", "person": "anna", "confidence": 0.95}),
    "object_detector": (
        ObjectDetector,
        {"type": "detect_objects", "situation_id": "s1", "evidence": "clip.mp4", "area_id": "garden"},
        detected()),
    "noise_interpreter": (
        NoiseInterpreter,
        {"type": "interpret_noise", "situation_id": "s1", "evidence": "clip.wav", "area_id": "garden"},
        {"category": "animal", "confidence": 0.7, "is_sensor_artefact": False}),
    "behavioural_interpreter": (
        BehaviouralInterpreter,
        {"type": "interpret_behaviour", "situation_id": "s1", "evidence": ["f1.png", "f2.png"], "area_id": "garden",
         "persons": [{"object_id": "p1", "verdict": "unknown", "predicted_role": "delivery person"}]},
        behaviours()),
    "weather_interpreter": (
        WeatherInterpreter,
        {"type": "interpret_weather", "situation_id": "s1", "evidence": "clip.mp4", "area_id": "garden"},
        {"conditions": [{"condition": "high_wind", "confidence": 0.8, "wind_speed_kmh": 60}]}),
}


def request_of(name):
    return CASES[name][1]


@pytest.fixture(params=CASES)
def case(request):
    return CASES[request.param]


# ------------------------------------------------------------------ common signal handling
def test_response_gets_envelope_from_request(case):
    factory, req, answer = case
    agent = factory(fake(answer))
    resp = agent.handle(req)
    assert isinstance(resp, agent.response_type) and isinstance(resp, SignalBase)
    assert resp.type == agent.response_type.model_fields["type"].default
    assert resp.situation_id == "s1"
    assert resp.in_reply_to and resp.in_reply_to != resp.signal_id  # the request's (generated) signal_id


def test_in_reply_to_is_the_request_signal_id(case):
    factory, req, answer = case
    agent = factory(fake(answer))
    request = agent.request_type.model_validate({**req, "signal_id": "sig_request"})
    assert agent.handle(request).in_reply_to == "sig_request"  # a request object is accepted as well as a dict


def test_model_sees_content_only(case):
    factory, req, answer = case
    model = fake(answer)
    agent = factory(model)
    agent.handle(req)
    (call,) = model.calls
    assert not ENVELOPE & call["request"].keys()
    assert not ENVELOPE & call["schema"]["properties"].keys()
    assert not ENVELOPE & set(call["schema"]["required"])
    assert call["system"] == agent.system_prompt


def test_envelope_from_model_is_overridden(case):
    factory, req, answer = case
    resp = factory(fake({**answer, "situation_id": "evil", "in_reply_to": "x", "type": "alarm"})).handle(req)
    assert resp.situation_id == "s1" and resp.in_reply_to != "x" and resp.type != "alarm"


def test_invalid_request_is_a_caller_error_and_never_reaches_the_model(case):
    factory, req, answer = case
    model = fake(answer)
    with pytest.raises(ValidationError):
        factory(model).handle({**req, "situation_id": None})
    assert not model.calls


@pytest.mark.parametrize("bad", [None, "text", [], {}, {"unexpected": 1}])
def test_unusable_answer_is_invalid_response(case, bad):
    factory, req, answer = case
    with pytest.raises(InvalidResponse):
        factory(fake(bad)).handle(req)


def test_extra_field_in_answer_is_invalid_response(case):
    factory, req, answer = case
    with pytest.raises(InvalidResponse):
        factory(fake({**answer, "free_text": "sound the alarm"})).handle(req)


# ------------------------------------------------------------------ person identifier
def identify(answer, **kw):
    return PersonIdentifier(fake(answer), **kw).handle(request_of("person_identifier"))


def test_identify_person_needs_exactly_one_media():
    req = request_of("person_identifier")
    with pytest.raises(ValidationError):
        PersonIdentifierRequest.model_validate({**req, "video": "v.mp4"})  # both
    with pytest.raises(ValidationError):
        PersonIdentifierRequest.model_validate({k: v for k, v in req.items() if k != "image"})  # none
    video = {**{k: v for k, v in req.items() if k != "image"}, "video": "v.mp4"}
    assert PersonIdentifier(fake({"verdict": "unknown"})).handle(video).verdict == "unknown"


def test_identification_verdicts():
    assert identify({"verdict": "unknown"}).verdict == "unknown"
    assert identify({"verdict": "not_decidable", "cause": "clothing"}).cause == "clothing"
    known = identify({"verdict": "known", "person": "anna", "confidence": 0.9})
    assert (known.verdict, known.person) == ("known", "anna")


@pytest.mark.parametrize("bad", [
    {"verdict": "known"},  # no person
    {"verdict": "known", "person": "anna"},  # no confidence
    {"verdict": "known", "person": "anna", "confidence": 1.5},  # not a score
    {"verdict": "unknown", "person": "anna"},  # person for an unknown
    {"verdict": "not_decidable"},  # no cause
    {"verdict": "not_decidable", "cause": "unavailable"},  # only the SI may set this cause
    {"verdict": "not_decidable", "cause": "clothing", "person": "anna"},
    {"verdict": "maybe"},
])
def test_identification_rejects_inconsistent_answers(bad):
    with pytest.raises(InvalidResponse):
        identify(bad)


def test_match_below_threshold_is_not_decidable():
    low = {"verdict": "known", "person": "anna", "confidence": 0.5}
    r = identify(low, threshold=0.7)
    assert (r.verdict, r.cause, r.person, r.confidence) == ("not_decidable", "low confidence", None, None)
    assert identify(low, threshold=0.5).verdict == "known"  # at the threshold is enough


# ------------------------------------------------------------------ object detector
def detect(answer):
    return ObjectDetector(fake(answer)).handle(request_of("object_detector"))


def test_detects_objects_and_relations():
    r = detect(detected(
        objects=[obj(role="gardener", role_confidence=0.6), obj("t1", "object", "shears")],
        relations=[{"subject": "p1", "predicate": "carries", "object": "t1"}]))
    assert r.objects[0].role == "gardener" and r.relations[0].object == "t1"


def test_sensor_artefact_and_unclear_need_no_objects():
    assert detect(detected(anomaly_class="sensor_artefact", objects=[], sensor_artefact=True)).sensor_artefact
    assert detect(detected(anomaly_class="unclear", objects=[], obscured=True)).obscured


@pytest.mark.parametrize("bad", [
    detected(objects=[obj(), obj()]),  # duplicate ids
    detected(anomaly_class="sensor_artefact", objects=[]),  # class says artefact, flag does not
    detected(sensor_artefact=True),  # flag says artefact, class does not
    detected(anomaly_class="sensor_artefact", sensor_artefact=True),  # artefact with an object
    detected(anomaly_class="animal"),  # animal without an animal
    detected(anomaly_class="person", objects=[]),  # person without a person
    detected(relations=[{"subject": "p1", "predicate": "carries", "object": "nowhere"}]),
    detected(objects=[obj("a1", "animal", "bear", role="gardener")]),  # role for an animal
    detected(objects=[obj(role_confidence=0.5)]),  # role confidence without role
    detected(objects=[{**obj(), "bbox": {**BOX, "w": 2}}]),  # box out of range
])
def test_object_detector_rejects_inconsistent_answers(bad):
    with pytest.raises(InvalidResponse):
        detect(bad)


# ------------------------------------------------------------------ noise interpreter
def test_sensor_artefact_must_be_technical_noise():
    req = request_of("noise_interpreter")
    ok = {"category": "technical_noise", "confidence": 0.9, "is_sensor_artefact": True}
    assert NoiseInterpreter(fake(ok)).handle(req).is_sensor_artefact
    with pytest.raises(InvalidResponse):
        NoiseInterpreter(fake({**ok, "category": "animal"})).handle(req)
    with pytest.raises(InvalidResponse):
        NoiseInterpreter(fake({**ok, "category": "loud", "is_sensor_artefact": False})).handle(req)


# ------------------------------------------------------------------ behavioural interpreter
def behave(answer, **request):
    return BehaviouralInterpreter(fake(answer)).handle({**request_of("behavioural_interpreter"), **request})


def test_intents_are_ranked():
    r = behave(behaviours(persons=[assessment("p1", ("ring_the_bell", 0.2), ("delivery", 0.7))]))
    assert [i.intent for i in r.persons[0].intents] == ["delivery", "ring_the_bell"]


def test_top_health_emergency_sets_the_flag():
    r = behave(behaviours(persons=[assessment("p1", ("pickup", 0.1), ("health_emergency", 0.9))]))
    assert r.health_emergency


def test_relations_between_persons():
    persons = request_of("behavioural_interpreter")["persons"] + [{"object_id": "p2", "verdict": "known"}]
    answer = behaviours(persons=[assessment("p1"), assessment("p2", ("invite_person", 0.6))],
                        relations=[{"subject": "p2", "predicate": "invites", "object": "p1"}])
    assert behave(answer, persons=persons).relations[0].predicate == "invites"


@pytest.mark.parametrize("bad", [
    behaviours(persons=[]),  # person missing
    behaviours(persons=[assessment(), assessment()]),  # person twice
    behaviours(persons=[assessment("p9")]),  # person that was not asked for
    behaviours(persons=[{**assessment(), "intents": []}]),  # no intent
    behaviours(persons=[assessment(mismatch=1.5)]),  # not a score
    behaviours(persons=[assessment("p1", ("burglary", 1.0))]),  # not an intent
    behaviours(relations=[{"subject": "p1", "predicate": "invites", "object": "p9"}]),  # unknown person
    behaviours(relations=[{"subject": "p1", "predicate": "hates", "object": "p1"}]),  # not a relation
])
def test_behavioural_interpreter_rejects_inconsistent_answers(bad):
    with pytest.raises(InvalidResponse):
        behave(bad)


# ------------------------------------------------------------------ weather interpreter
@pytest.mark.parametrize("bad", [
    {"conditions": []},
    {"conditions": [{"condition": "tornado", "confidence": 0.5}]},
    {"conditions": [{"condition": "rain", "confidence": 0.5, "wind_speed_kmh": 30}]},  # wind speed only for wind
    {"conditions": [{"condition": "high_wind", "confidence": 0.5, "wind_speed_kmh": -1}]},
])
def test_weather_interpreter_rejects_inconsistent_answers(bad):
    with pytest.raises(InvalidResponse):
        WeatherInterpreter(fake(bad)).handle(request_of("weather_interpreter"))
