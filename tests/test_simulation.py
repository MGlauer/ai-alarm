"""The simulated outside world: weather fetcher, speaker, text gateway, scripted model, CCTV/audio processor."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from world import T0, Clock

from ai_alarm.agents.mock import ScriptedModel
from ai_alarm.demos import DEMOS, BOX, Demo, ScriptedEvent
from ai_alarm.hardware import SimulatedSpeaker, SimulatedTextGateway
from ai_alarm.sensors import CctvAudioProcessor
from ai_alarm.weather import ForecastEntry, WeatherForecastFetcher


# ------------------------------------------------------------------ weather forecast fetcher
def entry(hours_from, hours_to, *conditions, wind=10.0):
    return ForecastEntry(valid_from=T0 + timedelta(hours=hours_from), valid_to=T0 + timedelta(hours=hours_to),
                         conditions=list(conditions), wind_speed_kmh=wind)


class Service:
    """A weather service that counts how often it is asked."""
    def __init__(self, entries):
        self.entries, self.calls, self.fail = entries, 0, False

    def __call__(self):
        self.calls += 1
        if self.fail:
            raise ConnectionError("service down")
        return self.entries


def test_forecast_entries_are_those_valid_at_the_requested_time():
    fetcher = WeatherForecastFetcher(Service([entry(0, 2, "rain"), entry(2, 4, "sunny"), entry(1, 3, "cloudy")]),
                                     Clock(T0))
    forecast = fetcher.get_forecast(T0 + timedelta(hours=1, minutes=30))
    assert [e.conditions for e in forecast.entries] == [["rain"], ["cloudy"]]
    assert fetcher.conditions_at(T0 + timedelta(hours=1, minutes=30)) == {"rain", "cloudy"}
    assert fetcher.conditions_at(T0 + timedelta(hours=9)) == set()


def test_the_service_is_asked_at_most_once_an_hour():
    clock, service = Clock(T0), Service([entry(0, 9, "fog")])
    fetcher = WeatherForecastFetcher(service, clock)
    first = fetcher.get_forecast(T0)
    assert (service.calls, first.from_cache, first.fetched_at) == (1, False, T0)

    clock.advance(3599)
    second = fetcher.get_forecast(T0)
    assert (service.calls, second.from_cache, second.fetched_at) == (1, True, T0)

    clock.advance(1)
    third = fetcher.get_forecast(T0)
    assert (service.calls, third.from_cache, third.fetched_at) == (2, False, T0 + timedelta(hours=1))


def test_a_failing_service_falls_back_to_the_cache_and_is_not_hammered():
    clock, service = Clock(T0), Service([entry(0, 9, "fog")])
    fetcher = WeatherForecastFetcher(service, clock)
    fetcher.get_forecast(T0)
    service.fail = True
    clock.advance(3600)
    stale = fetcher.get_forecast(T0)
    assert stale.from_cache and stale.fetched_at == T0 and stale.entries[0].conditions == ["fog"]
    fetcher.get_forecast(T0)
    assert service.calls == 2  # asked once when it was due, not again for every request


def test_a_failing_service_without_cache_raises_and_is_asked_again():
    service = Service([])
    service.fail = True
    fetcher = WeatherForecastFetcher(service, Clock(T0))
    for _ in range(2):
        with pytest.raises(ConnectionError):
            fetcher.get_forecast(T0)
    assert service.calls == 2


def test_the_cache_can_be_cleared():
    service = Service([entry(0, 9, "fog")])
    fetcher = WeatherForecastFetcher(service, Clock(T0))
    fetcher.get_forecast(T0)
    fetcher.clear_cache()
    assert not fetcher.get_forecast(T0).from_cache and service.calls == 2


# ------------------------------------------------------------------ hardware
def test_speaker_remembers_what_it_said():
    speaker = SimulatedSpeaker(Clock(T0))
    speaker.speak("Leave the area.")
    assert speaker.spoken == [(T0, "Leave the area.")]


def test_text_gateway_delivers_unless_nobody_can_be_reached():
    gateway = SimulatedTextGateway(Clock(T0))
    assert gateway.send("+49 1", "hello") is True and gateway.sent == [(T0, "+49 1", "hello")]
    gateway.reachable = False
    assert gateway.send("+49 2", "hello") is False and len(gateway.sent) == 1


# ------------------------------------------------------------------ scripted model
@pytest.mark.parametrize("request_, key", [
    ({"evidence": "clip.mp4", "area_id": "garden"}, "clip.mp4"),
    ({"evidence": ["f1.png", "f2.png"], "persons": []}, "f1.png"),  # frames: the first one
    ({"image": "img.png", "object_id": "p1"}, "img.png"),
    ({"video": "clip.mp4", "object_id": "p1"}, "clip.mp4"),
])
def test_scripted_model_answers_by_evidence(request_, key):
    model = ScriptedModel(lambda: {key: {"answer": key}})
    assert model(system="", request=request_, schema={}) == {"answer": key}


def test_scripted_model_fails_for_evidence_without_a_script_and_follows_script_changes():
    script = {}
    model = ScriptedModel(lambda: script)
    with pytest.raises(LookupError, match="no scripted answer"):
        model(system="", request={"evidence": "clip.mp4"}, schema={})
    script["clip.mp4"] = None  # an unusable answer is a scripted answer too
    assert model(system="", request={"evidence": "clip.mp4"}, schema={}) is None


def test_scripted_model_can_answer_for_one_object_of_the_evidence():
    script = {"clip.mp4#p2": {"verdict": "unknown"}, "clip.mp4": {"verdict": "known"}}
    model = ScriptedModel(lambda: script)
    assert model(system="", request={"video": "clip.mp4", "object_id": "p2"}, schema={}) == {"verdict": "unknown"}
    assert model(system="", request={"video": "clip.mp4", "object_id": "p1"}, schema={}) == {"verdict": "known"}


# ------------------------------------------------------------------ CCTV/audio processor
DEMO = Demo(
    "test", "test demo",
    (ScriptedEvent(0.2, "mic_garden", "b.wav"), ScriptedEvent(0.0, "cam_garden", "a.mp4", BOX)),
    {}, audio={"garden": "ambient.wav"})


@pytest.fixture
def processor(session_factory):
    return CctvAudioProcessor(session_factory, {"test": DEMO}, clock=Clock(T0))


def test_processor_plays_the_events_of_a_demo_in_order(processor):
    emitted = []
    processor.play("test", emitted.append, time_scale=0, wait=True)

    assert [(e.type, e.sensor_id, e.area_id, e.evidence) for e in emitted] == [
        ("video_event", "cam_garden", "garden", "a.mp4"), ("audio_event", "mic_garden", "garden", "b.wav")]
    assert emitted[0].bbox.model_dump() == BOX and emitted[1].bbox is None  # a box is for video only
    assert emitted[0].start_time == T0 and len({e.event_id for e in emitted}) == 2


def test_processor_waits_between_events_according_to_the_time_scale(processor):
    emitted = []
    processor.play("test", emitted.append, time_scale=1.0, wait=True)  # 0.2 s between the two events
    assert len(emitted) == 2


def test_processor_offers_the_latest_audio_of_the_area_of_the_running_demo(processor):
    assert processor.latest_audio("garden", T0) is None  # nothing played yet
    processor.play("test", lambda e: None, time_scale=0, wait=True)
    assert processor.latest_audio("garden", T0) == "ambient.wav" and processor.latest_audio("entry", T0) is None
    assert processor.demo is DEMO


def test_unknown_demo(processor):
    with pytest.raises(LookupError, match="unknown demo"):
        processor.play("nothing", lambda e: None)


def test_a_failing_event_does_not_stop_the_demo(processor):
    seen = []

    def emit(signal):
        seen.append(signal.evidence)
        if signal.evidence == "a.mp4":
            raise RuntimeError("controller failed")

    processor.play("test", emit, time_scale=0, wait=True)
    assert seen == ["a.mp4", "b.wav"]


def test_events_of_an_unknown_sensor_are_skipped(session_factory):
    demo = Demo("x", "", (ScriptedEvent(0, "cam_moon", "a.mp4"), ScriptedEvent(0, "cam_garden", "b.mp4")), {})
    emitted = []
    CctvAudioProcessor(session_factory, {"x": demo}).play("x", emitted.append, time_scale=0, wait=True)
    assert [e.evidence for e in emitted] == ["b.mp4"]


def test_the_shipped_demos_only_use_sensors_of_the_demo_house(session_factory):
    from ai_alarm.db.models import Sensor
    with session_factory() as s:
        sensors = {sensor.id for sensor in s.query(Sensor)}
    assert {e.sensor_id for demo in DEMOS.values() for e in demo.events} <= sensors
    for demo in DEMOS.values():  # every clip has a scripted answer of the object detector or the noise interpreter
        first_agents = demo.answers.get("object_detector", {}).keys() | demo.answers.get("noise_interpreter", {}).keys()
        assert demo.description and all(e.evidence in first_agents for e in demo.events)
