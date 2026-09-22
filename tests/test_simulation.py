"""The simulated outside world: weather fetcher, speaker, text gateway, scripted model, CCTV/audio processor."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO

import pytest
from world import T0, Clock

from ai_alarm.agents.mock import ScriptedModel
from ai_alarm.demos import DEMOS, BOX, Demo, ScriptedEvent
from ai_alarm.hardware import SimulatedSpeaker, SimulatedTextGateway
from ai_alarm.media import (
    render_audio_icon, render_camera_clip, render_camera_frame, render_noise_clip, render_placeholder,
)
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
def processor(session_factory, tmp_path):
    # `assets_dir` points at an empty, never-created directory: none of these tests' demos are shipped ones (see
    # demos.SITUATION_ASSETS), so no pre-rendered asset would ever match regardless -- this just keeps every test
    # here independent of whatever the real repo-level media/ happens to contain.
    return CctvAudioProcessor(session_factory, {"test": DEMO}, clock=Clock(T0), assets_dir=tmp_path / "no-assets")


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


# ------------------------------------------------------------------ synthetic media
def test_camera_frame_is_a_valid_png_that_varies_with_its_inputs():
    from PIL import Image

    def frame(**over):
        args = dict(sensor_id="cam_garden", area_id="garden", area_name="Garden", at=T0, objects=[],
                    weather=None, night=False)
        return render_camera_frame(**{**args, **over})

    plain = frame()
    img = Image.open(BytesIO(plain))
    assert img.format == "PNG" and img.size == (480, 270)

    with_person = frame(objects=[{"kind": "person", "bbox": BOX}])
    assert with_person != plain  # drawing something changes the picture
    assert frame(night=True) != plain
    assert frame(weather="wind") != plain
    assert frame(weather="artefact") != frame(weather="wind")
    assert frame(area_id="entry") != frame(area_id="house")  # the ground colour is area-specific


def test_audio_icon_is_stable_per_sensor():
    a1 = render_audio_icon(sensor_id="mic_garden", area_name="Garden", at=T0)
    a2 = render_audio_icon(sensor_id="mic_garden", area_name="Garden", at=T0)
    b = render_audio_icon(sensor_id="mic_entry", area_name="Entryway", at=T0)
    assert a1 == a2 != b  # deterministic per sensor, not flickering noise


def test_placeholder_is_a_valid_png():
    from PIL import Image

    png = render_placeholder(sensor_id="cam_garden", area_name="Garden", at=T0)
    assert Image.open(BytesIO(png)).size == (480, 270)


# ------------------------------------------------------------------ processor.frame()
@pytest.fixture
def demo_with_a_detected_object(session_factory, tmp_path):
    demo = Demo("seen", "test", (ScriptedEvent(0, "cam_garden", "a.mp4", BOX),), {
        "object_detector": {"a.mp4": {"anomaly_class": "person", "objects": [{"kind": "person", "bbox": BOX}]}}})
    return CctvAudioProcessor(session_factory, {"seen": demo}, clock=Clock(T0), assets_dir=tmp_path / "no-assets")


def test_processor_frame_draws_the_scripted_objects(demo_with_a_detected_object):
    gif, mimetype = demo_with_a_detected_object.frame(evidence="a.mp4", kind="video", sensor_id="cam_garden",
                                                       area_id="garden", area_name="Garden", at=T0)
    from PIL import Image

    img = Image.open(BytesIO(gif))
    assert mimetype == "image/gif" and img.format == "GIF" and img.size == (480, 270) and img.n_frames > 1


def test_processor_frame_falls_back_to_a_placeholder_for_unknown_evidence(demo_with_a_detected_object):
    known, _ = demo_with_a_detected_object.frame(evidence="a.mp4", kind="video", sensor_id="cam_garden",
                                                 area_id="garden", area_name="Garden", at=T0)
    unknown, mimetype = demo_with_a_detected_object.frame(evidence="nope.mp4", kind="video", sensor_id="cam_garden",
                                                          area_id="garden", area_name="Garden", at=T0)
    placeholder = render_placeholder(sensor_id="cam_garden", area_name="Garden", at=T0)
    assert mimetype == "image/png" and unknown == placeholder and known != placeholder


def test_processor_frame_is_an_icon_for_audio(processor):
    png, mimetype = processor.frame(evidence="b.wav", kind="audio", sensor_id="mic_garden", area_id="garden",
                                    area_name="Garden", at=T0)
    assert mimetype == "image/png" and png == render_audio_icon(sensor_id="mic_garden", area_name="Garden", at=T0)


def test_idle_frame_shows_the_area_with_no_objects(processor):
    day, mimetype = processor.idle_frame(sensor_id="cam_garden", kind="camera", area_id="garden",
                                         area_name="Garden", at=datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc))
    expected = render_camera_frame(sensor_id="cam_garden", area_id="garden", area_name="Garden",
                                   at=datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc), objects=[], weather=None,
                                   night=False)
    assert mimetype == "image/png" and day == expected


def test_idle_frame_is_dark_at_night(processor):
    night, _ = processor.idle_frame(sensor_id="cam_garden", kind="camera", area_id="garden", area_name="Garden",
                                    at=datetime(2026, 9, 21, 23, 0, tzinfo=timezone.utc))
    day, _ = processor.idle_frame(sensor_id="cam_garden", kind="camera", area_id="garden", area_name="Garden",
                                  at=datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc))
    assert night != day


def test_idle_frame_is_an_icon_for_a_microphone(processor):
    png, mimetype = processor.idle_frame(sensor_id="mic_garden", kind="microphone", area_id="garden",
                                         area_name="Garden", at=T0)
    assert mimetype == "image/png" and png == render_audio_icon(sensor_id="mic_garden", area_name="Garden", at=T0)


def test_processor_frame_uses_the_demos_weather_hint(processor, tmp_path):
    weather_demo = Demo("gust", "test", (ScriptedEvent(0, "cam_garden", "gust.mp4"),),
                        {"object_detector": {"gust.mp4": {"anomaly_class": "weather_environment", "objects": []}}})
    calm_demo = Demo("still", "test", (ScriptedEvent(0, "cam_garden", "still.mp4"),),
                     {"object_detector": {"still.mp4": {"anomaly_class": "unclear", "objects": []}}})
    proc = CctvAudioProcessor(processor.session_factory, {"gust": weather_demo, "still": calm_demo}, Clock(T0),
                              assets_dir=tmp_path / "no-assets")

    kw = dict(kind="video", sensor_id="cam_garden", area_id="garden", area_name="Garden", at=T0)
    windy, _ = proc.frame(evidence="gust.mp4", **kw)
    calm, _ = proc.frame(evidence="still.mp4", **kw)
    assert windy != calm  # the swaying tree (the wind hint) is the only difference between the two clips


def test_processor_frame_decides_night_from_the_real_time_not_the_demos_own_scripted_flag(processor, tmp_path):
    """Two cameras used to be able to disagree about whether it was day or night at the same moment, because each
    demo hard-coded its own `ScriptedEvent.night` regardless of when it was actually played. Night must instead
    come from the timestamp every camera already agrees on."""
    night_demo = Demo("prowler", "test", (ScriptedEvent(0, "cam_garden", "prowler.mp4", night=True),),
                      {"object_detector": {"prowler.mp4": {"anomaly_class": "person", "objects": []}}})
    proc = CctvAudioProcessor(processor.session_factory, {"prowler": night_demo}, Clock(T0),
                              assets_dir=tmp_path / "no-assets")
    kw = dict(evidence="prowler.mp4", kind="video", sensor_id="cam_garden", area_id="garden", area_name="Garden")

    at_noon = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
    clip, _ = proc.frame(at=at_noon, **kw)
    daytime = render_camera_clip(sensor_id="cam_garden", area_id="garden", area_name="Garden", at=at_noon,
                                 objects=[], weather=None, night=False)
    assert clip == daytime  # scripted night=True, but at noon it must still look like day

    at_night = datetime(2026, 9, 21, 23, 0, tzinfo=timezone.utc)
    night_clip, _ = proc.frame(at=at_night, **kw)
    assert night_clip != clip  # the same demo, played at an actual night hour, does go dark


# ------------------------------------------------------------------ processor prefers pre-rendered assets
@pytest.fixture
def assets_dir(tmp_path):
    d = tmp_path / "assets"
    (d / "scenes" / "sensors").mkdir(parents=True)
    (d / "scenes" / "audio").mkdir(parents=True)
    return d


def test_idle_frame_prefers_the_prerendered_default_scene_of_the_sensor(session_factory, assets_dir):
    (assets_dir / "scenes" / "sensors" / "cam_garden.png").write_bytes(b"PRERENDERED-PNG")
    proc = CctvAudioProcessor(session_factory, {"test": DEMO}, clock=Clock(T0), assets_dir=assets_dir)

    png, mimetype = proc.idle_frame(sensor_id="cam_garden", kind="camera", area_id="garden", area_name="Garden",
                                    at=T0)
    assert (png, mimetype) == (b"PRERENDERED-PNG", "image/png")


def test_idle_frame_ignores_the_prerendered_default_scene_at_night(session_factory, assets_dir):
    """The default scene is a permanently daylit picture (see scripts/generate_media.py); using it regardless of
    the time would leave this camera looking like day while another, freshly-drawn one goes dark at the same
    moment -- exactly the kind of disagreement between cameras this is meant to avoid."""
    (assets_dir / "scenes" / "sensors" / "cam_garden.png").write_bytes(b"PRERENDERED-PNG")
    proc = CctvAudioProcessor(session_factory, {"test": DEMO}, clock=Clock(T0), assets_dir=assets_dir)

    png, mimetype = proc.idle_frame(sensor_id="cam_garden", kind="camera", area_id="garden", area_name="Garden",
                                    at=datetime(2026, 9, 21, 23, 0, tzinfo=timezone.utc))
    assert mimetype == "image/png" and png != b"PRERENDERED-PNG"


def test_idle_frame_falls_back_to_drawing_one_when_no_prerendered_scene_exists(session_factory, assets_dir):
    proc = CctvAudioProcessor(session_factory, {"test": DEMO}, clock=Clock(T0), assets_dir=assets_dir)
    png, mimetype = proc.idle_frame(sensor_id="cam_garden", kind="camera", area_id="garden", area_name="Garden",
                                    at=T0)
    assert mimetype == "image/png" and png != b"PRERENDERED-PNG"


def test_frame_builds_a_fresh_animated_clip_from_the_real_area_for_shipped_demos(session_factory, assets_dir):
    """Nothing about a situation is ever read from disk (unlike the old media/scenes/situations/, removed): the
    clip is composed fresh from the real area/night/weather of the request, so it can never drift out of sync with
    them the way a pre-baked-per-demo scene could (and did)."""
    proc = CctvAudioProcessor(session_factory, DEMOS, clock=Clock(T0), assets_dir=assets_dir)
    evidence = DEMOS["gardener"].events[0].evidence

    gif, mimetype = proc.frame(evidence=evidence, kind="video", sensor_id="cam_garden", area_id="garden",
                               area_name="Garden", at=T0)
    assert mimetype == "image/gif"
    from PIL import Image

    img = Image.open(BytesIO(gif))
    assert img.format == "GIF" and img.n_frames > 1


def test_with_sprite_picks_a_known_persons_own_sprite(processor):
    demo = Demo("id-demo", "test", (), {
        "person_identifier": {"clip.gif#p1": {"verdict": "known", "person": "gustav", "confidence": 0.9}}})
    proc = CctvAudioProcessor(processor.session_factory, {"id-demo": demo}, clock=Clock(T0))
    obj = {"object_id": "p1", "kind": "person"}
    assert proc._with_sprite(obj, "clip.gif")["sprite"] == "gardener"


def test_with_sprite_falls_back_to_role_then_to_a_generic_person(processor):
    gardener = {"object_id": "p1", "kind": "person", "role": "gardener"}
    assert processor._with_sprite(gardener, "nope.gif")["sprite"] == "gardener"
    stranger = {"object_id": "p1", "kind": "person"}
    assert processor._with_sprite(stranger, "nope.gif")["sprite"] == "unknown_person"


def test_with_sprite_matches_an_animal_by_label(processor):
    assert processor._with_sprite({"kind": "animal", "label": "bear"}, "x")["sprite"] == "animal_bear"
    assert processor._with_sprite({"kind": "animal", "label": "brown bear"}, "x")["sprite"] == "animal_bear"
    assert "sprite" not in processor._with_sprite({"kind": "animal", "label": "small dog"}, "x")


def test_idle_audio_prefers_the_shared_ambient_sound(session_factory, assets_dir):
    (assets_dir / "scenes" / "audio" / "ambient.wav").write_bytes(b"PRERENDERED-WAV")
    proc = CctvAudioProcessor(session_factory, {"test": DEMO}, clock=Clock(T0), assets_dir=assets_dir)
    assert proc.idle_audio("mic_garden") == b"PRERENDERED-WAV"


def test_idle_audio_falls_back_to_a_synthesised_sound_when_the_ambient_file_is_missing(session_factory, assets_dir):
    proc = CctvAudioProcessor(session_factory, {"test": DEMO}, clock=Clock(T0), assets_dir=assets_dir)
    assert proc.idle_audio("mic_garden") == render_noise_clip("ambient", seed="mic_garden")
