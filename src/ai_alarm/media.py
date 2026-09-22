"""Synthetic media for the sensors and the demos: camera clips and sounds.

 A camera clip shows what a sensor recorded; any bounding box a viewer draws on top of one comes from
the real (mocked) object detector's answer, propagated through the pipeline (`PersonAssessment.bbox`,
`AnimalAssessment.bbox`), never from this module.

Uses only Pillow and the standard library: a GIF for motion, a plain PCM WAV for sound. People and animals are
drawn from the character/animal cutouts in media/sprites/ (see scripts/generate_media.py) -- there used to be a
second, py-avataaars-based system generating a unique cartoon portrait per person; it was removed in favour of
reusing this same sprite art everywhere a person needs a picture, including their knowledge-base profile
(`sprite_for_person`, used by `ai_alarm.api`).
"""
from __future__ import annotations

import math
import struct
import wave
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from random import Random
from typing import Any

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 480, 270  # small: a sketch, not a photo
GROUND_Y = int(HEIGHT * 0.55)

_GROUND = {"garden": (58, 110, 64), "entry": (150, 140, 128), "house": (176, 160, 132)}
_SKY_DAY, _SKY_NIGHT = (176, 205, 224), (20, 26, 42)
_SHAPE_COLOUR = {"person": (222, 196, 150), "animal": (120, 78, 46), "object": (150, 150, 160)}

# The character/animal cutouts of media/sprites/ (see scripts/generate_media.py) -- reused here, pasted onto a
# freshly drawn, always-correct-for-the-request background, instead of ever baking a whole scene to disk in
# advance (which drifted out of sync with the area/night/weather a piece of evidence actually belongs to).
SPRITES_DIR = Path(__file__).resolve().parents[2] / "media" / "sprites"
_sprite_cache: dict[str, Image.Image | None] = {}

# Which sprite (media/sprites/<name>.png) stands in for a person or animal, resolved fresh every time rather than
# baked into a static file: a known person's own identity, then their role, then a generic figure; an animal by
# its label. Shared by `sensors/processor.py` (what a camera shows) and `sprite_for_person` below (a person's
# knowledge-base picture), so the same character always looks the same everywhere they appear.
IDENTITY_SPRITES = {"anna": "family_anna", "gustav": "gardener"}
ROLE_SPRITES = {"gardener": "gardener"}
ANIMAL_SPRITES = {"bear": "animal_bear", "cat": "animal_cat"}


def _load_sprite(name: str) -> Image.Image | None:
    """A cached sprite by name (without ".png"), or None if it does not exist -- e.g. an animal label with no
    bespoke art, which then falls back to the plain silhouette below."""
    if name not in _sprite_cache:
        path = SPRITES_DIR / f"{name}.png"
        _sprite_cache[name] = Image.open(path).convert("RGBA") if path.is_file() else None
    return _sprite_cache[name]


def sprite_for_person(person_id: str, role_names: list[str]) -> str:
    """The sprite that represents a knowledge-base person: their own identity if they are one of the named
    characters, else their first role with a dedicated sprite, else a generic figure. Never empty -- unlike a
    per-person generated portrait, there is always at least a generic picture to show."""
    if person_id in IDENTITY_SPRITES:
        return IDENTITY_SPRITES[person_id]
    return next((ROLE_SPRITES[r] for r in role_names if r in ROLE_SPRITES), "unknown_person")


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def _burn_in(draw: ImageDraw.ImageDraw, *, sensor_id: str, area_name: str, at: datetime, blink: bool = True) -> None:
    """The on-screen display of a CCTV camera: sensor id, area, timestamp, a small recording dot (`blink=False`
    for every other frame of a clip, so the dot flashes instead of sitting there solid)."""
    font = _font(13)
    draw.rectangle([0, 0, WIDTH, 20], fill=(0, 0, 0, 150))
    draw.text((6, 4), f"{sensor_id.upper()} · {area_name.upper()}", font=font, fill=(255, 255, 255))
    stamp = at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    draw.text((WIDTH - draw.textlength(stamp, font=font) - 6, 4), stamp, font=font, fill=(255, 255, 255))
    if blink:
        draw.ellipse([WIDTH - 15, HEIGHT - 15, WIDTH - 7, HEIGHT - 7], fill=(220, 40, 40))


def _tree(draw: ImageDraw.ImageDraw, x: float, y: float, *, canopy_w: float, canopy_h: float, sway: float = 0.0
          ) -> None:
    """A simple tree. `sway` is a horizontal pixel offset of the canopy only -- the sole way wind ever shows up in
    a scene: movement in the foliage, never a drawn gust or streak (`render_camera_clip`'s only user of it)."""
    trunk_w = canopy_w * 0.14
    draw.rectangle([x - trunk_w / 2, y - canopy_h * 0.15, x + trunk_w / 2, y + canopy_h * 0.55], fill=(90, 65, 45))
    cx = x + sway
    draw.ellipse([cx - canopy_w / 2, y - canopy_h, cx + canopy_w / 2, y - canopy_h * 0.15], fill=(52, 105, 58))


def _background(draw: ImageDraw.ImageDraw, area_id: str, *, sway: float = 0.0) -> None:
    """The ground and a little fixed scenery for an area -- the same look as its default idle scene
    (media/scenes/sensors/<id>.png, see scripts/generate_media.py), so a detected anomaly and the ordinary view of
    the same camera are recognisably the same place. The sky is the caller's job (it depends on night, not area)."""
    draw.rectangle([0, GROUND_Y, WIDTH, HEIGHT], fill=_GROUND.get(area_id, (90, 90, 90)))
    if area_id == "entry":
        draw.rectangle([0, HEIGHT * 0.15, WIDTH, GROUND_Y], fill=(196, 182, 158))
        draw.rectangle([WIDTH * 0.42, HEIGHT * 0.30, WIDTH * 0.58, GROUND_Y], fill=(96, 66, 46))
    elif area_id == "garden":
        _tree(draw, WIDTH * 0.84, GROUND_Y, canopy_w=110, canopy_h=100, sway=sway)
        for i in range(3):
            x = WIDTH * 0.10 + i * 22
            draw.ellipse([x, GROUND_Y - 14, x + 16, GROUND_Y + 2], fill=(200, 90, 110))
    elif area_id == "house":
        draw.rectangle([0, HEIGHT * 0.15, WIDTH, GROUND_Y], fill=(214, 206, 190))
        draw.rectangle([WIDTH * 0.60, HEIGHT * 0.22, WIDTH * 0.82, HEIGHT * 0.42], fill=(150, 190, 215))


def _silhouette(img: Image.Image, draw: ImageDraw.ImageDraw, obj: dict[str, Any], bob: float = 0.0) -> None:
    """The object's bounding box, filled in with only what physically belongs in the scene: no label or box outline
    (those are the agent's *annotation* of this frame, never part of the recording). `bob` is a vertical pixel
    offset, for a clip's walking/bouncing motion; it is 0 for a single still frame.

    If `obj["sprite"]` names one of media/sprites/ (set by the caller from the object's role/label/identity, e.g.
    sensors/processor.py), that cutout is pasted in, scaled to the bounding box. Failing that, a plain generic
    shape (e.g. for an animal label with no bespoke sprite, or a prop like a crowbar)."""
    bbox = obj["bbox"]
    x0, y0 = bbox["x"] * WIDTH, bbox["y"] * HEIGHT + bob
    x1, y1 = x0 + bbox["w"] * WIDTH, y0 + bbox["h"] * HEIGHT

    sprite = _load_sprite(obj["sprite"]) if obj.get("sprite") else None
    if sprite is not None:
        scaled = sprite.resize((max(1, round(x1 - x0)), max(1, round(y1 - y0))))
        img.paste(scaled, (round(x0), round(y0)), scaled)
        return

    colour = _SHAPE_COLOUR.get(obj["kind"], (170, 170, 170))
    if obj["kind"] == "person":
        cx, head_r = (x0 + x1) / 2, (x1 - x0) * 0.32
        draw.ellipse([cx - head_r, y0, cx + head_r, y0 + head_r * 2], fill=colour)
        draw.rounded_rectangle([x0 + (x1 - x0) * 0.15, y0 + head_r * 1.7, x1 - (x1 - x0) * 0.15, y1], radius=8,
                               fill=colour)
    elif obj["kind"] == "animal":
        draw.ellipse([x0, y0 + (y1 - y0) * 0.3, x1, y1], fill=colour)
        draw.ellipse([x0 - (x1 - x0) * 0.15, y0, x0 + (x1 - x0) * 0.55, y0 + (y1 - y0) * 0.55], fill=colour)
    else:
        draw.rectangle([x0, y0, x1, y1], fill=colour, outline=(70, 70, 70))


def render_camera_frame(
    *, sensor_id: str, area_id: str, area_name: str, at: datetime, objects: list[dict[str, Any]],
    weather: str | None, night: bool,
) -> bytes:
    img = Image.new("RGB", (WIDTH, HEIGHT), _SKY_NIGHT if night else _SKY_DAY)
    draw = ImageDraw.Draw(img, "RGBA")
    _background(draw, area_id)
    if weather == "wind":
        for i in range(6):
            y = 32 + i * 14
            draw.line([(20 + i * 10, y), (WIDTH - 20 - i * 6, y - 6)], fill=(255, 255, 255, 100), width=2)
    elif weather == "artefact":
        draw.ellipse([WIDTH * 0.55, 18, WIDTH * 0.85, 118], fill=(255, 255, 210, 130))
    for obj in objects:
        if obj.get("bbox"):
            _silhouette(img, draw, obj)
    _burn_in(draw, sensor_id=sensor_id, area_name=area_name, at=at)
    return _encode(img)


def render_audio_icon(*, sensor_id: str, area_name: str, at: datetime) -> bytes:
    """A microphone has no picture; this stands in for it: a waveform, stable per sensor so it does not flicker."""
    img = Image.new("RGB", (WIDTH, HEIGHT), (24, 28, 36))
    draw = ImageDraw.Draw(img)
    mid, x, rnd = HEIGHT // 2, 18, Random(sensor_id)
    while x < WIDTH - 18:
        h = rnd.randint(8, 76)
        draw.line([(x, mid - h // 2), (x, mid + h // 2)], fill=(110, 170, 220), width=3)
        x += 9
    _burn_in(draw, sensor_id=sensor_id, area_name=area_name, at=at)
    return _encode(img)


def render_placeholder(*, sensor_id: str, area_name: str, at: datetime) -> bytes:
    """For evidence the simulation does not recognise, so an <img> never breaks."""
    img = Image.new("RGB", (WIDTH, HEIGHT), (60, 64, 72))
    draw = ImageDraw.Draw(img)
    font = _font(14)
    text = "No preview available"
    draw.text(((WIDTH - draw.textlength(text, font=font)) / 2, HEIGHT / 2 - 8), text, font=font, fill=(205, 209, 216))
    _burn_in(draw, sensor_id=sensor_id, area_name=area_name, at=at)
    return _encode(img)


def _encode(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ====================================================================== animated footage
def render_camera_clip(
    *, sensor_id: str, area_id: str, area_name: str, at: datetime, objects: list[dict[str, Any]],
    weather: str | None, night: bool, frames: int = 10, frame_ms: int = 150,
) -> bytes:
    """A short looping animation of the same scene `render_camera_frame` draws as a still: a person or animal has a
    walking bob, a lens flare pulses, the recording dot blinks, and -- unlike a still, which cannot show movement at
    all -- wind sways a tree's canopy, the only way it is ever shown (never a drawn gust or streak). Encoded as an
    animated GIF (a video codec would be its own large dependency; a GIF needs only Pillow and plays anywhere)."""
    pictures = []
    for i in range(frames):
        phase = i / frames  # 0..1, one full loop; every animated element is periodic in `phase` so the loop is seamless
        img = Image.new("RGB", (WIDTH, HEIGHT), _SKY_NIGHT if night else _SKY_DAY)
        draw = ImageDraw.Draw(img, "RGBA")
        sway = math.sin(phase * 2 * math.pi) * 12 if weather == "wind" else 0.0
        _background(draw, area_id, sway=sway)
        if weather == "artefact":
            pulse = 0.6 + 0.4 * math.sin(phase * 2 * math.pi)
            draw.ellipse([WIDTH * 0.55, 18, WIDTH * 0.85, 118], fill=(255, 255, 210, int(130 * pulse)))
        bob = math.sin(phase * 2 * math.pi) * 4  # a gentle step cycle, shared by everything in the scene
        for obj in objects:
            if obj.get("bbox"):
                _silhouette(img, draw, obj, bob=bob)
        _burn_in(draw, sensor_id=sensor_id, area_name=area_name, at=at, blink=(i % 2 == 0))
        pictures.append(img)
    buf = BytesIO()
    pictures[0].save(buf, format="GIF", save_all=True, append_images=pictures[1:], duration=frame_ms, loop=0,
                     disposal=2)
    return buf.getvalue()


# ====================================================================== sound
def render_noise_clip(category: str, *, seed: str, duration_s: float = 2.5, sample_rate: int = 8000) -> bytes:
    """A short synthesised sound for a noise category (matching `NoiseCategory`: human_activity, animal, weather,
    technical_noise; anything else, including a microphone's idle ambience, gets a faint noise floor). Deterministic
    per `seed`, so the same evidence always sounds the same. A plain mono 16-bit PCM WAV: no audio DSP library."""
    synth = {"human_activity": _synth_footsteps, "animal": _synth_animal, "weather": _synth_wind,
            "technical_noise": _synth_hum}.get(category, _synth_ambient)
    samples = synth(Random(seed), int(duration_s * sample_rate), sample_rate)
    return _encode_wav(samples, sample_rate)


def _clip16(x: float) -> int:
    return max(-32000, min(32000, int(x)))


def _synth_wind(rnd: Random, n: int, rate: int) -> list[int]:
    """White noise through a one-pole low-pass filter (rumble, not a hiss), with a slow gusting envelope."""
    out, y = [], 0.0
    for i in range(n):
        y += 0.05 * (rnd.uniform(-9000, 9000) - y)
        gust = 0.5 + 0.5 * math.sin(2 * math.pi * 0.15 * i / rate + 0.6)
        out.append(_clip16(y * gust))
    return out


def _synth_footsteps(rnd: Random, n: int, rate: int) -> list[int]:
    """A faint room-noise floor, with a short percussive thump every ~0.55s."""
    out = [_clip16(rnd.uniform(-300, 300)) for _ in range(n)]
    step = int(rate * 0.55)
    thump = int(rate * 0.08)
    for start in range(0, n, step):
        for k in range(min(thump, n - start)):
            envelope = math.exp(-k / (rate * 0.02))
            out[start + k] = _clip16(out[start + k] + rnd.uniform(-9000, 9000) * envelope)
    return out


def _synth_animal(rnd: Random, n: int, rate: int) -> list[int]:
    """A low, wavering growl: a tone with vibrato, mixed with a little noise, rising and falling in volume."""
    out = []
    for i in range(n):
        t = i / rate
        pitch = 90 + math.sin(2 * math.pi * 6 * t) * 8
        tone = math.sin(2 * math.pi * pitch * t)
        noise = rnd.uniform(-1, 1) * 0.3
        envelope = 0.6 + 0.4 * math.sin(2 * math.pi * 0.4 * t)
        out.append(_clip16((tone * 0.7 + noise) * 7000 * envelope))
    return out


def _synth_hum(rnd: Random, n: int, rate: int) -> list[int]:
    """A steady mains-like hum with the occasional click: a sensor fault, not a living thing."""
    out = []
    for i in range(n):
        tone = math.sin(2 * math.pi * 50 * i / rate) * 4000
        glitch = 6000 if rnd.random() < 0.0006 else 0
        out.append(_clip16(tone + glitch))
    return out


def _synth_ambient(rnd: Random, n: int, rate: int) -> list[int]:
    """Near silence: a very faint noise floor, for a microphone that has not reported anything."""
    return [_clip16(rnd.uniform(-250, 250)) for _ in range(n)]


def _encode_wav(samples: list[int], sample_rate: int) -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buf.getvalue()
