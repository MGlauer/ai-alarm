"""The knowledge base content of the demo house."""
from __future__ import annotations

from typing import Callable

from sqlalchemy.orm import Session

from ai_alarm.db.models import (
    Area, AreaConnection, Contact, EntryPoint, PermissionRule, Person, PersonImage, Role, Sensor,
)


def seed_demo_house(factory: Callable[[], Session]) -> None:
    """The house of the demos: entry (the entryway) is connected to garden and house. Anna (family) and Gustav
    (gardener) are known; contacts: Bob (primary) and Anna (when she is in the area). Creating and editing these is
    out of the scope of the challenge, so they are seeded."""
    family, delivery, gardener = Role(name="family"), Role(name="delivery person"), Role(name="gardener")
    entry, garden, house = Area(id="entry", name="Entryway", is_entryway=True), Area(id="garden", name="Garden"), \
        Area(id="house", name="House")
    with factory() as s:
        s.add_all([
            family, delivery, gardener, entry, garden, house,
            AreaConnection(area_a_id="entry", area_b_id="garden"),
            AreaConnection(area_a_id="entry", area_b_id="house"),
            EntryPoint(id="door", name="Front door", area_id="entry", is_open=False),
            Sensor(id="cam_entry", kind="camera", area_id="entry", x=0, y=0, orientation_deg=0),
            Sensor(id="cam_garden", kind="camera", area_id="garden", x=5, y=0, orientation_deg=90),
            Sensor(id="mic_garden", kind="microphone", area_id="garden", x=5, y=1),
            Sensor(id="cam_house", kind="camera", area_id="house", x=0, y=5, orientation_deg=180),
            Person(id="anna", name="Anna", familiarity=1.0, roles=[family],
                   images=[PersonImage(uri="anna_1.png"), PersonImage(uri="anna_2.png")]),
            Person(id="gustav", name="Gustav", familiarity=0.8, roles=[gardener],
                   images=[PersonImage(uri="gustav_1.png", source="cctv")]),
            Contact(id="c_bob", name="Bob", phone="+49 100", is_primary=True),
            Contact(id="c_anna", name="Anna", phone="+49 200", person_id="anna"),
            PermissionRule(description="family may enter all areas", applies_to="role", role_name="family",
                           all_areas=True),
            PermissionRule(description="delivery people: entryway", applies_to="role", role_name="delivery person",
                           areas=[entry]),
            PermissionRule(description="gardeners: garden and entryway", applies_to="role", role_name="gardener",
                           areas=[garden, entry]),
            PermissionRule(description="unfamiliar persons: entryway for two minutes, unlimited if invited",
                           applies_to="unfamiliar", areas=[entry], max_duration_s=120, invitation_lifts_limit=True),
        ])
        s.commit()
