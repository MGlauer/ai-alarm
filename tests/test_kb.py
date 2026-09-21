"""Knowledge base service: what agents and the controller read from and write to the database."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from ai_alarm.db.models import EntryPoint, Person, PersonPresence
from ai_alarm.kb import UnknownArea

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)


def test_reference_images_only_list_persons_that_have_images(kb):
    assert kb.reference_images() == [
        {"person_id": "anna", "images": ["anna_1.png", "anna_2.png"]},
        {"person_id": "gustav", "images": ["gustav_1.png"]},
    ]


def test_person_lookup(kb):
    anna = kb.person("anna")
    assert (anna.name, anna.roles, anna.familiarity) == ("Anna", ("family",), 1.0)
    assert kb.person("nobody") is None and not kb.person_exists("nobody") and kb.person_exists("neighbour")
    assert kb.role_names() == ["delivery person", "family", "gardener"]


def test_area_info_lists_connected_areas(kb):
    info = kb.area_info("entry")
    assert info["is_entryway"] and [a["id"] for a in info["connected_to"]] == ["garden", "house"]
    assert [a["id"] for a in kb.area_info("garden")["connected_to"]] == ["entry"]
    with pytest.raises(UnknownArea):
        kb.area_info("moon")


def test_open_entry_point(kb, session_factory):
    assert not kb.has_open_entry_point()
    with session_factory() as s:
        s.get(EntryPoint, "door").is_open = True
        s.commit()
    assert kb.has_open_entry_point()


# (area, roles, identified, familiar, invited, stay_s) -> permitted
@pytest.mark.parametrize("area, roles, identified, familiar, invited, stay, permitted", [
    ("house", ["family"], True, True, False, 0, True),  # family may enter all areas
    ("house", ["family"], False, False, False, 0, False),  # ... but "looks like family" opens nothing
    ("entry", ["delivery person"], False, False, False, 0, True),  # delivery: entryway ...
    ("garden", ["delivery person"], False, False, False, 0, False),  # ... and nothing else
    ("garden", ["gardener"], True, True, False, 0, True),
    ("house", ["gardener"], True, True, False, 0, False),
    ("entry", [], False, False, False, 0, True),  # unfamiliar: entryway for a limited time ...
    ("entry", [], False, False, False, 120, True),
    ("entry", [], False, False, False, 121, False),
    ("entry", [], False, False, True, 9999, True),  # ... unless invited
    ("garden", [], False, False, True, 0, False),  # an invitation opens nothing else
    ("entry", [], True, True, False, 9999, False),  # a familiar person without a matching role: no rule applies
])
def test_entry_permitted(kb, area, roles, identified, familiar, invited, stay, permitted):
    assert kb.entry_permitted(area, roles, identified, familiar, invited, stay) is permitted


def test_record_detection_creates_person_and_presence(kb, session_factory):
    entered = kb.record_detection("unk_1", "entry", 0.6, NOW, predicted_role="delivery person")
    assert entered == NOW
    assert kb.record_detection("unk_1", "entry", 0.7, NOW + timedelta(minutes=1)) == NOW  # still the same visit
    with session_factory() as s:
        person = s.get(Person, "unk_1")
        assert (person.suspicion, person.name, [r.name for r in person.roles]) == (0.7, None, ["delivery person"])


def test_record_detection_only_attaches_roles_the_kb_knows(kb, session_factory):
    kb.record_detection("unk_1", "entry", 0.1, NOW, predicted_role="astronaut")
    with session_factory() as s:
        assert s.get(Person, "unk_1").roles == []


def test_a_person_is_in_one_area_at_a_time(kb, session_factory):
    kb.record_detection("anna", "entry", 0.0, NOW)
    kb.record_detection("anna", "garden", 0.0, NOW + timedelta(minutes=1))
    assert [p.id for p in kb.people_present("garden")] == ["anna"] and kb.people_present("entry") == []
    with session_factory() as s:
        left = s.scalars(select(PersonPresence).where(PersonPresence.area_id == "entry")).one()
        assert left.left_at == NOW + timedelta(minutes=1)


def test_close_presences_and_recipients(kb):
    assert [r.contact_id for r in kb.recipients()] == ["c_bob"]  # Anna is not in the area
    kb.record_detection("anna", "garden", 0.0, NOW)
    assert [r.contact_id for r in kb.recipients()] == ["c_anna", "c_bob"]
    kb.close_presences("garden", NOW)
    assert kb.people_present() == [] and [r.contact_id for r in kb.recipients()] == ["c_bob"]
