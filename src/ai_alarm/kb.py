"""Knowledge Base (Service): persons, areas, permission rules, sensors and presences, on top of the database.

Every method opens its own short session and returns plain values, so callers never hold database objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ai_alarm.db.models import (
    Area,
    AreaConnection,
    Contact,
    EntryPoint,
    PermissionRule,
    Person,
    PersonPresence,
    Role,
)


class UnknownArea(LookupError):
    """An area id that is not in the knowledge base. A bug of the caller."""


@dataclass(frozen=True)
class PersonInfo:
    id: str
    name: str | None
    roles: tuple[str, ...]
    familiarity: float


@dataclass(frozen=True)
class Recipient:
    contact_id: str
    name: str
    phone: str


def _info(person: Person) -> PersonInfo:
    return PersonInfo(
        person.id,
        person.name,
        tuple(sorted(r.name for r in person.roles)),
        person.familiarity,
    )


class KnowledgeBase:
    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    # ---------------------------------------------------------------- persons
    def person(self, person_id: str) -> PersonInfo | None:
        with self.session_factory() as s:
            person = s.get(Person, person_id)
            return _info(person) if person else None

    def person_exists(self, person_id: str) -> bool:
        return self.person(person_id) is not None

    def reference_images(self) -> list[dict[str, Any]]:
        """The reference images of all persons that have any, for the person identifier."""
        with self.session_factory() as s:
            persons = s.scalars(
                select(Person).where(Person.images.any()).order_by(Person.id)
            )
            return [
                {"person_id": p.id, "images": [i.uri for i in p.images]}
                for p in persons
            ]

    def role_names(self) -> list[str]:
        with self.session_factory() as s:
            return list(s.scalars(select(Role.name).order_by(Role.name)))

    def record_detection(
        self,
        person_id: str,
        area_id: str,
        suspicion: float,
        now: datetime,
        predicted_role: str | None = None,
    ) -> datetime:
        """Store a detected person (created if new) with its suspicion and, if the KB knows that role, the predicted
        role, and mark the person as present in the area. Returns when the person entered the area.
        """
        with self.session_factory() as s:
            person = s.get(Person, person_id)
            if person is None:
                person = Person(id=person_id)
                s.add(person)
            person.suspicion = suspicion
            role = s.get(Role, predicted_role) if predicted_role else None
            if role is not None and role not in person.roles:
                person.roles.append(role)

            open_presences = s.scalars(
                select(PersonPresence).where(
                    PersonPresence.person_id == person_id,
                    PersonPresence.left_at.is_(None),
                )
            ).all()
            here = next((p for p in open_presences if p.area_id == area_id), None)
            for p in open_presences:
                if p is not here:
                    p.left_at = now  # a person is in one area at a time
            if here is None:
                here = PersonPresence(
                    person_id=person_id,
                    area_id=area_id,
                    entered_at=now,
                    last_seen_at=now,
                )
                s.add(here)
            here.last_seen_at = now
            entered_at = here.entered_at
            s.commit()
            return entered_at

    # ---------------------------------------------------------------- areas
    def area_info(self, area_id: str) -> dict[str, Any]:
        """The area with the areas it is connected to. Raises `UnknownArea`."""
        with self.session_factory() as s:
            area = s.get(Area, area_id)
            if area is None:
                raise UnknownArea(area_id)
            links = s.scalars(
                select(AreaConnection).where(
                    or_(
                        AreaConnection.area_a_id == area_id,
                        AreaConnection.area_b_id == area_id,
                    )
                )
            )
            neighbours = [
                link.area_b if link.area_a_id == area_id else link.area_a
                for link in links
            ]
            return {
                "id": area.id,
                "name": area.name,
                "is_entryway": area.is_entryway,
                "is_drop_off": area.is_drop_off,
                "connected_to": [
                    {"id": a.id, "name": a.name}
                    for a in sorted(neighbours, key=lambda a: a.id)
                ],
            }

    def has_open_entry_point(self) -> bool:
        with self.session_factory() as s:
            return (
                s.scalars(select(EntryPoint.id).where(EntryPoint.is_open)).first()
                is not None
            )

    def entry_permitted(
        self,
        area_id: str,
        roles: list[str],
        identified: bool,
        familiar: bool,
        invited: bool,
        stay_s: float,
    ) -> bool:
        """Is a person with these roles allowed to be in the area (whitelist: no matching rule = not allowed)?

        `identified`: the roles come from the knowledge base. Roles predicted from an image are not trusted for rules
        that open all areas, so that "looks like family" never opens the whole house.
        `stay_s`: how long the person has been in the area, for rules with a time limit.
        """
        with self.session_factory() as s:
            for rule in s.scalars(select(PermissionRule)):
                if not (rule.all_areas or any(a.id == area_id for a in rule.areas)):
                    continue
                if rule.applies_to == "role":
                    if rule.role_name in roles and (identified or not rule.all_areas):
                        return True
                elif (
                    not familiar
                ):  # rule for unfamiliar persons (also those that are not decidable)
                    limit_lifted = invited and rule.invitation_lifts_limit
                    if (
                        rule.max_duration_s is None
                        or stay_s <= rule.max_duration_s
                        or limit_lifted
                    ):
                        return True
            return False

    # ---------------------------------------------------------------- presence
    def people_present(self, area_id: str | None = None) -> list[PersonInfo]:
        with self.session_factory() as s:
            query = (
                select(Person)
                .join(PersonPresence)
                .where(PersonPresence.left_at.is_(None))
            )
            if area_id is not None:
                query = query.where(PersonPresence.area_id == area_id)
            return [_info(p) for p in s.scalars(query.distinct())]

    def close_presences(self, area_id: str, now: datetime) -> None:
        with self.session_factory() as s:
            for p in s.scalars(
                select(PersonPresence).where(
                    PersonPresence.area_id == area_id, PersonPresence.left_at.is_(None)
                )
            ):
                p.left_at = now
            s.commit()

    # ---------------------------------------------------------------- contacts
    def recipients(self) -> list[Recipient]:
        """Primary contacts and the contacts who are currently in the surveilled area."""
        with self.session_factory() as s:
            present = select(PersonPresence.person_id).where(
                PersonPresence.left_at.is_(None)
            )
            contacts = s.scalars(
                select(Contact)
                .where(or_(Contact.is_primary, Contact.person_id.in_(present)))
                .order_by(Contact.id)
            )
            return [Recipient(c.id, c.name, c.phone) for c in contacts]
