"""The model of an agent in the simulation: it answers what the script says for the evidence it is given.

For the challenge every agent may be replaced by a mock. A script maps the reference of a piece of evidence to the
answer of the agent; for a request about one object (identify_person) `"<evidence>#<object id>"` answers for that
object alone. An answer of `None` is an unusable answer; evidence without an entry makes the model fail like an
unreachable backend would. Both show up as "agent not available" in the workflow.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping


class ScriptedModel:
    def __init__(self, script: Callable[[], Mapping[str, Any]]):
        self.script = script  # a function, so that the script can change while the system runs

    def __call__(self, *, system: str, request: dict[str, Any], schema: dict[str, Any]) -> Any:
        evidence = request.get("evidence") or request.get("image") or request.get("video")
        if isinstance(evidence, list):  # frames
            evidence = evidence[0]
        script = self.script()
        keys = [f"{evidence}#{request['object_id']}", evidence] if "object_id" in request else [evidence]
        for key in keys:
            if key in script:
                return script[key]
        raise LookupError(f"no scripted answer for {keys[0]!r}")
