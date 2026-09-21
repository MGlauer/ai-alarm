"""
This module implements the base agent class.

An agent is a stateless function: request signal in, response signal out. The base class owns everything
that is the same for all agents:

1. validate the incoming request against its schema,
2. hand the request content and the response schema to the model,
3. add the signal envelope (id, time, situation, reply-to) to the model's answer,
4. validate the outgoing response against its schema and the agent's consistency rules.

Subclasses only declare their signal types and prompt, and may override `postprocess`.
Timeouts, retries and the treatment of an invalid response as "no response" are the caller's job (the SI).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Generic, Protocol, TypeVar

from pydantic import ValidationError

from ai_alarm.signals import SignalBase

Req = TypeVar("Req", bound=SignalBase)
Resp = TypeVar("Resp", bound=SignalBase)

# Set by the agent, never by the model.
ENVELOPE_FIELDS = frozenset(SignalBase.model_fields) | {"type"}


class Model(Protocol):
    """The model backend (mock, Ollama, ...): system prompt, request content and response schema in, JSON out."""

    def __call__(self, *, system: str, request: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]: ...


class InvalidResponse(Exception):
    """The model's answer does not match the response schema or violates a consistency rule."""


class Agent(Generic[Req, Resp]):
    name: str  # used for logging and configuration, e.g. "object_detector"
    prompt: str  # role and rules for the model
    request_type: type[Req]
    response_type: type[Resp]

    def __init__(self, model: Model):
        self.model = model

    def handle(self, request: Req | dict[str, Any]) -> Resp:
        """Process one request signal. Raises `InvalidResponse` if the model's answer is unusable.

        A malformed *request* raises pydantic's `ValidationError`: that is a bug of the caller, not of the model.
        Errors of the model backend itself (timeout, connection) are not caught here.
        """
        req = self.request_type.model_validate(request)
        answer = self.model(
            system=self.system_prompt,
            request=req.model_dump(mode="json", exclude=ENVELOPE_FIELDS),
            schema=self.content_schema,
        )
        if not isinstance(answer, dict):
            raise InvalidResponse(f"{self.name}: expected a JSON object, got {type(answer).__name__}")
        try:
            resp = self.response_type.model_validate({**answer, **self._envelope(req)})
        except ValidationError as e:
            raise InvalidResponse(f"{self.name}: {e}") from e
        return self.postprocess(req, resp)

    def postprocess(self, request: Req, response: Resp) -> Resp:
        """Hook for the agent's consistency rules.

        Raise `InvalidResponse` for a violation, or return a corrected response where the rules say so.
        """
        return response

    @property
    def system_prompt(self) -> str:
        return f"{self.prompt.strip()}\n\nAnswer only with a JSON object that matches the given schema."

    @property
    def content_schema(self) -> dict[str, Any]:
        """JSON schema of the response without the envelope, i.e. only what the model has to produce."""
        schema = self.response_type.model_json_schema()
        schema["properties"] = {k: v for k, v in schema["properties"].items() if k not in ENVELOPE_FIELDS}
        schema["required"] = [k for k in schema.get("required", []) if k not in ENVELOPE_FIELDS]
        return schema

    def _envelope(self, request: Req) -> dict[str, Any]:
        return {
            "type": self.response_type.model_fields["type"].default,
            "signal_id": f"sig_{uuid.uuid4().hex}",
            "sent_at": datetime.now(timezone.utc),
            "situation_id": request.situation_id,
            "in_reply_to": request.signal_id,
        }
