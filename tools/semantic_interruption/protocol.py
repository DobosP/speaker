"""Strict JSONL contract for an isolated Anyreach shadow worker.

This module defines messages only.  It does not import a candidate runtime,
load model or benchmark bytes, map actions into interruption policy, or cause
effects.  Conversation text is deliberately hidden from dataclass reprs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from typing import Final, Mapping, TypeAlias

from tools.semantic_interruption.anyreach import AnyreachAction, CANDIDATE_ID


PROTOCOL_VERSION: Final = 1
MAX_LINE_BYTES: Final = 64 * 1024
MAX_MESSAGE_CHARS: Final = 16 * 1024
MAX_TOTAL_MESSAGE_CHARS: Final = 32 * 1024
MAX_BENCHMARK_ORDINAL: Final = 23
CPU_PROVIDER: Final = "CPUExecutionProvider"
MESSAGE_ROLE_ORDER: Final = ("user", "assistant", "user")
MODEL_ACTION_TOKENS: Final = (
    (AnyreachAction.CONTINUE_LISTENING, 151_665),
    (AnyreachAction.START_SPEAKING, 151_666),
    (AnyreachAction.START_LISTENING, 151_667),
    (AnyreachAction.CONTINUE_SPEAKING, 151_668),
)
MODEL_ACTION_ORDER: Final = tuple(action for action, _token in MODEL_ACTION_TOKENS)
MODEL_TOKEN_ORDER: Final = tuple(token for _action, token in MODEL_ACTION_TOKENS)

_SAFE_ID_RE: Final = re.compile(r"[a-z0-9][a-z0-9_.-]{0,95}\Z")
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_MAX_JSON_DEPTH: Final = 16
_MAX_JSON_NODES: Final = 4_096
_MAX_JSON_STRING_CHARS: Final = MAX_LINE_BYTES
_MIN_JSON_INTEGER: Final = -(1 << 63)
_MAX_JSON_INTEGER: Final = (1 << 63) - 1


class ProtocolError(RuntimeError):
    """A wire value failed closed without exposing private input detail."""


def _protocol_version(value: object) -> int:
    if type(value) is not int or value != PROTOCOL_VERSION:
        raise ProtocolError()
    return value


def _safe_id(value: object) -> str:
    if type(value) is not str or _SAFE_ID_RE.fullmatch(value) is None:
        raise ProtocolError()
    return value


def _sha256(value: object) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ProtocolError()
    return value


def _ordinal(value: object) -> int:
    if type(value) is not int or value < 0 or value > MAX_BENCHMARK_ORDINAL:
        raise ProtocolError()
    return value


def _finite_float(value: object) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ProtocolError()
    return value


@dataclass(frozen=True)
class ChatMessage:
    """One private benchmark message; content never appears in its repr."""

    role: str
    content: str = field(repr=False)

    def __post_init__(self) -> None:
        _validate_message(self)

    def as_dict(self) -> dict[str, object]:
        return {"role": self.role, "content": self.content}


def _validate_message(message: object) -> None:
    if type(message) is not ChatMessage:
        raise ProtocolError()
    if (
        type(message.role) is not str
        or message.role not in {"user", "assistant"}
        or type(message.content) is not str
        or not message.content
        or len(message.content) > MAX_MESSAGE_CHARS
        or "\x00" in message.content
    ):
        raise ProtocolError()


def _messages(value: object) -> tuple[ChatMessage, ChatMessage, ChatMessage]:
    if type(value) is not list or len(value) != len(MESSAGE_ROLE_ORDER):
        raise ProtocolError()
    parsed: list[ChatMessage] = []
    for raw, expected_role in zip(value, MESSAGE_ROLE_ORDER, strict=True):
        if type(raw) is not dict or set(raw) != {"role", "content"}:
            raise ProtocolError()
        message = ChatMessage(role=raw.get("role"), content=raw.get("content"))
        if message.role != expected_role:
            raise ProtocolError()
        parsed.append(message)
    if sum(len(message.content) for message in parsed) > MAX_TOTAL_MESSAGE_CHARS:
        raise ProtocolError()
    return (parsed[0], parsed[1], parsed[2])


def _validate_messages(value: object) -> None:
    if type(value) is not tuple or len(value) != len(MESSAGE_ROLE_ORDER):
        raise ProtocolError()
    messages = value
    for message in messages:
        _validate_message(message)
    if (
        tuple(message.role for message in messages) != MESSAGE_ROLE_ORDER
        or sum(len(message.content) for message in messages) > MAX_TOTAL_MESSAGE_CHARS
    ):
        raise ProtocolError()


@dataclass(frozen=True)
class ScoreRequest:
    request_id: str
    ordinal: int
    messages: tuple[ChatMessage, ChatMessage, ChatMessage] = field(repr=False)
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _protocol_version(self.protocol_version)
        _safe_id(self.request_id)
        _ordinal(self.ordinal)
        _validate_messages(self.messages)

    def as_dict(self) -> dict[str, object]:
        return {
            "v": self.protocol_version,
            "id": self.request_id,
            "op": "score",
            "ordinal": self.ordinal,
            "messages": [ChatMessage.as_dict(message) for message in self.messages],
        }


@dataclass(frozen=True)
class ShutdownRequest:
    request_id: str
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _protocol_version(self.protocol_version)
        _safe_id(self.request_id)

    def as_dict(self) -> dict[str, object]:
        return {
            "v": self.protocol_version,
            "id": self.request_id,
            "op": "shutdown",
        }


@dataclass(frozen=True)
class ReadyEvent:
    artifact_manifest_sha256: str
    artifact_set_sha256: str
    worker_manifest_sha256: str
    runtime_receipt_sha256: str
    candidate_id: str = CANDIDATE_ID
    provider: str = CPU_PROVIDER
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _protocol_version(self.protocol_version)
        if (
            type(self.candidate_id) is not str
            or self.candidate_id != CANDIDATE_ID
            or type(self.provider) is not str
            or self.provider != CPU_PROVIDER
        ):
            raise ProtocolError()
        _sha256(self.artifact_manifest_sha256)
        _sha256(self.artifact_set_sha256)
        _sha256(self.worker_manifest_sha256)
        _sha256(self.runtime_receipt_sha256)

    def as_dict(self) -> dict[str, object]:
        return {
            "v": self.protocol_version,
            "type": "ready",
            "candidate_id": self.candidate_id,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "artifact_set_sha256": self.artifact_set_sha256,
            "worker_manifest_sha256": self.worker_manifest_sha256,
            "runtime_receipt_sha256": self.runtime_receipt_sha256,
            "provider": self.provider,
        }


def stable_four_way_softmax(
    logits: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Return one stable softmax over all four publisher actions."""

    if type(logits) is not tuple or len(logits) != len(MODEL_ACTION_ORDER):
        raise ProtocolError()
    checked = tuple(_finite_float(value) for value in logits)
    maximum = max(checked)
    weights = tuple(math.exp(value - maximum) for value in checked)
    denominator = math.fsum(weights)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ProtocolError()
    probabilities = tuple(weight / denominator for weight in weights)
    if (
        len(probabilities) != 4
        or any(
            not math.isfinite(probability) or probability < 0.0 or probability > 1.0
            for probability in probabilities
        )
        or not math.isclose(
            math.fsum(probabilities),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ProtocolError()
    return probabilities  # type: ignore[return-value]


@dataclass(frozen=True)
class ArgmaxDecision:
    """A unique top action, or every exactly tied top action in model order."""

    unique_action: AnyreachAction | None
    tied_actions: tuple[AnyreachAction, ...]

    def __post_init__(self) -> None:
        if self.unique_action is not None:
            if (
                type(self.unique_action) is not AnyreachAction
                or type(self.tied_actions) is not tuple
                or self.tied_actions
            ):
                raise ProtocolError()
            return
        if (
            type(self.tied_actions) is not tuple
            or len(self.tied_actions) < 2
            or len(self.tied_actions) > len(MODEL_ACTION_ORDER)
            or any(type(action) is not AnyreachAction for action in self.tied_actions)
            or tuple(
                action for action in MODEL_ACTION_ORDER if action in self.tied_actions
            )
            != self.tied_actions
            or len(set(self.tied_actions)) != len(self.tied_actions)
        ):
            raise ProtocolError()

    @property
    def is_tie(self) -> bool:
        return self.unique_action is None


def deterministic_argmax(
    logits: tuple[float, float, float, float],
) -> ArgmaxDecision:
    """Classify the exact maximum without silently breaking a tie."""

    if type(logits) is not tuple or len(logits) != len(MODEL_ACTION_ORDER):
        raise ProtocolError()
    checked = tuple(_finite_float(value) for value in logits)
    maximum = max(checked)
    top = tuple(
        action
        for action, value in zip(MODEL_ACTION_ORDER, checked, strict=True)
        if value == maximum
    )
    if len(top) == 1:
        return ArgmaxDecision(unique_action=top[0], tied_actions=())
    return ArgmaxDecision(unique_action=None, tied_actions=top)


@dataclass(frozen=True)
class ScoreEvent:
    request_id: str
    ordinal: int
    logits: tuple[float, float, float, float]
    protocol_version: int = PROTOCOL_VERSION
    probabilities: tuple[float, float, float, float] = field(
        init=False,
        repr=False,
    )
    argmax: ArgmaxDecision = field(init=False)

    def __post_init__(self) -> None:
        _protocol_version(self.protocol_version)
        _safe_id(self.request_id)
        _ordinal(self.ordinal)
        probabilities = stable_four_way_softmax(self.logits)
        argmax = deterministic_argmax(self.logits)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "argmax", argmax)

    def as_dict(self) -> dict[str, object]:
        return {
            "v": self.protocol_version,
            "id": self.request_id,
            "type": "score",
            "ordinal": self.ordinal,
            "scores": [
                {
                    "action": action.value,
                    "token_id": token_id,
                    "logit": logit,
                }
                for (action, token_id), logit in zip(
                    MODEL_ACTION_TOKENS,
                    self.logits,
                    strict=True,
                )
            ],
        }


@dataclass(frozen=True)
class ErrorEvent:
    request_id: str
    code: str
    fatal: bool
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _protocol_version(self.protocol_version)
        _safe_id(self.request_id)
        _safe_id(self.code)
        if type(self.fatal) is not bool:
            raise ProtocolError()

    def as_dict(self) -> dict[str, object]:
        return {
            "v": self.protocol_version,
            "id": self.request_id,
            "type": "error",
            "code": self.code,
            "fatal": self.fatal,
        }


@dataclass(frozen=True)
class ShutdownEvent:
    request_id: str
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _protocol_version(self.protocol_version)
        _safe_id(self.request_id)

    def as_dict(self) -> dict[str, object]:
        return {
            "v": self.protocol_version,
            "id": self.request_id,
            "type": "shutdown",
        }


@dataclass(frozen=True)
class ByeEvent:
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _protocol_version(self.protocol_version)

    def as_dict(self) -> dict[str, object]:
        return {"v": self.protocol_version, "type": "bye"}


Request: TypeAlias = ScoreRequest | ShutdownRequest
Response: TypeAlias = ReadyEvent | ScoreEvent | ErrorEvent | ShutdownEvent | ByeEvent


def _bad_constant(_value: str) -> object:
    raise ProtocolError()


def _rebound_json_value(
    value: object,
    *,
    depth: int,
    remaining_nodes: list[int],
    string_chars: list[int],
    active_containers: set[int],
) -> object:
    if depth > _MAX_JSON_DEPTH or remaining_nodes[0] <= 0:
        raise ProtocolError()
    remaining_nodes[0] -= 1
    value_type = type(value)

    if value_type is dict:
        if len(value) > remaining_nodes[0] or id(value) in active_containers:
            raise ProtocolError()
        active_containers.add(id(value))
        try:
            snapshot = dict.copy(value)
            if len(snapshot) > remaining_nodes[0]:
                raise ProtocolError()
            rebound: dict[str, object] = {}
            for key, item in dict.items(snapshot):
                if type(key) is not str:
                    raise ProtocolError()
                string_chars[0] += len(key)
                if string_chars[0] > _MAX_JSON_STRING_CHARS:
                    raise ProtocolError()
                rebound[key] = _rebound_json_value(
                    item,
                    depth=depth + 1,
                    remaining_nodes=remaining_nodes,
                    string_chars=string_chars,
                    active_containers=active_containers,
                )
            return rebound
        finally:
            active_containers.remove(id(value))

    if value_type is list:
        if len(value) > remaining_nodes[0] or id(value) in active_containers:
            raise ProtocolError()
        active_containers.add(id(value))
        try:
            snapshot = list.copy(value)
            if len(snapshot) > remaining_nodes[0]:
                raise ProtocolError()
            return [
                _rebound_json_value(
                    item,
                    depth=depth + 1,
                    remaining_nodes=remaining_nodes,
                    string_chars=string_chars,
                    active_containers=active_containers,
                )
                for item in snapshot
            ]
        finally:
            active_containers.remove(id(value))

    if value_type is str:
        string_chars[0] += len(value)
        if string_chars[0] > _MAX_JSON_STRING_CHARS:
            raise ProtocolError()
        return value
    if value_type is int:
        if value < _MIN_JSON_INTEGER or value > _MAX_JSON_INTEGER:
            raise ProtocolError()
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ProtocolError()
        return value
    if value_type is bool or value is None:
        return value
    raise ProtocolError()


def encode_message(value: Mapping[str, object]) -> bytes:
    """Encode one canonical UTF-8 JSON object followed by exactly one LF."""

    if type(value) is not dict:
        raise ProtocolError()
    try:
        rebound = _rebound_json_value(
            value,
            depth=0,
            remaining_nodes=[_MAX_JSON_NODES],
            string_chars=[0],
            active_containers=set(),
        )
        if type(rebound) is not dict:
            raise ProtocolError()
        encoded = (
            json.dumps(
                rebound,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    except ProtocolError:
        raise
    except (
        MemoryError,
        OverflowError,
        RecursionError,
        RuntimeError,
        TypeError,
        UnicodeError,
        ValueError,
    ):
        raise ProtocolError() from None
    if not encoded or len(encoded) > MAX_LINE_BYTES:
        raise ProtocolError()
    return encoded


def strict_json_line(raw: bytes) -> dict[str, object]:
    """Parse one bounded canonical line and reject duplicate keys at any depth."""

    if (
        type(raw) is not bytes
        or not raw
        or len(raw) > MAX_LINE_BYTES
        or not raw.endswith(b"\n")
        or b"\n" in raw[:-1]
    ):
        raise ProtocolError()

    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ProtocolError()
            result[key] = value
        return result

    try:
        value = json.loads(
            raw[:-1].decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=_bad_constant,
        )
    except ProtocolError:
        raise
    except (OverflowError, RecursionError, UnicodeError, ValueError):
        raise ProtocolError() from None
    if type(value) is not dict or encode_message(value) != raw:
        raise ProtocolError()
    return value


def encode_request(value: Request) -> bytes:
    if type(value) is ScoreRequest:
        _validate_messages(value.messages)
        messages = tuple(
            ChatMessage(role=message.role, content=message.content)
            for message in value.messages
        )
        rebound: Request = ScoreRequest(
            request_id=value.request_id,
            ordinal=value.ordinal,
            messages=(messages[0], messages[1], messages[2]),
            protocol_version=value.protocol_version,
        )
    elif type(value) is ShutdownRequest:
        rebound = ShutdownRequest(
            request_id=value.request_id,
            protocol_version=value.protocol_version,
        )
    else:
        raise ProtocolError()
    if rebound != value:
        raise ProtocolError()
    encoded = encode_message(rebound.as_dict())
    if parse_request(encoded) != rebound:
        raise ProtocolError()
    return encoded


def encode_response(value: Response) -> bytes:
    if type(value) is ReadyEvent:
        rebound: Response = ReadyEvent(
            artifact_manifest_sha256=value.artifact_manifest_sha256,
            artifact_set_sha256=value.artifact_set_sha256,
            worker_manifest_sha256=value.worker_manifest_sha256,
            runtime_receipt_sha256=value.runtime_receipt_sha256,
            candidate_id=value.candidate_id,
            provider=value.provider,
            protocol_version=value.protocol_version,
        )
    elif type(value) is ScoreEvent:
        rebound = ScoreEvent(
            request_id=value.request_id,
            ordinal=value.ordinal,
            logits=value.logits,
            protocol_version=value.protocol_version,
        )
        if (
            type(value.probabilities) is not tuple
            or len(value.probabilities) != len(MODEL_ACTION_ORDER)
            or any(
                type(item) is not float or not math.isfinite(item)
                for item in value.probabilities
            )
            or type(value.argmax) is not ArgmaxDecision
        ):
            raise ProtocolError()
        ArgmaxDecision.__post_init__(value.argmax)
    elif type(value) is ErrorEvent:
        rebound = ErrorEvent(
            request_id=value.request_id,
            code=value.code,
            fatal=value.fatal,
            protocol_version=value.protocol_version,
        )
    elif type(value) is ShutdownEvent:
        rebound = ShutdownEvent(
            request_id=value.request_id,
            protocol_version=value.protocol_version,
        )
    elif type(value) is ByeEvent:
        rebound = ByeEvent(protocol_version=value.protocol_version)
    else:
        raise ProtocolError()
    if rebound != value:
        raise ProtocolError()
    encoded = encode_message(rebound.as_dict())
    if parse_response(encoded) != rebound:
        raise ProtocolError()
    return encoded


def parse_request(raw: bytes) -> Request:
    value = strict_json_line(raw)
    protocol_version = _protocol_version(value.get("v"))
    operation = value.get("op")
    if operation == "shutdown":
        if set(value) != {"v", "id", "op"}:
            raise ProtocolError()
        return ShutdownRequest(
            request_id=_safe_id(value.get("id")),
            protocol_version=protocol_version,
        )
    if operation != "score" or set(value) != {
        "v",
        "id",
        "op",
        "ordinal",
        "messages",
    }:
        raise ProtocolError()
    return ScoreRequest(
        request_id=_safe_id(value.get("id")),
        ordinal=_ordinal(value.get("ordinal")),
        messages=_messages(value.get("messages")),
        protocol_version=protocol_version,
    )


def _score_logits(value: object) -> tuple[float, float, float, float]:
    if type(value) is not list or len(value) != len(MODEL_ACTION_TOKENS):
        raise ProtocolError()
    logits: list[float] = []
    for raw, (expected_action, expected_token) in zip(
        value,
        MODEL_ACTION_TOKENS,
        strict=True,
    ):
        if type(raw) is not dict or set(raw) != {"action", "token_id", "logit"}:
            raise ProtocolError()
        if (
            raw.get("action") != expected_action.value
            or type(raw.get("action")) is not str
            or raw.get("token_id") != expected_token
            or type(raw.get("token_id")) is not int
        ):
            raise ProtocolError()
        logits.append(_finite_float(raw.get("logit")))
    return (logits[0], logits[1], logits[2], logits[3])


def parse_response(raw: bytes) -> Response:
    value = strict_json_line(raw)
    protocol_version = _protocol_version(value.get("v"))
    event_type = value.get("type")
    if event_type == "ready":
        if set(value) != {
            "v",
            "type",
            "candidate_id",
            "artifact_manifest_sha256",
            "artifact_set_sha256",
            "worker_manifest_sha256",
            "runtime_receipt_sha256",
            "provider",
        }:
            raise ProtocolError()
        return ReadyEvent(
            candidate_id=value.get("candidate_id"),
            artifact_manifest_sha256=_sha256(value.get("artifact_manifest_sha256")),
            artifact_set_sha256=_sha256(value.get("artifact_set_sha256")),
            worker_manifest_sha256=_sha256(value.get("worker_manifest_sha256")),
            runtime_receipt_sha256=_sha256(value.get("runtime_receipt_sha256")),
            provider=value.get("provider"),
            protocol_version=protocol_version,
        )
    if event_type == "score":
        if set(value) != {"v", "id", "type", "ordinal", "scores"}:
            raise ProtocolError()
        return ScoreEvent(
            request_id=_safe_id(value.get("id")),
            ordinal=_ordinal(value.get("ordinal")),
            logits=_score_logits(value.get("scores")),
            protocol_version=protocol_version,
        )
    if event_type == "error":
        if set(value) != {"v", "id", "type", "code", "fatal"}:
            raise ProtocolError()
        fatal = value.get("fatal")
        if type(fatal) is not bool:
            raise ProtocolError()
        return ErrorEvent(
            request_id=_safe_id(value.get("id")),
            code=_safe_id(value.get("code")),
            fatal=fatal,
            protocol_version=protocol_version,
        )
    if event_type == "shutdown":
        if set(value) != {"v", "id", "type"}:
            raise ProtocolError()
        return ShutdownEvent(
            request_id=_safe_id(value.get("id")),
            protocol_version=protocol_version,
        )
    if event_type == "bye":
        if set(value) != {"v", "type"}:
            raise ProtocolError()
        return ByeEvent(protocol_version=protocol_version)
    raise ProtocolError()
