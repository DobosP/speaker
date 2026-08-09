"""Bounded, dependency-free tests for the Anyreach worker wire contract."""

from __future__ import annotations

from dataclasses import replace
import json
import math

import pytest

from tools.semantic_interruption import anyreach
from tools.semantic_interruption import protocol


_DIGESTS = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)


def _messages(
    *,
    secret: str = "private benchmark text",
) -> tuple[protocol.ChatMessage, protocol.ChatMessage, protocol.ChatMessage]:
    return (
        protocol.ChatMessage("user", f"first {secret}"),
        protocol.ChatMessage("assistant", f"middle {secret}"),
        protocol.ChatMessage("user", f"last {secret}"),
    )


def _request() -> protocol.ScoreRequest:
    return protocol.ScoreRequest("case-0000", 0, _messages())


def _ready() -> protocol.ReadyEvent:
    return protocol.ReadyEvent(
        artifact_manifest_sha256=_DIGESTS[0],
        artifact_set_sha256=_DIGESTS[1],
        worker_manifest_sha256=_DIGESTS[2],
        runtime_receipt_sha256=_DIGESTS[3],
    )


def _score(
    logits: tuple[float, float, float, float] = (1.0, 2.0, 3.0, 4.0),
) -> protocol.ScoreEvent:
    return protocol.ScoreEvent("case-0000", 0, logits)


def test_protocol_v1_binds_exact_model_action_and_token_order() -> None:
    assert protocol.PROTOCOL_VERSION == 1
    assert protocol.MODEL_ACTION_TOKENS == (
        (anyreach.AnyreachAction.CONTINUE_LISTENING, 151_665),
        (anyreach.AnyreachAction.START_SPEAKING, 151_666),
        (anyreach.AnyreachAction.START_LISTENING, 151_667),
        (anyreach.AnyreachAction.CONTINUE_SPEAKING, 151_668),
    )
    assert (
        protocol.MODEL_ACTION_ORDER
        != anyreach.load_source_lock().benchmark_class_labels
    )


def test_score_request_round_trip_is_canonical_and_repr_hides_text() -> None:
    secret = "do-not-log-this"
    request = protocol.ScoreRequest("case-0007", 7, _messages(secret=secret))

    raw = protocol.encode_request(request)
    parsed = protocol.parse_request(raw)

    assert parsed == request
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 1
    assert raw == protocol.encode_message(json.loads(raw))
    assert secret not in repr(request)
    assert secret not in repr(request.messages[0])


def test_score_request_has_exact_user_assistant_user_messages() -> None:
    with pytest.raises(protocol.ProtocolError):
        protocol.ScoreRequest(
            "case-0000",
            0,
            (
                protocol.ChatMessage("user", "one"),
                protocol.ChatMessage("user", "two"),
                protocol.ChatMessage("user", "three"),
            ),
        )
    with pytest.raises(protocol.ProtocolError):
        protocol.ScoreRequest("case-0000", 0, _messages()[:2])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_content",
    ["", "nul\x00text", "x" * (protocol.MAX_MESSAGE_CHARS + 1)],
)
def test_message_content_is_nonempty_nul_free_and_bounded(bad_content: str) -> None:
    with pytest.raises(protocol.ProtocolError):
        protocol.ChatMessage("user", bad_content)


def test_total_message_content_is_bounded() -> None:
    messages = (
        protocol.ChatMessage("user", "a" * protocol.MAX_MESSAGE_CHARS),
        protocol.ChatMessage("assistant", "b" * protocol.MAX_MESSAGE_CHARS),
        protocol.ChatMessage("user", "c"),
    )
    with pytest.raises(protocol.ProtocolError):
        protocol.ScoreRequest("case-0000", 0, messages)


@pytest.mark.parametrize("field", ["role", "content"])
def test_score_request_revalidates_forged_message_fields(field: str) -> None:
    messages = list(_messages())
    forged = messages[0]
    object.__setattr__(
        forged,
        field,
        "assistant" if field == "role" else "x" * (protocol.MAX_MESSAGE_CHARS + 1),
    )

    with pytest.raises(protocol.ProtocolError):
        protocol.ScoreRequest("case-0000", 0, tuple(messages))  # type: ignore[arg-type]


def test_lifecycle_messages_round_trip() -> None:
    responses: tuple[protocol.Response, ...] = (
        _ready(),
        _score(),
        protocol.ErrorEvent("case-0000", "invalid-input", True),
        protocol.ShutdownEvent("shutdown", 1),
        protocol.ByeEvent(),
    )
    for response in responses:
        assert protocol.parse_response(protocol.encode_response(response)) == response

    shutdown = protocol.ShutdownRequest("shutdown")
    assert protocol.parse_request(protocol.encode_request(shutdown)) == shutdown


def test_ready_is_exact_candidate_cpu_and_receipt_gated() -> None:
    ready = _ready()

    assert ready.candidate_id == anyreach.CANDIDATE_ID
    assert ready.provider == "CPUExecutionProvider"
    assert protocol.parse_response(protocol.encode_response(ready)) == ready
    with pytest.raises(protocol.ProtocolError):
        replace(ready, provider="CUDAExecutionProvider")
    with pytest.raises(protocol.ProtocolError):
        replace(ready, candidate_id="other")
    with pytest.raises(protocol.ProtocolError):
        replace(ready, runtime_receipt_sha256="0" * 63)
    with pytest.raises(protocol.ProtocolError):
        replace(ready, provider=object())  # type: ignore[arg-type]


def test_score_event_round_trip_carries_self_describing_fixed_order() -> None:
    event = _score((-4.0, -3.0, -2.0, -1.0))
    raw = protocol.encode_response(event)
    value = json.loads(raw)

    assert protocol.parse_response(raw) == event
    assert [(item["action"], item["token_id"]) for item in value["scores"]] == [
        (action.value, token_id) for action, token_id in protocol.MODEL_ACTION_TOKENS
    ]
    assert [item["logit"] for item in value["scores"]] == list(event.logits)
    assert "probabilities" not in value
    assert "takeover" not in raw.decode("utf-8")


@pytest.mark.parametrize("field", ["action", "token_id"])
def test_score_event_rejects_action_or_token_substitution(field: str) -> None:
    value = _score().as_dict()
    scores = value["scores"]
    assert isinstance(scores, list)
    first = scores[0]
    assert isinstance(first, dict)
    first[field] = "start_speaking" if field == "action" else 151_666

    with pytest.raises(protocol.ProtocolError):
        protocol.parse_response(protocol.encode_message(value))


def test_score_event_rejects_reordered_score_records() -> None:
    value = _score().as_dict()
    scores = value["scores"]
    assert isinstance(scores, list)
    scores[0], scores[1] = scores[1], scores[0]

    with pytest.raises(protocol.ProtocolError):
        protocol.parse_response(protocol.encode_message(value))


def test_softmax_is_stable_over_all_four_logits() -> None:
    event = _score((1.0e308, -1.0e308, 1.0e308, 0.0))

    assert event.probabilities == (0.5, 0.0, 0.5, 0.0)
    assert math.fsum(event.probabilities) == 1.0
    assert event.argmax == protocol.ArgmaxDecision(
        unique_action=None,
        tied_actions=(
            anyreach.AnyreachAction.CONTINUE_LISTENING,
            anyreach.AnyreachAction.START_LISTENING,
        ),
    )


@pytest.mark.parametrize("winner_index", range(4))
def test_unique_argmax_is_deterministic_in_model_order(winner_index: int) -> None:
    logits = [-2.0, -2.0, -2.0, -2.0]
    logits[winner_index] = 5.0
    event = _score(tuple(logits))  # type: ignore[arg-type]

    assert not event.argmax.is_tie
    assert event.argmax.unique_action is protocol.MODEL_ACTION_ORDER[winner_index]
    assert event.argmax.tied_actions == ()


@pytest.mark.parametrize(
    ("logits", "expected"),
    [
        ((2.0, 2.0, 0.0, 0.0), (0, 1)),
        ((3.0, 0.0, 3.0, 0.0), (0, 2)),
        ((1.0, 1.0, 1.0, 1.0), (0, 1, 2, 3)),
    ],
)
def test_ties_are_explicit_and_never_broken_by_position(
    logits: tuple[float, float, float, float],
    expected: tuple[int, ...],
) -> None:
    decision = _score(logits).argmax

    assert decision.is_tie
    assert decision.unique_action is None
    assert decision.tied_actions == tuple(
        protocol.MODEL_ACTION_ORDER[index] for index in expected
    )


def test_unique_argmax_requires_an_exact_empty_tie_tuple() -> None:
    with pytest.raises(protocol.ProtocolError):
        protocol.ArgmaxDecision(
            unique_action=anyreach.AnyreachAction.START_LISTENING,
            tied_actions=None,  # type: ignore[arg-type]
        )
    with pytest.raises(protocol.ProtocolError):
        protocol.ArgmaxDecision(
            unique_action=None,
            tied_actions=(anyreach.AnyreachAction.START_LISTENING,) * 5,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_logits_fail_direct_and_encoding(value: float) -> None:
    with pytest.raises(protocol.ProtocolError):
        _score((value, 0.0, 0.0, 0.0))
    with pytest.raises(protocol.ProtocolError):
        protocol.encode_message({"value": value})


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}\n',
        b'{"value":NaN}\n',
        b'{"value":Infinity}\n',
        b'{"value":1}',
        b'{ "value":1}\n',
        b'{"value":1}\n{"value":2}\n',
        b"\xff\n",
        b"[]\n",
        b"",
    ],
)
def test_strict_json_line_rejects_duplicates_constants_and_noncanonical_bytes(
    raw: bytes,
) -> None:
    with pytest.raises(protocol.ProtocolError) as caught:
        protocol.strict_json_line(raw)
    assert str(caught.value) == ""


def test_nested_duplicate_json_keys_are_rejected() -> None:
    raw = b'{"messages":[{"content":"one","role":"user","role":"assistant"}]}\n'

    with pytest.raises(protocol.ProtocolError):
        protocol.strict_json_line(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("v", True),
        ("ordinal", False),
        ("ordinal", 24),
        ("id", "PRIVATE TEXT"),
    ],
)
def test_request_scalar_types_and_bounds_are_strict(field: str, value: object) -> None:
    raw = _request().as_dict()
    raw[field] = value

    with pytest.raises(protocol.ProtocolError):
        protocol.parse_request(protocol.encode_message(raw))


def test_json_integer_is_not_accepted_as_a_model_logit() -> None:
    value = _score().as_dict()
    scores = value["scores"]
    assert isinstance(scores, list) and isinstance(scores[0], dict)
    scores[0]["logit"] = 1

    with pytest.raises(protocol.ProtocolError):
        protocol.parse_response(protocol.encode_message(value))


def test_unknown_and_missing_fields_fail_closed() -> None:
    request = _request().as_dict()
    request["unknown"] = False
    response = _score().as_dict()
    response.pop("ordinal")

    with pytest.raises(protocol.ProtocolError):
        protocol.parse_request(protocol.encode_message(request))
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_response(protocol.encode_message(response))


def test_line_limit_is_enforced_before_json_parsing() -> None:
    raw = b"{" + b"x" * protocol.MAX_LINE_BYTES + b"}\n"

    with pytest.raises(protocol.ProtocolError):
        protocol.strict_json_line(raw)


def test_encode_message_rejects_nested_coercion_dispatch_cycles_and_oversize() -> None:
    class ExplodingList(list[object]):
        def __iter__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("list subclass was iterated")

    class ExplodingDict(dict[str, object]):
        def items(self):  # type: ignore[no-untyped-def]
            raise AssertionError("dict subclass items were dispatched")

    class ExplodingText(str):
        def __lt__(self, _other: object) -> bool:
            raise AssertionError("string subclass comparison was dispatched")

    cycle: list[object] = []
    cycle.append(cycle)
    rejected: tuple[dict[object, object], ...] = (
        {1: "integer key"},
        {True: "boolean key"},
        {None: "null key"},
        {ExplodingText("key"): "value"},
        {"nested": ExplodingList([1])},
        {"nested": ExplodingDict({"key": "value"})},
        {"tuple": (1, 2)},
        {"cycle": cycle},
        {"too_many": [None] * (protocol._MAX_JSON_NODES + 1)},
        {"huge_integer": 1 << 64},
    )

    for value in rejected:
        with pytest.raises(protocol.ProtocolError):
            protocol.encode_message(value)  # type: ignore[arg-type]


def test_encode_helpers_reject_the_wrong_message_direction() -> None:
    with pytest.raises(protocol.ProtocolError):
        protocol.encode_request(_score())  # type: ignore[arg-type]
    with pytest.raises(protocol.ProtocolError):
        protocol.encode_response(_request())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kind", "field", "value"),
    [
        ("request", "request_id", "PRIVATE TEXT"),
        ("request", "messages", ()),
        ("score", "ordinal", 24),
        ("score", "logits", (1.0, 2.0, 3.0, float("nan"))),
        ("score", "probabilities", (0.25, 0.25, 0.25, 0.25)),
    ],
)
def test_encode_helpers_reject_forged_in_process_messages(
    kind: str,
    field: str,
    value: object,
) -> None:
    message: protocol.Request | protocol.Response
    message = _request() if kind == "request" else _score()
    object.__setattr__(message, field, value)

    with pytest.raises(protocol.ProtocolError):
        if kind == "request":
            protocol.encode_request(message)  # type: ignore[arg-type]
        else:
            protocol.encode_response(message)  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ["request", "score", "probabilities"])
def test_encode_helpers_reject_unbounded_forged_fields_before_iteration(
    kind: str,
) -> None:
    def unbounded():  # type: ignore[no-untyped-def]
        raise AssertionError("forged iterable was consumed")
        yield None

    message: protocol.Request | protocol.Response
    message = _request() if kind == "request" else _score()
    object.__setattr__(
        message,
        (
            "messages"
            if kind == "request"
            else "probabilities"
            if kind == "probabilities"
            else "logits"
        ),
        unbounded(),
    )

    with pytest.raises(protocol.ProtocolError):
        if kind == "request":
            protocol.encode_request(message)  # type: ignore[arg-type]
        else:
            protocol.encode_response(message)  # type: ignore[arg-type]


def test_encode_helpers_never_dispatch_forged_nested_instance_methods() -> None:
    def dispatched() -> object:
        raise AssertionError("forged instance method was dispatched")

    request = _request()
    object.__setattr__(request.messages[0], "as_dict", dispatched)
    event = _score()
    object.__setattr__(event.argmax, "__post_init__", dispatched)

    assert protocol.parse_request(protocol.encode_request(request)) == request
    assert protocol.parse_response(protocol.encode_response(event)) == event
