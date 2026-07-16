from __future__ import annotations

import json
import logging

import pytest

import tools.stt_selector_eval as selector_eval


def _source(source_id: str, text: str) -> selector_eval.SourceHypothesis:
    return selector_eval.SourceHypothesis(source_id, text)


def test_private_text_is_hidden_from_source_case_and_decision_reprs():
    reference = "SENTINEL_PRIVATE_REFERENCE"
    hypothesis = "SENTINEL_PRIVATE_HYPOTHESIS"
    source = _source("production", hypothesis)
    case = selector_eval.VerifierCase(reference, (source,), "production")
    decision = selector_eval.SelectionDecision.selected(0)

    rendered = repr((source, case, decision))

    assert reference not in rendered
    assert hypothesis not in rendered
    assert "production" in rendered


def test_single_source_selector_returns_only_configured_input_index():
    sources = (
        _source("production", "search my bolt"),
        _source("gpu", "search my vault"),
    )

    selected = selector_eval.SingleSourceSelector("gpu").select(sources)
    missing = selector_eval.SingleSourceSelector("offline").select(sources)
    empty = selector_eval.SingleSourceSelector("gpu").select(
        (_source("gpu", "..."),)
    )

    assert selected.selected_index == 1
    assert sources[selected.selected_index].text == "search my vault"
    assert missing.selected_index is None
    assert missing.reason is selector_eval.SelectionReason.MISSING_SOURCE
    assert empty.selected_index is None
    assert empty.reason is selector_eval.SelectionReason.EMPTY_SOURCE


def test_consensus_selects_an_existing_exact_quorum_permutation_invariant():
    selector = selector_eval.ConsensusSelector(min_support=2)
    original = (
        _source("z_gpu", "Find, my vault!"),
        _source("a_offline", "find my vault"),
        _source("production", "find my bolt"),
    )

    first = selector.select(original)
    permuted = (original[2], original[0], original[1])
    second = selector.select(permuted)

    assert original[first.selected_index].source_id == "a_offline"
    assert permuted[second.selected_index].source_id == "a_offline"
    assert first.support_score == pytest.approx(8.0 / 9.0)


def test_consensus_abstains_on_differing_best_candidates_and_no_quorum():
    tied = selector_eval.ConsensusSelector(
        min_support=2,
        min_similarity=0.0,
    ).select(
        (
            _source("one", "alpha"),
            _source("two", "beta"),
        )
    )
    no_quorum = selector_eval.ConsensusSelector().select(
        (
            _source("one", "alpha"),
            _source("two", "beta"),
            _source("three", "gamma"),
        )
    )

    assert tied.reason is selector_eval.SelectionReason.TIED_CONSENSUS
    assert no_quorum.reason is selector_eval.SelectionReason.NO_CONSENSUS


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "[]",
        '{"choice": true}',
        '{"choice": 1.0}',
        '{"choice": -1}',
        '{"choice": 2}',
        '{"choice": "0"}',
        '{"choice": "ABSTAIN"}',
        '{"choice": 0, "choice": 1}',
        '{"choice": 0, "text": "invented"}',
        '{"answer": 0}',
    ],
)
def test_choice_parser_rejects_non_index_and_non_exact_json(content):
    with pytest.raises(selector_eval.InvalidChoiceOutput) as raised:
        selector_eval.parse_choice(content, 2)

    assert content not in str(raised.value)


def test_choice_parser_accepts_only_in_range_index_or_literal_abstain():
    assert selector_eval.parse_choice('{"choice": 0}', 2) == 0
    assert selector_eval.parse_choice('{"choice": 1}', 2) == 1
    assert selector_eval.parse_choice('{"choice": "abstain"}', 2) is None


class _TargetTransport:
    def __init__(self, target: str) -> None:
        self.target = target
        self.payloads: list[dict[str, object]] = []

    def request(self, payload):
        self.payloads.append(payload)
        messages = payload["messages"]
        prompt = json.loads(messages[1]["content"])
        choice = next(
            candidate["index"]
            for candidate in prompt["candidates"]
            if candidate["transcript"] == self.target
        )
        return {"message": {"content": json.dumps({"choice": choice})}}


def test_ollama_selector_requires_unanimous_canonical_choice_without_logging(caplog):
    private_target = "SENTINEL_PRIVATE_BEST_TRANSCRIPT"
    sources = (
        _source("production", "the wrong words"),
        _source("gpu", private_target),
        _source("offline", "other wrong words"),
    )
    transport = _TargetTransport(private_target)
    selector = selector_eval.LocalOllamaChoiceSelector(
        "local-model",
        repeats=2,
        max_sources=3,
        transport=transport,
    )

    with caplog.at_level(logging.DEBUG):
        decision = selector.select(sources)

    assert decision.selected_index == 1
    assert decision.attempts == 12
    assert len(transport.payloads) == 12
    assert private_target not in caplog.text
    for payload in transport.payloads:
        assert payload["stream"] is False
        assert payload["think"] is False
        assert payload["format"]["required"] == ["choice"]
        assert payload["format"]["additionalProperties"] is False
        integer_choice = payload["format"]["properties"]["choice"]["oneOf"][0]
        assert integer_choice == {"type": "integer", "minimum": 0, "maximum": 2}
        assert payload["options"] == {
            "temperature": 0.0,
            "seed": 0,
            "top_k": 1,
            "num_predict": 8,
        }


class _FirstPositionTransport:
    def request(self, _payload):
        return {"message": {"content": '{"choice": 0}'}}


def test_ollama_selector_abstains_when_position_choice_is_permutation_unstable():
    selector = selector_eval.LocalOllamaChoiceSelector(
        "local-model",
        repeats=2,
        max_sources=3,
        transport=_FirstPositionTransport(),
    )

    decision = selector.select(
        (
            _source("one", "alpha"),
            _source("two", "beta"),
            _source("three", "gamma"),
        )
    )

    assert decision.selected_index is None
    assert decision.reason is selector_eval.SelectionReason.UNSTABLE_CHOICE
    assert decision.attempts < 12


def test_ollama_selector_does_not_call_model_without_two_usable_sources():
    transport = _FirstPositionTransport()
    selector = selector_eval.LocalOllamaChoiceSelector(
        "local-model",
        source_ids=("one",),
        transport=transport,
    )

    decision = selector.select((_source("one", "alpha"),))

    assert decision.reason is selector_eval.SelectionReason.NO_CONSENSUS
    assert decision.attempts == 0


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (
            {"message": {"content": '{"choice": "abstain"}'}},
            selector_eval.SelectionReason.MODEL_ABSTAIN,
        ),
        (
            {"message": {"content": '{"choice": 0, "text": "rewrite"}'}},
            selector_eval.SelectionReason.INVALID_OUTPUT,
        ),
        (
            {"message": {"content": "free-form rewritten transcript"}},
            selector_eval.SelectionReason.INVALID_OUTPUT,
        ),
    ],
)
def test_ollama_selector_abstains_on_model_abstain_or_non_choice_output(
    response,
    reason,
):
    class _Transport:
        def request(self, _payload):
            return response

    selector = selector_eval.LocalOllamaChoiceSelector(
        "local-model",
        repeats=1,
        transport=_Transport(),
    )
    decision = selector.select(
        (_source("one", "alpha"), _source("two", "beta"))
    )

    assert decision.selected_index is None
    assert decision.reason is reason
    assert "rewrite" not in repr(decision)
    assert "free-form" not in repr(decision)


def test_ollama_selector_turns_provider_details_into_safe_abstention():
    class _FailingTransport:
        def request(self, _payload):
            raise RuntimeError("SENTINEL_PRIVATE_PROMPT_AND_PATH")

    selector = selector_eval.LocalOllamaChoiceSelector(
        "local-model",
        repeats=1,
        transport=_FailingTransport(),
    )
    decision = selector.select(
        (_source("one", "alpha"), _source("two", "beta"))
    )

    assert decision.reason is selector_eval.SelectionReason.PROVIDER_ERROR
    assert "SENTINEL" not in repr(decision)


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:11434",
        "http://0.0.0.0:11434",
        "http://192.168.1.20:11434",
        "http://ollama.internal:11434",
        "http://user:pass@127.0.0.1:11434",
        "http://127.0.0.1:11434/api",
        "http://127.0.0.1:11434?token=secret",
        "http://127.0.0.1:0",
        "http://127.0.0.1:",
    ],
)
def test_ollama_selector_rejects_every_non_direct_loopback_endpoint(endpoint):
    with pytest.raises(selector_eval.SelectionInputError):
        selector_eval.LocalOllamaChoiceSelector(
            "local-model",
            endpoint=endpoint,
            transport=_FirstPositionTransport(),
        )


def test_ollama_selector_validates_timeout_even_with_injected_transport():
    with pytest.raises(selector_eval.SelectionInputError):
        selector_eval.LocalOllamaChoiceSelector(
            "local-model",
            timeout_sec=-1,
            transport=_FirstPositionTransport(),
        )


def test_private_transport_posts_directly_without_auth_redirect_or_prompt_logging(
    monkeypatch,
    caplog,
):
    private_target = "SENTINEL_PRIVATE_DIRECT_TRANSPORT"
    requests = []

    class _Response:
        status = 200

        def __init__(self, content):
            self._content = content

        def read(self, _limit):
            return json.dumps(
                {"message": {"content": json.dumps({"choice": self._content})}}
            ).encode()

    class _Connection:
        def __init__(self, host, port, timeout):
            assert host == "127.0.0.1"
            assert port == 11434
            assert timeout == 3.0
            self.choice = None

        def set_debuglevel(self, level):
            assert level == 0

        def request(self, method, path, *, body, headers):
            assert method == "POST"
            assert path == "/api/chat"
            assert not any(key.lower() == "authorization" for key in headers)
            payload = json.loads(body)
            prompt = json.loads(payload["messages"][1]["content"])
            self.choice = next(
                item["index"]
                for item in prompt["candidates"]
                if item["transcript"] == private_target
            )
            requests.append((path, tuple(sorted(headers))))

        def getresponse(self):
            return _Response(self.choice)

        def close(self):
            pass

    monkeypatch.setattr(selector_eval.http.client, "HTTPConnection", _Connection)
    selector = selector_eval.LocalOllamaChoiceSelector(
        "local-model",
        repeats=1,
        max_sources=2,
        timeout_sec=3.0,
    )
    with caplog.at_level(logging.DEBUG):
        decision = selector.select(
            (
                _source("gpu", private_target),
                _source("offline", "other words"),
            )
        )

    assert decision.selected_index == 0
    assert decision.attempts == 2
    assert len(requests) == 2
    assert private_target not in caplog.text


class _FixedSelector:
    def __init__(self, index: int, latency_ns: int = 2_000_000) -> None:
        self.index = index
        self.latency_ns = latency_ns

    def select(self, _sources):
        return selector_eval.SelectionDecision.selected(
            self.index,
            latency_ns=self.latency_ns,
            support_score=0.75,
        )


def test_aggregate_evaluator_reports_accuracy_safety_support_and_latency_only():
    case = selector_eval.VerifierCase(
        "search my vault",
        (
            _source("production", "search my bolt"),
            _source("gpu", "search my vault"),
            _source("offline", "search my vault"),
        ),
        "production",
    )

    report = selector_eval.evaluate_variants(
        [case],
        {
            "single_gpu": selector_eval.SingleSourceSelector("gpu"),
            "consensus": selector_eval.ConsensusSelector(),
            "fixed": _FixedSelector(1),
        },
        keywords=("vault",),
    )

    assert report["aggregate_only"] is True
    assert report["clips"] == 1
    assert report["baseline"]["accuracy"]["word_errors"] == 1
    for name in ("single_gpu", "consensus", "fixed"):
        variant = report["variants"][name]
        assert variant["accuracy"]["word_errors"] == 0
        assert variant["comparison"]["promotable"] is True
        assert variant["unsupported_token_edits"] == 0
        assert variant["selection_coverage"] == 1.0
    assert report["variants"]["fixed"]["selection_support_mean"] == 0.75
    assert report["variants"]["fixed"]["selector_latency_ms"] == {
        "p50": 2.0,
        "p95": 2.0,
        "max": 2.0,
    }


def test_aggregate_evaluator_flags_unsupported_and_stop_command_regressions():
    case = selector_eval.VerifierCase(
        "continue now",
        (
            _source("production", "continue now"),
            _source("gpu", "stop"),
            _source("offline", "continue now"),
        ),
        "production",
    )

    variant = selector_eval.evaluate_variants(
        [case],
        {"unsafe": _FixedSelector(1)},
    )["variants"]["unsafe"]

    assert variant["unsupported_token_edits"] > 0
    assert variant["unsupported_control_edits"] == 1
    assert variant["stop_commands"]["false_activations"] == 1
    assert variant["stop_commands"]["regressions"] == 1
    assert variant["comparison"]["promotable"] is False


def test_aggregate_evaluator_falls_back_on_invalid_index_and_has_no_private_rows():
    private_reference = "SENTINEL_PRIVATE_REFERENCE"
    private_baseline = "SENTINEL_PRIVATE_BASELINE"
    private_hypothesis = "SENTINEL_PRIVATE_HYPOTHESIS"
    case = selector_eval.VerifierCase(
        private_reference,
        (
            _source("production", private_baseline),
            _source("gpu", private_hypothesis),
        ),
        "production",
    )

    report = selector_eval.evaluate_variants(
        [case],
        {"invalid": _FixedSelector(99)},
    )
    encoded = json.dumps(report, sort_keys=True)
    variant = report["variants"]["invalid"]

    assert variant["selections"] == 0
    assert variant["abstention_reasons"] == {"invalid_output": 1}
    assert private_reference not in encoded
    assert private_baseline not in encoded
    assert private_hypothesis not in encoded
    assert "rows" not in report


def test_aggregate_evaluator_sanitizes_selector_exception_details():
    class _ExplodingSelector:
        def select(self, _sources):
            raise RuntimeError("SENTINEL_PRIVATE_SELECTOR_DETAIL")

    case = selector_eval.VerifierCase(
        "alpha",
        (_source("production", "beta"),),
        "production",
    )

    report = selector_eval.evaluate_variants(
        [case],
        {"failure": _ExplodingSelector()},
    )
    encoded = json.dumps(report)

    assert report["variants"]["failure"]["abstention_reasons"] == {
        "provider_error": 1
    }
    assert "SENTINEL" not in encoded
