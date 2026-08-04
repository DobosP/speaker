from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

import tools.recorded_stt_eval as stt_eval
import tools.tool_route_gate as route_gate


@pytest.fixture
def full_classifier():
    return route_gate.ToolRouteClassifier(
        route_gate.ToolRouteGateProfile(
            vault_enabled=True,
            reminders_enabled=True,
            app_aliases=("notes",),
        )
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("tell me a short joke", "none"),
        ("search in my vault for the roadmap", "vault.search"),
        ("go in my vault and find the roadmap", "vault.search"),
        ("find in my vault the roadmap", "vault.search"),
        ("search the web for the weather", "web.search"),
        ("remind me to stretch in 10 minutes", "reminder.create"),
        ("list my reminders", "reminder.list"),
        ("cancel reminder 01234567", "reminder.cancel"),
        ("cancel my next reminder", "reminder.cancel"),
        ("open notes", "app.open"),
        ("stop", "control"),
        ("open terminal", "generic_action"),
        ("open notes https://example.invalid", "generic_action"),
        ("dictate a private sentence", "other_action"),
    ],
)
def test_classifier_uses_closed_production_routes(full_classifier, text, expected):
    assert full_classifier.classify(text) == expected


def test_disabled_profile_does_not_invent_optional_matchers():
    classifier = route_gate.ToolRouteClassifier(route_gate.ToolRouteGateProfile())

    assert classifier.classify("find in my vault the roadmap") == "none"
    assert classifier.classify("list my reminders") == "none"
    assert classifier.classify("open notes") == "generic_action"
    assert classifier.classify("search the web for weather") == "web.search"


def test_match_is_deterministic_and_never_invokes_or_launches(monkeypatch):
    opened: list[str] = []

    def forbidden_open(_self, desktop_id):
        opened.append(desktop_id)
        raise AssertionError("launcher must stay inert")

    def forbidden_dispatch(*_args, **_kwargs):
        raise AssertionError("dispatcher dispatch must stay inert")

    monkeypatch.setattr(route_gate._NoOpenLauncher, "open", forbidden_open)
    from core.trusted_apps import DeviceToolCommandDispatcher

    monkeypatch.setattr(DeviceToolCommandDispatcher, "dispatch", forbidden_dispatch)
    profile = route_gate.ToolRouteGateProfile(
        reminders_enabled=True,
        app_aliases=("notes",),
    )

    first = route_gate.ToolRouteClassifier(profile)
    second = route_gate.ToolRouteClassifier(profile)
    phrases = (
        "remind me to stretch in 10 minutes",
        "list my reminders",
        "cancel my next reminder",
        "open notes",
    )
    assert [first.classify(text) for text in phrases] == [
        "reminder.create",
        "reminder.list",
        "reminder.cancel",
        "app.open",
    ]
    assert [second.classify(text) for text in phrases] == [
        "reminder.create",
        "reminder.list",
        "reminder.cancel",
        "app.open",
    ]
    assert opened == []


def test_reminder_matching_never_constructs_store_or_manager(monkeypatch):
    import core.reminders as reminders

    def forbidden(*_args, **_kwargs):
        raise AssertionError("persistent reminder state must stay unreachable")

    monkeypatch.setattr(reminders.ReminderStore, "__init__", forbidden)
    monkeypatch.setattr(reminders.ReminderManager, "__init__", forbidden)

    classifier = route_gate.ToolRouteClassifier(
        route_gate.ToolRouteGateProfile(reminders_enabled=True)
    )
    assert classifier.classify("list my reminders") == "reminder.list"
    assert (
        classifier.classify("remind me to stretch in 10 minutes") == "reminder.create"
    )


@pytest.mark.parametrize(
    "tags",
    [
        (),
        ("private-diagnostic",),
        ("expected-tool.unknown",),
        ("expected-tool.none", "expected-tool.web.search"),
        ("expected-tool.none", "expected-tool.none"),
    ],
)
def test_expected_tag_grammar_fails_closed_without_detail(tags):
    canary = "SENTINEL_PRIVATE_TAG"
    hostile = (*tags, canary)

    with pytest.raises(route_gate.ToolRouteGateError) as caught:
        route_gate.expected_route(hostile)

    assert canary not in str(caught.value)
    assert canary not in repr(caught.value)


def test_accumulator_scores_all_expected_routes_aggregate_only(full_classifier):
    accumulator = route_gate.ToolRouteGateAccumulator(full_classifier)
    cases = (
        ("none", "tell me a joke"),
        ("vault.search", "search in my vault for the roadmap"),
        ("web.search", "search the web for weather"),
        ("reminder.create", "remind me to stretch in 10 minutes"),
        ("reminder.list", "list my reminders"),
        ("reminder.cancel", "cancel my next reminder"),
        ("app.open", "open notes"),
    )
    for expected, text in cases:
        accumulator.add_case((f"expected-tool.{expected}",), (text,))

    totals = accumulator.totals()
    report = totals.as_dict()
    assert totals.complete is True
    assert totals.annotated_cases == totals.decisions == totals.exact_cases == 7
    assert totals.expected_positive_cases == 6
    assert totals.expected_none_cases == 1
    assert [row["route"] for row in report["per_expected"]] == list(
        route_gate.EXPECTED_ROUTES
    )
    assert all(row["attempts"] == row["hits"] == 1 for row in report["per_expected"])
    assert "joke" not in repr(totals)


@pytest.mark.parametrize(
    ("selected", "decisions", "single", "empty", "multi"),
    [
        ((), 0, 0, 0, 0),
        (("",), 1, 1, 1, 0),
        (("search the web", "for weather"), 2, 0, 0, 1),
    ],
)
def test_zero_empty_and_multi_terminal_boundaries_fail_without_joining(
    full_classifier,
    selected,
    decisions,
    single,
    empty,
    multi,
):
    accumulator = route_gate.ToolRouteGateAccumulator(full_classifier)
    accumulator.add_case(("expected-tool.web.search",), selected)
    totals = accumulator.totals()

    assert totals.complete is False
    assert totals.decisions == decisions
    assert totals.single_decision_cases == single
    assert totals.empty_decisions == empty
    assert totals.multi_decision_cases == multi
    assert totals.misses == 1


def test_counter_taxonomy_distinguishes_miss_wrong_and_unsafe(full_classifier):
    accumulator = route_gate.ToolRouteGateAccumulator(full_classifier)
    rows = (
        ("vault.search", "search the web for weather"),
        ("none", "search the web for weather"),
        ("none", "stop"),
        ("none", "open terminal"),
    )
    for expected, selected in rows:
        accumulator.add_case((f"expected-tool.{expected}",), (selected,))
    totals = accumulator.totals()

    assert totals.misses == 1
    assert totals.wrong_tool == 1
    assert totals.unexpected_tool == 1
    assert totals.unexpected_control == 1
    assert totals.unexpected_action == 1
    assert totals.complete is False


def _valid_totals(**updates):
    values = {
        "annotated_cases": 2,
        "decisions": 2,
        "single_decision_cases": 2,
        "empty_decisions": 0,
        "expected_positive_cases": 1,
        "expected_none_cases": 1,
        "exact_cases": 2,
        "misses": 0,
        "wrong_tool": 0,
        "unexpected_tool": 0,
        "unexpected_control": 0,
        "unexpected_action": 0,
        "multi_decision_cases": 0,
        "attempts": {
            route: int(route in {"none", "web.search"})
            for route in route_gate.EXPECTED_ROUTES
        },
        "hits": {
            route: int(route in {"none", "web.search"})
            for route in route_gate.EXPECTED_ROUTES
        },
    }
    values.update(updates)
    return route_gate.ToolRouteGateTotals(**values)


def test_totals_detach_closed_maps_and_no_regression_is_monotonic():
    attempts = {
        route: int(route in {"none", "web.search"})
        for route in route_gate.EXPECTED_ROUTES
    }
    baseline = _valid_totals(
        exact_cases=1,
        misses=1,
        attempts=attempts,
        hits={route: int(route == "none") for route in route_gate.EXPECTED_ROUTES},
    )
    attempts["none"] = 99
    candidate = _valid_totals()

    assert baseline.attempts["none"] == 1
    assert route_gate.no_regression(baseline, candidate)
    assert candidate.complete


@pytest.mark.parametrize(
    "updates",
    [
        {"decisions": True},
        {"attempts": {route: 0 for route in route_gate.EXPECTED_ROUTES}},
        {"hits": {**{route: 0 for route in route_gate.EXPECTED_ROUTES}, "none": True}},
        {"attempts": {"SENTINEL_PRIVATE_KEY": 2}},
    ],
)
def test_totals_reject_malformed_contract_detail_free(updates):
    with pytest.raises(route_gate.ToolRouteGateError) as caught:
        _valid_totals(**updates)

    assert "SENTINEL_PRIVATE_KEY" not in str(caught.value)
    assert "SENTINEL_PRIVATE_KEY" not in repr(caught.value)


def test_profile_digest_is_canonical_and_alias_private():
    canary = "sentinelprivatealias"
    first = route_gate.ToolRouteGateProfile(True, True, (canary, "notes"))
    second = route_gate.ToolRouteGateProfile(True, True, ("NOTES", canary.upper()))

    assert route_gate.profile_digest(first) == route_gate.profile_digest(second)
    assert canary not in repr(first)
    assert canary not in route_gate.profile_digest(first)
    assert route_gate.profile_digest(first) != route_gate.profile_digest(
        route_gate.ToolRouteGateProfile(False, True, (canary, "notes"))
    )
    assert route_gate.profile_digest(first) != route_gate.profile_digest(
        route_gate.ToolRouteGateProfile(True, True, (canary, "other"))
    )


def _recorded_totals(rows, *, attested=True):
    measured = stt_eval._measure(rows)
    return stt_eval.EvaluationTotals(
        measured,
        measured,
        measured,
        len(rows),
        len(rows),
        {"decoded": len(rows)},
        selected_sources={"streaming": len(rows)} if attested else {},
        selected_sources_attested=attested,
    )


@dataclass(frozen=True)
class _Config:
    value: int = 1


def _install_cli_fakes(monkeypatch, items, evaluate):
    loaded = stt_eval._CorpusLoad(tuple(items), "c" * 64, object())
    monkeypatch.setattr(stt_eval, "_load_corpus", lambda _path: loaded)
    monkeypatch.setattr(stt_eval, "_verify_loaded_corpus", lambda _loaded: None)
    monkeypatch.setattr(stt_eval, "_load_config", _Config)
    monkeypatch.setattr(stt_eval, "_evaluate", evaluate)
    monkeypatch.setattr(stt_eval, "_config_digest", lambda _config: "d" * 64)
    monkeypatch.setattr(stt_eval, "_model_digest", lambda _config: "e" * 64)


def test_cli_allows_route_only_candidate_correction(capsys, monkeypatch):
    items = (
        stt_eval._CorpusItem(
            "alpha", object(), 16_000, None, ("expected-tool.vault.search",)
        ),
        stt_eval._CorpusItem("beta", object(), 16_000, None, ("expected-tool.none",)),
    )
    totals = _recorded_totals((("alpha", "alpha"), ("beta", "beta")))

    def evaluate(config, corpus, _keywords, *, route_gate=None):
        assert route_gate is not None
        selected = (
            ("search in my vault for alpha", "hello there")
            if config.value == 2
            else ("hello there", "hello there")
        )
        for item, text in zip(corpus, selected):
            route_gate.add_case(item.tags, (text,))
        return totals, ((0, 0), (0, 0))

    _install_cli_fakes(monkeypatch, items, evaluate)
    assert (
        stt_eval.main(
            ["--tool-route-gate", "--tool-route-vault-enabled", "--set", "value=2"]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["baseline"]["tool_route_gate"]["complete"] is False
    assert payload["candidate"]["tool_route_gate"]["complete"] is True
    assert payload["comparison"]["promotable"] is True


def test_cli_baseline_green_without_any_improvement_stays_rejected(capsys, monkeypatch):
    items = (
        stt_eval._CorpusItem(
            "alpha", object(), 16_000, None, ("expected-tool.web.search",)
        ),
        stt_eval._CorpusItem("beta", object(), 16_000, None, ("expected-tool.none",)),
    )
    totals = _recorded_totals((("alpha", "alpha"), ("beta", "beta")))

    def evaluate(_config, corpus, _keywords, *, route_gate=None):
        for item, text in zip(corpus, ("search the web for alpha", "hello there")):
            route_gate.add_case(item.tags, (text,))
        return totals, ((0, 0), (0, 0))

    _install_cli_fakes(monkeypatch, items, evaluate)
    assert stt_eval.main(["--tool-route-gate", "--set", "value=2"]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["baseline"]["tool_route_gate"]["complete"] is True
    assert payload["candidate"]["tool_route_gate"]["complete"] is True
    assert payload["comparison"]["promotable"] is False


def test_cli_candidate_route_red_blocks_accuracy_win(capsys, monkeypatch):
    items = (
        stt_eval._CorpusItem(
            "alpha beta", object(), 16_000, None, ("expected-tool.web.search",)
        ),
        stt_eval._CorpusItem("hello", object(), 16_000, None, ("expected-tool.none",)),
    )
    baseline = _recorded_totals((("alpha beta", "alpha wrong"), ("hello", "hello")))
    candidate = _recorded_totals((("alpha beta", "alpha beta"), ("hello", "hello")))

    def evaluate(config, corpus, _keywords, *, route_gate=None):
        for item in corpus:
            route_gate.add_case(item.tags, ("hello there",))
        if config.value == 2:
            return candidate, ((0, 0), (0, 0))
        return baseline, ((1, 5), (0, 0))

    _install_cli_fakes(monkeypatch, items, evaluate)
    assert stt_eval.main(["--tool-route-gate", "--set", "value=2"]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate"]["selected"]["word_errors"] == 0
    assert payload["candidate"]["tool_route_gate"]["complete"] is False
    assert payload["comparison"]["promotable"] is False


def test_cli_route_fix_cannot_bypass_candidate_source_attestation(
    capsys,
    monkeypatch,
):
    items = (
        stt_eval._CorpusItem(
            "alpha",
            object(),
            16_000,
            None,
            ("expected-tool.vault.search",),
        ),
        stt_eval._CorpusItem(
            "beta",
            object(),
            16_000,
            None,
            ("expected-tool.none",),
        ),
    )
    baseline = _recorded_totals((("alpha", "alpha"), ("beta", "beta")))
    candidate = _recorded_totals(
        (("alpha", "alpha"), ("beta", "beta")),
        attested=False,
    )

    def evaluate(config, corpus, _keywords, *, route_gate=None):
        selected = (
            ("search in my vault for alpha", "hello there")
            if config.value == 2
            else ("hello there", "hello there")
        )
        for item, text in zip(corpus, selected):
            route_gate.add_case(item.tags, (text,))
        return (
            (candidate, ((0, 0), (0, 0)))
            if config.value == 2
            else (baseline, ((0, 0), (0, 0)))
        )

    _install_cli_fakes(monkeypatch, items, evaluate)
    assert (
        stt_eval.main(
            ["--tool-route-gate", "--tool-route-vault-enabled", "--set", "value=2"]
        )
        == 3
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate"]["tool_route_gate"]["complete"] is True
    assert payload["candidate"]["selected_source_accounting_complete"] is False
    assert payload["comparison"]["promotable"] is False


def test_route_improvement_cannot_bypass_accuracy_regression():
    baseline = _recorded_totals((("alpha beta", "alpha beta"),))
    candidate = _recorded_totals((("alpha beta", "wrong words"),))

    comparison = stt_eval.compare_candidate(
        baseline,
        candidate,
        ((0, 0),),
        ((2, 9),),
        additional_improvement=True,
    )

    assert comparison.promotable is False


def test_cli_preflights_annotations_before_config_or_decode(capsys, monkeypatch):
    item = stt_eval._CorpusItem(
        "private", object(), 16_000, None, ("expected-tool.unknown",)
    )
    loaded = stt_eval._CorpusLoad((item,), "c" * 64, object())
    called: list[str] = []
    monkeypatch.setattr(stt_eval, "_load_corpus", lambda _path: loaded)
    monkeypatch.setattr(stt_eval, "_load_config", lambda: called.append("config"))

    assert stt_eval.main(["--tool-route-gate"]) == 2
    assert called == []
    assert json.loads(capsys.readouterr().out) == stt_eval._SAFE_ERROR


def test_cli_rejects_gate_on_legacy_before_config(capsys, monkeypatch):
    item = stt_eval._CorpusItem("legacy", object(), 16_000, None)
    loaded = stt_eval._CorpusLoad((item,), "c" * 64, stt_eval._LEGACY_CORPUS_WITNESS)
    called: list[str] = []
    monkeypatch.setattr(stt_eval, "_load_corpus", lambda _path: loaded)
    monkeypatch.setattr(stt_eval, "_load_config", lambda: called.append("config"))

    assert stt_eval.main(["--tool-route-gate"]) == 2
    assert called == []
    assert json.loads(capsys.readouterr().out) == stt_eval._SAFE_ERROR


@pytest.mark.parametrize(
    "args",
    [
        ["--tool-route-vault-enabled"],
        ["--tool-route-reminders-enabled"],
        ["--tool-route-app-alias", "notes"],
    ],
)
def test_cli_rejects_profile_options_without_gate_before_config(
    args,
    capsys,
    monkeypatch,
):
    item = stt_eval._CorpusItem("legacy", object(), 16_000, None)
    loaded = stt_eval._CorpusLoad((item,), "c" * 64, stt_eval._LEGACY_CORPUS_WITNESS)
    called: list[str] = []
    monkeypatch.setattr(stt_eval, "_load_corpus", lambda _path: loaded)
    monkeypatch.setattr(stt_eval, "_load_config", lambda: called.append("config"))

    assert stt_eval.main(args) == 2
    assert called == []
    assert json.loads(capsys.readouterr().out) == stt_eval._SAFE_ERROR


def test_cli_invalid_private_alias_fails_without_disclosure(
    capsys,
    monkeypatch,
):
    canary = "SENTINEL PRIVATE ALIAS"
    item = stt_eval._CorpusItem(
        "private",
        object(),
        16_000,
        None,
        ("expected-tool.none",),
    )
    loaded = stt_eval._CorpusLoad((item,), "c" * 64, object())
    called: list[str] = []
    monkeypatch.setattr(stt_eval, "_load_corpus", lambda _path: loaded)
    monkeypatch.setattr(stt_eval, "_load_config", lambda: called.append("config"))

    assert stt_eval.main(["--tool-route-gate", "--tool-route-app-alias", canary]) == 2
    output = capsys.readouterr().out
    assert called == []
    assert canary not in output
    assert json.loads(output) == stt_eval._SAFE_ERROR


def test_cli_profile_digest_never_discloses_alias(capsys, monkeypatch, tmp_path):
    canary = "sentinelprivatealias"
    items = (
        stt_eval._CorpusItem(
            "alpha", object(), 16_000, None, ("expected-tool.app.open",)
        ),
        stt_eval._CorpusItem("beta", object(), 16_000, None, ("expected-tool.none",)),
    )
    totals = _recorded_totals((("alpha", "alpha"), ("beta", "beta")))

    def evaluate(_config, corpus, _keywords, *, route_gate=None):
        for item, text in zip(corpus, (f"open {canary}", "hello there")):
            route_gate.add_case(item.tags, (text,))
        return totals, ((0, 0), (0, 0))

    _install_cli_fakes(monkeypatch, items, evaluate)
    report = tmp_path / "report.json"
    assert (
        stt_eval.main(
            [
                "--tool-route-gate",
                "--tool-route-app-alias",
                canary,
                "--output",
                str(report),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert canary not in output
    assert canary not in report.read_text(encoding="utf-8")
    payload = json.loads(output)
    assert len(payload["tool_route_profile_digest"]) == 64
    assert payload["baseline"]["tool_route_gate"]["complete"] is True


def test_ungated_cli_uses_exact_legacy_three_argument_evaluate_path(
    capsys,
    monkeypatch,
):
    item = stt_eval._CorpusItem("alpha", object(), 16_000, None)
    totals = _recorded_totals((("alpha", "alpha"),))

    def evaluate(_config, _corpus, _keywords):
        return totals, ((0, 0),)

    loaded = stt_eval._CorpusLoad((item,), "c" * 64, stt_eval._LEGACY_CORPUS_WITNESS)
    monkeypatch.setattr(stt_eval, "_load_corpus", lambda _path: loaded)
    monkeypatch.setattr(stt_eval, "_load_config", _Config)
    monkeypatch.setattr(stt_eval, "_evaluate", evaluate)
    monkeypatch.setattr(stt_eval, "_config_digest", lambda _config: "d" * 64)
    monkeypatch.setattr(stt_eval, "_model_digest", lambda _config: "e" * 64)

    assert stt_eval.main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "tool_route_profile_digest" not in payload
    assert "tool_route_gate" not in payload["baseline"]
