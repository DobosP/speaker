"""Autonomous memory tier: does the assistant store a fact and *use* it later?

Drives the real ``assistant.answer`` capability (the same provider the runtime
uses) across a temporary SQLite close/reopen with recall enabled -- no Postgres,
no audio. Two levels of proof:

* **plumbing** -- the recall block injected into the model's ``system`` on the
  recalling turn contains the stored fact (keyword-overlap recall found it);
* **semantic** -- the model's actual answer contains the fact (it used the
  recalled context). Needs a real small LLM; ``--llm echo`` skips this level.

A fact is stated in turn 1, buried under unrelated distractor turns, then asked
for again -- the canonical "remembered across turns" check.
"""
from __future__ import annotations

from hashlib import sha256
import json
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence
from urllib.parse import urlparse

from always_on_agent.capabilities import CapabilityRegistry
from always_on_agent.sqlite_memory import SqliteVecMemory
from always_on_agent.untrusted import fenced_data_contains
from core.capabilities import RecallConfig, attach_llm_capabilities
from core.config import apply_device_profile, load_config
from core.conversation import RecentContextConfig
from core.llm import EchoLLM, OllamaLLM
from tools.conversation_eval.provenance import (
    identity_bundle,
    identity_snapshot,
    json_sha256,
    repository_metadata,
    validate_identity_bundle,
)
from tools.conversation_eval.runner import LOCAL_OLLAMA_HEADERS, LOCAL_OLLAMA_HOST
from tools.setup_minicpm import LOCAL_MODEL
from .verdicts import DIAGNOSTIC_PASS, FAIL, PASS


_REPO = Path(__file__).resolve().parents[2]
_PROBE_CONTRACT_SCHEMA = 1
_DEFAULT_DEVICE = "desktop_gpu_4090"
_SHIPPED_MAIN_MODEL = "gemma3:12b"


def _uses_recalled_fact(answer: str, keyword: str) -> bool:
    """Return whether *answer* confidently attributes *keyword* to the user.

    A bare keyword check can turn a denial (``"I don't know; maybe teal"``) or
    an assistant-persona claim (``"My favorite color is teal"``) into a false
    green.  This deliberately small semantic grader accepts concise scalar and
    user-attributed answers, while failing closed on common denial, hedging,
    negation, and first-person self-attribution forms.
    """
    text = re.sub(r"\s+", " ", (answer or "").replace("’", "'")).strip()
    wanted = re.sub(r"\s+", " ", (keyword or "")).strip()
    if not text or not wanted:
        return False

    term = r"\s+".join(re.escape(part) for part in wanted.split())
    bounded_term = rf"(?<!\w){term}(?!\w)"
    if re.search(bounded_term, text, flags=re.IGNORECASE) is None:
        return False

    # Knowledge denials and hedges are not evidence of successful recall even
    # when they repeat the expected value as a possibility.
    uncertainty_patterns = (
        r"\b(?:maybe|perhaps|possibly|probably|presumably)\b",
        r"\b(?:may|might|could)\s+(?:well\s+)?be\b",
        r"\b(?:i\s+)?(?:do(?:n't| not)|can(?:not|'t))\s+"
        r"(?:know|recall|remember|confirm|determine|say|tell|answer|verify)\b",
        r"\b(?:i\s+)?can(?:not|'t)\s+be\s+(?:sure|certain)\b",
        r"\b(?:i(?:'m| am)?\s+)?(?:not sure|unsure)\b",
        r"\b(?:i\s+have\s+)?no\s+(?:idea|information|memory|record|way\s+to\s+know)\b",
        r"\b(?:i|we)\s+(?:do(?:n't| not)|can(?:not|'t))\s+"
        r"(?:have|possess)\b[^.!?;\n]{0,80}\b(?:favorite|favourite|preference|"
        r"taste|opinion)s?\b",
        r"\bno\s+(?:personal\s+)?(?:favorite|favourite|preference|taste|opinion)s?\b",
        r"\b(?:as\s+an?\s+|i(?:'m| am)\s+an?\s+)(?:ai|assistant|language\s+model)\b",
        r"\b(?:unknown|unclear|uncertain)\b",
        r"\b(?:i\s+)?(?:think|guess|suspect|doubt)\b",
        r"\b(?:i\s+)?(?:would|have)\s+to\s+guess\b",
        r"\b(?:seems|appears)\s+(?:to\s+be\s+)?",
    )
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in uncertainty_patterns):
        return False

    # Reject direct negation of the expected value, without rejecting useful
    # contrasts such as "not blue -- teal".
    negated_term_patterns = (
        rf"\b(?:isn't|isnt|wasn't|wasnt|aren't|arent|weren't|werent|not|never)\s+"
        rf"(?:actually\s+)?{bounded_term}",
        rf"{bounded_term}\s+(?:(?:is|was|are|were)\s+not|isn't|isnt|wasn't|wasnt|"
        rf"aren't|arent|weren't|werent)\b",
        rf"{bounded_term}\s*\?\s*no\b",
    )
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in negated_term_patterns):
        return False

    # Explicit second-person attribution wins over incidental first-person
    # framing (for example, "I remember that your favorite color is teal").
    user_attribution_patterns = (
        rf"\b(?:your|yours)\b[^.!?;\n]{{0,100}}{bounded_term}",
        rf"\byou\s+(?:said|mentioned|stated|told\s+me|prefer|preferred|like|liked|"
        rf"love|loved|chose|picked|selected)\b[^.!?;\n]{{0,100}}{bounded_term}",
        rf"{bounded_term}[^.!?;\n]{{0,80}}\b(?:is\s+yours|was\s+yours|as\s+you\s+"
        rf"(?:said|mentioned|told\s+me))\b",
    )
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in user_attribution_patterns):
        return True

    # Do not credit the model for assigning the remembered fact to itself.
    assistant_attribution_patterns = (
        rf"\bmy\s+[^.!?;\n]{{0,80}}(?:\b(?:is|was|are|were)\b|'s|:)\s+"
        rf"(?:definitely\s+|certainly\s+)?{bounded_term}",
        rf"\bmy\s+[^.!?;\n]{{0,80}}\?\s*{bounded_term}",
        rf"{bounded_term}[^.!?;\n]{{0,60}}\b(?:is|was)\s+(?:one\s+of\s+)?my\b",
        rf"\bmine\b[^.!?;\n]{{0,60}}{bounded_term}",
        rf"{bounded_term}[^.!?;\n]{{0,60}}\b(?:is|was)\s+mine\b",
        rf"\bi\s+(?:personally\s+)?(?:(?:would\s+)?(?:prefer|like|love|choose)|"
        rf"chose|picked|selected|"
        rf"said|mentioned|stated)\b[^.!?;\n]{{0,80}}{bounded_term}",
        rf"{bounded_term}[^.!?;\n]{{0,60}}\bi\s+(?:prefer|like|love|chose|choose|"
        rf"picked|selected)\b",
        rf"\b(?:for\s+me|personally)\b[^.!?;\n]{{0,60}}{bounded_term}",
    )
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in assistant_attribution_patterns):
        return False

    # Scalar ("Teal!"), copular ("It's teal"), and other neutral factual
    # phrasings are valid once the false-positive forms above are excluded.
    return True


def _uses_recalled_canary(
    answer: str,
    keyword: str,
    subject_terms: Sequence[str],
) -> bool:
    """Require one affirmative clause to bind the canary to its subject.

    Keyword and subject terms in different clauses are not grounding.  The
    probe also abstains on corrections, contrasts, alternatives, and competing
    subject assignments instead of trying to infer which clause should win.
    """
    text = re.sub(r"\s+", " ", (answer or "").replace("’", "'")).strip()
    wanted = re.sub(r"\s+", " ", (keyword or "")).strip()
    if not text or not wanted or not _uses_recalled_fact(text, wanted):
        return False

    # Explicit truth-value rejection can follow an otherwise affirmative-looking
    # assignment in the same punctuation clause.  Do not let the expected
    # token inside the rejected proposition mint evidence.
    if re.search(
        r"\b(?:false|incorrect|untrue|wrong)\b|\bnot\s+(?:really|true|correct)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return False

    normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized_wanted = re.sub(r"[^a-z0-9\s]", " ", wanted.lower())
    normalized_wanted = re.sub(r"\s+", " ", normalized_wanted).strip()
    # A concise scalar is unambiguous because the prompt asks for exactly one
    # synthetic value.
    if normalized in {
        normalized_wanted,
        f"it is {normalized_wanted}",
        f"it was {normalized_wanted}",
    }:
        return True

    # Corrections/contrasts/alternatives can contain both the expected token
    # and a competing answer.  Treat the complete answer as ambiguous.
    if re.search(
        r"(?:\b(?:but|however|although|though|yet|instead|rather|actually|"
        r"correction|except|or)\b|(?:^|[;,.!?])\s*no\s*[,;:—-])",
        text,
        flags=re.IGNORECASE,
    ):
        return False

    terms = [
        re.sub(r"\s+", " ", term.lower()).strip()
        for term in subject_terms
        if term.strip()
    ]
    if not terms:
        return False

    clauses = tuple(
        clause.strip(" ,:—-")
        for clause in re.split(
            r"(?:[.!?;\n]+|\s+[—–-]\s+|\s*,?\s+and\s+)",
            text,
            flags=re.IGNORECASE,
        )
        if clause.strip(" ,:—-")
    )
    keyword_pattern = r"(?<!\w)" + r"\s+".join(
        re.escape(part) for part in wanted.split()
    ) + r"(?!\w)"
    grounded: list[str] = []
    for clause in clauses:
        clause_normalized = re.sub(r"[^a-z0-9\s]", " ", clause.lower())
        clause_normalized = re.sub(r"\s+", " ", clause_normalized).strip()
        has_keyword = re.search(keyword_pattern, clause, re.IGNORECASE) is not None
        has_subject = all(
            re.search(rf"\b{re.escape(term)}\b", clause_normalized) is not None
            for term in terms
        )
        if has_keyword and has_subject:
            grounded.append(clause)
        elif has_subject and re.search(
            r"\b(?:is|was|are|were|called|named|codename)\b",
            clause_normalized,
        ):
            # A second assignment for the same subject competes with the
            # expected canary even if punctuation separated the values.
            return False
    if len(grounded) != 1:
        return False
    clause = grounded[0]
    if not _uses_recalled_fact(clause, wanted):
        return False
    keyword_match = re.search(keyword_pattern, clause, re.IGNORECASE)
    term_matches = [
        re.search(
            r"\b" + r"\s+".join(
                re.escape(part) for part in term.split()
            ) + r"\b",
            clause,
            re.IGNORECASE,
        )
        for term in terms
    ]
    if keyword_match is None or any(match is None for match in term_matches):
        return False
    concrete_terms = [match for match in term_matches if match is not None]
    if all(match.end() <= keyword_match.start() for match in concrete_terms):
        bridge = clause[max(match.end() for match in concrete_terms):keyword_match.start()]
        related = re.fullmatch(
            r"\s*(?:(?:is|was|are|were|called|named)\s*|:)\s*",
            bridge,
            re.IGNORECASE,
        ) is not None
    elif all(match.start() >= keyword_match.end() for match in concrete_terms):
        bridge = clause[keyword_match.end():min(match.start() for match in concrete_terms)]
        related = re.fullmatch(
            r"\s*(?:is|was|are|were)\s+"
            r"(?:(?:the|your|my|this|that)\s+)?(?:lighthouse\s+)?",
            bridge,
            re.IGNORECASE,
        ) is not None
    else:
        related = False
    if not related:
        return False
    # Reject assigning the canary to the assistant or a different subject.
    if re.search(
        r"\b(?:assistant(?:'s|s)?|another|other|their|his|her)\b"
        r"[^.!?;\n]{0,80}\bproject\b",
        clause.lower(),
    ):
        return False
    return True


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _safe_loopback_host(value: object) -> tuple[str, bool]:
    raw = str(value or LOCAL_OLLAMA_HOST).strip()
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    hostname = (parsed.hostname or "").lower()
    safe = bool(
        parsed.scheme == "http"
        and hostname in {"localhost", "127.0.0.1", "::1"}
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
        and parsed.path in {"", "/"}
    )
    if not safe:
        return LOCAL_OLLAMA_HOST, False
    try:
        port = parsed.port or 11434
    except ValueError:
        return LOCAL_OLLAMA_HOST, False
    display = f"[{hostname}]" if ":" in hostname else hostname
    return f"http://{display}:{port}", True


@dataclass(frozen=True)
class _ReleaseContract:
    identity_config: dict
    configured_roles: dict[str, str]
    host: str
    host_safe: bool
    shipped_roles: bool
    contract_sha256: str


def _release_contract(
    *,
    device: str,
    role_models: dict[str, str],
    fact: str,
    keyword: str,
    question: str,
    answer_subject_terms: Sequence[str],
    distractors: Sequence[str],
) -> _ReleaseContract:
    """Build the local, content-hashed contract for one real memory probe."""
    try:
        base = load_config(
            str(_REPO / "config.json"),
            local=str(_REPO / "config.local.json"),
        )
        resolved = apply_device_profile(base, device, strict=True)
        llm = resolved.get("llm", {}) or {}
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        llm = {}
    host, host_safe = _safe_loopback_host(llm.get("host"))
    configured_roles = {
        "main": str(llm.get("main_model", "")),
        "fast": str(llm.get("fast_model", "")),
    }
    backend_ok = str(llm.get("backend", "ollama") or "ollama").lower() == "ollama"
    shipped_roles = bool(
        backend_ok
        and role_models == configured_roles
        and role_models.get("fast") == LOCAL_MODEL
        and role_models.get("main") == _SHIPPED_MAIN_MODEL
    )
    contract = {
        "schema": _PROBE_CONTRACT_SCHEMA,
        "device": device,
        "host": host,
        "role_models": role_models,
        "configured_role_models": configured_roles,
        "transport": {
            "loopback_only": host_safe,
            "ambient_credentials": False,
        },
        "llm": {"think": False, "keep_alive": "5m", "timeout_sec": 120.0},
        "memory": {
            "backend": "sqlite",
            "recall_enabled": True,
            "recall_max_chars": 600,
            "write_recent_context": False,
            "read_recent_context": True,
            "read_recent_as_messages": True,
        },
        "inputs": {
            "fact_sha256": _text_sha256(fact),
            "keyword_sha256": _text_sha256(keyword),
            "question_sha256": _text_sha256(question),
            "subject_terms_sha256": json_sha256(list(answer_subject_terms)),
            "distractors_sha256": json_sha256(list(distractors)),
        },
    }
    identity_config = {"llm": {"host": host}}
    return _ReleaseContract(
        identity_config=identity_config,
        configured_roles=configured_roles,
        host=host,
        host_safe=host_safe,
        shipped_roles=shipped_roles,
        contract_sha256=json_sha256(contract),
    )


def _capture_model_identity(
    role_models: dict[str, str], identity_config: dict
) -> dict[str, object]:
    """Capture model evidence; tests replace this private I/O boundary."""
    try:
        value = identity_snapshot(role_models, identity_config)
        if isinstance(value, dict):
            return value
    except Exception as exc:  # noqa: BLE001 - evidence failure makes the gate red
        return {"error": type(exc).__name__}
    return {"error": "invalid identity snapshot"}


@dataclass(frozen=True)
class MemoryProvenance:
    applicable: bool
    ok: Optional[bool]
    contract_schema: int
    contract_sha256: str
    contract_stable: bool
    host: str
    host_safe: bool
    role_models: dict[str, str]
    configured_role_models: dict[str, str]
    shipped_roles: bool
    repository_before: dict[str, object]
    repository_after: dict[str, object]
    repository_stable: bool
    identity: dict[str, object]
    model_identity_stable: bool


def _diagnostic_provenance() -> MemoryProvenance:
    return MemoryProvenance(
        applicable=False,
        ok=None,
        contract_schema=_PROBE_CONTRACT_SCHEMA,
        contract_sha256="",
        contract_stable=False,
        host="",
        host_safe=True,
        role_models={},
        configured_role_models={},
        shipped_roles=False,
        repository_before={},
        repository_after={},
        repository_stable=False,
        identity={},
        model_identity_stable=False,
    )


def _build_release_provenance(
    *,
    role_models: dict[str, str],
    contract_before: _ReleaseContract,
    contract_after: _ReleaseContract,
    repository_before: dict[str, object],
    repository_after: dict[str, object],
    identity_before: dict[str, object],
    identity_after: dict[str, object],
) -> MemoryProvenance:
    if not isinstance(repository_before, dict):
        repository_before = {}
    if not isinstance(repository_after, dict):
        repository_after = {}
    revision = str(repository_before.get("revision", "") or "")
    repository_stable = bool(
        re.fullmatch(r"[0-9a-f]{40}", revision, re.IGNORECASE)
        and repository_before.get("dirty") is False
        and repository_after == repository_before
    )
    if not isinstance(identity_before, dict):
        identity_before = {}
    if not isinstance(identity_after, dict):
        identity_after = {}
    identity = identity_bundle(identity_before, identity_after)
    model_identity_stable, _contracts = validate_identity_bundle(
        role_models,
        identity,
    )
    contract_stable = contract_before == contract_after
    shipped_roles = bool(
        contract_stable
        and contract_before.shipped_roles
        and contract_before.host_safe
        and role_models == contract_before.configured_roles
        and role_models.get("fast") == LOCAL_MODEL
        and role_models.get("main") == _SHIPPED_MAIN_MODEL
    )
    contract_valid = bool(
        contract_stable
        and re.fullmatch(
            r"[0-9a-f]{64}",
            contract_before.contract_sha256,
            re.IGNORECASE,
        )
    )
    ok = bool(
        contract_before.host_safe
        and shipped_roles
        and contract_valid
        and repository_stable
        and model_identity_stable
    )
    return MemoryProvenance(
        applicable=True,
        ok=ok,
        contract_schema=_PROBE_CONTRACT_SCHEMA,
        contract_sha256=contract_before.contract_sha256,
        contract_stable=contract_stable,
        host=contract_before.host,
        host_safe=contract_before.host_safe,
        role_models=dict(role_models),
        configured_role_models=dict(contract_before.configured_roles),
        shipped_roles=shipped_roles,
        repository_before=repository_before,
        repository_after=repository_after,
        repository_stable=repository_stable,
        identity=identity,
        model_identity_stable=model_identity_stable,
    )


class _CapturingLLM:
    """Wraps a real LLM, recording the ``system`` prompt + output of each call
    so the probe can assert on both the injected recall block and the answer."""

    def __init__(self, inner, *, role: str, attempts: list[str]):
        self.inner = inner
        self.role = role
        self.attempts = attempts
        self.systems: list[str] = []
        self.outputs: list[str] = []
        self.histories: list[list[dict]] = []

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images=None,
        history=None,
    ) -> str:
        self.attempts.append(self.role)
        self.systems.append(system or "")
        self.histories.append(list(history or []))
        out = self.inner.generate(
            prompt, system=system, images=images, history=history
        )
        self.outputs.append(out)
        return out

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images=None,
        history=None,
    ) -> Iterator[str]:
        self.attempts.append(self.role)
        self.systems.append(system or "")
        self.histories.append(list(history or []))
        chunks: list[str] = []
        for tok in self.inner.stream(
            prompt, system=system, images=images, history=history
        ):
            chunks.append(tok)
            yield tok
        self.outputs.append("".join(chunks))


@dataclass
class MemoryResult:
    ok: bool
    plumbing_ok: bool
    complete: bool
    outcome: str
    fact: str
    keyword: str
    recall_available: bool         # plumbing: selector returned the fact
    recall_injected: bool          # model path only: fact appeared in its prompt
    recall_fenced: bool            # persisted fact stayed inside untrusted-data boundary
    answer_uses_fact: Optional[bool]  # semantic: answer grounds the fact (None for echo)
    answer: str
    answer_model: str
    answer_route: str
    answer_sensitivity: str
    attempt_order: tuple[str, ...]
    main_selected_first: bool
    topology_valid: bool
    recent_history_clean: bool
    controller_answer: bool
    backend_reopened_with_fresh_registry: bool
    provenance_ok: Optional[bool]
    provenance: MemoryProvenance
    recall_block_preview: str
    llm_label: str
    turns: int
    duration_sec: float
    detail: list[str] = field(default_factory=list)


def run_memory_probe(
    *,
    llm_kind: str = "ollama",
    model: str = "minicpm5-1b:q8",
    main_model: Optional[str] = "gemma3:12b",
    fact: str = "my lighthouse project codename was Amber Finch",
    keyword: str = "Amber Finch",
    question: str = "what was my lighthouse project codename?",
    answer_subject_terms: Sequence[str] = ("lighthouse", "project", "codename"),
    distractors: Sequence[str] = (
        "what time is it",
        "tell me a fun fact about the ocean",
        "how many days are in a week",
    ),
    device: str = _DEFAULT_DEVICE,
) -> MemoryResult:
    """Run a fact -> restart -> fenced strong-recall flow and return a verdict."""
    t0 = time.monotonic()
    detail: list[str] = []
    attempts: list[str] = []
    role_models: dict[str, str] = {}
    host = ""
    repository_before: dict[str, object] = {}
    repository_after: dict[str, object] = {}
    identity_before: dict[str, object] = {}
    identity_after: dict[str, object] = {}
    contract_before: _ReleaseContract | None = None
    contract_after: _ReleaseContract | None = None

    if llm_kind == "echo":
        label = "echo"
        fast_llm = _CapturingLLM(
            EchoLLM(), role="fast", attempts=attempts
        )
        main_llm = _CapturingLLM(
            EchoLLM(), role="main", attempts=attempts
        )
        topology_valid = True  # deterministic plumbing mode has no model roles
    else:
        resolved_main = main_model or model
        role_models = {"main": resolved_main, "fast": model}
        # The first repository snapshot precedes every config/model read.  The
        # exact effective contract is recomputed before the final repository
        # snapshot so config/Git drift makes provenance red.
        repository_before = repository_metadata()
        contract_before = _release_contract(
            device=device,
            role_models=role_models,
            fact=fact,
            keyword=keyword,
            question=question,
            answer_subject_terms=answer_subject_terms,
            distractors=distractors,
        )
        host = contract_before.host
        identity_before = _capture_model_identity(
            role_models,
            contract_before.identity_config,
        )
        fast_llm = _CapturingLLM(
            OllamaLLM(
                model=model,
                host=host,
                keep_alive="5m",
                timeout=120.0,
                think=False,
                client_headers=LOCAL_OLLAMA_HEADERS,
            ),
            role="fast",
            attempts=attempts,
        )
        main_llm = (
            fast_llm
            if resolved_main == model
            else _CapturingLLM(
                OllamaLLM(
                    model=resolved_main,
                    host=host,
                    keep_alive="5m",
                    timeout=120.0,
                    think=False,
                    client_headers=LOCAL_OLLAMA_HEADERS,
                ),
                role="main",
                attempts=attempts,
            )
        )
        label = f"ollama/fast={model},main={resolved_main}"
        topology_valid = main_llm is not fast_llm

    with tempfile.TemporaryDirectory(prefix="speaker-memory-probe-") as tmp:
        path = str(Path(tmp) / "memory.db")
        memory = SqliteVecMemory(path)
        registry = CapabilityRegistry()
        attach_llm_capabilities(
            registry,
            main_llm,
            fast_llm=fast_llm,
            memory=memory,
            recall=RecallConfig(enabled=True, max_chars=600),
            recent_context=RecentContextConfig(enabled=False),
        )

        # Process/session 1: establish and bury the fact through the real
        # assistant capability, then close the backend to erase every controller
        # cache and trusted recent-history ring.
        try:
            registry.invoke("assistant.answer", fact, {})
            detail.append(f"turn 1 (state fact): {fact!r}")
            for i, d in enumerate(distractors, start=2):
                registry.invoke("assistant.answer", d, {})
                detail.append(f"turn {i} (distractor): {d!r}")
        finally:
            memory.close()

        # Process/session 2: a fresh backend + registry may see old rows only
        # through persistent recall, never native recent history or controller
        # state.  Inspect only this final model call.
        for captured in (fast_llm, main_llm):
            captured.systems.clear()
            captured.outputs.clear()
            captured.histories.clear()
        attempts.clear()
        memory = SqliteVecMemory(path)
        registry = CapabilityRegistry()
        attach_llm_capabilities(
            registry,
            main_llm,
            fast_llm=fast_llm,
            memory=memory,
            recall=RecallConfig(enabled=True, max_chars=600),
            recent_context=RecentContextConfig(enabled=True, as_messages=True),
        )
        try:
            reopened_recent_empty = memory.all() == []
            retrieval_block = memory.context_for_llm(question)
            result = registry.invoke("assistant.answer", question, {})
        finally:
            memory.close()
    if llm_kind != "echo":
        assert contract_before is not None
        identity_after = _capture_model_identity(
            role_models,
            contract_before.identity_config,
        )
        contract_after = _release_contract(
            device=device,
            role_models=role_models,
            fact=fact,
            keyword=keyword,
            question=question,
            answer_subject_terms=answer_subject_terms,
            distractors=distractors,
        )
        repository_after = repository_metadata()
    n_turns = 1 + len(distractors) + 1
    detail.append(f"turn {n_turns} (recall): {question!r}")

    controller_answer = bool(result.data.get("recalled_self_fact"))
    answered_by = main_llm if main_llm.outputs else fast_llm
    system_used = "" if controller_answer else (
        answered_by.systems[-1] if answered_by.systems else ""
    )
    answer = result.text or (answered_by.outputs[-1] if answered_by.outputs else "")
    answer_route = str(result.data.get("route", ""))
    answer_sensitivity = str(result.data.get("sensitivity", "") or "")
    attempt_order = tuple(attempts)
    main_selected_first = attempt_order == ("main",)
    final_history = answered_by.histories[-1] if answered_by.histories else []
    if controller_answer:
        answer_model = "control"
    elif llm_kind == "echo":
        answer_model = "echo"
    else:
        answer_model = (main_model or model) if answered_by is main_llm else model
    recall_available = keyword.lower() in retrieval_block.lower()
    recall_injected = keyword.lower() in system_used.lower()
    recall_fenced = recall_injected and fenced_data_contains(system_used, keyword)
    recent_history_clean = reopened_recent_empty and not final_history
    backend_reopened_with_fresh_registry = (
        recall_available and recent_history_clean
    )

    if controller_answer:
        answer_uses_fact = _uses_recalled_fact(answer or "", keyword)
    elif llm_kind == "echo":
        answer_uses_fact: Optional[bool] = None  # echo can't reason; plumbing only
    else:
        answer_uses_fact = _uses_recalled_canary(
            answer or "", keyword, answer_subject_terms
        )
    provenance = (
        _diagnostic_provenance()
        if llm_kind == "echo"
        else _build_release_provenance(
            role_models=role_models,
            contract_before=contract_before,
            contract_after=contract_after,
            repository_before=repository_before,
            repository_after=repository_after,
            identity_before=identity_before,
            identity_after=identity_after,
        )
    )
    provenance_ok = provenance.ok
    plumbing_ok = (
        recall_available
        and recall_injected
        and recall_fenced
        and answer_route == "main"
        and answer_sensitivity == "private"
        and main_selected_first
        and not controller_answer
        and recent_history_clean
        and backend_reopened_with_fresh_registry
    )
    complete = llm_kind != "echo"
    if (
        not plumbing_ok
        or (complete and not topology_valid)
        or (complete and provenance_ok is not True)
    ):
        outcome = FAIL
    elif not complete:
        outcome = DIAGNOSTIC_PASS
    else:
        outcome = PASS if bool(answer_uses_fact) else FAIL
    ok = outcome == PASS

    # a compact preview of where the recall block sits in the system prompt
    preview_source = system_used or retrieval_block
    lo = preview_source.lower()
    pos = lo.find(keyword.lower())
    preview = (
        preview_source[max(0, pos - 40): pos + 60]
        if pos >= 0 else preview_source[:120]
    ).strip()

    return MemoryResult(
        ok=ok,
        plumbing_ok=plumbing_ok,
        complete=complete,
        outcome=outcome,
        fact=fact,
        keyword=keyword,
        recall_available=recall_available,
        recall_injected=recall_injected,
        recall_fenced=recall_fenced,
        answer_uses_fact=answer_uses_fact,
        answer=answer.strip(),
        answer_model=answer_model,
        answer_route=answer_route,
        answer_sensitivity=answer_sensitivity,
        attempt_order=attempt_order,
        main_selected_first=main_selected_first,
        topology_valid=topology_valid,
        recent_history_clean=recent_history_clean,
        controller_answer=controller_answer,
        backend_reopened_with_fresh_registry=(
            backend_reopened_with_fresh_registry
        ),
        provenance_ok=provenance_ok,
        provenance=provenance,
        recall_block_preview=preview,
        llm_label=label,
        turns=n_turns,
        duration_sec=round(time.monotonic() - t0, 2),
        detail=detail,
    )
