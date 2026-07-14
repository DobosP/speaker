from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import time

from .events import Mode
from .models import IntentDecision, IntentKind, SpeechObservation
from .text import detect_language, keywords, normalize_text


_STOP_PHRASES = {
    "stop",
    "cancel",
    "cancel that",
    "quiet",
    "stop talking",
    "stop speaking",
    "never mind",
    "be quiet",
    "opreste",
    "anuleaza",
}
_CONFIRM_PHRASES = {"yes", "confirm", "approve", "do it", "ok do it", "da", "confirma"}
_DENY_PHRASES = {"no", "deny", "cancel command", "do not", "dont", "nu", "anuleaza comanda"}
_WAKE_TERMS = {"assistant", "computer", "jarvis", "asistent"}
_MODE_ALIASES = {
    "passive mode": Mode.PASSIVE,
    "assistant mode": Mode.ASSISTANT,
    "command mode": Mode.COMMAND,
    "search mode": Mode.SEARCH,
    "research mode": Mode.RESEARCH,
    "dictation mode": Mode.DICTATION,
    "meeting mode": Mode.MEETING,
    "mod pasiv": Mode.PASSIVE,
    "mod asistent": Mode.ASSISTANT,
    "mod cautare": Mode.SEARCH,
    "mod cercetare": Mode.RESEARCH,
    "mod dictare": Mode.DICTATION,
}

_EXPLICIT_NON_ASSISTANT_PREFIXES = (
    "research ", "cerceteaza ",
    "search ", "cauta ",
    "dictate ", "scrie ",
    "run ", "open ", "execute ",
)

_VAULT_SCOPE_MARKERS = (
    "my notes",
    "my vault",
    "my obsidian",
    "my second brain",
    "dobo brain",
    "dobo-brain",
    "paul brain",
)


def is_vault_scoped_request(text: str) -> bool:
    """Whether an explicit search names the user's configured notes vault.

    This stays metadata on the existing SEARCH intent rather than adding a new
    mode/enum to the cross-platform brain contract.
    """

    normalized = normalize_text(text)
    vault_positions = [
        normalized.find(marker)
        for marker in _VAULT_SCOPE_MARKERS
        if marker in normalized
    ]
    if not vault_positions:
        return False
    first_vault = min(vault_positions)
    if (
        normalized.startswith(("web ", "the web ", "online "))
        or " on the web" in normalized
        or " from the web" in normalized
        or normalized.endswith(" online")
    ):
        return False
    # An embedded web-search phrase is a source override only when it appears
    # before the private-vault marker ("search the web for my notes"). Text in
    # the requested note topic ("my notes about search the web design") is not.
    if any(
        0 <= normalized.find(marker) < first_vault
        for marker in ("search the web", "web search", "search online")
    ):
        return False
    return True


def _is_vault_lookup_request(text: str) -> bool:
    """Recognize a clear private-vault scope without an LLM tool choice."""

    return is_vault_scoped_request(text)


def is_assistant_mode_final_candidate(
    text: str,
    current_mode: Mode,
    *,
    has_pending_confirmation: bool = False,
    vault_search_enabled: bool = False,
) -> bool:
    """Whether assistant-mode default analysis can resolve ``text`` as ASSISTANT.

    This is a bounded, stateless preview for runtime handoff bookkeeping before
    the supervisor has parsed a queued final. It is deliberately scoped to an
    already-active ASSISTANT mode and recognizes only decisive built-in
    exclusions; passive wake activation and custom analyzers are outside this
    contract.
    """
    normalized = normalize_text(text)
    if current_mode != Mode.ASSISTANT or not normalized:
        return False
    if normalized in _STOP_PHRASES or normalized in _MODE_ALIASES:
        return False
    if has_pending_confirmation and normalized in (
        _CONFIRM_PHRASES | _DENY_PHRASES
    ):
        return False
    if vault_search_enabled and _is_vault_lookup_request(normalized):
        return False
    return not normalized.startswith(_EXPLICIT_NON_ASSISTANT_PREFIXES)


@dataclass(frozen=True)
class ModePolicy:
    passive_requires_explicit_action: bool = True
    assistant_auto_reply: bool = True
    search_auto_reply: bool = True
    research_parallel_tasks: int = 2
    command_requires_confirmation: bool = True
    vault_search_enabled: bool = False


class LiveSpeechAnalyzer:
    """
    Converts noisy partial/final transcripts into observations and decisions.

    This layer is deterministic by design. Slow LLM planning can be added after
    the decision, but control and mode switching must stay cheap and reliable.
    """

    def __init__(self, policy: ModePolicy | None = None):
        self.policy = policy or ModePolicy()
        self._last_partial = ""
        self._last_partial_at = 0.0

    def observe(self, text: str, *, is_final: bool) -> SpeechObservation:
        normalized = normalize_text(text)
        stability = self._stability(normalized) if not is_final else 1.0
        activation = self._activation_score(normalized)
        observation = SpeechObservation(
            text=text,
            normalized=normalized,
            is_final=is_final,
            language=detect_language(text),
            stability=stability,
            activation_score=activation,
            keywords=keywords(text),
        )
        if not is_final:
            self._last_partial = normalized
            self._last_partial_at = time.time()
        return observation

    def decide(
        self,
        observation: SpeechObservation,
        current_mode: Mode,
        *,
        has_pending_confirmation: bool = False,
    ) -> IntentDecision:
        text = observation.normalized
        if not text:
            return IntentDecision(IntentKind.IGNORE, 1.0, "", "empty")

        if text in _STOP_PHRASES:
            return IntentDecision(IntentKind.STOP, 1.0, observation.text, "stop_phrase")
        # CONFIRM/DENY are control-plane replies to a STAGED command confirmation,
        # so only treat a bare "yes"/"no" as one when a confirmation is ACTUALLY
        # pending. Otherwise a "yes" answering the assistant's own yes/no question
        # was fired at an empty queue ("Nothing to confirm." -- never even spoken)
        # and the turn was silently dropped. With nothing pending it falls through
        # to the normal conversational path below and is answered (with recent
        # context resolving the antecedent).
        # SECURITY: only a FINAL may confirm/deny a staged action. A CONFIRM/DENY
        # derived from a PARTIAL would carry the PRIOR turn's speaker-ID trust
        # (state.turn_owner_verified is only updated on finals), so an ambient
        # partial "yes" could launder the owner's trust onto a staged owner-verified
        # action (the ambient-yes race). Partials fall through to IGNORE below; the
        # FINAL "yes"/"no" acts. (STOP above stays partial-allowed for instant
        # barge-in -- cancelling is always fail-safe.)
        if has_pending_confirmation and observation.is_final:
            if text in _CONFIRM_PHRASES:
                return IntentDecision(IntentKind.CONFIRM, 0.98, observation.text, "confirm_phrase")
            if text in _DENY_PHRASES:
                return IntentDecision(IntentKind.DENY, 0.98, observation.text, "deny_phrase")

        mode = self._mode_from_text(text)
        if mode is not None:
            return IntentDecision(
                IntentKind.MODE_SWITCH,
                0.98,
                observation.text,
                "mode_phrase",
                target_mode=mode,
            )

        if not observation.is_final:
            return IntentDecision(IntentKind.IGNORE, 0.8, observation.text, "partial_non_control")

        explicit = self._explicit_intent(text, observation.text)
        if explicit is not None:
            return explicit

        if (
            current_mode == Mode.ASSISTANT
            and self.policy.vault_search_enabled
            and _is_vault_lookup_request(text)
        ):
            return IntentDecision(
                IntentKind.SEARCH,
                0.95,
                observation.text,
                "vault_lookup_phrase",
                mode=Mode.SEARCH,
                metadata={"search_scope": "vault"},
            )

        if current_mode == Mode.PASSIVE and self.policy.passive_requires_explicit_action:
            if observation.activation_score < 0.65:
                return IntentDecision(IntentKind.IGNORE, 0.9, observation.text, "passive_no_activation")
            return IntentDecision(
                IntentKind.ASSISTANT,
                observation.activation_score,
                self._strip_wake_word(observation.text),
                "wake_word_activation",
                mode=Mode.ASSISTANT,
            )

        if current_mode == Mode.SEARCH:
            metadata = (
                {"search_scope": "vault"}
                if is_vault_scoped_request(text)
                else {}
            )
            return IntentDecision(
                IntentKind.SEARCH,
                0.82,
                observation.text,
                "vault_search_mode" if metadata else "search_mode",
                mode=current_mode,
                metadata=metadata,
            )
        if current_mode == Mode.RESEARCH:
            metadata = (
                {"search_scope": "vault"}
                if is_vault_scoped_request(text)
                else {}
            )
            return IntentDecision(
                IntentKind.RESEARCH,
                0.82,
                observation.text,
                "vault_research_mode" if metadata else "research_mode",
                mode=current_mode,
                metadata=metadata,
            )
        if current_mode == Mode.DICTATION:
            return IntentDecision(IntentKind.DICTATION, 0.9, observation.text, "dictation_mode", speak=False)
        if current_mode == Mode.MEETING:
            return IntentDecision(IntentKind.MEETING_NOTE, 0.85, observation.text, "meeting_mode", speak=False)
        if current_mode == Mode.COMMAND:
            return IntentDecision(
                IntentKind.COMMAND,
                0.82,
                observation.text,
                "command_mode",
                mode=current_mode,
                requires_confirmation=self.policy.command_requires_confirmation,
            )
        return IntentDecision(IntentKind.ASSISTANT, 0.75, observation.text, "assistant_mode", mode=current_mode)

    def _explicit_intent(self, normalized: str, original: str) -> IntentDecision | None:
        # A polite prefix must not hide an otherwise explicit search/research
        # command.  Keep this deliberately narrow: stripping ``please`` from
        # every utterance would turn ordinary assistant requests into control
        # intents (and could make commands confirmation-bearing unexpectedly).
        if normalized.startswith("please "):
            courteous_normalized = normalized.removeprefix("please ")
            if courteous_normalized.startswith(("research ", "search ")):
                normalized = courteous_normalized
                original = _after_first_word(original)
        if normalized.startswith(("research ", "cerceteaza ")):
            query = _after_first_word(original)
            metadata = (
                {"search_scope": "vault"}
                if is_vault_scoped_request(query)
                else {}
            )
            return IntentDecision(
                IntentKind.RESEARCH,
                0.95,
                query,
                "vault_research_prefix" if metadata else "research_prefix",
                mode=Mode.RESEARCH,
                metadata=metadata,
            )
        if normalized.startswith(("search ", "cauta ")):
            query = _after_first_word(original)
            if normalize_text(query).startswith("for "):
                query = _after_first_word(query)
            metadata = (
                {"search_scope": "vault"}
                if is_vault_scoped_request(query)
                else {}
            )
            return IntentDecision(
                IntentKind.SEARCH,
                0.95,
                query,
                "vault_search_prefix" if metadata else "search_prefix",
                mode=Mode.SEARCH,
                metadata=metadata,
            )
        if normalized.startswith(("dictate ", "scrie ")):
            return IntentDecision(IntentKind.DICTATION, 0.95, _after_first_word(original), "dictation_prefix", mode=Mode.DICTATION, speak=False)
        if normalized.startswith(("run ", "open ", "execute ")):
            return IntentDecision(IntentKind.COMMAND, 0.9, original, "command_prefix", mode=Mode.COMMAND, requires_confirmation=True)
        return None

    @staticmethod
    def _mode_from_text(normalized: str) -> Mode | None:
        return _MODE_ALIASES.get(normalized)

    @staticmethod
    def _activation_score(normalized: str) -> float:
        words = set(normalized.split())
        score = 0.0
        if words & _WAKE_TERMS:
            score += 0.55
        if {"can", "could", "please", "help", "vreau"} & words:
            score += 0.2
        if {"search", "research", "cauta", "cerceteaza"} & words:
            score += 0.3
        return min(score, 1.0)

    def _stability(self, normalized: str) -> float:
        if not normalized:
            return 0.0
        if not self._last_partial:
            return 0.2
        return SequenceMatcher(None, self._last_partial, normalized).ratio()

    @staticmethod
    def _strip_wake_word(text: str) -> str:
        words = text.split()
        if words and normalize_text(words[0]) in _WAKE_TERMS:
            return " ".join(words[1:]).strip() or text
        return text


def _after_first_word(text: str) -> str:
    parts = text.split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""
