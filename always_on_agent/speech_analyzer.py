from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
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
    "my obsidian vault",
    "my obsidian notes",
    "my obsidian",
    "my second brain",
    "dobo brain",
    "paul brain",
    "pauls brain",
)
_VAULT_SCOPE_PATTERN = "|".join(re.escape(marker) for marker in _VAULT_SCOPE_MARKERS)
_VAULT_SCOPE_RE = re.compile(
    rf"(?<![a-z0-9])(?:{_VAULT_SCOPE_PATTERN})(?![a-z0-9])"
)
_VAULT_PUBLIC_SOURCE_PATTERN = (
    r"(?:(?:(?:on|in|inside|into|from|via|across|over|using|through|with|within)\s+)?"
    r"(?:(?:the\s+)?(?:(?:public|open)\s+)?(?:web|internet)|online)"
    r"(?:\s+(?:search\s+)?(?:documents?|pages?|sources?|results?))?)"
)
_VAULT_SOURCE_MODIFIER_PATTERN = (
    r"(?:(?:quite|rather|very)\s+[a-z0-9]+ly|[a-z0-9]+ly|also|first|just|"
    r"again|kindly|later|maybe|now|please|quickly|instead|today|tomorrow|"
    r"tonight|right\s+now|once\s+more|"
    r"as\s+(?:quickly|soon)\s+as\s+possible|if\s+possible|if\s+you\s+can|"
    r"when\s+you\s+get\s+a\s+chance|when(?:ever)?\s+you\s+can)"
)
_VAULT_PUBLIC_SOURCE_RE = re.compile(rf"\b{_VAULT_PUBLIC_SOURCE_PATTERN}\b")
_VAULT_PUBLIC_HEAD_RE = re.compile(
    rf"^(?:(?:search|research|find|look(?:\s+up)?|check|read|list|browse|"
    rf"consult|query|summarize)\s+"
    rf"(?:{_VAULT_SOURCE_MODIFIER_PATTERN}\s+){{0,3}}"
    rf"{_VAULT_PUBLIC_SOURCE_PATTERN}\b|"
    rf"go\s+{_VAULT_PUBLIC_SOURCE_PATTERN}\b|"
    rf"use\s+{_VAULT_PUBLIC_SOURCE_PATTERN}\b|"
    rf"(?:online|(?:the\s+)?(?:public\s+)?(?:web|internet))\s+"
    rf"(?:search|lookup|find|check)\b)"
)
_VAULT_SOURCE_ROUTABLE_REQUEST_RE = re.compile(
    r"^(?:(?:search|research|find|look(?:\s+up)?|check|read|list|browse|"
    rf"consult|query|summarize)\b|go\s+(?:in|into|inside|through|to|within)\b|"
    rf"go\s+{_VAULT_PUBLIC_SOURCE_PATTERN}\b|"
    rf"use\s+{_VAULT_PUBLIC_SOURCE_PATTERN}\b|"
    r"(?:what|whats|do|does|is|are)\b|"
    r"(?:online|(?:the\s+)?(?:public\s+)?(?:web|internet))\s+"
    r"(?:search|lookup|find|check)\b)"
)
_VAULT_SOURCE_MODIFIERS_RE = re.compile(
    rf"^(?:{_VAULT_SOURCE_MODIFIER_PATTERN}\s*){{1,3}}$"
)
_VAULT_TOPIC_JOIN_RE = re.compile(
    r"^(?:about|concerning|discussing|for|on|regarding)\b"
)
_VAULT_AFTER_SCOPE_RESPONSE_RE = re.compile(
    r"^(?:say|contain|contains|mention|mentions|have|show)\b\s*"
)
_VAULT_SCOPE_ACCESS_ACTION_PATTERN = (
    r"(?:access(?:ing)?|brows(?:e|ing)|check(?:ing)?|consult(?:ing)?|"
    r"find(?:ing)?|go(?:ing)?|include|including|list(?:ing)?|look(?:ing)?|query(?:ing)?|"
    r"read(?:ing)?|research(?:ing)?|search(?:ing)?|touch(?:ing)?|us(?:e|ing))"
)
_VAULT_SCOPE_ACCESS_PREPOSITION_PATTERN = (
    r"(?:at|from|in|inside(?:\s+of)?|into|of|through|to|using|with|within)"
)
_VAULT_SCOPE_EXCLUSION_OPERATOR_PATTERN = (
    r"(?:instead\s+of|rather\s+than|other\s+than|"
    r"(?:anything|anywhere|everything|everywhere)\s+(?:but|except(?:\s+for)?)|"
    r"but\s+not|and\s+not|except(?:\s+for)?|apart\s+from|excluding|"
    r"exclude|omit(?:ting)?|ignore|ignoring|outside(?:\s+of)?|without|"
    r"neither|nor|(?:keep|stay)\s+out(?:\s+of)?|(?:leave|leaving)\s+out)"
)
_VAULT_SCOPE_NEGATOR_PATTERN = (
    r"(?:do\s+not|dont|never|not|avoid(?:ing)?|skip(?:ping)?)"
)
_VAULT_SCOPE_NEGATION_FILLER_PATTERN = (
    r"(?:actually|also|completely|entirely|ever|just|please|really)"
)
_VAULT_SCOPE_EXCLUSION_OBJECT_PATTERN = (
    r"(?:(?:all|any|every|some|the|these|those)\s+)?"
    r"(?:anything|content|contents|data|files?|information|material|notes?|results?)"
    r"(?:\s+(?:found|originating|sourced|stored|taken))?"
)
_VAULT_SCOPE_EXCLUSION_TAIL_RE = re.compile(
    rf"(?:{_VAULT_SCOPE_EXCLUSION_OPERATOR_PATTERN}|"
    rf"{_VAULT_SCOPE_NEGATOR_PATTERN})"
    rf"(?:\s+{_VAULT_SCOPE_NEGATION_FILLER_PATTERN}){{0,3}}"
    rf"(?:\s+either)?"
    rf"(?:\s+{_VAULT_SCOPE_ACCESS_ACTION_PATTERN})?"
    rf"(?:\s+{_VAULT_SCOPE_EXCLUSION_OBJECT_PATTERN})?"
    rf"(?:\s+{_VAULT_SCOPE_ACCESS_PREPOSITION_PATTERN})?"
    rf"(?:\s+either)?\s*$"
)
_VAULT_SCOPE_BROAD_EXCLUSION_TAIL_RE = re.compile(
    r"(?P<operator>"
    r"(?:filter|filtering)(?:\s+out)?|minus|exclude|excluding|without|"
    r"omit|omitting|ignore|ignoring|outside(?:\s+of)?|"
    r"skip|skipping(?:\s+over)?|"
    r"(?:do\s+not|dont|never)\s+"
    r"(?:access|browse|check|consult|include|look|read|search|touch|use))"
    r"(?:\s+[a-z0-9]+){0,8}\s*$"
)
_VAULT_SCOPE_TOPICAL_EXCLUSION_RE = re.compile(
    r"\b(?:about|concerning|discussing|for|on|regarding)\s*$"
)
_VAULT_SCOPE_NEGATED_CHOICE_TAIL_RE = re.compile(
    rf"(?:neither(?:\s+[a-z0-9]+){{1,12}}\s+nor|"
    rf"(?:(?:do\s+not|dont|never)\s+{_VAULT_SCOPE_ACCESS_ACTION_PATTERN}"
    rf"(?:\s+{_VAULT_SCOPE_ACCESS_PREPOSITION_PATTERN})?|"
    rf"not(?:\s+{_VAULT_SCOPE_ACCESS_PREPOSITION_PATTERN})?)"
    rf"\s+either(?:\s+[a-z0-9]+){{1,12}}\s+(?:or|nor))"
    rf"(?:\s+{_VAULT_SCOPE_ACCESS_PREPOSITION_PATTERN})?\s*$"
)
_VAULT_SCOPE_COORDINATION_RE = re.compile(
    rf"^\s+(?:and(?:\s+also)?|nor|or)"
    rf"(?:\s+{_VAULT_SCOPE_ACCESS_PREPOSITION_PATTERN})?\s+$"
)
_VAULT_SCOPE_POST_EXCLUDED_RE = re.compile(
    r"^\s+(?:is\s+)?excluded(?:\s+(?:please|for\s+me))?\s*$"
)
_VAULT_SCOPE_WITH_RE = re.compile(r"\bwith\s*$")
_VAULT_SCOPE_LEAVE_RE = re.compile(r"\bleav(?:e|ing)\s*$")
_VAULT_SCOPE_POST_OUT_ACTION_RE = re.compile(
    r"\b(?:filter|filtering|keep|keeping|leave|leaving)\s*$"
)
_VAULT_SCOPE_ALONE_RE = re.compile(
    r"^\s+(?:(?:completely|entirely)\s+)?(?:alone|out)"
    r"(?:\s+of\s+it)?(?:\s+(?:please|for\s+me))?\s*$"
)
_VAULT_POSITIVE_SCOPE_CLUSTER_RE = re.compile(
    r"^\s*(?:(?:and|or)(?:\s+also)?|as\s+well\s+as)\s*$"
)
_VAULT_SOURCE_CORRECTION_CUE_PATTERN = r"(?:actually|but|no|then)"
_VAULT_SOURCE_SELECTION_ACTION_PATTERN = (
    r"(?:browse|check|consult|find|go|look(?:\s+up)?|query|read|research|search|use)"
)
_VAULT_PRIVATE_SOURCE_CORRECTION_TAIL_RE = re.compile(
    rf"\b(?P<cue>{_VAULT_SOURCE_CORRECTION_CUE_PATTERN})\s+"
    rf"(?:(?:actually|please|wait)\s+)*"
    rf"(?P<action>{_VAULT_SOURCE_SELECTION_ACTION_PATTERN})"
    rf"(?:\s+(?:back|only))?"
    rf"(?:\s+(?:in|inside|into|through|to|within))?\s+$"
)
_VAULT_PUBLIC_SOURCE_CORRECTION_RE = re.compile(
    rf"\b{_VAULT_SOURCE_CORRECTION_CUE_PATTERN}\s+"
    rf"(?:(?:actually|please|wait)\s+)*"
    rf"(?:{_VAULT_SOURCE_SELECTION_ACTION_PATTERN}\s+)?"
    rf"{_VAULT_PUBLIC_SOURCE_PATTERN}\b\s+"
    rf"(?:instead|after\s+all)\b"
)
_VAULT_NEGATED_PUBLIC_CORRECTION_RE = re.compile(
    rf"\b(?:actually|but|then)\s+"
    rf"(?:do\s+not|dont|never|not|avoid(?:ing)?|without)\s+"
    rf"(?:{_VAULT_SOURCE_SELECTION_ACTION_PATTERN}\s+)?"
    rf"{_VAULT_PUBLIC_SOURCE_PATTERN}\b"
)
_VAULT_LITERAL_TOPIC_AFTER_SCOPE_RE = re.compile(
    r"^\s+(?:about|concerning|discussing|for|regarding)\s+"
    r"(?:the\s+)?(?:phrase|text|title|words?)\b"
)
_VAULT_NEGATED_PUBLIC_SOURCE_RE = re.compile(
    rf"\b(?:"
    rf"(?:do\s+not|dont)\s+(?:(?:go|search|look|browse|use)\s+)?"
    rf"{_VAULT_PUBLIC_SOURCE_PATTERN}|"
    rf"without\s+(?:(?:going|searching|looking|browsing|using)\s+)?"
    rf"{_VAULT_PUBLIC_SOURCE_PATTERN}|"
    rf"(?:not|never|no|neither|nor|avoid(?:ing)?|skip(?:ping)?|instead\s+of|rather\s+than|except(?:\s+for)?)\s+"
    rf"(?:(?:going|searching|looking|browsing|using)\s+)?"
    rf"{_VAULT_PUBLIC_SOURCE_PATTERN})\b"
)
_VAULT_LEADING_NEGATED_PUBLIC_PREAMBLE_RE = re.compile(
    rf"^(?:without|not)\s+"
    rf"(?:(?:browsing|checking|consulting|going|looking|searching|using)\s+)?"
    rf"{_VAULT_PUBLIC_SOURCE_PATTERN}\s+"
)
_VAULT_COURTESY_RE = re.compile(
    r"^(?:(?:please|kindly|assistant|computer|jarvis|asistent)\s+"
    r"|(?:(?:can|could|would|will)\s+you)\s+"
    r"|(?:i\s+(?:want|need)\s+you\s+to)\s+)+"
)
_VAULT_READ_IMPERATIVE_RE = re.compile(
    r"^(?:search|research|find|look(?:\s+up)?|check|read|list|browse|consult|query|summarize)\b"
)
_VAULT_GO_IMPERATIVE_RE = re.compile(
    r"^go\s+(?:in|into|inside(?:\s+of)?|through|to|within)\b"
)
_VAULT_MUTATING_ACTION_PATTERN = (
    r"(?:act|add|amend|annotate|append|apply|approve|archive|attach|burn|change|"
    r"classify|clear|compress|contact|convert|copy|create|cut|decrypt|discard|dispose|"
    r"delete|destroy|download|duplicate|edit|email|embed|encrypt|erase|execute|"
    r"export|favorite|fax|format|forward|give|hide|import|insert|label|link|lock|mark|"
    r"merge|mess|modify|notify|"
    r"make|move|open|overwrite|paste|pin|print|publish|purge|relocate|remove|rename|"
    r"replace|restore|revise|rewrite|post|put|quarantine|redact|run|save|scrub|send|"
    r"set|share|shred|sign|sort|star|submit|sync|tag|trash|unfavorite|unlink|unlock|"
    r"unpin|unpublish|unstar|update|upload|wipe|write)"
)
_VAULT_ACTION_FILLER_PATTERN = (
    r"(?:(?:please|also|kindly|just|now|then|do|immediately|permanently|quickly|"
    r"carefully|very|really|quite|rather|extremely)\s+|"
    r"(?:(?:can|could|would|will)\s+you\s+)|(?:[a-z0-9]+ly\s+))"
)
_VAULT_COMPOUND_ACTION_RE = re.compile(
    rf"\b(?P<connector>and|then)\s+"
    rf"(?:{_VAULT_ACTION_FILLER_PATTERN}){{0,5}}"
    rf"(?P<action>{_VAULT_MUTATING_ACTION_PATTERN})\b"
)
_VAULT_PURPOSE_ACTION_RE = re.compile(
    rf"\bto\s+(?:{_VAULT_ACTION_FILLER_PATTERN}){{0,5}}"
    rf"(?P<action>{_VAULT_MUTATING_ACTION_PATTERN})\b"
)
_VAULT_RESULT_TARGET_PATTERN = (
    r"(?:(?:all\s+of\s+)?(?:it|them|one|ones)|"
    r"(?:(?:all\s+of\s+)?(?:a|an|the|this|that|these|those|each|every|any|all|"
    r"my|your|selected|found|matched)\s+)?"
    r"(?:[a-z0-9]+\s+){0,2}"
    r"(?:document|documents|draft|drafts|entry|entries|file|files|item|items|"
    r"note|notes|page|pages|project|projects|record|records|result|results|"
    r"summary|summaries))"
)
_VAULT_STRICT_RESULT_REFERENCE_PATTERN = (
    r"(?:(?:all\s+of\s+)?(?:it|them|one|ones)|"
    r"(?:(?:all\s+of\s+)?(?:a|an|another|my|your|the|this|that|these|those|"
    r"each|every|any|all|selected|found|matched)\s+)"
    r"(?:[a-z0-9]+\s+){0,2}"
    r"(?:document|documents|draft|drafts|entry|entries|file|files|item|items|"
    r"note|notes|page|pages|project|projects|record|records|result|results|"
    r"summary|summaries))"
)
_VAULT_GENERIC_ACTION_BRIDGE_PATTERN = r"(?:[a-z0-9]+\s+){0,4}"
_VAULT_NON_MUTATING_RESULT_VERB_PATTERN = (
    r"(?:search|research|find|look|lookup|check|read|list|browse|consult|query|"
    r"summarize|use|about|after|also|am|and|are|away|back|be|been|before|being|but|"
    r"can|could|do|for|immediately|just|kindly|not|now|of|off|or|out|please|so|then|"
    r"very|will|with|without|would|you|"
    r"concerning|contain|contains|describe|discuss|discussing|explain|for|from|"
    r"go|has|have|in|inside|into|is|mention|mentions|of|on|regarding|say|says|"
    r"show|shows|tell|through|to|using|was|were|with|within)"
)
_VAULT_GENERIC_RESULT_ACTION_PATTERN = (
    rf"(?!(?:{_VAULT_NON_MUTATING_RESULT_VERB_PATTERN})\b)[a-z][a-z0-9]*"
)
_VAULT_GENERIC_RESULT_ACTION_RE = re.compile(
    rf"\b(?P<cue>and|then|to)\s+(?:{_VAULT_ACTION_FILLER_PATTERN}){{0,5}}"
    rf"(?P<action>{_VAULT_GENERIC_RESULT_ACTION_PATTERN})\s+"
    rf"{_VAULT_GENERIC_ACTION_BRIDGE_PATTERN}"
    rf"(?P<reference>{_VAULT_STRICT_RESULT_REFERENCE_PATTERN})\b"
)
_VAULT_GENERIC_REQUEST_FILLER_ACTION_RE = re.compile(
    rf"\b(?:(?:also|immediately|just|kindly|now|permanently|please|then)\s+|"
    rf"(?:(?:can|could|would|will)\s+you\s+)){{1,4}}"
    rf"{_VAULT_GENERIC_RESULT_ACTION_PATTERN}\s+"
    rf"{_VAULT_GENERIC_ACTION_BRIDGE_PATTERN}"
    rf"(?P<reference>{_VAULT_STRICT_RESULT_REFERENCE_PATTERN})\b"
)
_VAULT_GENERIC_FUTURE_PURPOSE_ACTION_RE = re.compile(
    rf"\bso\s+(?:that\s+)?(?:i|we)\s+"
    rf"(?:can|could|may|might|will|would)\s+"
    rf"(?:{_VAULT_ACTION_FILLER_PATTERN}){{0,5}}"
    rf"{_VAULT_GENERIC_RESULT_ACTION_PATTERN}\s+"
    rf"{_VAULT_GENERIC_ACTION_BRIDGE_PATTERN}"
    rf"(?P<reference>{_VAULT_STRICT_RESULT_REFERENCE_PATTERN})\b"
)
_VAULT_DIRECT_GENERIC_RESULT_ACTION_RE = re.compile(
    rf"^(?:{_VAULT_ACTION_FILLER_PATTERN}){{0,5}}"
    rf"{_VAULT_GENERIC_RESULT_ACTION_PATTERN}\s+"
    rf"{_VAULT_GENERIC_ACTION_BRIDGE_PATTERN}"
    rf"(?P<reference>{_VAULT_STRICT_RESULT_REFERENCE_PATTERN})\b"
)
_VAULT_DIRECT_BOUNDED_REFERENCE_PATTERN = (
    r"(?:(?:this|that)|(?:its|their)\s+[a-z0-9]+)"
)
_VAULT_DIRECT_BOUNDED_REFERENCE_ACTION_RE = re.compile(
    rf"^(?:{_VAULT_ACTION_FILLER_PATTERN}){{0,5}}"
    rf"{_VAULT_GENERIC_RESULT_ACTION_PATTERN}\s+"
    rf"{_VAULT_GENERIC_ACTION_BRIDGE_PATTERN}"
    rf"{_VAULT_DIRECT_BOUNDED_REFERENCE_PATTERN}"
    r"(?:\s+(?:immediately|kindly|now|please|right\s+now))?$"
)
_VAULT_DIRECT_PHRASAL_ACTION_RE = re.compile(
    r"^(?:do\s+away\s+with|get\s+rid\s+of)\s+"
    r"(?:it|them|this|that|(?:a|an|another|my|the|this|that|your)\s+"
    r"(?:document|draft|entry|file|item|note|page|project|record|result|summary))\b"
)
_VAULT_BOUNDED_REFERENCE_SUFFIX_PATTERN = (
    rf"(?=$|\s+(?:(?:from|in|inside|within)\s+)?"
    rf"(?:{_VAULT_SCOPE_PATTERN})(?![a-z0-9])\s*$|"
    r"\s+(?:immediately|kindly|now|please|right\s+now)\s*$)"
)
_VAULT_BOUNDED_REFERENCE_ACTION_RE = re.compile(
    rf"\b(?P<cue>and|then|to)\s+"
    rf"(?:{_VAULT_ACTION_FILLER_PATTERN}){{0,5}}"
    rf"(?P<action>{_VAULT_GENERIC_RESULT_ACTION_PATTERN})\s+"
    rf"{_VAULT_GENERIC_ACTION_BRIDGE_PATTERN}"
    rf"(?P<reference>{_VAULT_DIRECT_BOUNDED_REFERENCE_PATTERN})"
    rf"{_VAULT_BOUNDED_REFERENCE_SUFFIX_PATTERN}"
)
_VAULT_BOUNDED_PHRASAL_ACTION_RE = re.compile(
    rf"\b(?P<cue>and|then|to)\s+do\s+away\s+with\s+"
    rf"(?P<reference>{_VAULT_STRICT_RESULT_REFERENCE_PATTERN}|"
    rf"{_VAULT_DIRECT_BOUNDED_REFERENCE_PATTERN})"
    rf"{_VAULT_BOUNDED_REFERENCE_SUFFIX_PATTERN}"
)
_VAULT_PASSIVE_ACTION_RE = re.compile(
    rf"\b(?:and|then)\s+(?:{_VAULT_ACTION_FILLER_PATTERN}){{0,5}}"
    rf"(?:(?:have|get)\s+{_VAULT_RESULT_TARGET_PATTERN}\s+"
    r"(?:[a-z0-9]+ly\s+){0,2}"
    r"(?:[a-z0-9]+(?:ed|en)|cut|made|run|sent|shared|wiped|written)\b|"
    rf"get\s+rid\s+of\s+{_VAULT_RESULT_TARGET_PATTERN}\b)"
)
_VAULT_DIRECT_PASSIVE_ACTION_RE = re.compile(
    rf"^(?:{_VAULT_ACTION_FILLER_PATTERN}){{0,5}}"
    rf"(?:(?:have|get)\s+{_VAULT_RESULT_TARGET_PATTERN}\s+"
    r"(?:[a-z0-9]+ly\s+){0,2}"
    r"(?:[a-z0-9]+(?:ed|en)|cut|made|run|sent|shared|wiped|written)\b|"
    rf"get\s+rid\s+of\s+{_VAULT_RESULT_TARGET_PATTERN}\b)"
)
_VAULT_FUTURE_PURPOSE_ACTION_RE = re.compile(
    rf"\bso\s+(?:that\s+)?(?:i|we)\s+(?:can|could|may|might|will|would)\s+"
    rf"(?:{_VAULT_ACTION_FILLER_PATTERN}){{0,5}}"
    rf"{_VAULT_MUTATING_ACTION_PATTERN}\b"
)
_VAULT_TEMPORAL_RESULT_ACTION_RE = re.compile(
    rf"\b(?:before|after)\s+(?:{_VAULT_ACTION_FILLER_PATTERN}){{0,3}}"
    rf"[a-z0-9]+ing\s+{_VAULT_RESULT_TARGET_PATTERN}\b"
)
_VAULT_RESULT_PASSIVE_PURPOSE_RE = re.compile(
    rf"\bso\s+(?:that\s+)?{_VAULT_RESULT_TARGET_PATTERN}\s+"
    r"(?:can|could|may|might|will|would)\s+be\s+"
    r"(?:[a-z0-9]+(?:ed|en)|cut|made|run|sent|shared|wiped|written)\b"
)
_VAULT_RESULT_TARGET_RE = re.compile(rf"\b{_VAULT_RESULT_TARGET_PATTERN}\b")
_VAULT_TOPICAL_ACTION_PREFIX_RE = re.compile(
    r"(?:\b(?:how|when|whether|why|where|what)(?:\s+[a-z0-9]+){0,3}"
    r"|\b(?:advice|approaches|documentation|docs|examples?|guidance|instructions?|"
    r"methods?|procedures?|steps?|strategies|techniques|tutorials?|ways?))\s*$"
)
_VAULT_LOOKUP_COMMAND_PATTERN = (
    r"(?:search|research|find|look(?:\s+up)?|lookup|check|read|list|browse|consult|query|summarize)"
)
_VAULT_CONFIG_SUBJECT_PATTERN = (
    r"(?:access|availability|capabilities|capability|commands?|features?|"
    r"function|functionality|functions|index|indexing|modes?|performance|"
    r"permissions?|status|support)"
)
_VAULT_STATUS_COPULA_PATTERN = (
    r"(?:is|are|was|were|(?:has|have|had)(?:\s+not)?\s+been|"
    r"(?:will|would|can|could|may|might|must|should)(?:\s+not)?\s+be)"
)
_VAULT_STATUS_COMPLEMENT_PATTERN = (
    r"(?:active|allowed|available|blocked|broken|configured|disabled|enabled|"
    r"degraded|down|failing|flaky|functioning|healthy|inactive|limited|okay|ok|on|"
    r"operational|permitted|readable|restricted|revoked|slow|"
    r"turned\s+(?:off|on)|unavailable|unreliable|usable|working)"
)
_VAULT_STATUS_COMPLEMENT_CLAUSE_PATTERN = (
    rf"(?:(?:[a-z0-9]+ly|not|no\s+longer)\s+){{0,3}}"
    rf"{_VAULT_STATUS_COMPLEMENT_PATTERN}"
)
_VAULT_STATUS_PREDICATE_PATTERN = (
    r"(?:(?:appears?|remains?|seems?)(?:\s+to\s+be)?\s+"
    rf"{_VAULT_STATUS_COMPLEMENT_CLAUSE_PATTERN}|"
    r"(?:can|cant|cannot|could|may|might|must|should|will|wont|would)\s+"
    r"(?:(?:currently|still|not)\s+){0,3}work|"
    r"(?:currently|still)\s+(?:fails?|works?)|continues?\s+to\s+(?:fail|work)|"
    r"does\s+(?:(?:currently|still|not)\s+){0,3}work|doesnt\s+work|"
    r"has\s+(?:not\s+)?stopp?ed\s+working|"
    r"isnt\s+(?:active|available|disabled|enabled|working)|keeps?\s+(?:failing|working)|"
    r"no\s+longer\s+works?|never\s+works?|stopp?ed\s+working|used\s+to\s+work|"
    r"has\s+(?:degraded|failed)|works?|fails?|failed|broke|broken)"
)
_VAULT_STATUS_TRAILING_PATTERN = (
    r"(?:(?:\s+(?:across|for|from|in|inside|of|through|to|within)))*\s*"
)
_VAULT_DECLARATIVE_PREFIX_RE = re.compile(
    rf"^(?:{_VAULT_LOOKUP_COMMAND_PATTERN}\s+(?:"
    rf"{_VAULT_CONFIG_SUBJECT_PATTERN}\s+(?:"
    rf"{_VAULT_STATUS_COPULA_PATTERN}(?:\s+{_VAULT_STATUS_COMPLEMENT_CLAUSE_PATTERN})?"
    rf"|{_VAULT_STATUS_PREDICATE_PATTERN})"
    rf"|{_VAULT_STATUS_COPULA_PATTERN}(?:\s+{_VAULT_STATUS_COMPLEMENT_CLAUSE_PATTERN})?)"
    rf"|{_VAULT_LOOKUP_COMMAND_PATTERN}\s+{_VAULT_STATUS_PREDICATE_PATTERN}"
    rf"|read\s+(?:access|permission|permissions))"
    rf"{_VAULT_STATUS_TRAILING_PATTERN}$"
)
_VAULT_STATUS_SUBJECT_PATTERN = (
    rf"(?:the\s+)?(?:(?:vault\s+)?{_VAULT_LOOKUP_COMMAND_PATTERN}"
    rf"(?:\s+{_VAULT_CONFIG_SUBJECT_PATTERN})?|"
    rf"write(?:\s+{_VAULT_CONFIG_SUBJECT_PATTERN})?)"
)
_VAULT_STATUS_QUESTION_PREFIX_RE = re.compile(
    rf"^(?:is|are|was|were)\s+{_VAULT_STATUS_SUBJECT_PATTERN}\s+"
    rf"{_VAULT_STATUS_COMPLEMENT_CLAUSE_PATTERN}"
    rf"{_VAULT_STATUS_TRAILING_PATTERN}$"
)
_VAULT_STATUS_TRAILING_MODIFIER_PATTERN = (
    r"(?:(?:\s+(?:correctly|currently|now|recently|still|today))?"
    rf"{_VAULT_STATUS_TRAILING_PATTERN})"
)
_VAULT_STATUS_DECLARATION_RE = re.compile(
    rf"^{_VAULT_STATUS_SUBJECT_PATTERN}\s+(?:"
    rf"{_VAULT_STATUS_COPULA_PATTERN}\s+{_VAULT_STATUS_COMPLEMENT_CLAUSE_PATTERN}|"
    rf"{_VAULT_STATUS_COMPLEMENT_CLAUSE_PATTERN}|"
    rf"{_VAULT_STATUS_PREDICATE_PATTERN})"
    rf"{_VAULT_STATUS_TRAILING_MODIFIER_PATTERN}$"
)
_VAULT_STATUS_META_QUESTION_RE = re.compile(
    rf"^(?:check|find\s+out|look\s+up)\s+(?:if|whether)\s+"
    rf"{_VAULT_STATUS_SUBJECT_PATTERN}\s+(?:"
    rf"{_VAULT_STATUS_COPULA_PATTERN}\s+{_VAULT_STATUS_COMPLEMENT_CLAUSE_PATTERN}|"
    rf"{_VAULT_STATUS_COMPLEMENT_CLAUSE_PATTERN}|"
    rf"{_VAULT_STATUS_PREDICATE_PATTERN})"
    rf"{_VAULT_STATUS_TRAILING_MODIFIER_PATTERN}$"
)
_VAULT_STATUS_SCOPE_TOKEN = "personalvault"
_VAULT_STATUS_LOCATION_PATTERN = (
    rf"(?:across|for|from|in|inside|of|through|to|within)\s+"
    rf"{_VAULT_STATUS_SCOPE_TOKEN}"
)
_VAULT_CANONICAL_STATUS_SUBJECT_PATTERN = (
    rf"(?:the\s+)?(?:obsidian\s+)?(?:(?:(?:{_VAULT_STATUS_SCOPE_TOKEN}|vault)\s+)?"
    rf"(?:{_VAULT_LOOKUP_COMMAND_PATTERN}|searching)(?:\s+{_VAULT_CONFIG_SUBJECT_PATTERN})?|"
    rf"write(?:\s+{_VAULT_CONFIG_SUBJECT_PATTERN})?)"
    rf"(?:\s+{_VAULT_STATUS_LOCATION_PATTERN})?"
)
_VAULT_CANONICAL_STATUS_STATE_PATTERN = (
    rf"{_VAULT_STATUS_COMPLEMENT_CLAUSE_PATTERN}"
    r"(?:\s+(?:now|right\s+now|today|correctly|properly|normally|"
    r"as\s+expected|at\s+the\s+moment))?"
)
_VAULT_CANONICAL_STATUS_ASSERTION_PATTERN = (
    rf"(?:{_VAULT_STATUS_COPULA_PATTERN}\s+"
    rf"{_VAULT_CANONICAL_STATUS_STATE_PATTERN}|"
    rf"{_VAULT_CANONICAL_STATUS_STATE_PATTERN}|"
    rf"{_VAULT_STATUS_PREDICATE_PATTERN})"
)
_VAULT_CANONICAL_STATUS_SUFFIX_PATTERN = (
    rf"(?:\s+{_VAULT_STATUS_LOCATION_PATTERN})?\s*"
)
_VAULT_CANONICAL_STATUS_DECLARATION_RE = re.compile(
    rf"^{_VAULT_CANONICAL_STATUS_SUBJECT_PATTERN}\s+"
    rf"{_VAULT_CANONICAL_STATUS_ASSERTION_PATTERN}"
    rf"{_VAULT_CANONICAL_STATUS_SUFFIX_PATTERN}$"
)
_VAULT_CANONICAL_STATUS_DIRECT_QUESTION_RE = re.compile(
    rf"^(?:is|are|was|were)\s+{_VAULT_CANONICAL_STATUS_SUBJECT_PATTERN}\s+"
    rf"{_VAULT_CANONICAL_STATUS_STATE_PATTERN}"
    rf"{_VAULT_CANONICAL_STATUS_SUFFIX_PATTERN}$"
)
_VAULT_CANONICAL_STATUS_EMBEDDED_QUESTION_RE = re.compile(
    rf"^(?:(?:{_VAULT_LOOKUP_COMMAND_PATTERN})(?:\s+out)?|determine|see|verify)\s+"
    rf"(?:if|whether)\s+{_VAULT_CANONICAL_STATUS_SUBJECT_PATTERN}\s+"
    rf"{_VAULT_CANONICAL_STATUS_ASSERTION_PATTERN}"
    rf"{_VAULT_CANONICAL_STATUS_SUFFIX_PATTERN}$"
)
_VAULT_CANONICAL_STATUS_INVERTED_QUESTION_RE = re.compile(
    rf"^(?:(?:do|does|did)\s+{_VAULT_CANONICAL_STATUS_SUBJECT_PATTERN}\s+"
    r"(?:(?:currently|still|not)\s+){0,3}(?:work|fail)|"
    rf"(?:has|have|had)\s+{_VAULT_CANONICAL_STATUS_SUBJECT_PATTERN}\s+"
    rf"(?:(?:not\s+)?been\s+{_VAULT_CANONICAL_STATUS_STATE_PATTERN}|"
    r"(?:not\s+)?(?:failed|degraded|stopp?ed\s+working))|"
    rf"(?:can|could|may|might|must|should|will|would)\s+"
    rf"{_VAULT_CANONICAL_STATUS_SUBJECT_PATTERN}\s+"
    rf"(?:(?:(?:currently|still|not)\s+){{0,3}}work|"
    rf"(?:not\s+)?be\s+{_VAULT_CANONICAL_STATUS_STATE_PATTERN}))"
    rf"{_VAULT_CANONICAL_STATUS_SUFFIX_PATTERN}$"
)
_VAULT_CANONICAL_STATUS_CAPABILITY_RE = re.compile(
    rf"^(?:check|determine|find\s+out|see|verify)\s+(?:if|whether)\s+"
    rf"i\s+(?:can|could|may|might)\s+"
    rf"(?:{_VAULT_LOOKUP_COMMAND_PATTERN}|searching)"
    rf"(?:\s+(?:in|inside|through|within))?\s+{_VAULT_STATUS_SCOPE_TOKEN}$"
)
_VAULT_CANONICAL_STATUS_OF_RE = re.compile(
    rf"^(?:check|find|look(?:\s+up)?|show)\s+(?:the\s+)?status\s+of\s+"
    rf"(?:obsidian\s+)?(?:{_VAULT_LOOKUP_COMMAND_PATTERN}|searching)"
    rf"(?:\s+(?:in|inside|through|within))?\s+{_VAULT_STATUS_SCOPE_TOKEN}$"
)
_VAULT_CANONICAL_STATUS_WHY_RE = re.compile(
    rf"^(?:check|determine|find\s+out|see)\s+why\s+"
    rf"{_VAULT_CANONICAL_STATUS_SUBJECT_PATTERN}\s+"
    rf"{_VAULT_CANONICAL_STATUS_ASSERTION_PATTERN}"
    rf"{_VAULT_CANONICAL_STATUS_SUFFIX_PATTERN}$"
)
_VAULT_DECLARATIVE_AFTER_SCOPE_RE = re.compile(
    r"^\s+(?:is|are|was|were|has\s+been|have\s+been|had\s+been|will\s+be|"
    r"would\s+be|can\s+be|could\s+be|should\s+be)\b"
)
_VAULT_INTERROGATIVE_LOOKUP_RE = re.compile(
    r"^(?:(?:search|research|find|look(?:\s+up)?|check|read|list|browse|consult|query|summarize)"
    r"\s+(?:for\s+)?|)(?:how|what|when|where|which|who|why)\b"
)
_VAULT_LEADING_ACTION_RE = re.compile(
    rf"^(?P<action>{_VAULT_MUTATING_ACTION_PATTERN})\b"
)
_VAULT_REQUEST_FILLER_ACTION_RE = re.compile(
    rf"\b(?:(?:also|immediately|just|kindly|now|permanently|please|then)\s+|"
    rf"(?:(?:can|could|would|will)\s+you\s+)){{1,4}}"
    rf"(?P<action>{_VAULT_MUTATING_ACTION_PATTERN})\b"
)
_VAULT_TOPICAL_ACTION_HEAD_RE = re.compile(
    rf"\b(?:browse|find|list|query|read|research|search|summarize|"
    rf"{_VAULT_MUTATING_ACTION_PATTERN})\s*$"
)
_VAULT_COMPOUND_TOPIC_TAIL_RE = re.compile(
    r"^(?:amplification|conflicts?|constructors?|contention|dialogs?|formats?|"
    r"management|policies|procedures?|questions?|semantics|strateg(?:y|ies)|"
    r"syntax|templates?)\b"
)
_VAULT_REQUESTED_ACTION_TAIL_RE = re.compile(
    rf"^(?:$|(?:all\s+of\s+)?(?:it|them|one|ones)\b|"
    rf"(?:a|all|an|another|any|each|every|my|some|the|these|this|those|that|your)\b|"
    rf"(?:from|in|inside|within)\s+(?:{_VAULT_SCOPE_PATTERN})(?![a-z0-9]))"
)
_VAULT_STRONG_ACTION_TAIL_RE = re.compile(
    r"^(?:(?:all\s+of\s+)?(?:it|them|one|ones)\b|"
    r"(?:a|all|an|another|any|each|every|my|some|the|these|this|those|that|your)\b|"
    r"(?:duplicates?|entries|entry|files?|items?|notes?)"
    r"(?:$|\s+(?:called|from|in|inside|named|to|with|within)\b|"
    r"\s+(?:(?:if\s+possible|if\s+you\s+can|right\s+now|"
    r"when(?:ever)?\s+you\s+can)\b|[a-z0-9]+ly\b|"
    r"(?:immediately|kindly|now|permanently|please|very)\b)))"
)
_VAULT_ACTION_DESTINATION_TAIL_RE = re.compile(
    r"^(?:into|to|with)\b"
)
_VAULT_ANY_ACTION_RE = re.compile(
    rf"\b(?P<action>{_VAULT_MUTATING_ACTION_PATTERN})\b"
)
_VAULT_CONVERSATIONAL_RES = (
    re.compile(
        rf"^(?:what\s+(?:is|s)|whats)\s+(?:in|inside|within)\s+"
        rf"(?:{_VAULT_SCOPE_PATTERN})(?![a-z0-9])"
    ),
    re.compile(
        rf"^what\s+(?:do|does)\s+(?:{_VAULT_SCOPE_PATTERN})\s+"
        r"(?:say|contain|mention|have|show)\b"
    ),
    re.compile(
        rf"^(?:do|does)\s+(?:{_VAULT_SCOPE_PATTERN})\s+"
        r"(?:say|contain|mention|have|show)\b"
    ),
    re.compile(
        rf"^where\s+(?:in|inside|within)\s+(?:{_VAULT_SCOPE_PATTERN})(?![a-z0-9])"
    ),
    re.compile(
        rf"^(?:is|are)\s+.+\s+(?:in|inside|within)\s+"
        rf"(?:{_VAULT_SCOPE_PATTERN})(?![a-z0-9])"
    ),
    re.compile(
        rf"^do\s+i\s+have\s+.+\s+(?:in|inside|within)\s+"
        rf"(?:{_VAULT_SCOPE_PATTERN})(?![a-z0-9])"
    ),
)


def _is_vault_status_request(command: str) -> bool:
    """Recognize a bounded status statement/question independent of scope order."""

    canonical = _VAULT_SCOPE_RE.sub(
        f" {_VAULT_STATUS_SCOPE_TOKEN} ", command
    )
    canonical = re.sub(r"\s+", " ", canonical).strip()
    return any(
        pattern.fullmatch(canonical) is not None
        for pattern in (
            _VAULT_CANONICAL_STATUS_DECLARATION_RE,
            _VAULT_CANONICAL_STATUS_DIRECT_QUESTION_RE,
            _VAULT_CANONICAL_STATUS_EMBEDDED_QUESTION_RE,
            _VAULT_CANONICAL_STATUS_INVERTED_QUESTION_RE,
            _VAULT_CANONICAL_STATUS_CAPABILITY_RE,
            _VAULT_CANONICAL_STATUS_OF_RE,
            _VAULT_CANONICAL_STATUS_WHY_RE,
        )
    )


def _generic_action_targets_scope(
    command: str,
    action: re.Match[str],
    scopes: tuple[re.Match[str], ...],
) -> bool:
    """Do not mistake a personal-source noun phrase for an action target."""

    if "reference" not in action.re.groupindex:
        return False
    return any(
        scope.start() == action.start("reference")
        and scope.end() == action.end("reference")
        for scope in scopes
    )


def _topic_names_search_utility(topic: str) -> bool:
    """Whether the infinitive describes a bounded tool-like search topic."""

    topic_head = re.split(
        r"\b(?:about|concerning|discussing|for|on|regarding)\b",
        topic,
    )[-1].strip()
    return re.fullmatch(
        r"(?:(?:a|an|some|the)\s+)?"
        r"(?:programs?|scripts?|software|systems?|tools?|utilities)",
        topic_head,
    ) is not None


def _generic_action_is_topical(
    command: str,
    action: re.Match[str],
    scopes: tuple[re.Match[str], ...],
) -> bool:
    """Recognize an infinitive inside the bounded lookup topic slot."""

    prefix = command[: action.start()].rstrip()
    if _VAULT_TOPICAL_ACTION_PREFIX_RE.search(prefix):
        return True
    if action.groupdict().get("cue") != "to":
        return False
    preceding_scopes = tuple(scope for scope in scopes if scope.end() <= action.start())
    if preceding_scopes:
        topic_slot = command[preceding_scopes[-1].end() : action.start()].strip()
        return _topic_names_search_utility(topic_slot)
    leading_lookup = _VAULT_READ_IMPERATIVE_RE.match(prefix)
    if leading_lookup is None:
        return False
    topic = prefix[leading_lookup.end() :].strip()
    return _topic_names_search_utility(topic)


def _purpose_action_is_topical(
    command: str,
    action: re.Match[str],
    scopes: tuple[re.Match[str], ...],
) -> bool:
    """Distinguish a searched-for utility from an action on a found result."""

    preceding_scopes = tuple(scope for scope in scopes if scope.end() <= action.start())
    if preceding_scopes:
        topic = command[preceding_scopes[-1].end() : action.start()].strip()
    else:
        prefix = command[: action.start()].strip()
        leading_lookup = _VAULT_READ_IMPERATIVE_RE.match(prefix)
        if leading_lookup is None:
            return False
        topic = prefix[leading_lookup.end() :].strip()
    return _topic_names_search_utility(topic)


def _has_requested_mutating_action(command: str) -> bool:
    """Recognize an action clause without rejecting action words in a topic."""

    scopes = tuple(_VAULT_SCOPE_RE.finditer(command))
    generic_future_actions = tuple(
        action
        for action in _VAULT_GENERIC_FUTURE_PURPOSE_ACTION_RE.finditer(command)
        if not _generic_action_targets_scope(command, action, scopes)
    )
    generic_filler_actions = tuple(
        action
        for action in _VAULT_GENERIC_REQUEST_FILLER_ACTION_RE.finditer(command)
        if not _generic_action_targets_scope(command, action, scopes)
        and not _generic_action_is_topical(command, action, scopes)
    )
    if (
        _VAULT_PASSIVE_ACTION_RE.search(command)
        or _VAULT_FUTURE_PURPOSE_ACTION_RE.search(command)
        or generic_future_actions
        or generic_filler_actions
        or _VAULT_RESULT_PASSIVE_PURPOSE_RE.search(command)
    ):
        return True
    temporal_action = _VAULT_TEMPORAL_RESULT_ACTION_RE.search(command)
    if temporal_action is not None and any(
        scope.end() <= temporal_action.start() for scope in scopes
    ):
        return True
    for action in _VAULT_GENERIC_RESULT_ACTION_RE.finditer(command):
        if _generic_action_targets_scope(command, action, scopes):
            continue
        if _generic_action_is_topical(command, action, scopes):
            continue
        return True
    for action in (
        *_VAULT_BOUNDED_REFERENCE_ACTION_RE.finditer(command),
        *_VAULT_BOUNDED_PHRASAL_ACTION_RE.finditer(command),
    ):
        if _generic_action_targets_scope(command, action, scopes):
            continue
        if action.group("cue") == "to" and _generic_action_is_topical(
            command, action, scopes
        ):
            continue
        if action.group("cue") in {"and", "then"} and not any(
            scope.end() <= action.start() for scope in scopes
        ):
            prefix = command[: action.start()].strip()
            leading_lookup = _VAULT_READ_IMPERATIVE_RE.match(prefix)
            topic = (
                prefix[leading_lookup.end() :].strip()
                if leading_lookup is not None
                else prefix
            )
            if _VAULT_RESULT_TARGET_RE.search(topic) is None:
                continue
        return True

    compound_actions = tuple(_VAULT_COMPOUND_ACTION_RE.finditer(command))
    purpose_actions = tuple(_VAULT_PURPOSE_ACTION_RE.finditer(command))
    structured_action_starts = {
        action.start("action") for action in (*compound_actions, *purpose_actions)
    }
    for action in compound_actions:
        if action.group("connector") == "then":
            return True
        prefix = command[: action.start()].strip()
        leading_lookup = _VAULT_READ_IMPERATIVE_RE.match(prefix)
        if leading_lookup is not None:
            prefix = prefix[leading_lookup.end() :].strip()
        # Coordinated action words are often the subject of a lookup ("save and
        # open dialogs").  A preceding non-action topic ("speaker and delete
        # it") is an additional request and must fail closed.
        paired_action_topic = bool(
            prefix and _VAULT_TOPICAL_ACTION_HEAD_RE.search(prefix)
        )
        action_tail = command[action.end() :].strip()
        if _VAULT_REQUESTED_ACTION_TAIL_RE.match(action_tail):
            return True
        if (
            any(scope.end() <= action.start() for scope in scopes)
            and not paired_action_topic
        ):
            return True
        if _VAULT_ACTION_DESTINATION_TAIL_RE.match(action_tail):
            return True
        if (
            _VAULT_STRONG_ACTION_TAIL_RE.match(action_tail)
            and not paired_action_topic
        ):
            return True
        if not paired_action_topic:
            # A small set of action-noun compounds remains topical ("merge
            # conflicts", "restore procedures"). Every other coordinated
            # mutator is a second requested action, including a bare object
            # before the source ("and delete project in my vault").
            if _VAULT_COMPOUND_TOPIC_TAIL_RE.match(action_tail):
                continue
            return True
    for action in purpose_actions:
        prefix = command[: action.start()]
        action_tail = command[action.end() :].strip()
        if _purpose_action_is_topical(command, action, scopes):
            continue
        if _VAULT_TOPICAL_ACTION_PREFIX_RE.search(prefix):
            continue
        if (
            re.search(
                r"\b(?:about|concerning|discussing|on|regarding)\b",
                prefix,
            )
            and not _VAULT_REQUESTED_ACTION_TAIL_RE.match(action_tail)
        ):
            continue
        return True
    for action in _VAULT_REQUEST_FILLER_ACTION_RE.finditer(command):
        if action.start("action") not in structured_action_starts:
            return True
    for action in _VAULT_ANY_ACTION_RE.finditer(command):
        if action.start() == 0 or action.start("action") in structured_action_starts:
            continue
        prefix = command[: action.start()].rstrip()
        action_tail = command[action.end() :].strip()
        explicit_post_scope_topic = any(
            scope.end() <= action.start()
            and _VAULT_TOPIC_JOIN_RE.match(command[scope.end() :].strip())
            for scope in scopes
        )
        if explicit_post_scope_topic and not action_tail:
            continue
        explanatory = re.search(
            r"\b(?:how|that|which|who)(?:\s+[a-z0-9]+){0,4}\s*$",
            prefix,
        )
        directly_topical = re.search(
            r"\b(?:about|concerning|discussing|for|on|regarding)\s*$",
            prefix,
        )
        if explanatory is not None or directly_topical is not None:
            continue
        if _VAULT_REQUESTED_ACTION_TAIL_RE.match(action_tail):
            return True
        if (
            _VAULT_STRONG_ACTION_TAIL_RE.match(action_tail)
        ):
            return True
    return False


def _has_direct_action_after_scope(after_scope: str) -> bool:
    """Fail closed on an action placed directly after the private source."""

    tail = after_scope.strip()
    if _VAULT_NEGATED_PUBLIC_CORRECTION_RE.match(tail):
        return False
    if re.match(
        rf"^(?:{_VAULT_RESULT_TARGET_PATTERN})\s+"
        r"(?:about|concerning|discussing|on|regarding)\b",
        tail,
    ):
        return False
    infinitive = re.match(
        rf"^(?P<topic>(?:[a-z0-9]+\s+){{1,8}})to\s+"
        rf"{_VAULT_GENERIC_RESULT_ACTION_PATTERN}\s+"
        rf"{_VAULT_GENERIC_ACTION_BRIDGE_PATTERN}"
        rf"(?:{_VAULT_STRICT_RESULT_REFERENCE_PATTERN}|"
        rf"{_VAULT_DIRECT_BOUNDED_REFERENCE_PATTERN})\b",
        tail,
    )
    if (
        infinitive is not None
        and not _VAULT_LEADING_ACTION_RE.match(tail)
        and _VAULT_RESULT_TARGET_RE.search(infinitive.group("topic")) is None
    ):
        return False
    coordinated = re.match(r"^(?:and|then)(?:\s+then)?\s+", tail)
    if coordinated is not None:
        requested = _VAULT_COURTESY_RE.sub("", tail[coordinated.end() :].strip())
        requested = re.sub(
            r"^(?:(?:also|first|just|kindly|now|please|quickly|then)\s+)+",
            "",
            requested,
        )
        if (
            _VAULT_READ_IMPERATIVE_RE.match(requested)
            or _VAULT_GO_IMPERATIVE_RE.match(requested)
            or re.match(r"^(?:show|tell)\s+(?:me\s+)?", requested)
            or re.match(
                rf"^use\s+(?:(?:{_VAULT_SCOPE_PATTERN})(?![a-z0-9])|"
                rf"{_VAULT_PUBLIC_SOURCE_PATTERN}\b)",
                requested,
            )
            or _VAULT_SCOPE_RE.match(requested)
            or _VAULT_PUBLIC_SOURCE_RE.match(requested)
        ):
            return False
        return True
    if re.match(r"^so\s+(?:that\s+)?(?:i|we)\b", tail):
        return True
    return bool(
        _VAULT_LEADING_ACTION_RE.match(tail)
        or _VAULT_DIRECT_PASSIVE_ACTION_RE.match(tail)
        or _VAULT_DIRECT_GENERIC_RESULT_ACTION_RE.match(tail)
        or _VAULT_DIRECT_BOUNDED_REFERENCE_ACTION_RE.match(tail)
        or _VAULT_DIRECT_PHRASAL_ACTION_RE.match(tail)
    )


def _is_vault_non_lookup_request(text: str) -> bool:
    """Whether any personal-source mention declares or requests an action."""

    normalized = normalize_text(text)
    command = _VAULT_COURTESY_RE.sub("", normalized)
    polarities = _vault_scope_polarities(command)
    if not polarities:
        return False
    if _is_vault_status_request(command):
        return True
    if _VAULT_LEADING_ACTION_RE.match(command):
        return True
    if _has_requested_mutating_action(command):
        return True
    interrogative = _VAULT_INTERROGATIVE_LOOKUP_RE.match(command) is not None
    for scope, negated in polarities:
        if negated:
            continue
        before_scope = command[: scope.start()]
        after_scope = command[scope.end() :]
        if (
            _VAULT_DECLARATIVE_PREFIX_RE.match(before_scope)
            or _VAULT_STATUS_QUESTION_PREFIX_RE.match(before_scope)
            or _VAULT_STATUS_DECLARATION_RE.match(before_scope)
            or _VAULT_STATUS_META_QUESTION_RE.match(before_scope)
        ):
            return True
        scope_is_topical = re.search(
            r"\b(?:about|concerning|discussing|on|regarding)\b",
            before_scope,
        ) is not None
        if (
            _VAULT_DECLARATIVE_AFTER_SCOPE_RE.match(after_scope)
            and not interrogative
            and not scope_is_topical
        ):
            return True
        if _has_direct_action_after_scope(after_scope):
            return True
    return False


def _vault_scope_polarities(
    command: str,
) -> tuple[tuple[re.Match[str], bool], ...]:
    """Return personal-source mentions paired with an explicit exclusion bit."""

    result: list[tuple[re.Match[str], bool]] = []
    for scope in _VAULT_SCOPE_RE.finditer(command):
        before = command[: scope.start()]
        after = command[scope.end() :]
        precise_exclusion = _VAULT_SCOPE_EXCLUSION_TAIL_RE.search(before)
        broad_exclusion = _VAULT_SCOPE_BROAD_EXCLUSION_TAIL_RE.search(before)
        precise_exclusion_is_topical = bool(
            precise_exclusion is not None
            and _VAULT_SCOPE_TOPICAL_EXCLUSION_RE.search(
                before[: precise_exclusion.start()]
            )
        )
        broad_exclusion_is_topical = bool(
            broad_exclusion is not None
            and _VAULT_SCOPE_TOPICAL_EXCLUSION_RE.search(before[: broad_exclusion.start()])
        )
        private_after_negated_public = any(
            re.fullmatch(
                rf"\s*{_VAULT_SOURCE_SELECTION_ACTION_PATTERN}"
                rf"(?:\s+(?:in|inside|into|through|to|within))?\s*",
                before[public_exclusion.end() :],
            )
            for public_exclusion in _VAULT_NEGATED_PUBLIC_SOURCE_RE.finditer(before)
        )
        negated = bool(
            (
                precise_exclusion is not None
                and not precise_exclusion_is_topical
                and not private_after_negated_public
            )
            or _VAULT_SCOPE_NEGATED_CHOICE_TAIL_RE.search(before)
            or (
                broad_exclusion is not None
                and not broad_exclusion_is_topical
                and not private_after_negated_public
            )
            or (
                _VAULT_SCOPE_POST_EXCLUDED_RE.fullmatch(after)
                and (
                    _VAULT_SCOPE_WITH_RE.search(before)
                    or after.lstrip().startswith("is ")
                )
            )
            or (
                _VAULT_SCOPE_POST_OUT_ACTION_RE.search(before)
                and _VAULT_SCOPE_ALONE_RE.fullmatch(after)
            )
        )
        if result and result[-1][1]:
            bridge = command[result[-1][0].end() : scope.start()]
            negated = negated or bool(
                _VAULT_SCOPE_COORDINATION_RE.fullmatch(bridge)
            )
        result.append((scope, negated))
    return tuple(result)


def _has_public_source_after_scope(after_scope: str) -> bool:
    """Recognize only a source in the slot immediately after a vault marker."""

    tail = _VAULT_AFTER_SCOPE_RESPONSE_RE.sub("", after_scope.strip())
    source = re.match(
        rf"^(?:(?:{_VAULT_SOURCE_MODIFIER_PATTERN})\s+){{0,3}}"
        rf"{_VAULT_PUBLIC_SOURCE_PATTERN}\b",
        tail,
    )
    if source is None:
        return False
    remainder = tail[source.end() :].strip()
    if not remainder or _VAULT_SOURCE_MODIFIERS_RE.fullmatch(remainder):
        return True
    if _VAULT_TOPIC_JOIN_RE.match(remainder):
        return True
    topic_join = re.search(
        r"\b(?:about|concerning|discussing|for|on|regarding)\b",
        remainder,
    )
    if topic_join is not None and _VAULT_SOURCE_MODIFIERS_RE.fullmatch(
        remainder[: topic_join.start()].strip()
    ):
        return True
    return bool(
        re.fullmatch(
            rf"(?:in|inside|into|through|within)\s+"
            rf"(?:{_VAULT_SCOPE_PATTERN})(?![a-z0-9])",
            remainder,
        )
    )


def _first_positive_scope_cluster_end(
    command: str,
    polarities: tuple[tuple[re.Match[str], bool], ...],
) -> int | None:
    """Return the end of the first directly coordinated positive source group."""

    positive_index = next(
        (index for index, (_scope, negated) in enumerate(polarities) if not negated),
        None,
    )
    if positive_index is None:
        return None
    end = polarities[positive_index][0].end()
    for scope, negated in polarities[positive_index + 1 :]:
        if negated:
            break
        bridge = command[end : scope.start()]
        if _VAULT_POSITIVE_SCOPE_CLUSTER_RE.fullmatch(bridge) is None:
            break
        end = scope.end()
    return end


def _last_explicit_source_correction(
    command: str,
    polarities: tuple[tuple[re.Match[str], bool], ...],
) -> str | None:
    """Resolve contrastive source corrections in textual order."""

    events: list[tuple[int, str]] = []
    public_sources = tuple(_VAULT_PUBLIC_SOURCE_RE.finditer(command))
    positive_scopes = tuple(scope for scope, negated in polarities if not negated)
    for scope in positive_scopes:
        before = command[: scope.start()]
        after = command[scope.end() :]
        correction = _VAULT_PRIVATE_SOURCE_CORRECTION_TAIL_RE.search(before)
        has_correction_suffix = re.match(
            r"^\s+(?:instead|after\s+all)\b", after
        ) is not None
        terse_correction = bool(
            correction is not None
            and correction.group("cue") != "then"
            and correction.group("action") in {"go", "search", "use"}
        )
        if (
            correction is not None
            and (has_correction_suffix or terse_correction)
            and any(
                source.end() <= correction.start() for source in public_sources
            )
        ):
            events.append((scope.start(), "private"))
        for public_exclusion in _VAULT_NEGATED_PUBLIC_SOURCE_RE.finditer(before):
            between = before[public_exclusion.end() :]
            if (
                any(
                    source.end() <= public_exclusion.start()
                    for source in public_sources
                )
                and re.fullmatch(
                    rf"\s*{_VAULT_SOURCE_SELECTION_ACTION_PATTERN}"
                    rf"(?:\s+(?:in|inside|into|through|to|within))?\s*",
                    between,
                )
            ):
                events.append((scope.start(), "private"))
    for correction in _VAULT_PUBLIC_SOURCE_CORRECTION_RE.finditer(command):
        prior_scopes = tuple(
            scope for scope in positive_scopes if scope.end() <= correction.start()
        )
        literal_topic = bool(
            prior_scopes
            and _VAULT_LITERAL_TOPIC_AFTER_SCOPE_RE.match(
                command[prior_scopes[-1].end() : correction.start()]
            )
        )
        if prior_scopes and not literal_topic:
            events.append((correction.start(), "public"))
    for correction in _VAULT_NEGATED_PUBLIC_CORRECTION_RE.finditer(command):
        if any(scope.end() <= correction.start() for scope in positive_scopes):
            events.append((correction.start(), "private"))
    return max(events, default=(-1, ""), key=lambda event: event[0])[1] or None


def _has_public_source_at_request_head(command: str) -> bool:
    """Recognize a public head only when its following words fit source syntax."""

    source = _VAULT_PUBLIC_HEAD_RE.match(command)
    if source is None:
        return False
    remainder = command[source.end() :].strip()
    if not remainder:
        return True
    direct_join = re.compile(
        rf"^(?:about|for|to|what|where|using|with)\b|"
        rf"^(?:{_VAULT_SCOPE_PATTERN})(?![a-z0-9])"
    )
    if direct_join.match(remainder):
        return True
    join = re.search(
        rf"\b(?:about|for|to|what|where|using|with|"
        rf"{_VAULT_SCOPE_PATTERN})\b",
        remainder,
    )
    if join is not None and _VAULT_SOURCE_MODIFIERS_RE.fullmatch(
        remainder[: join.start()].strip()
    ):
        return True
    return bool(
        re.match(
            r"^(?:and|then)\s+(?:browse|check|find|list|look(?:\s+up)?|"
            r"query|read|research|search)\b",
            remainder,
        )
    )


def is_vault_public_source_request(text: str) -> bool:
    """Whether a personal-source request explicitly chooses web/online search."""

    normalized = normalize_text(text)
    command = _VAULT_COURTESY_RE.sub("", normalized)
    polarities = _vault_scope_polarities(command)
    if (
        not polarities
        or not _VAULT_SOURCE_ROUTABLE_REQUEST_RE.match(command)
        or _is_vault_non_lookup_request(normalized)
    ):
        return False
    positive_scopes = tuple(scope for scope, negated in polarities if not negated)
    public_negated = _VAULT_NEGATED_PUBLIC_SOURCE_RE.search(command) is not None
    correction = _last_explicit_source_correction(command, polarities)
    if correction is not None:
        return correction == "public"
    # A structurally selected public source wins before scanning later topic
    # text for negated web words ("web ... notes about avoiding the internet").
    if _has_public_source_at_request_head(command):
        return True
    if not positive_scopes:
        return bool(
            not public_negated and _VAULT_PUBLIC_SOURCE_RE.search(command)
        )
    if public_negated:
        return False
    cluster_end = _first_positive_scope_cluster_end(command, polarities)
    return bool(
        cluster_end is not None
        and _has_public_source_after_scope(command[cluster_end:])
    )


def is_vault_scoped_request(text: str) -> bool:
    """Whether a request names the user's configured notes vault.

    The personal-vault marker, not one fixed leading verb, owns the scope: voice
    phrasing such as ``search in``, ``go in``, ``find in``, or ``look through``
    therefore shares one deterministic local route. This stays metadata on the
    existing SEARCH intent rather than adding a new mode/enum to the portable
    brain contract.
    """

    normalized = normalize_text(text)
    command = _VAULT_COURTESY_RE.sub("", normalized)
    polarities = _vault_scope_polarities(command)
    if not any(not negated for _scope, negated in polarities):
        return False
    return not is_vault_public_source_request(normalized)


def _is_explicit_vault_lookup_request(text: str) -> bool:
    """Recognize an imperative read/list request for the private vault."""

    normalized = normalize_text(text)
    if not is_vault_scoped_request(normalized):
        return False
    command = _VAULT_COURTESY_RE.sub("", normalized)
    if _is_vault_non_lookup_request(normalized):
        return False
    command = _VAULT_LEADING_NEGATED_PUBLIC_PREAMBLE_RE.sub("", command)
    return bool(
        _VAULT_READ_IMPERATIVE_RE.match(command)
        or _VAULT_GO_IMPERATIVE_RE.match(command)
    )


def _is_conversational_vault_lookup_request(text: str) -> bool:
    """Recognize narrow questions that ask what the private vault contains."""

    normalized = normalize_text(text)
    question = _VAULT_COURTESY_RE.sub("", normalized)
    return (
        is_vault_scoped_request(normalized)
        and not _is_vault_non_lookup_request(normalized)
        and any(pattern.search(question) for pattern in _VAULT_CONVERSATIONAL_RES)
    )


def is_vault_lookup_request(text: str) -> bool:
    """Recognize a clear private-vault read without an LLM tool choice."""

    return (
        _is_explicit_vault_lookup_request(text)
        or _is_conversational_vault_lookup_request(text)
    )


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
    if vault_search_enabled and is_vault_lookup_request(normalized):
        return False
    if (
        normalized.startswith(("research ", "search "))
        and is_vault_scoped_request(normalized)
        and not is_vault_lookup_request(normalized)
    ):
        return True
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

        if self.policy.vault_search_enabled and _is_explicit_vault_lookup_request(text):
            # An explicit personal-vault read is a source-scoped command even
            # when the optional provider is present. Without it, the ordinary
            # PRIVATE assistant route remains available and cannot become web.
            return IntentDecision(
                IntentKind.SEARCH,
                0.95,
                observation.text,
                "vault_lookup_phrase",
                mode=Mode.SEARCH,
                metadata={"search_scope": "vault"},
            )

        if (
            current_mode == Mode.ASSISTANT
            and self.policy.vault_search_enabled
            and _is_conversational_vault_lookup_request(text)
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

        if (
            current_mode in (Mode.SEARCH, Mode.RESEARCH)
            and is_vault_scoped_request(text)
            and _is_vault_non_lookup_request(text)
        ):
            return IntentDecision(
                IntentKind.ASSISTANT,
                0.9,
                observation.text,
                "vault_non_lookup",
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
        if (
            normalized.startswith(("research ", "search "))
            and is_vault_scoped_request(original)
            and not is_vault_lookup_request(original)
        ):
            # A reserved search prefix must not turn a compound mutation or a
            # declaration about the private source into a vault read.
            return None
        if normalized.startswith(("research ", "cerceteaza ")):
            query = _after_first_word(original)
            metadata = (
                {"search_scope": "vault"}
                if is_vault_scoped_request(original)
                else {}
            )
            return IntentDecision(
                IntentKind.RESEARCH,
                0.95,
                original if metadata else query,
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
                if is_vault_scoped_request(original)
                else {}
            )
            return IntentDecision(
                IntentKind.SEARCH,
                0.95,
                original if metadata else query,
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
