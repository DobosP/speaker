"""Verified MiniCPM5 XML planning adapter for the in-process phone model.

The checkpoint's native grammar is useful, but it is not an execution API.
This module translates the registry's current single-string tools into schemas,
builds canonical call/result history, and parses one bounded decision.  The
ReAct controller remains the sole allowlist and execution authority.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Collection, Mapping, Sequence
from xml.parsers import expat

from always_on_agent.planner_steps import (
    PlannerCall,
    PlannerExchange,
    PlannerStep,
    PlannerTool,
)
from always_on_agent.untrusted import wrap_untrusted

from .llm import LLAMACPP_TOOL_FORMAT_MINICPM5, LlamaCppLLM


_MAX_XML_BYTES = 8192
_MAX_QUERY_CHARS = 128
_MAX_QUERY_BYTES = 2048
_MAX_FINAL_CHARS = 4096
_MAX_HISTORY_RESULT_CHARS = 384
_MAX_CURRENT_REQUEST_CHARS = 256
_MAX_RECENT_CHARS = 128
_MAX_TOOL_DESCRIPTION_CHARS = 96
_MAX_TOOL_COUNT = 4
_MAX_HISTORY_EXCHANGES = 1
_SAFE_TOOL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
_STRIPPED_XML_ATTR = re.compile(r"(?:^|\s)name\s*=\s*['\"]", re.IGNORECASE)
_XML_DECLARATION = re.compile(r"<\s*[!?]", re.IGNORECASE)
_PROTOCOL_STEMS = (
    "think",
    "function",
    "function_call",
    "function_calls",
    "param",
    "parameter",
    "parameters",
    "tool",
    "tools",
    "tool_call",
    "tool_calls",
    "tool_response",
    "tool_sep",
    "tool_def_sep",
    "argument",
    "arguments",
)
_TAG_WORD = re.compile(r"<\s*/?\s*([A-Za-z_|]+)", re.IGNORECASE)
_NEUTRALIZED_TAG_WORD = re.compile(r"\^\s*/?\s*([A-Za-z_|]+)", re.IGNORECASE)
_NEUTRALIZED_ENTITY_TAG_WORD = re.compile(
    r"(?is)\+(?:amp;)*(?:lt;|#0*60;|#x0*3c;)\s*/?\s*([A-Za-z_|]+)"
)
_SPECIAL_S_TAG = re.compile(r"<\s*/?\s*s\s*>", re.IGNORECASE)
_NEUTRALIZED_ENTITY_S_TAG = re.compile(
    r"(?is)\+(?:amp;)*(?:lt;|#0*60;|#x0*3c;)\s*/?\s*s\s*"
    r"\+(?:amp;)*(?:gt;|#0*62;|#x0*3e;)"
)
_SPACED_SPECIAL_TAG_PREFIX = re.compile(
    r"(?is)(?:<|\^|\+(?:amp;)*(?:lt;|#0*60;|#x0*3c;))\s*/?\s*\|\s*"
    r"(?:i|im|t|th|tho|thou|thoug|though|thought)\s*(?=[_|>~+]|$)"
)
_ENCODED_TAG = re.compile(
    r"(?is)(?:&(?:amp;)+(?:lt|gt);|&(?:lt|gt);|"
    r"&(?:amp;)+#(?:0*(?:60|62)|x0*(?:3c|3e));|"
    r"&#(?:0*60|0*62);|&#x0*(?:3c|3e);)",
)
_DIRECT_JSON_TOOL_PROTOCOL = re.compile(
    r"(?is)(?:[\"'](?:tool_calls?|function_calls?)[\"']\s*:"
    r"|\b(?:tool_calls?|function_calls?)\s*[:=])",
)
_JSON_NAME_KEY = re.compile(r"(?is)[\"']name[\"']\s*:")
_JSON_ARGUMENT_KEY = re.compile(
    r"(?is)[\"'](?:query|arguments|parameters)[\"']\s*:"
)
_BARE_NAME_KEY = re.compile(r"(?is)\bname\s*[:=]")
_BARE_ARGUMENT_KEY = re.compile(
    r"(?is)\b(?:query|arguments|parameters)\s*[:=]"
)
_NEUTRALIZED_PROTOCOL = re.compile(
    r"(?is)(?:\^\s*/?\s*(?:think|function|param|tools?|tool_(?:call|response|sep)|"
    r"arguments?|parameters?|s)(?=\s|~|$)[^~]*~"
    r"|\^\s*\|(?:im_|thought_)"
    r"|\^\^\^(?:END_)?UNTRUSTED::)",
)
_TEXT_PROTOCOL_DIRECTIVE = re.compile(
    r"(?m)^\s*(?:TOOL\s+[A-Za-z_][\w.-]*\s*:|"
    r"(?i:tool\s+[A-Za-z_][\w-]*(?:\.[\w.-]+)+\s*:|final\s*:|"
    r"(?:function|tool)_call\s*:))",
)
_NATIVE_SYSTEM = (
    "You are the planning loop of a local voice assistant. Decide exactly one "
    "next step. If a listed tool is needed, emit exactly one complete function "
    "XML object with exactly one query parameter and no surrounding prose. If "
    "no tool is needed, answer the user's request directly with no XML. Never "
    "invent a tool or call more than one. If the request is ambiguous, ask one "
    "short clarification instead of guessing."
)

_NATIVE_REMINDER = (
    "Your previous output was invalid. Return either exactly one bare "
    '<function name="tool"><param name="query">value</param></function> object '
    "using a listed tool, or a direct answer with no XML. Do not put prose "
    "before or after a function object."
)


class MiniCPMToolParseError(ValueError):
    """Non-PII parser failure; the raw model response is never included."""

    def __init__(self, code: str):
        super().__init__(f"invalid MiniCPM5 tool call: {code}")
        self.code = code


@dataclass(frozen=True, slots=True)
class MiniCPMToolCall:
    name: str
    query: str


def _reject(code: str) -> None:
    raise MiniCPMToolParseError(code)


def _is_protocol_tag_word(word: str) -> bool:
    """Reserve unambiguous native/control stems while preserving ``x<t``."""

    word = word.lower()
    if len(word) < 3:
        return False
    special_stems = ("|im_", "|thought_")
    if any(stem.startswith(word) or word.startswith(stem) for stem in special_stems):
        return True
    return any(stem.startswith(word) for stem in _PROTOCOL_STEMS)


def _has_protocol_tag_prefix(text: str) -> bool:
    """Recognize raw and history-neutralized known protocol tag prefixes."""

    for pattern in (_TAG_WORD, _NEUTRALIZED_TAG_WORD, _NEUTRALIZED_ENTITY_TAG_WORD):
        if any(_is_protocol_tag_word(match.group(1)) for match in pattern.finditer(text)):
            return True
    return bool(
        _SPECIAL_S_TAG.search(text)
        or _NEUTRALIZED_ENTITY_S_TAG.search(text)
        or _SPACED_SPECIAL_TAG_PREFIX.search(text)
    )


def parse_minicpm_tool_call(
    raw: str,
    *,
    allowed_tools: Collection[str],
) -> MiniCPMToolCall:
    """Parse exactly one strict MiniCPM function element.

    Expat is used as an event parser so comments, processing instructions,
    declarations, namespaces, DTDs, custom entities, nested tags, and text in
    structural positions can all be rejected explicitly.  Only predefined or
    numeric XML entities and CDATA inside the one ``query`` parameter survive.
    """

    if not isinstance(raw, str) or not raw or raw.startswith("\ufeff"):
        _reject("empty-or-bom")
    try:
        encoded = raw.encode("utf-8", errors="strict")
    except UnicodeError:
        _reject("invalid-unicode")
    if len(encoded) > _MAX_XML_BYTES:
        _reject("xml-too-large")

    allowed = frozenset(allowed_tools)
    stack: list[str] = []
    query_parts: list[str] = []
    query_chars = 0
    function_name = ""
    saw_param = False
    root_closed = False

    parser = expat.ParserCreate(namespace_separator="}")
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)

    def start(name: str, attrs: Mapping[str, str]) -> None:
        nonlocal function_name, saw_param
        if root_closed:
            _reject("multiple-roots")
        if not stack:
            if name != "function" or set(attrs) != {"name"}:
                _reject("invalid-function")
            candidate = attrs["name"]
            if candidate not in allowed or not _SAFE_TOOL_NAME.fullmatch(candidate):
                _reject("unknown-function")
            function_name = candidate
            stack.append(name)
            return
        if stack == ["function"]:
            if saw_param or name != "param" or dict(attrs) != {"name": "query"}:
                _reject("invalid-parameter")
            saw_param = True
            stack.append(name)
            return
        _reject("nested-element")

    def end(name: str) -> None:
        nonlocal root_closed
        if not stack or stack[-1] != name:
            _reject("mismatched-element")
        stack.pop()
        if not stack:
            root_closed = True

    def characters(data: str) -> None:
        nonlocal query_chars
        if not data:
            return
        if stack == ["function", "param"]:
            query_chars += len(data)
            if query_chars > _MAX_QUERY_CHARS:
                _reject("query-too-large")
            query_parts.append(data)
            return
        if data.strip():
            _reject("text-outside-query")

    def start_cdata() -> None:
        if stack != ["function", "param"]:
            _reject("cdata-outside-query")

    def forbidden(*_args) -> None:
        _reject("forbidden-xml-feature")

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.CharacterDataHandler = characters
    parser.StartCdataSectionHandler = start_cdata
    parser.XmlDeclHandler = forbidden
    parser.CommentHandler = forbidden
    parser.ProcessingInstructionHandler = forbidden
    parser.StartNamespaceDeclHandler = forbidden
    parser.StartDoctypeDeclHandler = forbidden
    parser.EntityDeclHandler = forbidden
    parser.UnparsedEntityDeclHandler = forbidden
    parser.ElementDeclHandler = forbidden
    parser.AttlistDeclHandler = forbidden
    parser.NotationDeclHandler = forbidden
    parser.SkippedEntityHandler = forbidden
    parser.ExternalEntityRefHandler = forbidden

    try:
        parser.Parse(encoded, True)
    except MiniCPMToolParseError:
        raise
    except (expat.ExpatError, ValueError, TypeError):
        _reject("malformed-xml")

    if stack or not root_closed or not saw_param or not function_name:
        _reject("incomplete-call")
    query = "".join(query_parts).strip()
    if not query:
        _reject("empty-query")
    try:
        if len(query.encode("utf-8", errors="strict")) > _MAX_QUERY_BYTES:
            _reject("query-too-large")
    except UnicodeError:
        _reject("invalid-query")
    return MiniCPMToolCall(function_name, query)


def _cancelled(cancel: object | None) -> bool:
    check = getattr(cancel, "is_set", None)
    if not callable(check):
        return False
    try:
        return bool(check())
    except Exception:
        return False


def _clean_text(value: str) -> str:
    """Remove tokenizer-hostile controls without altering ordinary Unicode."""

    out: list[str] = []
    for char in str(value):
        codepoint = ord(char)
        if codepoint < 32 and char not in "\t\n\r":
            out.append("\ufffd")
        elif 0xD800 <= codepoint <= 0xDFFF:
            out.append("\ufffd")
        else:
            out.append(char)
    return "".join(out)


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = " …[truncated]"
    return value[: max(0, limit - len(marker))] + marker


def _clip_tail(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = "[truncated]… "
    if limit <= len(marker):
        return marker[:limit]
    return marker + value[-(limit - len(marker)):]


def _history_query(query: str) -> str:
    # OpenBMB does not define an encoding for CDATA's own terminator and
    # tokenizes protocol tags even inside values. Neutralize structural ASCII
    # before canonical replay without expanding the phone context budget.
    return _clip(_neutralize_xml(query), _MAX_QUERY_CHARS)


def _history_result(result: str, limit: int, *, untrusted: bool) -> str:
    # The official template inserts tool-result strings raw between
    # <tool_response> tags. Neutralize all structural characters so an
    # attacker-controlled web result cannot close that envelope or synthesize a
    # later function call. Stable-width neutralization keeps the post-encoding
    # cap honest. The existing spotlighting directive remains intact.
    bounded = (
        wrap_untrusted(
            result,
            source="web",
            compact=True,
            max_chars=limit,
        )
        if untrusted
        else _clip(result, limit)
    )
    return _neutralize_xml(_clip(bounded, limit))


_XML_NEUTRALIZE = str.maketrans({"<": "^", ">": "~", "&": "+"})


def _neutralize_xml(value: str) -> str:
    """Stable-width ASCII encoding for model-controlled structural text."""

    return _clean_text(value).translate(_XML_NEUTRALIZE)


class MiniCPMXmlPlannerBackend:
    """Stateless typed backend bound to one verified local llama.cpp client."""

    name = LLAMACPP_TOOL_FORMAT_MINICPM5

    def __init__(self, llm: LlamaCppLLM):
        if llm.tool_format != LLAMACPP_TOOL_FORMAT_MINICPM5:
            raise ValueError("LlamaCppLLM did not opt into MiniCPM5 native tools")
        self._llm = llm

    @staticmethod
    def _schemas(tools: Sequence[PlannerTool]) -> list[dict[str, object]]:
        schemas: list[dict[str, object]] = []
        for tool in tools[:_MAX_TOOL_COUNT]:
            if not _SAFE_TOOL_NAME.fullmatch(tool.name):
                continue
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": _clip(
                            _neutralize_xml(tool.description),
                            _MAX_TOOL_DESCRIPTION_CHARS,
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The complete query for this tool.",
                                }
                            },
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                    },
                }
            )
        return schemas

    @staticmethod
    def _messages(
        query: str,
        recent: str,
        exchanges: Sequence[PlannerExchange],
        reminder: bool,
    ) -> list[dict[str, object]]:
        current = _clip(
            _neutralize_xml(query),
            _MAX_CURRENT_REQUEST_CHARS,
        )
        bounded_recent = _clip_tail(
            _neutralize_xml(recent),
            _MAX_RECENT_CHARS,
        )
        user = f"Recent context: {bounded_recent}\n" if bounded_recent else ""
        user += f"User request: {current}\nDecide the next step."
        messages: list[dict[str, object]] = [
            {"role": "system", "content": _NATIVE_SYSTEM},
            {"role": "user", "content": user},
        ]
        bounded_exchanges = exchanges[-_MAX_HISTORY_EXCHANGES:]
        result_limit = _MAX_HISTORY_RESULT_CHARS
        for exchange in bounded_exchanges:
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": exchange.call.name,
                                # The official template requires a mapping and
                                # calls .items(); an OpenAI JSON string fails.
                                "arguments": {
                                    "query": _history_query(exchange.call.query)
                                },
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "content": _history_result(
                        exchange.result,
                        result_limit,
                        untrusted=exchange.untrusted,
                    ),
                }
            )
        if reminder:
            messages.append({"role": "user", "content": _NATIVE_REMINDER})
        return messages

    def next_step(
        self,
        *,
        query: str,
        recent: str,
        tools: Sequence[PlannerTool],
        exchanges: Sequence[PlannerExchange],
        reminder: bool,
        cancel,
        first_token_hook,
    ) -> PlannerStep:
        if _cancelled(cancel):
            return PlannerStep(malformed=True)
        schemas = self._schemas(tools)
        allowed = frozenset(
            str(schema["function"]["name"])  # type: ignore[index]
            for schema in schemas
        )
        completion = self._llm.complete_minicpm_tool_chat(
            messages=self._messages(query, recent, exchanges, reminder),
            tools=schemas,
            first_token_hook=first_token_hook,
            cancel_event=cancel,
        )
        if _cancelled(cancel):
            return PlannerStep(malformed=True)
        raw = completion.text.strip()
        if completion.finish_reason != "stop" or not raw:
            return PlannerStep(malformed=True)

        lowered = raw.lower()
        has_function_grammar = any(
            marker in lowered
            for marker in ("<function", "</function", "<param", "</param")
        )
        if has_function_grammar:
            try:
                call = parse_minicpm_tool_call(raw, allowed_tools=allowed)
            except MiniCPMToolParseError:
                return PlannerStep(malformed=True)
            return PlannerStep(call=PlannerCall(call.name, call.query))

        final = self.validate_final(raw)
        if final is None:
            return PlannerStep(malformed=True)
        return PlannerStep(final=final)

    @staticmethod
    def validate_final(text: str) -> str | None:
        """Keep raw/stripped MiniCPM tool grammar out of the spoken seam."""

        final = str(text or "").strip()
        lowered = final.lower()
        if (
            not final
            or len(final) > _MAX_FINAL_CHARS
            or _STRIPPED_XML_ATTR.search(final)
            or _XML_DECLARATION.search(final)
            or _has_protocol_tag_prefix(final)
            or _ENCODED_TAG.search(final)
            or _DIRECT_JSON_TOOL_PROTOCOL.search(final)
            or (
                (_JSON_NAME_KEY.search(final) or _BARE_NAME_KEY.search(final))
                and (
                    _JSON_ARGUMENT_KEY.search(final)
                    or _BARE_ARGUMENT_KEY.search(final)
                )
            )
            or "&lt;" in lowered
            or "&gt;" in lowered
            or any(char in final for char in "＜＞＆")
            or _NEUTRALIZED_PROTOCOL.search(final)
            or _TEXT_PROTOCOL_DIRECTIVE.search(final)
        ):
            return None
        return final


def build_minicpm_planner_backend(llm) -> MiniCPMXmlPlannerBackend | None:
    """Return the local native backend only for an explicit model opt-in."""

    local = getattr(llm, "local_main", llm)
    if (
        isinstance(local, LlamaCppLLM)
        and local.tool_format == LLAMACPP_TOOL_FORMAT_MINICPM5
        and local._think is False
    ):
        return MiniCPMXmlPlannerBackend(local)
    return None
