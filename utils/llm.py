"""
LLM module with memory integration and streaming support.

Supports:
- Local LLM via Ollama (batch and streaming)
- Multi-layer memory context
- Conversation history
- Streaming token generation for sentence-by-sentence TTS
"""
import os
import re
import time
from typing import Optional, List, Dict, Any, Generator, Callable
from dotenv import load_dotenv
import ollama

from utils.backend_trace import record_ollama

load_dotenv()


def _ollama_chat(*, trace_kind: str, purpose: str, model: str, **chat_kw):
    """Single logical ``ollama.chat`` call (streaming counts once via ``trace_kind``)."""
    record_ollama(trace_kind, model, purpose)
    return ollama.chat(model=model, **chat_kw)

LLM_GENERATION_PROFILES = {
    "edge": {"temperature": 0.2, "top_p": 0.8, "num_predict": 96},
    "balanced": {"temperature": 0.3, "top_p": 0.9, "num_predict": 160},
    "max_quality": {"temperature": 0.4, "top_p": 0.95, "num_predict": 256},
}


class LLM:
    """Base LLM class."""

    def get_response(
        self, text: str, context: str = None, history: List[Dict] = None
    ) -> str:
        raise NotImplementedError("Subclasses should implement this method")

    def get_streaming_response(
        self,
        text: str,
        context: str = None,
        history: List[Dict] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Yield response sentence-by-sentence as they become available."""
        if should_cancel and should_cancel():
            return
        full = self.get_response(text, context=context, history=history)
        yield full


class LocalLLM(LLM):
    """Local LLM using Ollama with optional streaming."""

    SYSTEM_PROMPT = """You are a helpful voice assistant with memory of past conversations.

IMPORTANT RULES:
- Keep responses SHORT: 1-2 sentences maximum (this is voice, not text)
- NO emojis, NO filler words like "Ah, great!", "I see!", etc.
- Be direct and helpful
- Use memory context below to answer questions about past conversations
- If asked about past conversations, reference the context directly

{context}"""

    def __init__(self, model: str = "llama2", generation_profile: str = "balanced"):
        self.model = model
        self.generation_profile = generation_profile
        self.options = LLM_GENERATION_PROFILES.get(
            generation_profile, LLM_GENERATION_PROFILES["balanced"]
        )
        try:
            ollama.list()
        except Exception:
            raise ConnectionError(
                "Ollama server not running. Please start it with 'ollama serve'"
            )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _build_messages(
        self,
        text: str,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """Build the messages list for the Ollama API."""
        messages = []

        # Filter out "Recent Conversation" from context (it's in history)
        context_text = ""
        if context:
            context_lines = context.split("\n")
            filtered_lines = []
            skip_section = False
            for line in context_lines:
                if "=== Recent Conversation ===" in line:
                    skip_section = True
                    continue
                elif line.startswith("===") and skip_section:
                    skip_section = False
                    filtered_lines.append(line)
                elif not skip_section:
                    filtered_lines.append(line)
            context_text = "\n".join(filtered_lines).strip()

        system_content = self.SYSTEM_PROMPT.format(
            context=(
                f"\n\nMemory Context:\n{context_text}" if context_text else ""
            )
        )
        messages.append({"role": "system", "content": system_content})

        if history:
            for msg in history[-8:]:
                messages.append(
                    {"role": msg["role"], "content": msg["content"]}
                )

        messages.append({"role": "user", "content": text})
        return messages

    def _postprocess(self, response_text: str) -> str:
        """Clean up verbose patterns, emojis, and enforce length limits."""
        verbose_patterns = [
            "Ah, ", "I see! ", "Great! ", "Of course! ", "Well, ",
            "You know, ", "Actually, ", "Basically, ", "So, ",
            "In our previous conversation, ", "As we discussed, ",
            "As I mentioned, ", "Let me tell you, ",
        ]
        for pattern in verbose_patterns:
            if response_text.startswith(pattern):
                response_text = response_text[len(pattern):].strip()

        response_text = self._EMOJI_RE.sub("", response_text)

        response_text = response_text.replace("...", ".")
        response_text = response_text.replace("!!", "!")
        response_text = response_text.replace("??", "?")

        # Limit length (~150 chars for voice)
        if len(response_text) > 150:
            sentences = response_text.split(". ")
            shortened = []
            total_len = 0
            for sent in sentences[:2]:
                sent = sent.strip()
                if not sent:
                    continue
                if total_len + len(sent) > 150:
                    break
                shortened.append(sent)
                total_len += len(sent) + 2
            if shortened:
                response_text = ". ".join(shortened)
                if not response_text.endswith((".", "!", "?")):
                    response_text += "."
            else:
                words = response_text.split()
                response_text = " ".join(words[:25])
                if not response_text.endswith((".", "!", "?")):
                    response_text += "..."

        return response_text.strip()

    @staticmethod
    def _is_stream_chunk_speakable(text: str) -> bool:
        """Drop punctuation-only / filler fragments so TTS does not speak '.' alone."""
        t = (text or "").strip()
        if not t:
            return False
        return bool(re.search(r"\w", t, flags=re.UNICODE))

    _EMOJI_RE = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )

    def _postprocess_stream_delta(self, fragment: str) -> str:
        """Per-chunk cleanup for Ollama deltas (no global length truncation)."""
        if not fragment:
            return ""
        return self._EMOJI_RE.sub("", fragment)

    # ── Batch response (existing) ────────────────────────────────────────

    def get_response(
        self,
        text: str,
        context: str = None,
        history: List[Dict[str, str]] = None,
    ) -> str:
        messages = self._build_messages(text, context, history)
        response = _ollama_chat(
            trace_kind="batch",
            purpose="main_turn",
            model=self.model,
            messages=messages,
            options=self.options,
        )
        return self._postprocess(response["message"]["content"].strip())

    # ── Streaming response (NEW) ─────────────────────────────────────────

    def _pop_stream_chunk(
        self,
        buffer: str,
        stream_mode: str,
        sentence_endings: re.Pattern,
        min_phrase_words: int,
        max_phrase_words: int,
    ) -> tuple[str, str]:
        """Pop one speakable chunk from buffer; return (chunk, remainder)."""
        if stream_mode == "sentence":
            match = sentence_endings.search(buffer)
            if match is None:
                return "", buffer
            end = match.start() + 1
            return buffer[:end].strip(), buffer[end:].strip()
        words = buffer.split()
        if len(words) < min_phrase_words:
            return "", buffer
        for m in re.finditer(r"[,;:\.!?]", buffer):
            end = m.end()
            prefix = buffer[:end].strip()
            if len(prefix.split()) >= min_phrase_words:
                return prefix, buffer[end:].lstrip()
        if len(words) >= max_phrase_words:
            return (
                " ".join(words[:max_phrase_words]),
                " ".join(words[max_phrase_words:]),
            )
        return "", buffer

    def _pop_word_stream_chunks(self, buffer: str) -> tuple[list[str], str]:
        """Split leading whitespace-delimited tokens; keep remainder."""
        emitted: list[str] = []
        rest = buffer
        while True:
            if not rest:
                break
            m = re.match(r"^(\S+\s+)", rest)
            if m:
                piece = m.group(1)
                emitted.append(piece)
                rest = rest[len(piece) :]
                continue
            if len(rest.split(None, 1)) == 1 and rest.strip():
                # trailing token without following space (handled on stream end)
                break
            break
        return emitted, rest

    def get_streaming_response(
        self,
        text: str,
        context: str = None,
        history: List[Dict[str, str]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
        stream_mode: str = "phrase",
        min_phrase_words: int = 6,
        max_phrase_words: int = 12,
        coalesce_tokens: bool = True,
        coalesce_min_words: int = 2,
        coalesce_max_words: int = 6,
        coalesce_flush_sec: float = 0.35,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """
        Stream the LLM response in speakable chunks.

        ``stream_mode``:
            - ``phrase``: yield ~6–12 words or up to the first strong punctuation.
            - ``sentence``: yield only after sentence-ending punctuation (legacy).
            - ``token``: yield each Ollama delta as soon as it arrives (fastest TTS).
              With ``coalesce_tokens`` (default True), buffers into ~min–max words.

            - ``word``: yield each whitespace-delimited word (smoother TTS than raw tokens).
        """
        messages = self._build_messages(text, context, history)
        stream = _ollama_chat(
            trace_kind="stream",
            purpose="main_turn",
            model=self.model,
            messages=messages,
            options=self.options,
            stream=True,
        )

        buffer = ""
        sentence_endings = re.compile(r"(?<=[.!?])\s")
        smode = (stream_mode or "phrase").strip().lower()
        if smode not in ("phrase", "sentence", "token", "word"):
            smode = "phrase"

        if smode == "token":
            if coalesce_tokens and coalesce_max_words >= 1:
                punct_re = re.compile(r"[,.!?;:]")
                buf = ""
                t_first: float | None = None
                for chunk in stream:
                    if should_cancel and should_cancel():
                        break
                    delta = chunk.get("message", {}).get("content", "")
                    if not delta:
                        continue
                    buf += self._postprocess_stream_delta(delta)
                    while buf.strip():
                        words = buf.split()
                        if not words:
                            buf = ""
                            t_first = None
                            break
                        now = time.monotonic()
                        if t_first is None:
                            t_first = now
                        elapsed = now - t_first
                        has_punct = punct_re.search(buf) is not None
                        nw = len(words)
                        if nw >= coalesce_max_words:
                            piece = " ".join(words[:coalesce_max_words])
                            buf = " ".join(words[coalesce_max_words:])
                            if self._is_stream_chunk_speakable(piece):
                                yield piece.strip()
                            t_first = time.monotonic() if buf.strip() else None
                            continue
                        if nw >= coalesce_min_words and (
                            has_punct or elapsed >= coalesce_flush_sec
                        ):
                            piece = buf.strip()
                            buf = ""
                            t_first = None
                            if self._is_stream_chunk_speakable(piece):
                                yield piece
                            break
                        break
                if buf.strip():
                    piece = buf.strip()
                    if self._is_stream_chunk_speakable(piece):
                        yield piece
                return

            for chunk in stream:
                if should_cancel and should_cancel():
                    break
                delta = chunk.get("message", {}).get("content", "")
                if not delta:
                    continue
                cleaned = self._postprocess_stream_delta(delta)
                if cleaned and self._is_stream_chunk_speakable(cleaned):
                    yield cleaned
            return

        if smode == "word":
            if coalesce_tokens and coalesce_max_words >= 1:
                punct_re = re.compile(r"[,.!?;:]")
                buf = ""
                t_first: float | None = None
                for chunk in stream:
                    if should_cancel and should_cancel():
                        break
                    delta = chunk.get("message", {}).get("content", "")
                    if not delta:
                        continue
                    buffer += delta
                    parts, buffer = self._pop_word_stream_chunks(buffer)
                    for piece in parts:
                        cleaned = self._postprocess_stream_delta(piece)
                        if not cleaned or not self._is_stream_chunk_speakable(cleaned):
                            continue
                        w = cleaned.rstrip()
                        buf = f"{buf} {w}".strip() if buf else w
                        while buf.strip():
                            words = buf.split()
                            if not words:
                                buf = ""
                                t_first = None
                                break
                            now = time.monotonic()
                            if t_first is None:
                                t_first = now
                            elapsed = now - t_first
                            has_punct = punct_re.search(buf) is not None
                            nw = len(words)
                            if nw >= coalesce_max_words:
                                piece_out = " ".join(words[:coalesce_max_words])
                                buf = " ".join(words[coalesce_max_words:])
                                if self._is_stream_chunk_speakable(piece_out):
                                    yield piece_out.strip()
                                t_first = time.monotonic() if buf.strip() else None
                                continue
                            if nw >= coalesce_min_words and (
                                has_punct or elapsed >= coalesce_flush_sec
                            ):
                                piece_out = buf.strip()
                                buf = ""
                                t_first = None
                                if self._is_stream_chunk_speakable(piece_out):
                                    yield piece_out
                                break
                            break
                if buffer.strip():
                    cleaned = self._postprocess_stream_delta(buffer.strip())
                    if cleaned and self._is_stream_chunk_speakable(cleaned):
                        w = cleaned.rstrip()
                        buf = f"{buf} {w}".strip() if buf else w
                if buf.strip():
                    piece_out = buf.strip()
                    if self._is_stream_chunk_speakable(piece_out):
                        yield piece_out
                return

            for chunk in stream:
                if should_cancel and should_cancel():
                    break
                delta = chunk.get("message", {}).get("content", "")
                if not delta:
                    continue
                buffer += delta
                parts, buffer = self._pop_word_stream_chunks(buffer)
                for piece in parts:
                    cleaned = self._postprocess_stream_delta(piece)
                    if cleaned and self._is_stream_chunk_speakable(cleaned):
                        yield cleaned.rstrip()
            if buffer.strip():
                cleaned = self._postprocess_stream_delta(buffer.strip())
                if cleaned and self._is_stream_chunk_speakable(cleaned):
                    yield cleaned
            return

        for chunk in stream:
            if should_cancel and should_cancel():
                break
            token = chunk.get("message", {}).get("content", "")
            if not token:
                continue
            buffer += token

            while True:
                emitted, buffer = self._pop_stream_chunk(
                    buffer,
                    smode,
                    sentence_endings,
                    min_phrase_words,
                    max_phrase_words,
                )
                if not emitted:
                    break
                cleaned = self._postprocess(emitted)
                if cleaned and self._is_stream_chunk_speakable(cleaned):
                    yield cleaned

        if buffer.strip():
            cleaned = self._postprocess(buffer.strip())
            if cleaned and self._is_stream_chunk_speakable(cleaned):
                yield cleaned

    # ── Conversation summarisation ───────────────────────────────────────

    def summarize_conversation(
        self, messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        conversation = "\n".join(
            [f"{m['role']}: {m['content']}" for m in messages]
        )
        prompt = f"""Summarize this conversation segment. Extract:
1. A brief summary (1-2 sentences)
2. Main topics discussed (list of keywords)
3. Any user preferences or personal information learned

Conversation:
{conversation}

Respond in this exact format:
SUMMARY: <your summary>
TOPICS: <comma-separated topics>
USER_INFO: <comma-separated user preferences/info, or "none">"""

        response = _ollama_chat(
            trace_kind="batch",
            purpose="summarize_conversation",
            model=self.model,
            options=self.options,
            messages=[{"role": "user", "content": prompt}],
        )

        result: Dict[str, Any] = {
            "summary": "",
            "topics": [],
            "user_preferences": [],
        }

        for line in response["message"]["content"].split("\n"):
            line = line.strip()
            if line.startswith("SUMMARY:"):
                result["summary"] = line[8:].strip()
            elif line.startswith("TOPICS:"):
                topics = line[7:].strip()
                result["topics"] = [
                    t.strip() for t in topics.split(",") if t.strip()
                ]
            elif line.startswith("USER_INFO:"):
                info = line[10:].strip()
                if info.lower() != "none":
                    result["user_preferences"] = [
                        i.strip() for i in info.split(",") if i.strip()
                    ]

        return result

    def extract_user_info(self, text: str) -> Dict[str, str]:
        prompt = f"""Extract any personal information from this message.
Look for: name, location, age, occupation, interests, preferences.

Message: "{text}"

If found, respond with key: value pairs, one per line.
If nothing found, respond with "NONE"."""

        response = _ollama_chat(
            trace_kind="batch",
            purpose="extract_user_info",
            model=self.model,
            options=self.options,
            messages=[{"role": "user", "content": prompt}],
        )

        result: Dict[str, str] = {}
        content = response["message"]["content"].strip()

        if content.upper() != "NONE":
            for line in content.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    result[key.strip().lower()] = value.strip()

        return result


def get_llm(
    llm_type: str = "local",
    model: str = "llama2",
    generation_profile: str = "balanced",
) -> LLM:
    if llm_type == "local":
        return LocalLLM(model=model, generation_profile=generation_profile)
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")
