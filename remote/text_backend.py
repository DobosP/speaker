"""Private, one-turn text inference pipe for the optional Go serving boundary.

This module owns no HTTP listener, audio, runtime, tools, or action authority.
``python -m remote.text_backend --config config.json`` accepts exactly one
bounded UTF-8 JSON object on stdin and emits exactly one bounded JSON reply.
The real backend remains the local Python ``core.llm`` implementation. The
explicit ``--echo`` switch is a stdlib-only synthetic qualification worker.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import ipaddress
import json
import logging
import os
from pathlib import Path
import sys
from typing import BinaryIO, Callable, Iterator
from urllib.parse import urlsplit

MAX_REQUEST_BYTES = 16 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
MAX_CONFIG_BYTES = 1024 * 1024
ERROR_MESSAGE = b"text backend error\n"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _decode_object(raw: bytes) -> dict:
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def _local_ollama_host(configured: object = None) -> str:
    """Resolve the SDK's effective host and pin it to an HTTP loopback origin."""
    if configured is not None and not isinstance(configured, str):
        raise ValueError("invalid Ollama host")
    host = configured or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
    if any(ord(char) <= 32 or ord(char) == 127 for char in host):
        raise ValueError("invalid Ollama host")
    url = urlsplit(host)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
        or url.path not in {"", "/"}
        or "%" in url.hostname
    ):
        raise ValueError("invalid Ollama host")
    # Pin the localhost spelling rather than trusting DNS/hosts resolution.
    hostname = url.hostname
    if hostname.lower() == "localhost":
        address = ipaddress.ip_address("127.0.0.1")
    else:
        address = ipaddress.ip_address(hostname)
    if not address.is_loopback:
        raise ValueError("Ollama must be loopback")
    port = url.port
    if port is not None and port < 1:
        raise ValueError("invalid Ollama port")
    literal = f"[{address}]" if address.version == 6 else str(address)
    return f"{url.scheme}://{literal}" + (f":{port}" if port is not None else "")


def _make_llm(config: dict):
    """Build the retained local core LLM without a cloud/action/runtime factory."""
    llm_cfg = config.get("llm") or {}
    if not isinstance(llm_cfg, dict):
        raise ValueError("invalid LLM config")
    backend = llm_cfg.get("backend", "ollama")
    if backend not in {"ollama", "llamacpp"}:
        raise ValueError("unsupported text backend")
    # Validate the endpoint before importing the retained Python core.
    host = _local_ollama_host(llm_cfg.get("host")) if backend == "ollama" else None
    from core.llm import EchoLLM, LlamaCppLLM, OllamaLLM

    if backend == "llamacpp":
        path = llm_cfg.get("main_model_path")
        if not path:
            # Preserve the legacy factory's explicit missing-model fallback.
            return EchoLLM()
        return LlamaCppLLM(
            path,
            n_ctx=llm_cfg.get("n_ctx", 4096),
            n_threads=llm_cfg.get("n_threads"),
            n_threads_batch=llm_cfg.get("n_threads_batch"),
            n_gpu_layers=llm_cfg.get("n_gpu_layers", 0),
            chat_format=llm_cfg.get("chat_format"),
            options=llm_cfg.get("options"),
        )

    class LocalOllamaLLM(OllamaLLM):
        def _client_kwargs(self) -> dict:
            # Both sync and async core paths use this hook. The one-turn worker
            # uses generate(), and no environment proxy or redirect may expand
            # the configured local origin into a remote text transport.
            return {
                **super()._client_kwargs(),
                "trust_env": False,
                "follow_redirects": False,
            }

    model = llm_cfg.get("main_model") or config.get("llm_model", "gemma3:12b")
    return LocalOllamaLLM(
        model=model,
        host=host,
        options=llm_cfg.get("options"),
        keep_alive=llm_cfg.get("keep_alive"),
    )


@contextmanager
def _discard_provider_output() -> Iterator[None]:
    """Keep Python prints, logging and native fd writes out of the protocol.

    The worker is an isolated one-turn process. Redirecting process-wide fds
    here is deliberately limited to provider construction and generation.
    """
    saved = []
    previous_logging_level = logging.root.manager.disable
    with open(os.devnull, "w", encoding="utf-8") as sink:
        try:
            for descriptor in (1, 2):
                saved.append((descriptor, os.dup(descriptor)))
                os.dup2(sink.fileno(), descriptor)
            logging.disable(logging.CRITICAL)
            with redirect_stdout(sink), redirect_stderr(sink):
                yield
        finally:
            logging.disable(previous_logging_level)
            for descriptor, original in reversed(saved):
                os.dup2(original, descriptor)
                os.close(original)


def _encode_reply(reply: object) -> bytes:
    if not isinstance(reply, str):
        raise ValueError("invalid text backend reply")
    # Reject before building an unbounded serialized copy. The exact UTF-8 JSON
    # cap below also accounts for multibyte characters and JSON escaping.
    if len(reply) > MAX_RESPONSE_BYTES:
        raise ValueError("text backend reply too large")
    payload = (
        json.dumps({"reply": reply.strip()}, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ValueError("text backend reply too large")
    return payload


def handle_request(
    raw: bytes,
    config: dict,
    *,
    echo: bool = False,
    llm_factory: Callable | None = None,
) -> bytes:
    """Validate and perform one local text turn, without emitting diagnostics."""
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("text request too large")
    request = _decode_object(raw)
    if set(request) != {"message"} or not isinstance(request["message"], str):
        raise ValueError("invalid text request")
    message = request["message"].strip()
    # Reject invalid escaped Unicode before forwarding it to a provider.
    message.encode("utf-8")
    if not message:
        return _encode_reply("")
    if echo:
        return _encode_reply(f"You said: {message}")
    failed = False
    with _discard_provider_output():
        llm = None
        try:
            llm = (llm_factory or _make_llm)(config)
            reply = llm.generate(message)
        except BaseException:
            # Provider SystemExit/KeyboardInterrupt must not bypass the fixed
            # machine error. Dispose of the provider exception while its output
            # is still suppressed, so traceback-held clients/destructors cannot
            # write diagnostic bytes after stdout/stderr have been restored.
            failed = True
        finally:
            llm = None
    if failed:
        raise RuntimeError("text backend error") from None
    return _encode_reply(reply)


def _load_config(path: str) -> dict:
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_CONFIG_BYTES + 1)
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError("config too large")
    return _decode_object(raw)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        # argparse's default error repeats user-provided arguments/paths.
        raise ValueError("invalid worker arguments")


def main(
    argv: list[str] | None = None,
    *,
    stdin: BinaryIO | None = None,
    stdout: BinaryIO | None = None,
    stderr: BinaryIO | None = None,
) -> int:
    parser = _ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="explicit local JSON config path")
    parser.add_argument("--echo", action="store_true", help="synthetic qualification only")
    source = stdin if stdin is not None else sys.stdin.buffer
    target = stdout if stdout is not None else sys.stdout.buffer
    errors = stderr if stderr is not None else sys.stderr.buffer
    try:
        args = parser.parse_args(argv)
        if not args.echo and not args.config:
            raise ValueError("explicit config required")
        # No model/core import or configuration is needed for synthetic echo.
        config = {} if args.echo else _load_config(args.config)
        raw = source.read(MAX_REQUEST_BYTES + 1)
        payload = handle_request(raw, config, echo=args.echo)
        target.write(payload)
        target.flush()
        return 0
    except Exception:
        errors.write(ERROR_MESSAGE)
        errors.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
