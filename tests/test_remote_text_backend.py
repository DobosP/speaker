"""Headless contracts for the retained private Python text-inference pipe."""

import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from remote import text_backend as worker

ROOT = Path(__file__).resolve().parents[1]


def _main(args, raw):
    output, errors = io.BytesIO(), io.BytesIO()
    status = worker.main(args, stdin=io.BytesIO(raw), stdout=output, stderr=errors)
    return status, output.getvalue(), errors.getvalue()


def _config(tmp_path, contents=None):
    path = tmp_path / "synthetic-config.json"
    path.write_text(json.dumps({} if contents is None else contents), encoding="utf-8")
    return str(path)


def test_llamacpp_text_server_uses_bounded_thread_pair():
    # Migrated verbatim behavioral contract from test_token_server.py. Neither
    # constructor imports llama_cpp or loads the synthetic model path.
    from core.llm_threads import resolve_llamacpp_thread_pair

    llm = worker._make_llm({"llm": {"backend": "llamacpp", "main_model_path": "model.gguf"}})
    expected = resolve_llamacpp_thread_pair()
    assert (llm.n_threads, llm.n_threads_batch) == (
        expected.n_threads,
        expected.n_threads_batch,
    )


def test_llamacpp_text_server_preserves_explicit_batch_override():
    llm = worker._make_llm(
        {
            "llm": {
                "backend": "llamacpp",
                "main_model_path": "model.gguf",
                "n_threads": 2,
                "n_threads_batch": 3,
            }
        }
    )
    assert (llm.n_threads, llm.n_threads_batch) == (2, 3)


def test_missing_llamacpp_model_preserves_echo_fallback():
    llm = worker._make_llm({"llm": {"backend": "llamacpp"}})
    assert llm.generate("synthetic") == "You said: synthetic"


@pytest.mark.parametrize("backend", ["cloud", "groq", "openai", "echo", "unknown", "", None])
def test_unknown_backend_fails_closed(backend):
    with pytest.raises(ValueError):
        worker._make_llm({"llm": {"backend": backend}})


@pytest.mark.parametrize(
    "host",
    [
        "http://127.0.0.1:11434",
        "https://127.0.0.2:443/",
        "http://[::1]:11434",
        "http://localhost:11434",
    ],
)
def test_only_loopback_ollama_transport_is_constructed(monkeypatch, host):
    monkeypatch.setenv("OLLAMA_HOST", "https://public.invalid")
    llm = worker._make_llm(
        {"llm": {"host": host, "main_model": "synthetic", "options": {"num_predict": 7}, "keep_alive": 0}}
    )
    kwargs = llm._client_kwargs()
    assert kwargs["host"] == worker._local_ollama_host(host)
    assert kwargs["trust_env"] is False
    assert kwargs["follow_redirects"] is False
    assert llm.model == "synthetic"
    assert llm._options == {"num_predict": 7}
    assert llm._keep_alive == 0
    assert llm._client is None  # No SDK client, model, or socket during construction.


def test_ollama_default_is_pinned_loopback(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert worker._local_ollama_host() == "http://127.0.0.1:11434"


def test_effective_ollama_host_includes_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://[::1]:12000")
    assert worker._make_llm({})._host == "http://[::1]:12000"
    monkeypatch.setenv("OLLAMA_HOST", "http://192.168.1.10:11434")
    with pytest.raises(ValueError):
        worker._make_llm({})


@pytest.mark.parametrize(
    "host",
    [
        "http://192.168.1.2:11434",
        "https://public.invalid",
        "http://0.0.0.0:11434",
        "http://[::]:11434",
        "http://[::ffff:127.0.0.1]:11434",
        "http://127.0.0.1.public.invalid",
        "http://localhost.public.invalid",
        "http://localhost.:11434",
        "127.0.0.1:11434",
        "ftp://127.0.0.1:11434",
        "http://synthetic:credential@127.0.0.1:11434",
        "http://127.0.0.1:11434/?key=synthetic",
        "http://127.0.0.1:11434/#synthetic",
        "http://127.0.0.1:11434/api/chat",
        "http://127.0.0.1:65536",
        "http://127.0.0.1:0",
        "http://[::1%eth0]:11434",
        " http://127.0.0.1:11434",
        "http://127.0.0.1:11434\n",
        True,
    ],
)
def test_nonlocal_or_ambiguous_host_is_rejected(host):
    with pytest.raises(ValueError):
        worker._make_llm({"llm": {"host": host}})


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b" ",
        b"{",
        b"[]",
        b"null",
        b"{}",
        b'{"message":null}',
        b'{"message":true}',
        b'{"message":7}',
        b'{"message":[]}',
        b'{"message":{}}',
        b'{"message":"synthetic","extra":"value"}',
        b'{"message":"first","message":"second"}',
        b'{"message":"synthetic"} {"message":"second"}',
        b'{"message":"synthetic"} trailing',
        b'{"message":"\xff"}',
        b'{"message":"\\ud800"}',
    ],
)
def test_invalid_request_returns_only_fixed_failure(raw):
    assert _main(["--echo"], raw) == (1, b"", worker.ERROR_MESSAGE)


def test_one_request_trimmed_reply_and_utf8_protocol():
    raw = json.dumps({"message": "  synthetic română  "}, ensure_ascii=False).encode("utf-8")
    assert _main(["--echo"], raw) == (
        0,
        '{"reply":"You said: synthetic română"}\n'.encode("utf-8"),
        b"",
    )


def test_empty_message_never_constructs_backend():
    def forbidden_factory(_config):
        raise AssertionError("empty request constructed a backend")

    assert worker.handle_request(b'{"message":"  "}', {}, llm_factory=forbidden_factory) == b'{"reply":""}\n'


def test_request_cap_counts_utf8_bytes_and_reads_bounded_once():
    overhead = len(b'{"message":""}')
    exact = b'{"message":"' + b"a" * (worker.MAX_REQUEST_BYTES - overhead) + b'"}'
    assert _main(["--echo"], exact)[0] == 0

    class BoundedSource(io.BytesIO):
        calls = []

        def read(self, size=-1):
            self.calls.append(size)
            return super().read(size)

    source = BoundedSource(exact + b" " * 500)
    output, errors = io.BytesIO(), io.BytesIO()
    assert worker.main(["--echo"], stdin=source, stdout=output, stderr=errors) == 1
    assert source.calls == [worker.MAX_REQUEST_BYTES + 1]
    assert output.getvalue() == b""
    assert errors.getvalue() == worker.ERROR_MESSAGE
    multibyte = json.dumps({"message": "ă" * 9000}, ensure_ascii=False).encode("utf-8")
    assert _main(["--echo"], multibyte) == (1, b"", worker.ERROR_MESSAGE)


def test_exact_output_cap_includes_json_and_newline():
    overhead = len(b'{"reply":""}\n')
    reply = "a" * (worker.MAX_RESPONSE_BYTES - overhead)
    assert len(worker._encode_reply(reply)) == worker.MAX_RESPONSE_BYTES
    with pytest.raises(ValueError):
        worker._encode_reply(reply + "a")


@pytest.mark.parametrize("reply", [None, 7, [], "a" * (worker.MAX_RESPONSE_BYTES + 1), "ă" * 40000, "\x00" * 12000, "\ud800"])
def test_invalid_or_oversize_reply_fails_closed(reply):
    class FakeLLM:
        def generate(self, message):
            return reply

    with pytest.raises((ValueError, UnicodeError)):
        worker.handle_request(b'{"message":"synthetic"}', {}, llm_factory=lambda _config: FakeLLM())


def test_backend_exception_has_no_protocol_detail(monkeypatch, tmp_path):
    class FakeLLM:
        def generate(self, message):
            raise RuntimeError("synthetic provider credential/path detail")

    monkeypatch.setattr(worker, "_make_llm", lambda _config: FakeLLM())
    assert _main(["--config", _config(tmp_path)], b'{"message":"synthetic"}') == (
        1,
        b"",
        worker.ERROR_MESSAGE,
    )


def test_explicit_config_only_and_generic_config_failures(tmp_path):
    assert _main([], b'{"message":"synthetic"}') == (1, b"", worker.ERROR_MESSAGE)
    assert _main(["--unexpected-synthetic-argument"], b'{}') == (1, b"", worker.ERROR_MESSAGE)
    missing = str(tmp_path / "missing-synthetic-config.json")
    assert _main(["--config", missing], b'{"message":"synthetic"}') == (1, b"", worker.ERROR_MESSAGE)
    invalid = tmp_path / "invalid-synthetic-config.json"
    invalid.write_bytes(b"[]")
    assert _main(["--config", str(invalid)], b'{"message":"synthetic"}') == (1, b"", worker.ERROR_MESSAGE)
    invalid.write_bytes(b" " * (worker.MAX_CONFIG_BYTES + 1))
    assert _main(["--config", str(invalid)], b'{"message":"synthetic"}') == (1, b"", worker.ERROR_MESSAGE)


def test_main_forwards_only_text_and_trims_reply(monkeypatch, tmp_path):
    calls = []

    class FakeLLM:
        def generate(self, message):
            calls.append(message)
            return "  synthetic reply  "

    def factory(config):
        calls.append(config)
        return FakeLLM()

    monkeypatch.setattr(worker, "_make_llm", factory)
    config = {"llm": {"backend": "llamacpp", "main_model_path": "synthetic.gguf"}}
    assert _main(["--config", _config(tmp_path, config)], b'{"message":" synthetic "}') == (
        0,
        b'{"reply":"synthetic reply"}\n',
        b"",
    )
    assert calls == [config, "synthetic"]


def test_echo_cli_without_site_packages_or_any_core_import():
    script = (
        "import sys; from remote.text_backend import main; "
        "status = main(['--echo']); "
        "assert not any(m == 'core' or m.startswith('core.') for m in sys.modules); "
        "raise SystemExit(status)"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=ROOT,
        input=b'{"message":"synthetic"}',
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert (result.returncode, result.stdout, result.stderr) == (
        0,
        b'{"reply":"You said: synthetic"}\n',
        b"",
    )


def test_provider_python_and_native_output_cannot_corrupt_pipe(tmp_path):
    script = r"""
import os
import sys
from remote import text_backend as worker
class FakeLLM:
    def generate(self, message):
        print("synthetic Python provider diagnostic")
        print("synthetic stderr provider diagnostic", file=sys.stderr)
        os.write(1, b"synthetic native stdout diagnostic\n")
        os.write(2, b"synthetic native stderr diagnostic\n")
        return " synthetic reply "
def factory(config):
    print("synthetic constructor diagnostic")
    os.write(1, b"synthetic native constructor diagnostic\n")
    return FakeLLM()
worker._make_llm = factory
raise SystemExit(worker.main(["--config", sys.argv[1]]))
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, _config(tmp_path)],
        cwd=ROOT,
        input=b'{"message":"synthetic"}',
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert (result.returncode, result.stdout, result.stderr) == (
        0,
        b'{"reply":"synthetic reply"}\n',
        b"",
    )


@pytest.mark.parametrize("stage", ["factory", "generate"])
@pytest.mark.parametrize("failure", [SystemExit, KeyboardInterrupt])
def test_provider_base_exceptions_and_disposal_emit_only_fixed_failure(monkeypatch, tmp_path, capfd, stage, failure):
    def diagnostic():
        print("synthetic Python provider diagnostic")
        print("synthetic Python stderr diagnostic", file=sys.stderr)
        os.write(1, b"synthetic native stdout diagnostic\n")
        os.write(2, b"synthetic native stderr diagnostic\n")

    class FakeLLM:
        def generate(self, message):
            diagnostic()
            raise failure("synthetic-provider-private-detail")

        def __del__(self):
            diagnostic()

    def factory(config):
        diagnostic()
        if stage == "factory":
            # The exception traceback also holds this disposable client.
            client = FakeLLM()
            raise failure("synthetic-provider-private-detail")
        return FakeLLM()

    monkeypatch.setattr(worker, "_make_llm", factory)
    assert _main(["--config", _config(tmp_path)], b'{"message":"synthetic"}') == (
        1,
        b"",
        worker.ERROR_MESSAGE,
    )
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""
