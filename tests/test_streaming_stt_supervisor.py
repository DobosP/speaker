from __future__ import annotations

import os
from pathlib import Path
import threading
import time

import pytest

from tests.streaming_stt_helpers import (
    scripted_case,
    stage_test_source_bundle,
    write_fixture,
)
from tools.streaming_stt.corpus import load_corpus
from tools.streaming_stt.manifest import load_worker_manifest
from tools.streaming_stt.protocol import (
    MAX_STDERR_BYTES,
    PcmInput,
    StreamConfig,
    TranscribeRequest,
)
from tools.streaming_stt.supervisor import StreamingWorker, WorkerError


def _worker(
    manifest_path: Path,
    scratch: Path,
    **kwargs,
) -> StreamingWorker:
    manifest = load_worker_manifest(manifest_path)
    bundle = stage_test_source_bundle(scratch, manifest.worker.path)
    return StreamingWorker(manifest, scratch, bundle, **kwargs)


def _request(
    scratch: Path,
    corpus_path: Path,
    digest: str,
    *,
    request_id: str = "case-0",
) -> TranscribeRequest:
    corpus = load_corpus(corpus_path)
    source = corpus.cases[0]
    path = scratch / "case.f32le"
    path.write_bytes(source.audio_bytes)
    path.chmod(0o600)
    return TranscribeRequest(
        request_id=request_id,
        pcm=PcmInput(
            path=path.resolve(),
            sha256=digest,
            samples=source.samples,
        ),
        stream=StreamConfig(
            chunk_samples=2,
            pace="burst",
            partial_interval_ms=100,
            tail_padding_samples=1,
        ),
    )


def _fixture(
    tmp_path: Path,
    *,
    hang_sec: float = 0.0,
    case_timeout_sec: float = 2.0,
) -> tuple[Path, Path, str]:
    manifest, corpus, digests = write_fixture(
        tmp_path,
        [
            {
                "values": [0.0, 0.1, -0.1, 0.0],
                "expected_text": "stop now",
                "commands": ["stop"],
            }
        ],
        [
            scripted_case(
                partials=[
                    (2, "stop", 50.0, 1.0),
                    (5, "stop now", 100.0, 1.5),
                ],
                final="stop now",
                elapsed_ms=150.0,
                hang_sec=hang_sec,
            )
        ],
        case_timeout_sec=case_timeout_sec,
    )
    return manifest, corpus, digests[0]


def test_real_worker_is_a_new_process_session_and_closes_cleanly(tmp_path):
    manifest_path, corpus_path, digest = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _worker(manifest_path, scratch)

    ready = worker.start()
    pid = worker.pid
    assert pid is not None
    if os.name == "posix":
        assert os.getpgid(pid) == pid
    trace = worker.transcribe(_request(scratch, corpus_path, digest))
    worker.close()

    assert ready.adapter == "fake-json-v1"
    assert [partial.seq for partial in trace.partials] == [0, 1]
    assert trace.final.seq == 2
    assert trace.partials[-1].samples_seen == trace.final.samples_seen == 5
    assert trace.final.chunks == 3
    assert worker._process is not None  # noqa: SLF001 - lifecycle proof
    assert worker._process.poll() == 0  # noqa: SLF001 - lifecycle proof


def test_worker_launch_argv_is_exact_isolated_and_bytecode_free(tmp_path):
    manifest_path, _, _ = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    captured: dict[str, object] = {}

    def rejecting_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        raise OSError

    worker = _worker(
        manifest_path,
        scratch,
        popen_factory=rejecting_popen,
    )
    with pytest.raises(WorkerError, match="worker_start"):
        worker.start()

    manifest = worker.manifest
    bundle_fd = worker._bundle_descriptor  # noqa: SLF001 - exact launch proof
    assert bundle_fd == -1
    command = captured["command"]
    assert isinstance(command, list)
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    bundle_fd, worker_fd, python_fd = kwargs["pass_fds"]
    assert command == [
        str(manifest.python.path),
        "-I",
        "-B",
        f"/proc/self/fd/{worker_fd}",
        "--manifest",
        str(manifest.path),
        "--scratch-root",
        str(scratch.resolve()),
        "--source-bundle-manifest",
        f"/proc/self/fd/{bundle_fd}/source-bundle.json",
        "--source-bundle-sha256",
        worker.source_bundle.tree_sha256,
        "--source-bundle-fd",
        str(bundle_fd),
    ]
    assert str(manifest.worker.path) not in command
    assert kwargs["cwd"] == str(scratch.resolve())
    assert kwargs["executable"] == f"/proc/self/fd/{python_fd}"
    assert kwargs["start_new_session"] is True
    assert kwargs["close_fds"] is True
    assert len(kwargs["pass_fds"]) == 3
    assert "shell" not in kwargs


@pytest.mark.skipif(os.name != "posix", reason="process-group proof is POSIX")
@pytest.mark.parametrize("failed_start", [1, 2])
def test_drainer_start_failure_reaps_worker_and_closes_launch_descriptors(
    tmp_path,
    monkeypatch,
    failed_start,
):
    manifest_path, _, _ = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _worker(manifest_path, scratch)
    original_start = threading.Thread.start
    starts = 0

    def fail_selected_start(thread):
        nonlocal starts
        starts += 1
        if starts == failed_start:
            raise RuntimeError
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_selected_start)
    monkeypatch.setattr("tools.streaming_stt.supervisor._TERM_GRACE_SEC", 0.1)
    monkeypatch.setattr("tools.streaming_stt.supervisor._KILL_GRACE_SEC", 0.5)

    with pytest.raises(WorkerError, match="worker_start"):
        worker.start()

    pid = worker.pid
    assert pid is not None
    assert worker._process is not None  # noqa: SLF001 - cleanup proof
    assert worker._process.poll() is not None  # noqa: SLF001 - cleanup proof
    with pytest.raises(ProcessLookupError):
        os.killpg(pid, 0)
    assert worker._stdout_thread is not None  # noqa: SLF001 - cleanup proof
    assert not worker._stdout_thread.is_alive()  # noqa: SLF001 - cleanup proof
    if worker._stderr_thread is not None:  # noqa: SLF001 - cleanup proof
        assert not worker._stderr_thread.is_alive()  # noqa: SLF001
    assert (worker._stderr_thread is None) is (failed_start == 1)  # noqa: SLF001
    assert worker._bundle_descriptor == -1  # noqa: SLF001 - cleanup proof
    assert worker._worker_descriptor == -1  # noqa: SLF001 - cleanup proof
    assert worker._python_descriptor == -1  # noqa: SLF001 - cleanup proof
    assert worker._process.stdin is not None  # noqa: SLF001 - cleanup proof
    assert worker._process.stdout is not None  # noqa: SLF001 - cleanup proof
    assert worker._process.stderr is not None  # noqa: SLF001 - cleanup proof
    assert worker._process.stdin.closed  # noqa: SLF001 - cleanup proof
    assert worker._process.stdout.closed  # noqa: SLF001 - cleanup proof
    assert worker._process.stderr.closed  # noqa: SLF001 - cleanup proof


def test_pipe_cleanup_skips_output_with_live_drainer(tmp_path):
    manifest_path, _, _ = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _worker(manifest_path, scratch)

    class CloseProbe:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class ProcessProbe:
        def __init__(self):
            self.stdin = CloseProbe()
            self.stdout = CloseProbe()
            self.stderr = CloseProbe()

    release = threading.Event()
    started = threading.Event()

    def live_drainer():
        started.set()
        release.wait()

    process = ProcessProbe()
    drainer = threading.Thread(target=live_drainer, daemon=True)
    drainer.start()
    assert started.wait(timeout=1.0)
    worker._process = process  # type: ignore[assignment]  # noqa: SLF001
    worker._stdout_thread = drainer  # noqa: SLF001 - bounded close seam

    try:
        worker._close_process_pipes()  # noqa: SLF001 - bounded close seam

        assert process.stdin.closed
        assert not process.stdout.closed
        assert process.stderr.closed
    finally:
        release.set()
    drainer.join(timeout=1.0)
    assert not drainer.is_alive()

    worker._close_process_pipes()  # noqa: SLF001 - stopped drainer closes

    assert process.stdout.closed


def test_worker_rejects_group_readable_scratch_before_launch(tmp_path):
    manifest_path, _, _ = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    scratch.chmod(0o750)
    worker = _worker(manifest_path, scratch)

    with pytest.raises(WorkerError, match="worker_prerequisite"):
        worker.start()

    assert worker.pid is None


def test_case_timeout_terminates_and_reaps_the_worker_group(tmp_path, monkeypatch):
    manifest_path, corpus_path, digest = _fixture(
        tmp_path,
        hang_sec=5.0,
        case_timeout_sec=0.1,
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _worker(manifest_path, scratch)
    worker.start()
    pid = worker.pid
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._TERM_GRACE_SEC",
        0.2,
    )
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._KILL_GRACE_SEC",
        0.5,
    )

    with pytest.raises(WorkerError, match="worker_timeout"):
        worker.transcribe(_request(scratch, corpus_path, digest))

    assert pid is not None
    assert worker._process is not None  # noqa: SLF001 - lifecycle proof
    assert worker._process.poll() is not None  # noqa: SLF001 - lifecycle proof
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.killpg(pid, 0)


@pytest.mark.skipif(os.name != "posix", reason="process-group proof is POSIX")
def test_timeout_kills_pipe_holding_descendant_before_joining_drainers(
    tmp_path,
    monkeypatch,
):
    worker_script = tmp_path / "pipe-holder-worker.py"
    worker_script.write_text(
        "import hashlib,json,platform,subprocess,sys,time\n"
        "from pathlib import Path\n"
        "manifest_path=Path(sys.argv[sys.argv.index('--manifest')+1])\n"
        "scratch=Path(sys.argv[sys.argv.index('--scratch-root')+1])\n"
        "raw=manifest_path.read_bytes()\n"
        "manifest=json.loads(raw)\n"
        "ready={'v':1,'type':'ready','model_id':manifest['model_id'],"
        "'manifest_sha256':hashlib.sha256(raw).hexdigest(),"
        "'source_bundle_sha256':"
        "sys.argv[sys.argv.index('--source-bundle-sha256')+1],"
        "'adapter':manifest['adapter'],'model_load_ms':0.0,"
        "'resources':{'rss_mb':1.0,'threads':1,'vram_mb':None},"
        "'runtime':{'python':platform.python_version(),'platform':sys.platform}}\n"
        "print(json.dumps(ready),flush=True)\n"
        "request=json.loads(sys.stdin.readline())\n"
        "marker=scratch/'descendant-ready'\n"
        'code="import signal,sys,time;'
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "open(sys.argv[1],'w').write('ready');time.sleep(60)\"\n"
        "child=subprocess.Popen([sys.executable,'-I','-B','-c',code,str(marker)])\n"
        "deadline=time.monotonic()+2.0\n"
        "while not marker.exists() and time.monotonic()<deadline:time.sleep(0.005)\n"
        "(scratch/'descendant-pid').write_text(str(child.pid))\n"
        "partial={'v':1,'type':'partial','id':request['id'],'seq':0,'text':'',"
        "'samples_seen':0,'elapsed_ms':0.0,'decode_ms':0.0}\n"
        "print(json.dumps(partial),flush=True)\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    manifest_path, corpus_path, digests = write_fixture(
        tmp_path,
        [{"values": [0.0], "expected_text": "x"}],
        [scripted_case(partials=[], final="x")],
        case_timeout_sec=0.3,
        worker_path=worker_script,
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _worker(manifest_path, scratch)
    worker.start()
    pid = worker.pid
    monkeypatch.setattr("tools.streaming_stt.supervisor._TERM_GRACE_SEC", 0.1)
    monkeypatch.setattr("tools.streaming_stt.supervisor._KILL_GRACE_SEC", 1.0)

    with pytest.raises(WorkerError, match="worker_timeout"):
        worker.transcribe(_request(scratch, corpus_path, digests[0]))

    assert pid is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(pid, 0)
    assert worker._stdout_thread is not None  # noqa: SLF001 - lifecycle proof
    assert worker._stderr_thread is not None  # noqa: SLF001 - lifecycle proof
    assert not worker._stdout_thread.is_alive()  # noqa: SLF001
    assert not worker._stderr_thread.is_alive()  # noqa: SLF001


def test_second_request_is_rejected_while_one_case_is_active(tmp_path):
    manifest_path, corpus_path, digest = _fixture(
        tmp_path,
        hang_sec=1.0,
        case_timeout_sec=0.2,
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _worker(manifest_path, scratch)
    worker.start()
    request = _request(scratch, corpus_path, digest)
    failures: list[WorkerError] = []

    def first_call() -> None:
        try:
            worker.transcribe(request)
        except WorkerError as exc:
            failures.append(exc)

    first = threading.Thread(target=first_call)
    first.start()
    deadline = time.monotonic() + 1.0
    while not worker._active and time.monotonic() < deadline:  # noqa: SLF001
        time.sleep(0.005)

    with pytest.raises(WorkerError, match="worker_state"):
        worker.transcribe(request)

    first.join(timeout=2.0)
    assert not first.is_alive()
    assert failures and failures[0].code == "worker_timeout"


def test_supervisor_rejects_pcm_outside_private_scratch_without_sending(
    tmp_path,
):
    manifest_path, corpus_path, digest = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    corpus = load_corpus(corpus_path)
    request = TranscribeRequest(
        request_id="outside",
        pcm=PcmInput(
            path=corpus.cases[0].source_path,
            sha256=digest,
            samples=corpus.cases[0].samples,
        ),
        stream=StreamConfig(2, "burst", 100, 0),
    )

    with _worker(manifest_path, scratch) as worker:
        with pytest.raises(WorkerError) as failure:
            worker.transcribe(request)

    assert str(failure.value) == "worker_prerequisite"
    assert str(corpus.cases[0].source_path) not in str(failure.value)


def test_worker_artifact_change_after_start_is_detected_on_close(tmp_path):
    manifest_path, _, _ = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    manifest = load_worker_manifest(manifest_path)
    bundle = stage_test_source_bundle(scratch, manifest.worker.path)
    worker = StreamingWorker(manifest, scratch, bundle)
    worker.start()
    artifact = manifest.artifact_by_name["fake-script"].path
    artifact.write_bytes(artifact.read_bytes() + b"x")

    with pytest.raises(WorkerError, match="worker_artifact_changed"):
        worker.close()


def test_staged_source_mutation_is_rejected_before_launch(tmp_path):
    manifest_path, _, _ = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _worker(manifest_path, scratch)
    staged_adapter = (
        worker.source_bundle.root / "tools" / "streaming_stt" / "adapters" / "fake.py"
    )
    staged_adapter.chmod(0o600)
    staged_adapter.write_bytes(staged_adapter.read_bytes() + b"\n")

    with pytest.raises(WorkerError, match="worker_artifact_changed"):
        worker.start()

    assert worker.pid is None


@pytest.mark.skipif(os.name != "posix", reason="bounded process signal check is POSIX")
def test_dead_worker_is_detected_without_waiting_for_full_timeout(tmp_path):
    manifest_path, _, _ = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _worker(manifest_path, scratch)
    worker.start()
    assert worker.pid is not None
    os.kill(worker.pid, 9)
    started = time.monotonic()

    with pytest.raises(WorkerError):
        worker._next_event(2.0)  # noqa: SLF001 - bounded EOF detection seam

    assert time.monotonic() - started < 1.0
    worker.close()


@pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
def test_worker_output_caps_fail_closed_without_pipe_deadlock(
    tmp_path,
    stream_name,
):
    worker_script = tmp_path / "noisy-worker.py"
    target = "stdout" if stream_name == "stdout" else "stderr"
    size = "4 * 1024 * 1024 + 1" if target == "stdout" else "256 * 1024 + 1"
    worker_script.write_text(
        "import sys,time\n"
        f"sys.{target}.buffer.write(b'x' * ({size}) + b'\\n')\n"
        f"sys.{target}.buffer.flush()\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )
    manifest_path, _, _ = write_fixture(
        tmp_path,
        [{"values": [0.0], "expected_text": "x"}],
        [scripted_case(partials=[], final="x")],
        worker_path=worker_script,
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    worker = _worker(manifest_path, scratch)

    with pytest.raises(WorkerError, match="worker_output_limit"):
        worker.start()

    assert worker._process is not None  # noqa: SLF001 - lifecycle proof
    assert worker._process.poll() is not None  # noqa: SLF001 - lifecycle proof
    if stream_name == "stderr":
        assert worker.stderr_summary["truncated"] is True
        assert worker.stderr_summary["bytes"] == MAX_STDERR_BYTES + 1
