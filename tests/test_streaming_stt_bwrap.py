from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import time

import pytest


_BWRAP = Path("/usr/bin/bwrap")


def _base_command(scratch: Path) -> list[str]:
    command = [
        str(_BWRAP),
        "--unshare-user-try",
        "--unshare-ipc",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--die-with-parent",
        "--new-session",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--ro-bind",
        "/etc/ld.so.cache",
        "/etc/ld.so.cache",
        "--bind",
        str(scratch),
        str(scratch),
        "--chdir",
        str(scratch),
    ]
    return command


def _sandbox_environment(scratch: Path) -> dict[str, str]:
    return {
        "HOME": str(scratch),
        "TMPDIR": str(scratch),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }


def _require_unprivileged_bwrap(scratch: Path) -> None:
    try:
        metadata = _BWRAP.lstat()
    except OSError:
        pytest.skip("unprivileged Bubblewrap unavailable: /usr/bin/bwrap missing")
    if not stat.S_ISREG(metadata.st_mode) or not metadata.st_mode & 0o111:
        pytest.skip(
            "unprivileged Bubblewrap unavailable: /usr/bin/bwrap is not executable"
        )
    try:
        probe = subprocess.run(
            [*_base_command(scratch), "--", "/usr/bin/true"],
            cwd=scratch,
            env=_sandbox_environment(scratch),
            check=False,
            capture_output=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("unprivileged Bubblewrap unavailable: boundary probe failed")
    if probe.returncode != 0:
        detail = probe.stderr.decode("utf-8", errors="replace").splitlines()
        reason = detail[0][:160] if detail else f"exit {probe.returncode}"
        pytest.skip(f"unprivileged Bubblewrap unavailable: {reason}")


@pytest.mark.skipif(os.name != "posix", reason="Bubblewrap boundary is Linux-only")
def test_bwrap_blocks_host_network_and_reaps_setsid_descendant(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    _require_unprivileged_bwrap(scratch)
    marker = scratch / "boundary.json"
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(0.1)
    port = listener.getsockname()[1]
    script = (
        "import json,os,signal,socket,subprocess,sys,time;"
        "probe=socket.socket();probe.settimeout(0.25);"
        "reachable=False;"
        "\ntry:\n probe.connect(('127.0.0.1',int(sys.argv[2])));reachable=True"
        "\nexcept OSError:\n pass"
        "\nfinally:\n probe.close()"
        "\ncode='import signal,time;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)';"
        "\nchild=subprocess.Popen([sys.executable,'-I','-S','-B','-c',code],"
        "start_new_session=True);"
        "\npayload={'pid':os.getpid(),'pgrp':os.getpgrp(),'sid':os.getsid(0),"
        "'child':child.pid,'host_network_reachable':reachable};"
        "\nopen(sys.argv[1],'w',encoding='utf-8').write(json.dumps(payload));"
        "\nsignal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"
    )
    process = subprocess.Popen(
        [
            *_base_command(scratch),
            "--",
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            "-c",
            script,
            str(marker),
            str(port),
        ],
        cwd=scratch,
        env=_sandbox_environment(scratch),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5.0
        while (
            not marker.exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert marker.exists(), process.stderr.read(1024) if process.stderr else b""
        payload = json.loads(marker.read_text(encoding="utf-8"))
        assert payload["host_network_reachable"] is False
        assert payload["pid"] >= 2
        assert payload["child"] > payload["pid"]
        assert payload["pgrp"] == payload["sid"]
        assert payload["sid"] != payload["pid"]
        assert os.getpgid(process.pid) == process.pid

        started = time.monotonic()
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3.0)
        assert time.monotonic() - started < 3.0
        process.communicate(timeout=1.0)
        with pytest.raises(ProcessLookupError):
            os.killpg(process.pid, 0)
        listener.settimeout(0.05)
        with pytest.raises(TimeoutError):
            listener.accept()
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                process.kill()
            process.wait(timeout=2.0)
        listener.close()
