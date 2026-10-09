"""Explicit pinned local Qwen profile installation; no defaults or audio changes.

Downloads only the selected public GGUF, verifies bytes before atomic publish,
then creates the measured project alias in an already running loopback Ollama.
The original upstream GGUF and Ollama's rewritten GGUF have distinct pins.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import ipaddress
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from core.voice_model_profile import (
    PROFILE_ALIAS, PROFILE_ASSET_RELATIVE, PROFILE_FILENAME, PROFILE_REVISION,
    PROFILE_RUNTIME_SHA256, PROFILE_SOURCE_BYTES, PROFILE_SOURCE_SHA256,
    PROFILE_SOURCE_URL, verify_voice_model_identity,
)


class SetupRefused(ValueError):
    """Only fixed code-owned failure messages leave this tool."""


def local_host(value: str) -> str:
    try:
        parts = urlsplit(value)
        host = parts.hostname
        ok = (host == "localhost" or bool(host and ipaddress.ip_address(host).is_loopback))
        if (parts.scheme != "http" or not ok or parts.port is None
                or parts.username or parts.password or parts.query or parts.fragment
                or parts.path not in {"", "/"}):
            raise ValueError
    except ValueError:
        raise SetupRefused("local_endpoint_required") from None
    return value.rstrip("/")


class PublicRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parts = urlsplit(newurl)
        host = parts.hostname or ""
        if (parts.scheme != "https" or parts.username or parts.password
                or not (host == "huggingface.co" or host.endswith(
                    (".xethub.hf.co", ".hf.co", ".huggingface.co")))):
            raise SetupRefused("download_redirect_refused")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _verify_file(path: Path) -> bool:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size != PROFILE_SOURCE_BYTES or before.st_nlink != 1:
            return False
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                return False
            h = sha256()
            while data := stream.read(8 * 1024 * 1024):
                h.update(data)
            after = os.fstat(stream.fileno())
        current = path.lstat()
        return (h.hexdigest() == PROFILE_SOURCE_SHA256
                and (opened.st_size, opened.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
                and (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
                == (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns))
    except OSError:
        return False


def ensure_asset(root: Path, *, opener=None, clock=time.monotonic) -> Path:
    root = root.absolute()
    asset = root / PROFILE_ASSET_RELATIVE
    asset.parent.mkdir(parents=True, exist_ok=True)
    if asset.parent.resolve() != asset.parent:
        raise SetupRefused("asset_parent_symlink_refused")
    if asset.exists() or asset.is_symlink():
        if not _verify_file(asset):
            raise SetupRefused("existing_asset_mismatch")
        return asset
    opener = opener or build_opener(ProxyHandler({}), PublicRedirect())
    partial = asset.with_name(PROFILE_FILENAME + ".atomic-part")
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        deadline = clock() + 600
        with os.fdopen(fd, "wb") as target:
            request = Request(PROFILE_SOURCE_URL + "?download=true",
                              headers={"User-Agent": "speaker-local-voice-model-setup"})
            with opener.open(request, timeout=30) as response:
                count, digest = 0, sha256()
                while True:
                    if clock() >= deadline:
                        raise SetupRefused("download_deadline")
                    block = response.read(min(8 * 1024 * 1024, PROFILE_SOURCE_BYTES - count + 1))
                    if not block:
                        break
                    count += len(block)
                    if count > PROFILE_SOURCE_BYTES:
                        raise SetupRefused("download_size_mismatch")
                    target.write(block)
                    digest.update(block)
            if count != PROFILE_SOURCE_BYTES or digest.hexdigest() != PROFILE_SOURCE_SHA256:
                raise SetupRefused("download_digest_mismatch")
            target.flush()
            os.fsync(target.fileno())
        # Exclusive publication cannot replace another installer/user's file.
        os.link(partial, asset, follow_symlinks=False)
    finally:
        partial.unlink(missing_ok=True)
    receipt = asset.with_name(PROFILE_FILENAME + ".provenance.json")
    value = {"model": "Qwen/Qwen2.5-1.5B-Instruct-GGUF", "revision": PROFILE_REVISION,
             "sha256": PROFILE_SOURCE_SHA256, "bytes": PROFILE_SOURCE_BYTES, "license": "Apache-2.0"}
    try:
        with receipt.open("x", encoding="utf-8") as out:
            json.dump(value, out, indent=2)
    except FileExistsError:
        pass  # preserve any prior provenance; the bytes were independently verified
    return asset


def install_alias(asset: Path, host: str, *, client=None, run=subprocess.run) -> None:
    host = local_host(host)
    if not _verify_file(asset):
        raise SetupRefused("source_asset_mismatch")
    if client is None:
        import ollama
        client = ollama.Client(host=host, timeout=20, trust_env=False,
                               headers={"authorization": "Bearer speaker-local-model-setup"})
    listed = client.list()
    rows = listed.get("models", ()) if isinstance(listed, dict) else listed.models
    names = {row.get("model") if isinstance(row, dict) else row.model for row in rows}
    if PROFILE_ALIAS in names:
        if not verify_voice_model_identity(show=client.show).ok:
            raise SetupRefused("existing_alias_mismatch")
        return
    executable = shutil.which("ollama")
    if executable is None:
        raise SetupRefused("ollama_cli_missing")
    # The CLI handles local GGUF upload/import. Keep model content and native
    # stderr private, and bind the resulting alias before declaring readiness.
    with tempfile.TemporaryDirectory(prefix="speaker-voice-model-") as scratch:
        modelfile = Path(scratch) / "Modelfile"
        modelfile.write_text(
            f'FROM "{asset.as_posix()}"\nPARAMETER temperature 0.7\nPARAMETER top_p 0.95\n'
            'PARAMETER num_ctx 4096\nPARAMETER stop "<|im_end|>"\n', encoding="utf-8")
        env = {key: os.environ[key] for key in ("HOME", "USERPROFILE", "PATH", "SYSTEMROOT", "TEMP", "TMP")
               if key in os.environ}
        env.update(OLLAMA_HOST=host, NO_PROXY="127.0.0.1,localhost,::1")
        result = run([executable, "create", PROFILE_ALIAS, "-f", str(modelfile)],
                     env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     timeout=120, check=False)
        if result.returncode:
            raise SetupRefused("alias_import_failed")
    if not verify_voice_model_identity(show=client.show).ok:
        raise SetupRefused("imported_alias_mismatch")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("qwen2.5-1.5b",), required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    args = parser.parse_args(argv)
    try:
        host = local_host(args.host)
        asset = ensure_asset(args.root)
        install_alias(asset, host)
        print(json.dumps({"profile": args.profile, "alias": PROFILE_ALIAS,
                          "source_sha256": PROFILE_SOURCE_SHA256,
                          "runtime_sha256": PROFILE_RUNTIME_SHA256, "verified": True}))
        return 0
    except Exception as error:
        code = str(error) if type(error) is SetupRefused else "setup_failed"
        print("voice_model_setup_refused:" + code, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
