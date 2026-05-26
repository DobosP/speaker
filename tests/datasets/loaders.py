"""Loaders for open-source NLU datasets, used to build large test corpora.

Two real, permissively-licensed datasets are reused:

* CLINC150  -- clinc/oos-eval, CC BY 3.0. 150 intents over 10 domains plus an
  explicit out-of-scope set (ideal for false-trigger / precision testing of the
  English ConversationRouter).
* MASSIVE 1.0 -- alexa/massive, CC BY 4.0. 51-language virtual-assistant NLU
  corpus; we use en-US and ro-RO (ideal for the bilingual LiveSpeechAnalyzer).

Loading strategy (mirrors tests/conftest.py graceful degradation):
  1. Default: read the small, attributed slices committed under ``samples/`` --
     no network, deterministic, CI-safe (still hundreds of real utterances).
  2. Opt-in: set ``SPEAKER_DATASET_DOWNLOAD=1`` to fetch + cache the FULL datasets
     (cache under ``_cache/``, gitignored), scaling the corpora to many thousands
     of utterances. ``SPEAKER_DATASET_LIMIT`` (default 1500) caps records/source.

Loaders never raise: any download/parse failure falls back to the bundled slice.
"""
from __future__ import annotations

import functools
import json
import os
import pathlib
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass

_HERE = pathlib.Path(__file__).parent
_SAMPLES = _HERE / "samples"
_CACHE = _HERE / "_cache"

_CLINC_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
_MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz"


@dataclass(frozen=True)
class UtteranceRecord:
    text: str
    intent: str
    locale: str = "en-US"
    scenario: str = ""
    source: str = ""


def _download_enabled() -> bool:
    return os.environ.get("SPEAKER_DATASET_DOWNLOAD", "").lower() in ("1", "true", "yes")


def _limit() -> int:
    try:
        return int(os.environ.get("SPEAKER_DATASET_LIMIT", "1500"))
    except ValueError:
        return 1500


def _load_sample(name: str) -> list[UtteranceRecord]:
    data = json.loads((_SAMPLES / name).read_text(encoding="utf-8"))
    locale = data.get("locale", "en-US")
    source = data.get("_source", name)
    return [
        UtteranceRecord(
            text=r["text"],
            intent=r.get("intent", ""),
            locale=r.get("locale", locale),
            scenario=r.get("scenario", ""),
            source=source,
        )
        for r in data["records"]
        if r.get("text")
    ]


def _fetch(url: str, dest: pathlib.Path, timeout: float = 120.0) -> bool:
    try:
        _CACHE.mkdir(exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            return True
        # Download to a temp file and atomically rename, so an interrupted
        # download never leaves a partial/corrupt file in the cache.
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            data = resp.read()
        fd, tmp = tempfile.mkstemp(dir=str(_CACHE))
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, dest)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return dest.exists() and dest.stat().st_size > 0
    except Exception:
        return False


@functools.lru_cache(maxsize=None)
def load_clinc150() -> tuple[UtteranceRecord, ...]:
    """CLINC150 (en-US). Full set when downloading; else the bundled slice."""
    if _download_enabled():
        try:
            dest = _CACHE / "clinc_data_full.json"
            if _fetch(_CLINC_URL, dest):
                data = json.loads(dest.read_text(encoding="utf-8"))
                recs: list[UtteranceRecord] = []
                for split in ("train", "val", "test"):
                    for utt, intent in data.get(split, []):
                        recs.append(UtteranceRecord(utt, intent, "en-US", "", "CLINC150"))
                for split in ("oos_train", "oos_val", "oos_test"):
                    for utt, _ in data.get(split, []):
                        recs.append(UtteranceRecord(utt, "oos", "en-US", "", "CLINC150"))
                if recs:
                    return tuple(recs[: _limit()])
        except Exception:
            pass
    return tuple(_load_sample("clinc150_sample.json"))


@functools.lru_cache(maxsize=None)
def load_massive(locale: str = "en-US") -> tuple[UtteranceRecord, ...]:
    """MASSIVE 1.0 for one locale (e.g. 'en-US', 'ro-RO')."""
    if _download_enabled():
        try:
            tar_path = _CACHE / "massive-1.0.tar.gz"
            if _fetch(_MASSIVE_URL, tar_path, timeout=300.0):
                member = f"1.0/data/{locale}.jsonl"
                recs = []
                with tarfile.open(tar_path, "r:gz") as tf:
                    fh = tf.extractfile(member)
                    if fh is not None:
                        for line in fh.read().decode("utf-8").splitlines():
                            d = json.loads(line)
                            recs.append(
                                UtteranceRecord(
                                    d["utt"], d.get("intent", ""), locale,
                                    d.get("scenario", ""), "MASSIVE",
                                )
                            )
                if recs:
                    return tuple(recs[: _limit()])
        except Exception:
            pass
    sample = {"en-US": "massive_en-US_sample.json", "ro-RO": "massive_ro-RO_sample.json"}
    return tuple(_load_sample(sample.get(locale, "massive_en-US_sample.json")))


def router_corpus() -> list[UtteranceRecord]:
    """English utterances for ConversationRouter tests (CLINC150 + MASSIVE en)."""
    return [*load_clinc150(), *load_massive("en-US")]


def analyzer_corpus() -> list[UtteranceRecord]:
    """Bilingual utterances for LiveSpeechAnalyzer tests (MASSIVE en + ro + CLINC)."""
    return [*load_massive("en-US"), *load_massive("ro-RO"), *load_clinc150()]
