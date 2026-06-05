"""Probe a local Ollama model for fitness as the app's answering tier -- measured
on THIS box, through the SAME ``OllamaLLM`` client the runtime uses, so a model
swap is decided by evidence, not by a model card.

Reports, per model: the real resident **VRAM** (GPU vs CPU split, from
``/api/ps``), **text** answer correctness + time-to-first-token on a few canonical
probes, and **multimodal** -- whether the model actually sees an image (a solid
red square -> "what colour is this?"). Built to compare candidates (e.g.
gemma4:12b vs gemma4:e4b vs the current gemma3:4b) before committing a config swap.

    python -m tools.model_probe gemma3:4b
    python -m tools.model_probe gemma4:12b gemma4:e4b
    python -m tools.model_probe gemma4:12b --pull      # pull if missing first

Pulls are NOT done unless --pull (they need disk); without the model present a
probe is reported as "missing", not an error.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from typing import Optional

OLLAMA = "http://127.0.0.1:11434"

# (prompt, expected substring | None) -- terse so a small model answers fast.
TEXT_PROBES = [
    ("What is the capital of France? Answer with one word.", "paris"),
    ("What is seven times eight? Reply with the number only.", "56"),
    ("Name the largest ocean on Earth. One word.", "pacific"),
    ("In one short sentence, tell a story about a lighthouse keeper.", None),
]


def _get(path: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(OLLAMA + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post(path: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        OLLAMA + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _installed() -> set:
    try:
        return {m["name"] for m in _get("/api/tags").get("models", [])}
    except Exception:
        return set()


def _vram(model: str) -> Optional[dict]:
    """Resident VRAM/CPU split for a loaded model, in GB (None if not loaded)."""
    try:
        for m in _get("/api/ps").get("models", []):
            if m.get("name") == model or m.get("model") == model:
                total = m.get("size", 0)
                vram = m.get("size_vram", 0)
                return {"vram_gb": vram / 1e9, "cpu_gb": (total - vram) / 1e9, "total_gb": total / 1e9}
    except Exception:
        pass
    return None


def _red_png() -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (128, 128), (220, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def probe(model: str, *, pull: bool = False, num_ctx: int = 4096) -> dict:
    out: dict = {"model": model}
    if model not in _installed():
        if not pull:
            out["status"] = "missing (pull it or pass --pull)"
            return out
        print(f"[{model}] pulling (this needs disk)...", flush=True)
        try:
            _post("/api/pull", {"model": model, "stream": False}, timeout=3600)
        except Exception as exc:  # noqa: BLE001
            out["status"] = f"pull failed: {exc}"
            return out

    from core.llm import OllamaLLM

    llm = OllamaLLM(model=model, options={"num_ctx": num_ctx}, keep_alive="5m")

    # Warm + architecture check: a load that fails here usually means Ollama is too
    # old for this model's architecture -> report it instead of crashing.
    t0 = time.time()
    try:
        _ = llm.generate("Reply with the single word: ready.")
    except Exception as exc:  # noqa: BLE001
        out["status"] = f"load/generate failed (Ollama may be too old): {exc}"
        return out
    out["warm_load_s"] = round(time.time() - t0, 1)
    out["vram"] = _vram(model)

    # Text probes: correctness + TTFT (first streamed token).
    text_rows = []
    for prompt, expect in TEXT_PROBES:
        t = time.time()
        ttft = None
        chunks = []
        try:
            for tok in llm.stream(prompt):
                if ttft is None:
                    ttft = round(time.time() - t, 2)
                chunks.append(tok)
        except Exception as exc:  # noqa: BLE001
            text_rows.append({"prompt": prompt[:30], "error": str(exc)[:60]})
            continue
        ans = "".join(chunks).strip()
        text_rows.append({
            "prompt": prompt[:30],
            "ttft_s": ttft,
            "ok": (expect is None) or (expect in ans.lower()),
            "answer": ans[:70],
        })
    out["text"] = text_rows

    # Multimodal: does the model SEE the image? Solid red square -> "what colour".
    try:
        png = _red_png()
        t = time.time()
        ans = llm.generate("What is the dominant colour in this image? Answer with one word.", images=[png])
        out["multimodal"] = {
            "sees_image": "red" in ans.lower(),
            "latency_s": round(time.time() - t, 2),
            "answer": ans.strip()[:60],
        }
    except Exception as exc:  # noqa: BLE001
        out["multimodal"] = {"error": str(exc)[:80]}
    out["status"] = "ok"
    return out


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(prog="tools.model_probe", description=__doc__)
    p.add_argument("models", nargs="+", help="ollama model tags to probe")
    p.add_argument("--pull", action="store_true", help="pull a missing model first (needs disk)")
    p.add_argument("--num-ctx", type=int, default=4096)
    args = p.parse_args(argv)

    results = [probe(m, pull=args.pull, num_ctx=args.num_ctx) for m in args.models]

    print("\n" + "=" * 78)
    print(f"{'model':16} {'status':10} {'vram':>10} {'warm':>6} {'text ok':>8} {'sees img':>9}")
    print("-" * 78)
    for r in results:
        v = r.get("vram") or {}
        vram = f"{v.get('vram_gb', 0):.1f}/{v.get('cpu_gb', 0):.1f}c" if v else "-"
        text = r.get("text") or []
        ok = sum(1 for t in text if t.get("ok"))
        mm = r.get("multimodal") or {}
        sees = "yes" if mm.get("sees_image") else ("err" if mm.get("error") else "no")
        print(f"{r['model']:16} {r.get('status','?')[:10]:10} {vram:>10} "
              f"{str(r.get('warm_load_s','-')):>6} {ok}/{len(text):>6} {sees:>9}")
    print("=" * 78)
    print("(vram = GPU/CPU GB resident; text ok = correct answers; sees img = multimodal works)\n")
    for r in results:
        print(json.dumps(r, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
