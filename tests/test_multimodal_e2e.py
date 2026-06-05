"""End-to-end image plumbing through the ROUTED/RACED main-tier LLM chain.

``test_core_multimodal.py`` pins that a *bare* :class:`OllamaLLM` attaches
``images=`` to the user message. But the object the runtime actually hands the
capability layer as the main/multimodal model is not a bare client: when cloud
is configured, ``core.llm_factory._wrap_cloud`` builds one :class:`HedgeLLM`
per cloud-chain (local raced against the chain) and wraps them in a
:class:`SensitivityRouterLLM` that picks a chain per turn from the live
``capability_context``. That routing/race chain is exactly the layer that could
silently *drop* the image bytes (a missing ``images=images`` in
``HedgeLLM._worker`` or ``SensitivityRouterLLM.stream`` would not be caught by
the existing bare-client test).

These tests pin that image bytes survive that full forwarding chain down to the
chosen backing multimodal client, and that a text-only turn carries no images.
They build the chain the *same way the real factory does* and drive chain
selection through the *real* ``capability_context`` ContextVar that the
capability layer sets per turn -- so they exercise the production forwarding
path, not a re-implementation of it.

All fakes -- no Ollama, no openai, no GPU, no network.
"""
from __future__ import annotations

from typing import Iterator, Optional, Sequence

from core.llm import (
    HedgeLLM,
    LlamaCppLLM,
    SensitivityRouterLLM,
    capability_context,
)
from core.routing import ChainSelector

# A tiny but structurally valid PNG (8-byte signature + a real IHDR chunk for a
# 1x1 image). Using real bytes -- not a sentinel string -- proves the *bytes*
# object travels untouched through the routing/race chain.
_PNG_1x1 = (
    b"\x89PNG\r\n\x1a\n"  # PNG signature
    b"\x00\x00\x00\rIHDR"  # IHDR length + type
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"  # 1x1, RGBA
    b"\x1f\x15\xc4\x89"  # IHDR CRC
)


class RecordingMMClient:
    """Multimodal LLMClient fake: records exactly what ``images=`` it received.

    Stands in for the backing vision model (gemma3:4b/12b via Ollama, or a
    cloud OpenAI-compat endpoint) at the bottom of the routing/race chain.
    Records on BOTH ``generate`` and ``stream`` because :class:`HedgeLLM` and
    :class:`SensitivityRouterLLM` reach the backend through ``stream``.
    """

    def __init__(self, tag: str = "mm", reply: str = "a small icon", *, silent: bool = False):
        self.tag = tag
        self.reply = reply
        # ``silent`` models record the call but yield no token -- a "dead"/empty
        # source in HedgeLLM's race, so the other source wins deterministically
        # (mirrors a real backend that produced nothing). The real backends never
        # emit empty tokens, so we never yield "" -- yielding nothing is the
        # faithful way to lose the race.
        self.silent = silent
        # Each entry is the ``images`` argument as received (the actual object,
        # so identity/content can both be checked) for one call.
        self.image_calls: list[Optional[Sequence[object]]] = []
        self.prompts: list[str] = []

    def generate(self, prompt, *, system=None, images=None) -> str:
        self.prompts.append(prompt)
        self.image_calls.append(images)
        return "" if self.silent else self.reply

    def stream(self, prompt, *, system=None, images=None) -> Iterator[str]:
        self.prompts.append(prompt)
        self.image_calls.append(images)
        if not self.silent:
            yield self.reply


def _last_images(client: RecordingMMClient):
    assert client.image_calls, f"{client.tag} backend was never called"
    return client.image_calls[-1]


# --- the routed/raced main tier (the real _wrap_cloud shape) ----------------


def _build_routed_main_tier(local: RecordingMMClient, cloud: RecordingMMClient):
    """Construct the SensitivityRouterLLM-over-HedgeLLM exactly like the real
    ``core.llm_factory._wrap_cloud`` multi-provider path does: one HedgeLLM per
    chain (local raced against the chain's clouds) wrapped in a
    SensitivityRouterLLM keyed by sensitivity. ``hedge_delay_ms=0`` makes the
    cloud start immediately so the fake cloud (which yields at once) reliably
    wins the race -- deterministic, no sleeps."""
    private_chain = HedgeLLM(local=local, cloud=[], hedge_delay_ms=0)  # local-only
    public_chain = HedgeLLM(local=local, cloud=[cloud], hedge_delay_ms=0)
    selector = ChainSelector(
        {"public": "public", "private": "private"}, default_chain="private"
    )
    return SensitivityRouterLLM(
        {"private": private_chain, "public": public_chain},
        selector=selector,
        default_chain="private",
    )


def test_image_bytes_reach_chosen_cloud_chain_via_capability_context():
    """A multimodal turn whose context routes to the cloud chain delivers the
    PNG bytes, intact, to that chain's backing vision client -- driven through
    the REAL ``capability_context`` ContextVar the capability layer sets."""
    # Local is silent so the cloud reliably wins the public chain's race (a real
    # backend that produced nothing); both still record the ``images=`` they got.
    local = RecordingMMClient("local", silent=True)
    cloud = RecordingMMClient("cloud")
    main = _build_routed_main_tier(local, cloud)

    # The capability layer publishes per-turn context here before streaming;
    # the SensitivityRouterLLM reads it to pick the chain. "public" -> the
    # cloud-backed chain.
    token = capability_context.set({"sensitivity": "public"})
    try:
        out = "".join(main.stream("describe this image", images=[_PNG_1x1]))
    finally:
        capability_context.reset(token)

    assert out == "a small icon"
    # The cloud chain's backend got the call AND the exact PNG bytes.
    received = _last_images(cloud)
    assert received is not None and list(received) == [_PNG_1x1]
    # Same bytes object preserved through HedgeLLM's worker + the router.
    assert received[0] == _PNG_1x1


def test_private_turn_routes_image_to_local_vision_tier():
    """A private (on-device) turn must keep the image local: it reaches the
    local backing client, and the cloud client is never touched -- raw image
    bytes never leave the device (the §9.7 boundary)."""
    local = RecordingMMClient("local")
    cloud = RecordingMMClient("cloud")
    main = _build_routed_main_tier(local, cloud)

    token = capability_context.set({"sensitivity": "private"})
    try:
        list(main.stream("what is in this photo", images=[_PNG_1x1]))
    finally:
        capability_context.reset(token)

    assert list(_last_images(local)) == [_PNG_1x1]
    assert cloud.image_calls == []  # image never crossed to cloud


def test_text_only_turn_carries_no_images_through_router():
    """The negative: a text-only turn forwards ``images=None`` -- the chain
    must not fabricate an images payload."""
    local = RecordingMMClient("local")
    cloud = RecordingMMClient("cloud")
    main = _build_routed_main_tier(local, cloud)

    token = capability_context.set({"sensitivity": "private"})
    try:
        list(main.stream("what time is it"))
    finally:
        capability_context.reset(token)

    assert _last_images(local) is None


def test_hedge_worker_forwards_images_to_the_racing_winner():
    """Pin the race-layer forwarding directly: HedgeLLM's worker hands
    ``images=`` to whichever client it launches (here the cloud wins because it
    yields immediately and local is empty)."""
    local = RecordingMMClient("local", silent=True)  # produces no token -> loses
    cloud = RecordingMMClient("cloud", reply="cloud says: an apple")
    hedge = HedgeLLM(local=local, cloud=[cloud], hedge_delay_ms=0)

    out = "".join(hedge.stream("identify this", images=[_PNG_1x1]))

    assert "an apple" in out
    assert list(_last_images(cloud)) == [_PNG_1x1]


def test_llamacpp_ondevice_path_encodes_image_bytes_as_data_uri():
    """The on-device GGUF tier (no Ollama) takes a different multimodal path:
    raw bytes become a base64 ``data:`` URL inside an ``image_url`` content
    block. Pin that the bytes actually land in the chat payload (this client is
    not covered by the existing bare-OllamaLLM test)."""
    import base64

    captured: dict = {}

    class FakeLlama:
        def create_chat_completion(self, *, messages, stream=False, **kw):
            captured["messages"] = messages
            return {"choices": [{"message": {"content": "ok"}}]}

    llm = LlamaCppLLM("ignored.gguf", client=FakeLlama())
    llm.generate("describe", images=[_PNG_1x1])

    user_msg = captured["messages"][-1]
    assert user_msg["role"] == "user"
    parts = user_msg["content"]
    # First a text part, then an image_url part carrying the encoded PNG.
    assert parts[0] == {"type": "text", "text": "describe"}
    img_part = next(p for p in parts if p.get("type") == "image_url")
    expected = "data:image/png;base64," + base64.b64encode(_PNG_1x1).decode("ascii")
    assert img_part["image_url"]["url"] == expected
