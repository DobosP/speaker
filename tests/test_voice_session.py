"""Pure contracts for the unified local/trusted-LAN session selector."""
from __future__ import annotations

from threading import Event, Thread

import pytest

from core.voice_session import (
    AudioEgress,
    SessionMode,
    SessionTopology,
    VoiceSession,
    VoiceSessionError,
    VoiceSessionPlan,
    engine_for_session,
    validate_livekit_url,
)


def test_default_plan_is_safe_console_and_device_only():
    plan = VoiceSessionPlan.resolve(config={})

    assert plan.mode is SessionMode.CONSOLE
    assert plan.engine == "console"
    assert plan.audio_egress is AudioEgress.DEVICE_ONLY
    assert plan.topology is SessionTopology.DEVICE_LOCAL
    assert plan.sends_audio_over_network is False


@pytest.mark.parametrize(
    ("session", "engine"),
    [
        ("console", "console"),
        ("local", "sherpa"),
        ("replay", "replay"),
    ],
)
def test_device_local_sessions_map_to_existing_adapters(session, engine):
    plan = VoiceSessionPlan.resolve(config={}, requested_session=session)

    assert plan.engine == engine
    assert plan.sends_audio_over_network is False


def test_trusted_lan_requires_machine_local_setup_grant():
    with pytest.raises(VoiceSessionError, match="setup_assistant"):
        VoiceSessionPlan.resolve(config={}, requested_session="trusted-lan")


def test_trusted_lan_grant_is_explicit_and_does_not_mint_owner_identity():
    plan = VoiceSessionPlan.resolve(
        config={
            "voice_session": {"audio_egress": "trusted_lan"},
            "remote": {
                "room": "kitchen",
                "agent_identity": "brain",
                "publisher_identity": "kitchen-satellite",
            },
        },
        requested_session="trusted-lan",
    )

    assert plan.mode is SessionMode.TRUSTED_LAN
    assert plan.engine == "livekit_agents"
    assert plan.audio_egress is AudioEgress.TRUSTED_LAN
    assert plan.topology is SessionTopology.TRUSTED_LAN
    assert plan.publisher_identity == "kitchen-satellite"
    assert plan.sends_audio_over_network is True


def test_canonical_trusted_lan_stays_unselectable_until_wrapper_lands():
    plan = VoiceSessionPlan.resolve(
        config={"voice_session": {"audio_egress": "trusted_lan"}},
        requested_session="trusted-lan",
    )

    assert plan.is_selectable is False
    with pytest.raises(VoiceSessionError, match="wrapper.*live A/B"):
        plan.require_selectable()


def test_local_session_remains_available_after_lan_permission_is_granted():
    plan = VoiceSessionPlan.resolve(
        config={"voice_session": {"audio_egress": "trusted_lan"}},
        requested_session="local",
    )

    assert plan.mode is SessionMode.LOCAL
    assert plan.sends_audio_over_network is False


def test_legacy_livekit_is_subject_to_same_explicit_audio_policy():
    with pytest.raises(VoiceSessionError, match="not authorized"):
        VoiceSessionPlan.resolve(config={}, legacy_engine="livekit")

    plan = VoiceSessionPlan.resolve(
        config={"voice_session": {"audio_egress": "trusted_lan"}},
        legacy_engine="livekit",
    )
    assert plan.mode is SessionMode.TRUSTED_LAN
    assert plan.engine == "livekit"
    assert plan.is_selectable is True


def test_new_and_legacy_selectors_cannot_compete():
    with pytest.raises(VoiceSessionError, match="choose --session"):
        VoiceSessionPlan.resolve(
            config={},
            requested_session="local",
            legacy_engine="sherpa",
        )


@pytest.mark.parametrize(
    "config",
    [
        {"voice_session": []},
        {"voice_session": {"audio_egress": "internet"}},
    ],
)
def test_malformed_session_configuration_fails_closed(config):
    with pytest.raises(VoiceSessionError):
        VoiceSessionPlan.resolve(config=config)


def test_malformed_remote_config_only_blocks_a_remote_session():
    local = VoiceSessionPlan.resolve(config={"remote": []}, requested_session="local")
    assert local.mode is SessionMode.LOCAL

    with pytest.raises(VoiceSessionError, match="remote configuration"):
        VoiceSessionPlan.resolve(
            config={
                "voice_session": {"audio_egress": "trusted_lan"},
                "remote": [],
            },
            requested_session="trusted-lan",
        )


@pytest.mark.parametrize(
    "field",
    ["room", "agent_identity", "publisher_identity"],
)
def test_remote_identifiers_are_exact_bounded_values(field):
    kwargs = {field: "bad/value"}
    with pytest.raises(VoiceSessionError, match="must be 1-128"):
        VoiceSessionPlan.resolve(config={}, **kwargs)


def test_engine_for_session_supports_pre_config_cli_checks():
    assert engine_for_session("local") == "sherpa"
    assert engine_for_session(None) == "console"


@pytest.mark.parametrize(
    "url",
    [
        "ws://127.0.0.1:7880",
        "ws://[::1]:7880",
        "wss://192.168.1.20:7880",
        "wss://voice.home.arpa",
        "wss://livekit.local",
    ],
)
def test_livekit_url_accepts_loopback_or_encrypted_lan(url):
    assert validate_livekit_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "",
        "https://192.168.1.20",
        "ws://192.168.1.20:7880",
        "wss://example.com",
        "wss://project.livekit.cloud",
        "wss://user:secret@192.168.1.20:7880",
        "wss://livekit",
        "wss://134744072",
        "wss://0x08080808",
        "wss://0177.0.0.1",
    ],
)
def test_livekit_url_rejects_missing_public_or_unencrypted_lan(url):
    with pytest.raises(VoiceSessionError):
        validate_livekit_url(url)


def test_voice_session_releases_exactly_one_injected_runtime():
    class Runtime:
        def __init__(self):
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1

    runtime = Runtime()
    session = VoiceSession(VoiceSessionPlan.resolve(config={}), runtime)

    session.stop()
    session.stop()

    assert runtime.stop_calls == 1
    assert session.stopped is True


def test_voice_session_context_releases_runtime_after_body_failure():
    class Runtime:
        stop_calls = 0

        def stop(self):
            self.stop_calls += 1

    runtime = Runtime()
    session = VoiceSession(VoiceSessionPlan.resolve(config={}), runtime)

    with pytest.raises(RuntimeError, match="startup failed"):
        with session:
            raise RuntimeError("startup failed")

    assert runtime.stop_calls == 1


def test_repeated_stop_observes_the_same_failure_without_retrying_runtime():
    failure = RuntimeError("stop failed")

    class Runtime:
        stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            raise failure

    runtime = Runtime()
    session = VoiceSession(VoiceSessionPlan.resolve(config={}), runtime)

    with pytest.raises(RuntimeError) as first:
        session.stop()
    with pytest.raises(RuntimeError) as second:
        session.stop()

    assert first.value is failure
    assert second.value is failure
    assert runtime.stop_calls == 1
    assert session.stopped is True


def test_concurrent_stop_waits_for_the_single_runtime_teardown():
    entered = Event()
    release = Event()
    follower_started = Event()
    follower_done = Event()

    class Runtime:
        stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            entered.set()
            assert release.wait(1.0)

    runtime = Runtime()
    session = VoiceSession(VoiceSessionPlan.resolve(config={}), runtime)
    owner = Thread(target=session.stop)

    def follow_stop():
        follower_started.set()
        session.stop()
        follower_done.set()

    follower = Thread(target=follow_stop)
    owner.start()
    assert entered.wait(1.0)
    follower.start()
    assert follower_started.wait(1.0)
    assert follower_done.wait(0.05) is False

    release.set()
    owner.join(1.0)
    follower.join(1.0)

    assert owner.is_alive() is False
    assert follower.is_alive() is False
    assert follower_done.is_set()
    assert runtime.stop_calls == 1


def test_voice_session_refuses_an_object_without_runtime_lifecycle():
    with pytest.raises(TypeError, match="provide stop"):
        VoiceSession(VoiceSessionPlan.resolve(config={}), object())
