"""
Failure-discovery tests for realistic audio edge cases.

This suite is intentionally allowed to fail: it captures real-world behavior
we want the assistant to support but the current implementation does not fully
handle yet.  Local deterministic samples are saved under
``tests/fixture_audio/failure_discovery/`` so failures are reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures import (
    SR,
    apply_gain_db,
    apply_room_delay,
    babble_noise,
    click_burst,
    clipped_speech,
    hard_clip,
    mix,
    music_noise,
    nonstationary_noise,
    near_far_mix,
    plosive_burst,
    real_tts_echo,
    reverberant_echo,
    sample_rate_roundtrip,
    silence,
    tts_echo,
    tv_noise,
    voiced_speech,
)
from tests.harness import AudioHarness, make_recorder

pytestmark = [pytest.mark.discovery, pytest.mark.audio, pytest.mark.slow]


SAMPLE_DIR = Path(__file__).parent / "fixture_audio" / "failure_discovery"
CORPUS_VERSION = 5
MIN_CORPUS_CASES = 100


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))


def _fit(audio: np.ndarray, seconds: float) -> np.ndarray:
    n = int(seconds * SR)
    if len(audio) >= n:
        return audio[:n].astype(np.float32)
    return np.pad(audio, (0, n - len(audio))).astype(np.float32)


def _add_case(
    cases: list[dict],
    samples: dict[str, np.ndarray],
    *,
    name: str,
    kind: str,
    audio: np.ndarray,
    ref: str | None = None,
    expectation: str,
    description: str,
    importance: str,
):
    samples[name] = audio.astype(np.float32)
    cases.append(
        {
            "name": name,
            "kind": kind,
            "ref": ref,
            "expectation": expectation,
            "description": description,
            "importance": importance,
            "samples": int(len(audio)),
            "sample_rate": SR,
            "rms": round(_rms(audio), 6),
        }
    )


def _build_corpus() -> tuple[dict[str, np.ndarray], list[dict]]:
    samples: dict[str, np.ndarray] = {}
    cases: list[dict] = []
    base_ref = real_tts_echo(1.4, amplitude=0.09)
    synthetic_ref = tts_echo(1.4, amplitude=0.10)
    user_base = voiced_speech(0.9, amplitude=0.26)
    tv_base = tv_noise(1.0, amplitude=0.05)
    babble_base = babble_noise(1.0, amplitude=0.04)
    music_base = music_noise(1.0, amplitude=0.05)

    _add_case(
        cases,
        samples,
        name="control_synthetic_zero_lag_tts",
        kind="echo_control",
        audio=synthetic_ref,
        ref="control_synthetic_zero_lag_tts",
        expectation="no_interrupt",
        description="Synthetic zero-lag TTS echo should be blocked.",
        importance="Baseline guard: proves the suite can pass a known echo case.",
    )
    _add_case(
        cases,
        samples,
        name="reference_clean_tts",
        kind="reference",
        audio=base_ref,
        expectation="saved_only",
        description="Clean TTS reference used by room echo cases.",
        importance="Shared assistant reference for room echo simulations.",
    )

    important_cases = [
        # Assistant echo leaks: different spaces/devices, not repeated sweeps.
        ("echo_phone_table_30cm", "echo_leak", apply_room_delay(base_ref, 1.0, -1.5), "reference_clean_tts", "no_interrupt", "Phone lying beside laptop speaker.", "Common desk setup; tiny delay should still be echo."),
        ("echo_laptop_mic_70cm", "echo_leak", apply_room_delay(base_ref, 2.0, -3.0), "reference_clean_tts", "no_interrupt", "Laptop speaker to built-in microphone path.", "Most likely desktop deployment false-interrupt path."),
        ("echo_smart_speaker_2m", "echo_leak", apply_room_delay(base_ref, 7.0, -4.0), "reference_clean_tts", "no_interrupt", "Smart speaker two metres from user microphone.", "Typical living-room propagation delay."),
        ("echo_kitchen_counter_tile", "echo_leak", reverberant_echo(base_ref, direct_delay_ms=4, rt60_ms=180, direct_gain=0.80), "reference_clean_tts", "no_interrupt", "Kitchen counter with hard tile reflections.", "Hard reflective rooms are common and noisy."),
        ("echo_bathroom_tile", "echo_leak", reverberant_echo(base_ref, direct_delay_ms=5, rt60_ms=450, direct_gain=0.70), "reference_clean_tts", "no_interrupt", "Bathroom tile reverberation.", "Worst-case reverb tail should not self-interrupt."),
        ("echo_hallway_far_wall", "echo_leak", apply_room_delay(base_ref, 22.0, -5.0), "reference_clean_tts", "no_interrupt", "Hallway far-wall assistant reflection.", "Long narrow spaces produce strong late reflections."),
        ("echo_open_living_room_wall", "echo_leak", apply_room_delay(base_ref, 35.0, -3.0), "reference_clean_tts", "no_interrupt", "Living-room first wall reflection.", "Known failure mode for zero-lag echo checks."),
        ("echo_car_cabin_dashboard", "echo_leak", reverberant_echo(base_ref, direct_delay_ms=2, rt60_ms=130, direct_gain=0.90), "reference_clean_tts", "no_interrupt", "Small car cabin dashboard reflection.", "Car cabins create dense early reflections."),
        ("echo_projector_speaker_ceiling", "echo_leak", apply_room_delay(base_ref, 50.0, -7.0), "reference_clean_tts", "no_interrupt", "Ceiling projector speaker reflection.", "Distant media speakers should not look like user speech."),
        ("echo_bluetooth_speaker_shelf", "echo_leak", reverberant_echo(base_ref, direct_delay_ms=9, rt60_ms=260, direct_gain=0.65), "reference_clean_tts", "no_interrupt", "Bluetooth speaker on bookshelf.", "Common assistant playback device with shelf reflections."),
        ("echo_tv_soundbar_living_room", "echo_leak", reverberant_echo(base_ref, direct_delay_ms=12, rt60_ms=300, direct_gain=0.75), "reference_clean_tts", "no_interrupt", "TV soundbar leaking assistant speech.", "Soundbars are loud and far from microphones."),
        ("echo_conference_table", "echo_leak", reverberant_echo(base_ref, direct_delay_ms=6, rt60_ms=220, direct_gain=0.72), "reference_clean_tts", "no_interrupt", "Conference table speakerphone echo.", "Meeting rooms are key voice-agent environments."),
        ("echo_corner_room_bass_buildup", "echo_leak", apply_gain_db(reverberant_echo(base_ref, direct_delay_ms=8, rt60_ms=360, direct_gain=0.80), 2.5), "reference_clean_tts", "no_interrupt", "Speaker in room corner with gain buildup.", "Corner placement amplifies echo energy."),
        ("echo_curtained_room_soft_reverb", "echo_leak", reverberant_echo(base_ref, direct_delay_ms=6, rt60_ms=120, direct_gain=0.55), "reference_clean_tts", "no_interrupt", "Curtained room with soft reverb.", "Even damped rooms must remain safe."),
        ("echo_open_door_secondary_room", "echo_leak", apply_room_delay(base_ref, 70.0, -10.0), "reference_clean_tts", "no_interrupt", "Assistant audio heard from another room.", "Late low-level echo should not trigger barge-in."),
        ("echo_high_volume_near_field", "echo_leak", apply_gain_db(apply_room_delay(base_ref, 4.0, -1.0), 3.0), "reference_clean_tts", "no_interrupt", "High-volume near-field assistant playback.", "Loud assistant speech is likely during TTS."),
        ("echo_low_volume_far_field", "echo_leak", apply_room_delay(base_ref, 45.0, -12.0), "reference_clean_tts", "no_interrupt", "Low-volume far-field assistant leakage.", "Weak echoes should not accumulate false evidence."),
        ("echo_reverb_tail_only_after_pause", "echo_leak", np.concatenate([np.zeros(int(0.15 * SR), dtype=np.float32), reverberant_echo(base_ref, direct_delay_ms=4, rt60_ms=300, direct_gain=0.70)[:-int(0.15 * SR)]]), "reference_clean_tts", "no_interrupt", "Reverb tail after short playback pause.", "Pause gaps are common in chunked TTS."),
        ("echo_sample_rate_artifact", "echo_leak", sample_rate_roundtrip(apply_room_delay(base_ref, 11.0, -5.0), device_sr=44_100), "reference_clean_tts", "no_interrupt", "Echo after device sample-rate conversion.", "44.1 kHz consumer devices can add interpolation artifacts to echo."),
        ("echo_clipped_speaker_output", "echo_leak", hard_clip(apply_room_delay(base_ref * 2.0, 8.0, -2.0), limit=0.18), "reference_clean_tts", "no_interrupt", "Clipped speaker output echo.", "Small speakers clip assistant speech."),
        ("echo_tts_with_background_tv", "echo_leak", mix(apply_room_delay(base_ref, 10.0, -5.0), tv_base, snr_db=10.0), "reference_clean_tts", "no_interrupt", "Assistant echo mixed with TV bed.", "Real rooms contain background media during TTS."),
        ("echo_tts_with_music_bed", "echo_leak", mix(reverberant_echo(base_ref, direct_delay_ms=6, rt60_ms=220), music_base, snr_db=12.0), "reference_clean_tts", "no_interrupt", "Assistant echo mixed with music.", "Music can confuse VAD while echo is present."),
        ("echo_tts_with_babble_bed", "echo_leak", mix(reverberant_echo(base_ref, direct_delay_ms=6, rt60_ms=240), babble_base, snr_db=12.0), "reference_clean_tts", "no_interrupt", "Assistant echo mixed with distant people.", "Parties and meetings are hard echo cases."),
        ("echo_multi_path_two_reflections", "echo_leak", apply_room_delay(base_ref, 7.0, -4.0) + apply_room_delay(base_ref, 42.0, -8.0), "reference_clean_tts", "no_interrupt", "Two dominant echo paths.", "Multi-path echo is more realistic than one delay."),
        ("echo_multi_path_three_reflections", "echo_leak", apply_room_delay(base_ref, 5.0, -3.0) + apply_room_delay(base_ref, 28.0, -7.0) + apply_room_delay(base_ref, 65.0, -12.0), "reference_clean_tts", "no_interrupt", "Three dominant echo paths.", "Large rooms create several reflections."),
        ("echo_reverb_long_phrase_tail", "echo_leak", reverberant_echo(real_tts_echo(1.4, amplitude=0.11), direct_delay_ms=7, rt60_ms=520, direct_gain=0.75), "reference_clean_tts", "no_interrupt", "Long TTS phrase with heavy tail.", "Long assistant responses must not self-cancel."),

        # Background-only cases: important false-listen sources.
        ("background_tv_news_anchor", "background", tv_noise(1.0, amplitude=0.032), None, "no_callback", "TV news anchor-like broadband audio.", "TV speech must not become a user command."),
        ("background_tv_loud_commercial", "background", tv_noise(1.0, amplitude=0.050), None, "no_callback", "Loud TV commercial burst.", "Commercials are sudden and speech-like."),
        ("background_podcast_far_field", "background", babble_noise(1.0, amplitude=0.030, num_speakers=2), None, "no_callback", "Podcast playing across the room.", "Single/few-speaker media can look voiced."),
        ("background_party_babble", "background", babble_noise(1.0, amplitude=0.045, num_speakers=5), None, "no_callback", "Party babble without direct user intent.", "Multi-talker noise is common in homes."),
        ("background_kids_room_babble", "background", apply_gain_db(babble_noise(1.0, amplitude=0.030, num_speakers=4), 2.0), None, "no_callback", "Children talking in another room.", "Distant family speech should not trigger callbacks."),
        ("background_music_120bpm", "background", music_noise(1.0, amplitude=0.035, beat_hz=2.0), None, "no_callback", "Rhythmic 120 bpm music.", "Beat envelopes can accumulate speech score."),
        ("background_music_fast_tempo", "background", music_noise(1.0, amplitude=0.040, beat_hz=3.2), None, "no_callback", "Fast-tempo music.", "High beat rate probes soft-decay accumulation."),
        ("background_clipped_radio", "background", hard_clip(music_noise(1.0, amplitude=0.18, beat_hz=2.4), limit=0.10), None, "no_callback", "Clipped radio speaker.", "Cheap speakers distort harmonic content."),
        ("background_hvac_turns_on", "background", nonstationary_noise(1.0, quiet_amplitude=0.006, loud_amplitude=0.050), None, "no_callback", "HVAC fan ramps up mid-turn.", "Stale noise floors cause false speech starts."),
        ("background_dishwasher_pulse", "background", nonstationary_noise(1.0, quiet_amplitude=0.010, loud_amplitude=0.045, transition_at=0.35), None, "no_callback", "Dishwasher pump pulse.", "Appliance bursts are frequent home noise."),
        ("background_keyboard_close", "background", plosive_burst(count=6, burst_ms=18, gap_ms=90, amplitude=0.18), None, "no_callback", "Close keyboard clicks.", "Transient bursts should not become speech."),
        ("background_mic_plosive_no_speech", "background", plosive_burst(count=5, burst_ms=30, gap_ms=110, amplitude=0.30), None, "no_callback", "Breath/plosive bursts without words.", "Mic pops can fool duration accumulation."),
        ("background_door_slam_tail", "background", np.concatenate([click_burst(0.08, amplitude=0.45), reverberant_echo(tts_echo(0.92, amplitude=0.03), rt60_ms=280)]), None, "no_callback", "Door slam with room tail.", "Impulse sounds should not open a turn."),
        ("background_mic_bump", "background", click_burst(1.0, amplitude=0.20), None, "no_callback", "Microphone bump burst.", "Physical handling noise is common."),
        ("background_fan_plus_music", "background", mix(nonstationary_noise(1.0, 0.018, 0.030), music_noise(1.0, amplitude=0.028), snr_db=3.0), None, "no_callback", "Fan plus music bed.", "Layered noise is more realistic than one source."),
        ("background_tv_plus_babble", "background", mix(tv_base, babble_base, snr_db=4.0), None, "no_callback", "TV plus distant people.", "Mixed speech sources should not trigger intent."),
        ("background_low_bass_music", "background", apply_gain_db(music_noise(1.0, amplitude=0.025, beat_hz=1.2), 3.0), None, "no_callback", "Bass-heavy music pulse.", "Low-frequency energy can fool RMS gates."),
        ("background_window_traffic", "background", nonstationary_noise(1.0, quiet_amplitude=0.018, loud_amplitude=0.038, transition_at=0.6), None, "no_callback", "Traffic swell through window.", "Outdoor noise changes slowly."),
        ("background_coffee_grinder", "background", hard_clip(nonstationary_noise(1.0, 0.030, 0.070, transition_at=0.2), limit=0.13), None, "no_callback", "Coffee grinder burst.", "Loud appliance noise is highly nonstationary."),
        ("background_vacuum_far_room", "background", nonstationary_noise(1.0, quiet_amplitude=0.025, loud_amplitude=0.055, transition_at=0.1), None, "no_callback", "Vacuum in adjacent room.", "Sustained high noise should not be a command."),
        ("background_siren_outside", "background", music_noise(1.0, amplitude=0.045, beat_hz=0.8), None, "no_callback", "Siren-like tonal sweep approximation.", "Tonal external sounds can appear voiced."),
        ("background_baby_monitor", "background", mix(babble_noise(1.0, amplitude=0.025, num_speakers=1), tv_noise(1.0, amplitude=0.020), snr_db=6.0), None, "no_callback", "Baby monitor audio.", "Remote speech should not control assistant."),
        ("background_video_call_speaker", "background", babble_noise(1.0, amplitude=0.038, num_speakers=2), None, "no_callback", "Video call played through speakers.", "Assistant must ignore other calls."),
        ("background_white_noise_machine", "background", tv_noise(1.0, amplitude=0.042), None, "no_callback", "White-noise sleep machine.", "Noise machines run near smart speakers."),
        ("background_rain_on_window", "background", tv_noise(1.0, amplitude=0.030, seed=99), None, "no_callback", "Rain-like broadband noise.", "Weather noise is common and persistent."),
        ("background_chair_scrape", "background", hard_clip(click_burst(1.0, amplitude=0.24, seed=22), limit=0.16), None, "no_callback", "Chair scrape impulse train.", "Furniture noise should not start STT."),

        # User speech cases: must still be heard under realistic degradation.
        ("user_near_clean_question", "user", user_base, None, "callback", "Near user asks a normal question.", "Primary assistant function must work."),
        ("user_quiet_soft_speaker", "user", voiced_speech(0.9, amplitude=0.16), None, "callback", "Quiet soft-spoken user.", "Accessibility and low-volume speakers matter."),
        ("user_loud_close_talker", "user", voiced_speech(0.9, amplitude=0.38), None, "callback", "Loud close-talking user.", "Close talkers can saturate energy gates."),
        ("user_short_stop_command", "user", voiced_speech(0.32, amplitude=0.36), None, "callback", "Short stop command.", "Critical safety command must not be dropped."),
        ("user_short_yes_command", "user", voiced_speech(0.36, amplitude=0.32), None, "callback", "Short yes command.", "Brief confirmations are common."),
        ("user_short_no_command", "user", voiced_speech(0.34, amplitude=0.34), None, "callback", "Short no command.", "Brief corrections must be captured."),
        ("user_over_far_tv", "user", near_far_mix(user_base, tv_base, near_db=0, far_db=-12), None, "callback", "User speaks while TV is on.", "Common living-room command path."),
        ("user_over_loud_tv", "user", mix(user_base, tv_noise(0.9, amplitude=0.08), snr_db=6.0), None, "callback", "User speaks over loud TV.", "Assistant should work with media playing."),
        ("user_over_party_babble", "user", mix(user_base, _fit(babble_noise(0.9, amplitude=0.05, num_speakers=5), 0.9), snr_db=7.0), None, "callback", "User speaks at a party.", "Multi-speaker background should not mask user."),
        ("user_over_music", "user", mix(user_base, _fit(music_noise(0.9, amplitude=0.06), 0.9), snr_db=7.0), None, "callback", "User speaks over music.", "Music during commands is common."),
        ("user_over_hvac", "user", mix(user_base, _fit(nonstationary_noise(0.9, 0.015, 0.040), 0.9), snr_db=8.0), None, "callback", "User speaks as HVAC turns on.", "Noise-floor adaptation should not drop speech."),
        ("user_sample_rate_44100", "user", sample_rate_roundtrip(user_base, device_sr=44_100), None, "callback", "User through 44.1 kHz capture path.", "USB/headset devices often use 44.1 kHz."),
        ("user_sample_rate_48000", "user", sample_rate_roundtrip(voiced_speech(0.9, amplitude=0.255, pitch_hz=172.0), device_sr=48_000), None, "callback", "User through 48 kHz capture path.", "Most OS audio stacks use 48 kHz with slightly different antialiasing behavior."),
        ("user_bluetooth_codec_artifact", "user", hard_clip(sample_rate_roundtrip(user_base, device_sr=32_000), limit=0.32), None, "callback", "Bluetooth codec-like artifact.", "Bluetooth microphones are common."),
        ("user_clipped_mic_gain_high", "user", clipped_speech(0.9, clip_level=0.32, amplitude=0.30), None, "callback", "High-gain clipped microphone.", "Bad gain staging should not drop commands."),
        ("user_laptop_fan_and_speech", "user", mix(user_base, _fit(nonstationary_noise(0.9, 0.020, 0.035), 0.9), snr_db=10.0), None, "callback", "Laptop fan under user speech.", "Local assistants run on laptops."),
        ("user_far_field_across_room", "user", apply_gain_db(user_base, -6.0), None, "callback", "User speaks across room.", "Far-field commands must still reach callback."),
        ("user_far_field_with_tv", "user", mix(apply_gain_db(user_base, -4.0), _fit(tv_noise(0.9, amplitude=0.04), 0.9), snr_db=5.0), None, "callback", "Far user with TV playing.", "Difficult but important living-room case."),
        ("user_kitchen_appliance", "user", mix(user_base, _fit(nonstationary_noise(0.9, 0.025, 0.060, transition_at=0.4), 0.9), snr_db=6.0), None, "callback", "User near kitchen appliance.", "Kitchen assistants are common."),
        ("user_car_cabin_noise", "user", mix(user_base, _fit(nonstationary_noise(0.9, 0.030, 0.050, transition_at=0.5), 0.9), snr_db=8.0), None, "callback", "User in car cabin noise.", "Mobile/vehicle voice use matters."),
        ("user_plosive_heavy_speech", "user", mix(user_base, _fit(plosive_burst(count=4, burst_ms=22, gap_ms=130, amplitude=0.18), 0.9), snr_db=9.0), None, "callback", "User speech with plosive mic pops.", "Close microphones often pop on consonants."),
        ("user_room_reverb_voice", "user", mix(user_base, _fit(reverberant_echo(tts_echo(0.9, amplitude=0.03), rt60_ms=260), 0.9), snr_db=9.0), None, "callback", "User speech in reverberant room.", "Room sound affects user speech too."),
        ("user_compressed_hard_limited", "user", hard_clip(user_base * 2.2, limit=0.35), None, "callback", "Hard-limited microphone speech.", "AGC/limiters can distort user voice."),
        ("user_low_battery_mic_noise", "user", mix(user_base, _fit(tv_noise(0.9, amplitude=0.025, seed=123), 0.9), snr_db=9.0), None, "callback", "User with noisy microphone electronics.", "Cheap microphones add broadband noise."),
        ("user_open_window_traffic", "user", mix(user_base, _fit(nonstationary_noise(0.9, 0.020, 0.045, transition_at=0.7), 0.9), snr_db=7.0), None, "callback", "User near open window traffic.", "Traffic swells should not drop commands."),
        ("user_speaker_change_pitch_low", "user", voiced_speech(0.9, pitch_hz=105.0, amplitude=0.28), None, "callback", "Low-pitched speaker.", "Low-pitched speakers can have lower VAD confidence."),
        ("user_speaker_change_pitch_high", "user", voiced_speech(0.9, pitch_hz=235.0, amplitude=0.24), None, "callback", "High-pitched speaker.", "High-pitched speakers can be over-filtered or misclassified."),
        ("user_fast_syllables", "user", voiced_speech(0.9, syllable_rate=7.0, amplitude=0.26), None, "callback", "Fast syllabic speech.", "Fast talkers should be captured."),
        ("user_slow_syllables", "user", voiced_speech(0.9, syllable_rate=2.0, amplitude=0.27), None, "callback", "Slow syllabic speech.", "Slow talkers with pauses should not be dropped."),
        ("user_near_speech_with_assistant_echo", "user", mix(user_base, _fit(apply_room_delay(base_ref, 8.0, -8.0), 0.9), snr_db=8.0), None, "callback", "User speech while assistant echo is present.", "Barge-in capture must preserve user speech."),
        ("user_clipped_with_babble", "user", mix(hard_clip(user_base * 2.0, limit=0.32), _fit(babble_base, 0.9), snr_db=8.0), None, "callback", "Clipped user over babble.", "Multiple degradations combine in practice."),
        ("user_quiet_with_music", "user", mix(voiced_speech(0.9, amplitude=0.17), _fit(music_base, 0.9), snr_db=6.0), None, "callback", "Quiet user over music.", "Low-volume users are easily missed."),
        ("user_emergency_help_short", "user", voiced_speech(0.42, amplitude=0.34, syllable_rate=5.5), None, "callback", "Short emergency help command.", "Safety-critical short commands must fire."),
        ("user_wake_followup_short", "user", voiced_speech(0.48, amplitude=0.26, syllable_rate=4.5), None, "callback", "Short follow-up after wakeword.", "Wakeword workflows often use short commands."),
        ("user_noisy_usb_mic", "user", mix(sample_rate_roundtrip(user_base, device_sr=48_000), _fit(tv_noise(0.9, amplitude=0.030, seed=321), 0.9), snr_db=10.0), None, "callback", "Noisy USB mic path.", "USB microphones vary widely."),
        ("user_table_reflection_comb_filter", "user", mix(user_base, _fit(apply_room_delay(user_base, 4.0, -9.0), 0.9), snr_db=6.0), None, "callback", "User speech plus desk reflection.", "Comb filtering should not defeat endpointing."),
        ("user_phone_speaker_feedback", "user", mix(user_base, _fit(music_noise(0.9, amplitude=0.035, beat_hz=2.5), 0.9), snr_db=10.0), None, "callback", "User near a phone playing audio.", "Phones frequently play background media."),
        ("user_repeated_small_pauses", "user", voiced_speech(0.9, amplitude=0.25, syllable_rate=1.5), None, "callback", "User speech with long inter-word pauses.", "Soft-decay should preserve natural pauses."),
        ("echo_tablet_kickstand_reflection", "echo_leak", reverberant_echo(base_ref, direct_delay_ms=3, rt60_ms=170, direct_gain=0.88), "reference_clean_tts", "no_interrupt", "Tablet on kickstand reflecting off desk.", "Tablets are common always-on assistant devices."),
        ("echo_open_back_headphones_leak", "echo_leak", apply_room_delay(base_ref, 2.5, -14.0), "reference_clean_tts", "no_interrupt", "Open-back headphone leakage into mic.", "Headset users can still leak TTS into microphones."),
        ("echo_dual_desktop_monitors", "echo_leak", apply_room_delay(base_ref, 6.0, -5.0) + apply_room_delay(base_ref, 18.0, -9.0), "reference_clean_tts", "no_interrupt", "Desktop speakers reflecting between monitors.", "Desktop geometry creates strong multi-path reflections."),
        ("echo_smart_display_countertop", "echo_leak", reverberant_echo(base_ref, direct_delay_ms=5, rt60_ms=210, direct_gain=0.82), "reference_clean_tts", "no_interrupt", "Smart display on countertop.", "Smart displays combine speaker and mic in reflective kitchens."),
        ("background_alarm_beep_sequence", "background", plosive_burst(count=8, burst_ms=40, gap_ms=80, amplitude=0.22), None, "no_callback", "Alarm-like beep sequence.", "Alarms are loud periodic bursts near microphones."),
        ("background_phone_notification_cluster", "background", plosive_burst(count=4, burst_ms=35, gap_ms=160, amplitude=0.26), None, "no_callback", "Cluster of phone notifications.", "Notification sounds should not start listening."),
        ("background_mic_cable_crackle", "background", hard_clip(click_burst(1.0, amplitude=0.28, seed=44), limit=0.11), None, "no_callback", "Microphone cable crackle.", "Hardware noise can look like voiced transients."),
        ("background_roommate_laugh_far", "background", babble_noise(1.0, amplitude=0.033, num_speakers=1), None, "no_callback", "Roommate laugh in background.", "Non-command human vocalization should not trigger STT."),
        ("user_whisper_like_low_energy", "user", voiced_speech(0.9, amplitude=0.14, pitch_hz=180.0), None, "callback", "Whisper-like low-energy user.", "Low-energy users should still be served when close to mic."),
        ("user_command_after_loud_noise", "user", np.concatenate([click_burst(0.08, amplitude=0.35), voiced_speech(0.82, amplitude=0.27)]), None, "callback", "Command immediately after a loud noise.", "Users often speak right after moving objects or tapping devices."),
    ]

    for name, kind, audio, ref, expectation, description, importance in important_cases:
        _add_case(
            cases,
            samples,
            name=name,
            kind=kind,
            audio=_fit(audio, 1.0 if kind != "echo_leak" else 1.4),
            ref=ref,
            expectation=expectation,
            description=description,
            importance=importance,
        )

    assert len(cases) >= MIN_CORPUS_CASES, len(cases)
    assert len({case["name"] for case in cases}) == len(cases)

    return samples, cases


def _save_samples() -> dict[str, np.ndarray]:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    samples, cases = _build_corpus()
    for stale in SAMPLE_DIR.glob("*.npy"):
        stale.unlink()
    for name, audio in samples.items():
        np.save(SAMPLE_DIR / f"{name}.npy", audio.astype(np.float32))
    metadata = {
        "version": CORPUS_VERSION,
        "case_count": len(cases),
        "sample_rate": SR,
        "cases": cases,
    }
    (SAMPLE_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return samples


def _load_samples() -> dict[str, np.ndarray]:
    metadata_path = SAMPLE_DIR / "metadata.json"
    if not metadata_path.exists():
        return _save_samples()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("version") != CORPUS_VERSION
        or metadata.get("case_count", 0) < MIN_CORPUS_CASES
    ):
        return _save_samples()
    expected = {case["name"] for case in metadata.get("cases", [])}
    missing = [name for name in expected if not (SAMPLE_DIR / f"{name}.npy").exists()]
    if missing:
        return _save_samples()
    return {
        name: np.load(SAMPLE_DIR / f"{name}.npy").astype(np.float32)
        for name in expected
    }


SAMPLES = _load_samples()
METADATA = json.loads((SAMPLE_DIR / "metadata.json").read_text(encoding="utf-8"))
CASES = METADATA["cases"]
ACTION_CASES = [
    case for case in CASES if case["expectation"] in {"callback", "no_callback", "no_interrupt"}
]


def _interrupts_for_tts_leak(
    mic_audio: np.ndarray,
    tts_ref: np.ndarray,
    *,
    gate_sec: float = 0.0,
    inter_chunk_delay: float = 0.0,
    zero_delays: bool = True,
) -> list[dict]:
    interrupts: list[dict] = []
    rec = make_recorder(
        on_interrupt=lambda info=None: interrupts.append(info or {}),
        barge_in_min_speech_sec=0.10,
        barge_in_min_delay_after_ref_sec=gate_sec,
        barge_in_min_delay_sec=0.0,
        barge_in_cooldown_sec=0.0,
        aec_enabled=False,
    )
    rec._noise_floor = 0.004
    with AudioHarness(rec) as harness:
        harness.set_tts_speaking(audio_ref=tts_ref, zero_delays=zero_delays)
        harness.inject(mic_audio, inter_chunk_delay=inter_chunk_delay)
        harness.drain(timeout=5.0)
    return interrupts


def _callbacks_for_user_audio(audio: np.ndarray) -> list[np.ndarray]:
    callbacks: list[np.ndarray] = []
    rec = make_recorder(
        callback=lambda captured: callbacks.append(captured),
        silence_duration=0.08,
        aec_enabled=False,
    )
    with AudioHarness(rec) as harness:
        harness.inject(audio, inter_chunk_delay=0.02)
        harness.inject(silence(0.35), inter_chunk_delay=0.02)
        harness.drain(timeout=5.0)
    return callbacks


def test_local_failure_samples_are_saved():
    assert (SAMPLE_DIR / "metadata.json").exists()
    assert METADATA["case_count"] >= MIN_CORPUS_CASES
    assert len(list(SAMPLE_DIR.glob("*.npy"))) >= MIN_CORPUS_CASES


def test_saved_samples_are_16khz_float32():
    for case in CASES:
        name = case["name"]
        audio = SAMPLES[name]
        assert audio.dtype == np.float32, name
        assert len(audio) >= int(0.75 * SR), name


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda c: c["name"])
def test_failure_discovery_corpus(case):
    audio = SAMPLES[case["name"]]
    expectation = case["expectation"]

    if expectation == "callback":
        callbacks = _callbacks_for_user_audio(audio)
        assert callbacks, case["description"]
        return

    if expectation == "no_callback":
        callbacks = _callbacks_for_user_audio(audio)
        assert callbacks == [], case["description"]
        return

    ref = SAMPLES[case["ref"]]
    if case["kind"] == "echo_control":
        interrupts = _interrupts_for_tts_leak(
            audio,
            ref,
            gate_sec=0.20,
            inter_chunk_delay=1024 / SR,
            zero_delays=False,
        )
    else:
        interrupts = _interrupts_for_tts_leak(audio, ref)
    assert interrupts == [], (
        f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
    )
