"""
Runtime profile and latency-budget guard tests.
"""
import unittest
import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.dev]

from main import (
    RESOURCE_PROFILES,
    RUNTIME_PROFILES,
    validate_runtime_config,
    validate_profile_transport_config,
)


class TestRuntimeProfiles(unittest.TestCase):
    def test_profiles_exist(self):
        self.assertIn("low", RESOURCE_PROFILES)
        self.assertIn("mid", RESOURCE_PROFILES)
        self.assertIn("high", RESOURCE_PROFILES)
        self.assertIn("edge", RUNTIME_PROFILES)
        self.assertIn("balanced", RUNTIME_PROFILES)
        self.assertIn("max_quality", RUNTIME_PROFILES)

    def test_profile_latency_budgets_shape(self):
        # Coarse budget assertions to prevent accidental regressions in defaults.
        low = RESOURCE_PROFILES["low"]
        mid = RESOURCE_PROFILES["mid"]
        high = RESOURCE_PROFILES["high"]
        self.assertLessEqual(low["aec_filter_ms"], mid["aec_filter_ms"])
        self.assertLessEqual(mid["aec_filter_ms"], high["aec_filter_ms"])
        self.assertTrue(low["streaming_llm"])
        self.assertTrue(mid["streaming_llm"])
        self.assertTrue(high["streaming_llm"])

    def test_validate_runtime_config_rejects_invalid_values(self):
        issues = validate_runtime_config(
            {
                "barge_in_min_delay_sec": -1.0,
                "barge_in_min_delay_after_ref_sec": -1.0,
                "barge_in_min_rms_ratio": 0.5,
                "echo_corr_threshold": 1.5,
                "aec_filter_ms": 5.0,
            }
        )
        self.assertGreaterEqual(len(issues), 4)

    def test_validate_runtime_config_accepts_typical_values(self):
        issues = validate_runtime_config(
            {
                "barge_in_min_delay_sec": 0.5,
                "barge_in_min_delay_after_ref_sec": 0.7,
                "barge_in_min_rms_ratio": 3.0,
                "echo_corr_threshold": 0.45,
                "aec_filter_ms": 120.0,
            }
        )
        self.assertEqual(issues, [])

    def test_validate_profile_transport_config(self):
        issues = validate_profile_transport_config(
            deployment_profile="mid",
            runtime_profile="balanced",
            transport_mode="hybrid",
        )
        self.assertEqual(issues, [])

    def test_validate_profile_transport_config_flags_risky_combo(self):
        issues = validate_profile_transport_config(
            deployment_profile="low",
            runtime_profile="edge",
            transport_mode="webrtc",
        )
        self.assertTrue(any("not recommended" in i for i in issues))


if __name__ == "__main__":
    unittest.main(verbosity=2)
