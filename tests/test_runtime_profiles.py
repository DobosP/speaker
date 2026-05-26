"""
Tests for runtime profile helpers in STT/LLM layers.
"""
import unittest

from utils.stt import resolve_stt_runtime
from utils.llm import LLM_GENERATION_PROFILES


class TestRuntimeLayerProfiles(unittest.TestCase):
    def test_resolve_stt_runtime_edge(self):
        cfg = resolve_stt_runtime("edge")
        self.assertEqual(cfg["model_type"], "whispercpp")

    def test_resolve_stt_runtime_override_model(self):
        cfg = resolve_stt_runtime("balanced", model_id="small")
        self.assertEqual(cfg["model_id"], "small")
        self.assertEqual(cfg["model_type"], "whisper")

    def test_llm_generation_profiles_present(self):
        self.assertIn("edge", LLM_GENERATION_PROFILES)
        self.assertIn("balanced", LLM_GENERATION_PROFILES)
        self.assertIn("max_quality", LLM_GENERATION_PROFILES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
