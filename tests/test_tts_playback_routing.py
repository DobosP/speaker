"""Unit tests for TTS output device routing (no microphone or speakers required)."""
import os
import sys
import unittest
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.dev

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio import resolve_output_device  # noqa: E402


class TestResolveOutputDevice(unittest.TestCase):
    def test_explicit_output_wins(self):
        self.assertEqual(resolve_output_device(4, 2), 2)

    def test_input_combo_device_reused(self):
        dev = {"max_output_channels": 2, "hostapi": 1, "name": "Analog in/out"}
        with patch("utils.audio.sd.query_devices", return_value=dev):
            self.assertEqual(resolve_output_device(7, None), 7)

    def test_hdmi_default_routes_to_analog_on_same_card(self):
        in_info = {
            "max_output_channels": 0,
            "hostapi": 1,
            "name": "HDA Intel PCH: ALC285 Analog",
        }
        hdmi_out = {
            "max_output_channels": 2,
            "hostapi": 0,
            "name": "HDA NVidia: BenQ GL2580",
        }
        analog_out = {
            "max_output_channels": 4,
            "hostapi": 1,
            "name": "HDA Intel PCH: ALC285 Analog",
        }
        all_devs = [hdmi_out, analog_out, in_info]

        def query(arg=None):
            if arg is None:
                return all_devs
            if arg == 4:
                return in_info
            if arg == 0:
                return hdmi_out
            if arg == 1:
                return analog_out
            return in_info

        with patch("utils.audio.sd.default") as default:
            default.device = [4, 0]
            with patch("utils.audio.sd.query_devices", side_effect=query):
                self.assertEqual(resolve_output_device(None, None), 1)


if __name__ == "__main__":
    unittest.main()
