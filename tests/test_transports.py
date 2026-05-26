"""
Tests for transport abstraction and session mux.
"""
import unittest

from utils.transports import SessionEnvelope, SessionMux, TransportMode


class TestSessionMux(unittest.TestCase):
    def test_local_mode_activation(self):
        mux = SessionMux(mode=TransportMode.LOCAL_LAN)
        mux.start()
        self.assertEqual(mux.active_transports(), ["local_lan"])
        mux.stop()

    def test_webrtc_mode_activation(self):
        mux = SessionMux(mode=TransportMode.WEBRTC)
        mux.start()
        self.assertEqual(mux.active_transports(), ["webrtc"])
        mux.stop()

    def test_hybrid_broadcast(self):
        mux = SessionMux(mode=TransportMode.HYBRID)
        mux.start()
        envelope = SessionEnvelope(
            session_id="s1",
            event_type="assistant_sentence",
            payload={"text": "hello"},
        )
        mux.broadcast(envelope)
        local = mux.local.pop_outbound(timeout=0.01)
        rtc = mux.webrtc.pop_outbound(timeout=0.01)
        self.assertIsNotNone(local)
        self.assertIsNotNone(rtc)
        self.assertEqual(local.event_type, "assistant_sentence")
        self.assertEqual(rtc.payload["text"], "hello")
        mux.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
