"""
Tests for OVOS-style capability registry adapters.
"""
import unittest

from utils.capabilities import (
    CapabilityRegistry,
    CapabilityRequest,
    CapabilityResponse,
    create_default_registry,
)


class TestCapabilityRegistry(unittest.TestCase):
    def test_register_and_invoke(self):
        registry = CapabilityRegistry()

        def provider(req: CapabilityRequest) -> CapabilityResponse:
            return CapabilityResponse(ok=True, data={"value": req.payload.get("x", 0)})

        registry.register("demo.value", provider)
        resp = registry.invoke(CapabilityRequest(name="demo.value", payload={"x": 7}))
        self.assertTrue(resp.ok)
        self.assertEqual(resp.data["value"], 7)

    def test_missing_capability(self):
        registry = CapabilityRegistry()
        resp = registry.invoke(CapabilityRequest(name="missing", payload={}))
        self.assertFalse(resp.ok)
        self.assertIn("not found", resp.error.lower())

    def test_default_registry_has_builtins(self):
        registry = create_default_registry()
        caps = registry.list_capabilities()
        self.assertIn("system.time", caps)
        self.assertIn("debug.echo", caps)


if __name__ == "__main__":
    unittest.main(verbosity=2)
