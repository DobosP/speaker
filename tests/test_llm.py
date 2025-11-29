"""
Unit tests for the LLM module.
"""
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm import get_llm, LocalLLM


class TestLLMModule(unittest.TestCase):
    """Test the LLM module."""
    
    def test_invalid_llm_type(self):
        """Should raise error for unsupported LLM type."""
        with self.assertRaises(ValueError):
            get_llm(llm_type="invalid_type")
    
    def test_local_llm_creation(self):
        """Should create LocalLLM instance (requires Ollama running)."""
        try:
            llm = get_llm(llm_type="local", model="llama2")
            self.assertIsInstance(llm, LocalLLM)
        except ConnectionError as e:
            self.skipTest(f"Ollama not running: {e}")
    
    def test_llm_response(self):
        """Should get a response from LLM (requires Ollama running)."""
        try:
            llm = get_llm(llm_type="local", model="llama2")
            response = llm.get_response("Say 'test' and nothing else.")
            self.assertIsInstance(response, str)
            self.assertTrue(len(response) > 0)
        except ConnectionError as e:
            self.skipTest(f"Ollama not running: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)

