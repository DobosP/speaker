"""Unit tests for partial STT model resolution (no model downloads)."""
import pytest

from utils.stt import resolve_partial_stt_config

pytestmark = pytest.mark.dev


def test_partial_never_defaults_to_same_large_final_model():
    r = resolve_partial_stt_config(
        partial_model=None,
        partial_backend=None,
        final_model_id="large-v3-turbo",
        n_threads=4,
    )
    assert r["model_id"] == "tiny"
    assert r["backend"] == "whispercpp"


def test_explicit_heavy_partial_is_downgraded():
    r = resolve_partial_stt_config(
        partial_model="large-v3-turbo",
        partial_backend="whispercpp",
        final_model_id="large-v3-turbo",
        n_threads=4,
    )
    assert r["model_id"] == "tiny"


def test_moonshine_backend_normalizes_model_id():
    r = resolve_partial_stt_config(
        partial_model="tiny",
        partial_backend="moonshine",
        final_model_id="base",
        n_threads=2,
    )
    assert r["model_id"] == "moonshine:tiny"
    assert r["backend"] == "moonshine"
