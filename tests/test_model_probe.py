from tools.model_probe import _final_status


def test_text_only_model_passes_when_text_is_healthy():
    rows = [{"ok": True}, {"ok": True}]
    mm = {"supported": False, "error": "image input is not supported"}
    assert _final_status(rows, mm, require_vision=False) == "ok"


def test_vision_role_fails_on_unsupported_images():
    rows = [{"ok": True}]
    mm = {"supported": False, "error": "image input is not supported"}
    assert _final_status(rows, mm, require_vision=True) == "fail"


def test_any_bad_text_probe_fails_even_for_text_only_role():
    assert _final_status(
        [{"ok": True}, {"ok": False}], {"sees_image": True}, require_vision=False
    ) == "fail"
