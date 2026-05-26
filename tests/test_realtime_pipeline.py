"""Unit tests for realtime pipeline queue helpers."""
import queue

import pytest

from utils.realtime_pipeline import TtsItem, flush_queue

pytestmark = pytest.mark.dev


def test_flush_queue_drains():
    q: queue.Queue = queue.Queue()
    q.put(1)
    q.put(2)
    n = flush_queue(q)
    assert n == 2
    assert q.empty()


def test_tts_item_frozen():
    item = TtsItem(text="hi", speak_session=3)
    assert item.speak_session == 3
