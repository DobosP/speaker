"""pytest session configuration.

Ensures the repository root is importable so tests can ``import core``,
``always_on_agent``, ``utils.memory``, and ``tests.sandbox``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
