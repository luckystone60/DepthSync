from __future__ import annotations

import sys
from tools.run_flow_model import _peak_working_set_bytes


def test_peak_working_set_is_reported_on_windows() -> None:
    value = _peak_working_set_bytes()
    if sys.platform == "win32":
        assert isinstance(value, int)
        assert value > 0
