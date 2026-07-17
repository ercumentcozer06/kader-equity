from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


PATH = Path(__file__).resolve().parents[1] / "backtest" / "gex_master" / "canonical_playbook.py"
SPEC = importlib.util.spec_from_file_location("canonical_playbook", PATH)
C = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(C)


def minute_day(rows):
    idx = pd.date_range("2026-01-05 09:30", periods=len(rows), freq="1min", tz="America/New_York")
    return pd.DataFrame(rows, index=idx)


def test_fill_occurs_after_confirmation_bar():
    day = minute_day([{"o": 100 + i, "h": 101 + i, "l": 99 + i, "c": 100.5 + i, "v": 1}
                      for i in range(10)])
    ts, px = C.fill_after_bar(day, day.index[0])
    assert ts == day.index[5]
    assert px == day.iloc[5]["o"]


def test_same_minute_stop_and_target_is_stop_first():
    day = minute_day([
        {"o": 100, "h": 103, "l": 98, "c": 101, "v": 1},
        {"o": 101, "h": 101, "l": 100, "c": 100, "v": 1},
    ])
    got = C.simulate(day, day.index[0], entry=100, side=1, stop=99, target=102)
    assert got["reason"] == "stop"
    assert got["exit"] == 99
    assert got["r_multiple_gross"] == -1


def test_short_stop_target_geometry():
    day = minute_day([
        {"o": 100, "h": 100.2, "l": 97.5, "c": 98, "v": 1},
        {"o": 98, "h": 98.1, "l": 97.9, "c": 98, "v": 1},
    ])
    got = C.simulate(day, day.index[0], entry=100, side=-1, stop=101, target=98)
    assert got["reason"] == "target"
    assert got["r_multiple_gross"] == 2


def test_acceptance_requires_cross_from_other_side():
    idx = pd.date_range("2026-01-05 09:30", periods=3, freq="5min", tz="America/New_York")
    b = pd.DataFrame({"o": [99, 99.5, 101], "h": [100, 101.5, 102], "l": [98, 99, 100],
                      "c": [99.5, 101.2, 101.5], "v": [1, 1, 1]}, index=idx)
    ev = C.first_accept(b, level=100, em=1, side=1)
    assert ev is not None
    assert ev[0] == idx[1]


def test_rejection_requires_touch_and_close_back_inside():
    idx = pd.date_range("2026-01-05 09:30", periods=2, freq="5min", tz="America/New_York")
    b = pd.DataFrame({"o": [99, 99], "h": [99.8, 100.2], "l": [98.5, 98.8],
                      "c": [99.5, 99.8], "v": [1, 1]}, index=idx)
    ev = C.first_reject(b, level=100, em=1, side_after=-1)
    assert ev is not None
    assert ev[0] == idx[1]
