"""2026-08-03 sağlık denetimi regresyonları: K2 PIT tarihi + NY piyasa saati."""
from __future__ import annotations

from datetime import date, datetime, timezone
import json


def test_market_date_uses_new_york_not_host_calendar():
    from modules.market_clock import market_date

    # İstanbul'da 3 Ağustos olsa da New York hâlâ 2 Ağustos.
    now = datetime(2026, 8, 2, 22, 40, tzinfo=timezone.utc)
    assert market_date(now) == date(2026, 8, 2)


def test_state_latest_ignores_future_dated_snapshot(tmp_path, monkeypatch):
    from engine import state as S

    d = tmp_path / "gamma_spy"
    d.mkdir()
    (d / "2026-08-02.json").write_text(
        json.dumps({"as_of": "2026-08-02", "spot": 100}), encoding="utf-8")
    (d / "2026-08-03.json").write_text(
        json.dumps({"as_of": "2026-08-03", "spot": 999}), encoding="utf-8")
    monkeypatch.setattr(S, "CACHE", tmp_path)
    monkeypatch.setattr(S, "market_date", lambda: date(2026, 8, 2))

    assert S._latest("gamma_spy")["spot"] == 100


def test_state_future_snapshot_is_fail_closed(monkeypatch):
    import run
    from engine import state as S

    gamma = {
        "as_of": "2026-08-03", "spot": 100.0, "net_gex_bn": -1.0,
        "exp_move_1d": 1.0, "gex_flip": 101.0, "call_wall": 102.0,
        "put_wall": 98.0,
    }
    surface = {
        "as_of": "2026-08-03", "spot": 100.0,
        "surface": {"30d": {"atm_iv": 16.0}},
    }
    monkeypatch.setattr(S, "market_date", lambda: date(2026, 8, 2))
    monkeypatch.setattr(
        run, "build_decision",
        lambda cfg: {"direction": "LONG", "position_target": 1.0,
                     "call_status": "current", "overlays": {}},
    )
    monkeypatch.setattr(S, "_latest", lambda sub: surface if sub.startswith("surface") else gamma)
    monkeypatch.setattr(S, "_realized_vol_ewma", lambda *a, **k: 14.0)

    _model, _state, meta = S.build_state({}, "SPY")
    assert meta["snapshot_age_days"] == -1
    assert meta["stale"] is True
    assert meta["data_junk"] is True
    assert any("gelecekte" in x for x in meta["data_fails"])


def test_k2_live_default_uses_market_date_not_frozen_tail(monkeypatch):
    from modules import market_clock
    from modules import supply_demand_derisk as D

    monkeypatch.setattr(market_clock, "market_date", lambda: date(2026, 7, 31))
    out = D.evaluate({}, live_tide_score=7.7839, live_tide_dir=1)

    assert out is not None
    assert out["as_of"] == "2026-07-31"
    assert out["mega_ceiling_usd"] > 8e10
    assert out["mega_hi"] is True
    assert out["supply_z"] == 1.29


def test_buyback_unhealthy_status_counts_as_failed_substep(monkeypatch):
    import importlib
    import run_daily

    class FakeSupplyComponents:
        @staticmethod
        def auto_pull_buyback(write=True):
            return {"status": "no_listing"}

    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib, "import_module",
        lambda name: FakeSupplyComponents if name == "screen.fetch_supply_components" else real_import(name),
    )

    assert run_daily._substep("buyback-test", run_daily._bb_pull) is False


def test_constan_partial_failure_is_visible_degraded(monkeypatch):
    import pytest
    import run_daily

    monkeypatch.setattr(run_daily, "_ipo_pull", lambda: None)
    monkeypatch.setattr(run_daily, "_bb_pull", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(run_daily, "_net_supply_pull", lambda: None)
    monkeypatch.setattr(run_daily, "_balance_derive", lambda: None)

    with pytest.raises(RuntimeError, match="DEGRADED"):
        run_daily._refresh_constan()


def test_forward_validation_serializes_strict_json():
    import math
    from backtest.gex_master.forward_validation import strict_json_value

    clean = strict_json_value({"a": float("nan"), "b": [float("inf"), 1.0]})
    assert clean == {"a": None, "b": [None, 1.0]}
    assert not any(isinstance(x, float) and not math.isfinite(x) for x in clean["b"])
