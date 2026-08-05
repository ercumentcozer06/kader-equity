"""es_basis_unwind overlay kilitleri (deploy 2026-08-04, Emir karari).

1) factor-math birim  2) trim-only + tetik-nadirlik  3) TARIHSEL KILIT: 22-Jun-26 tetigi (lab reproduce)
4) evaluate fail-closed (parquet yok -> notr + available=False)  5) config okundu (enabled + esikler)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules import es_basis_unwind as EB   # noqa: E402

PARQ = ROOT / "data" / "cache" / "es_basis_daily.parquet"


def test_unwind_factor_math():
    assert EB.unwind_factor(0.80, -2.0) == 0.0          # zengin + hizli donus -> flat
    assert EB.unwind_factor(0.50, -2.0) == 1.0          # zengin degil -> notr
    assert EB.unwind_factor(0.80, -1.0) == 1.0          # donus yok -> notr
    assert EB.unwind_factor(None, -2.0) == 1.0          # veri yok -> notr
    assert EB.unwind_factor(0.80, None) == 1.0
    assert EB.unwind_factor(0.80, -2.0, floor=0.25) == 0.25


def test_series_trim_only_and_rare():
    fs = EB.unwind_factor_series(pd.read_parquet(PARQ)["spread_bps"])
    assert set(fs.unique()) <= {0.0, 1.0}               # binary trim-only
    assert 0.0 < (fs == 0.0).mean() < 0.05              # tetik var ama NADIR (<%5 gun)


def test_jun26_trigger_lock():
    """Lab-reproduce kilidi: 22-Jun-26 mania-unwind tetigi (T+1 SPX -1.44/NDX -3.22 epizodu)."""
    fs = EB.unwind_factor_series(pd.read_parquet(PARQ)["spread_bps"])
    assert fs.loc["2026-06-22"] == 0.0
    assert fs.loc["2026-06-18"] == 1.0                  # tetik oncesi isgunu notr (06-19 Juneteenth tatil)


def test_should_write_guards():
    """Gecmis gune dokunma; settle-satiri (F1 dolu) ezilmez; canli-append (F1 NaN) gun-ici guncellenir."""
    import numpy as np
    base = pd.DataFrame({"F1": [7500.0, np.nan], "spread_bps": [30.0, -97.0]},
                        index=pd.to_datetime(["2026-07-31", "2026-08-04"]))
    assert EB._should_write(pd.Timestamp("2026-08-03"), base) is False   # gecmis
    assert EB._should_write(pd.Timestamp("2026-08-04"), base) is True    # canli-append -> guncelle
    assert EB._should_write(pd.Timestamp("2026-08-05"), base) is True    # yeni gun
    settle = base.copy(); settle.loc[pd.Timestamp("2026-08-04"), "F1"] = 7660.0
    assert EB._should_write(pd.Timestamp("2026-08-04"), settle) is False  # settle EZILMEZ


def test_evaluate_fail_closed_when_parquet_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(EB, "PARQ", tmp_path / "yok.parquet")
    out = EB.evaluate({"overlays": {"es_basis_unwind": {}}})
    assert out["factor"] == 1.0 and out["available"] is False and out["warning"]


def test_config_wired():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    ov = cfg["overlays"]["es_basis_unwind"]
    assert ov["enabled"] is True
    assert ov["p_thr"] == 0.75 and ov["dz_thr"] == -1.5 and ov["floor"] == 0.0
