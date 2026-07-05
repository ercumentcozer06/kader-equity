"""
screen/test_conditional_derisk — K2 kosullu de-risk kuralinin ON-KAYITLI davranisini KILITLER.

Ag YOK: donmus spine snapshot + repo-ici net_equity_supply.parquet okur. Calistir:
  C:/Users/admin/Downloads/kader-macro/.venv/Scripts/python.exe -X utf8 -m pytest screen/test_conditional_derisk.py -q
(repo kokunden). Kilitlenen sozlesme:
  * SABITLER on-kayitli degerlerinde (esik/trim oynamadi),
  * 2020 GERCEK-GUCLU-TALEP ceyreklerinde (Haz/Eyl/Ara PIT) fire YOK (kritik sessizlik),
  * 2021'de fire VAR,
  * trim zararsiz (dSharpe >= -0.03 her iki varlik) ve trim-only (faktor <= 1),
  * deterministik (iki kosu ayni fire-maskesi).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from screen import candidate_conditional_derisk as K2   # noqa: E402


@pytest.fixture(scope="module")
def S():
    return K2.build_signals()


def _asof(series, q):
    d = pd.Timestamp(q)
    sub = series[series.index <= d]
    return sub.iloc[-1]


def test_constants_are_prereg_frozen():
    # on-kayitli sabitler degismemis olmali (sonuclardan sonra oynama korumasi)
    assert K2.SUPPLY_Z_THR == 1.0
    assert K2.DEMAND_WEAK_LEVEL == 2.0
    assert K2.DECLINE_LB == 63
    assert K2.TRIM == 0.85
    assert K2.DSHARPE_FLOOR == -0.03


def test_2020_strong_demand_quarters_silent(S):
    """KRITIK: 2020 Haz/Eyl/Ara PIT (arz-yuksek AMA tide_score +5.9..+11.7, dir=1) -> fire YOK."""
    fire = S["fire"]
    for q in ("2020-06-15", "2020-09-14", "2020-12-14"):
        assert bool(_asof(fire.astype(float), q)) is False, f"2020 guclu-talep {q} YANLIS-ATESLEDI"
    # bu ceyreklerde tide_score gercekten guclu (level-kapisini gecmeli)
    ts = S["ts"]
    assert _asof(ts, "2020-06-15") > K2.DEMAND_WEAK_LEVEL
    assert _asof(ts, "2020-09-14") > K2.DEMAND_WEAK_LEVEL
    assert _asof(ts, "2020-12-14") > K2.DEMAND_WEAK_LEVEL


def test_2020_stray_fires_are_spine_flat_days(S):
    """2020'de varsa fire gunleri (pre-secim Eki-Kas mini-cozulmesi) spine'in KENDI FLAT
    gunleridir (tide_dir cogunlukla 0), 'guclu-talep raliyi kesme' degil."""
    fire, tdir, idx = S["fire"], S["tdir"], S["idx"]
    f2020 = fire[(idx.year == 2020) & fire.values].index
    if len(f2020) == 0:
        return
    # bu gunlerin tide_score'u DUSUK (level kapisi <= +2 ya da dir==0) olmali
    ts = S["ts"]
    assert (ts.reindex(f2020) <= K2.DEMAND_WEAK_LEVEL).all()
    # cogunlugu spine-FLAT (dir==0)
    assert (tdir.reindex(f2020) == 0).mean() >= 0.5


def test_2021_fires(S):
    fire, idx = S["fire"], S["idx"]
    assert int(fire[idx.year == 2021].sum()) > 0


def test_2021_collapse_quarter_fires_negative_forward(S):
    """2021Q4 PIT (tide_score coker, dir->0) atesler VE forward NEGATIF (trim hakli)."""
    fire = S["fire"]
    assert bool(_asof(fire.astype(float), "2021-12-13")) is True
    spx = S["prices"]["SPX"].dropna()
    f126 = K2.fwd_from(spx, pd.Timestamp("2021-12-13"), 126)
    assert f126 < 0.0


def test_trim_only_factor_bounded(S):
    """Faktor daima <= 1 (trim-only, rebound-safe — overlay gelenegi)."""
    fire, idx = S["fire"], S["idx"]
    fac = pd.Series(np.where(fire.values, K2.TRIM, 1.0), index=idx)
    assert fac.max() <= 1.0 + 1e-12
    assert fac.min() == pytest.approx(K2.TRIM)


def test_incremental_harmless_both_assets(S):
    """2019+ tide-ustu: her iki varlikta dSharpe >= -0.03 (zararsizlik on-kayitli esigi)."""
    fire, idx, tdir, prices = S["fire"], S["idx"], S["tdir"], S["prices"]
    fac = pd.Series(np.where(fire.values, K2.TRIM, 1.0), index=idx)
    for a in ("SPX", "NDX"):
        base = K2.strat_ret(tdir, prices[a])
        var = K2.strat_ret((tdir * fac).reindex(idx), prices[a])
        dsh = K2._sh(var) - K2._sh(base)
        assert dsh >= K2.DSHARPE_FLOOR, f"{a} dSharpe {dsh:+.3f} < {K2.DSHARPE_FLOOR}"


def test_deterministic(S):
    """Iki bagimsiz kosu ayni fire-maskesini uretir (deterministik, ag-bagimsiz)."""
    S2 = K2.build_signals()
    assert S["fire"].equals(S2["fire"])
