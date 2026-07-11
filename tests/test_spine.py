"""
test_spine — Faz 0 spine kilitleri + faithful reprodüksiyon + tazelik + Pass-0 invariant.

Donmuş equities-tide spine: RAW m2 + 8-modül + çapa SPX 1.43 / NQ 1.49 (parite kanıtı _tide_liveparity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spine import contract as C, tide as T
from backtest import engine as E


def test_frozen_locks():
    _scores, _prices, _vector, prov = C.read_frozen()
    assert prov["locks"]["raw_m2"] is True
    assert int(prov["locks"]["n_modules_nonzero"]) >= 8
    assert float(prov["anchor"]["SPX"]["sharpe"]) >= 1.40
    assert float(prov["anchor"]["NDX"]["sharpe"]) >= 1.40


def test_vector_canonical_and_contaminants_zero():
    _s, _p, vector, _prov = C.read_frozen()
    canon = {"m9": 0.563, "m5": 0.214, "m2": 0.118, "m0": 0.061, "m3": 0.025, "m6": 0.01, "m8": 0.006, "m4": 0.002}
    drift = max(abs(float(vector.get(k, 0.0)) - cv) for k, cv in canon.items())
    assert drift < 0.05, f"vektör kanonikten saptı: {drift}"
    # m1/m10/m11 = equity-kontaminantlar: sweep Dirichlet-winner ~5e-5 toz taşır (m9 .563'e karşı
    # ekonomik SIFIR; tide skoruna katkı ~±0.001). Tam sweep-winner'ı koruyoruz (faithful) → negligible eşik.
    for k in ("m1", "m10", "m11"):
        assert abs(float(vector.get(k, 0.0))) < 1e-3


def test_faithful_reproduce_anchor():
    """Donmuş spine, kader-macro/ağ olmadan çapa Sharpe'ı byte-yakın üretir."""
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    for a in ("SPX", "NDX"):
        st = E.backtest_dir(tdir, prices[a], lag=1)["strat"]
        assert np.isclose(round(st["sharpe"], 3), float(prov["anchor"][a]["sharpe"]), atol=0.002)


def test_freshness_gate():
    today = pd.Timestamp("2026-06-09").date()
    assert C.snapshot_freshness(pd.Timestamp("2026-05-01"), 5, today=today)["stale"] is True
    assert C.snapshot_freshness(pd.Timestamp("2026-06-07"), 5, today=today)["stale"] is False


def test_exec_lag_no_lookahead():
    """+1g lag dürüst spine; lag0 look-ahead şişirir (lag1 >= 1.40 olmalı, lag0 ondan farklı)."""
    scores, prices, vector, _prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    s0 = E.backtest_dir(tdir, prices["SPX"], lag=0)["strat"]["sharpe"]
    s1 = E.backtest_dir(tdir, prices["SPX"], lag=1)["strat"]["sharpe"]
    assert s1 >= 1.40
    assert not np.isclose(s0, s1, atol=1e-6)               # lag gerçekten önemli


def test_overlays_off_equals_tide():
    """Faithful invariant: overlay'ler OFF → position_target == tide_dir (spine'a byte-identik)."""
    import copy
    from config import load_config
    import run
    cfg = copy.deepcopy(load_config())
    cfg["overlays"] = {}                                   # tüm overlay'leri kapat
    d = run.build_decision(cfg)
    assert d["active_overlays"] == []
    assert float(d["position_target"]) == float(d["tide_dir"])
    assert d["direction"] in ("LONG", "FLAT")


def test_cor1m_froth_factor():
    """froth_factor: lineer ramp, [floor,1] sınırlı, veri-yok → 1.0 (asla agresif)."""
    from modules.cor1m_froth import froth_factor
    assert froth_factor(15, 8, 11, 0.0) == 1.0            # yüksek COR1M → normal
    assert froth_factor(8, 8, 11, 0.0) == 0.0            # lo → floor
    assert froth_factor(6, 8, 11, 0.0) == 0.0            # lo altı → floor (clip)
    assert abs(froth_factor(9.5, 8, 11, 0.0) - 0.5) < 1e-9   # orta → ramp
    assert froth_factor(None, 8, 11, 0.0) == 1.0         # veri yok → nötr
    assert froth_factor(float("nan"), 8, 11, 0.0) == 1.0


def test_cor1m_froth_superseded():
    """2026-07-08: cor1m_froth dispersion_ensemble ile SUPERSEDED (config enabled:false) → aktif DEĞİL.
    (froth_factor math testi test_cor1m_froth_factor'da korunur; modül geri-alınabilir fallback olarak durur.)"""
    from config import load_config
    import run
    d = run.build_decision(load_config())
    assert "cor1m_froth" not in d["active_overlays"]      # halef devraldı, çift-say yok


def test_dispersion_ensemble_factor():
    """ensemble_factor: yüksek froth_pct → trim (ramp), [floor,1] sınırlı, veri-yok → 1.0 (asla agresif)."""
    from modules.dispersion_ensemble import ensemble_factor
    assert ensemble_factor(0.50, 0.70, 0.95, 0.0) == 1.0           # düşük froth → normal
    assert ensemble_factor(0.70, 0.70, 0.95, 0.0) == 1.0           # lo sınırı → henüz trim yok
    assert abs(ensemble_factor(0.825, 0.70, 0.95, 0.0) - 0.5) < 1e-9  # orta → ramp 0.5
    assert ensemble_factor(0.95, 0.70, 0.95, 0.0) == 0.0           # hi → floor (derin froth = full trim)
    assert ensemble_factor(0.99, 0.70, 0.95, 0.0) == 0.0           # hi üstü → floor (clip)
    assert ensemble_factor(0.90, 0.70, 0.95, 0.4) == 0.4           # floor=0.4 → max %60 trim
    assert ensemble_factor(None, 0.70, 0.95, 0.0) == 1.0           # veri yok → nötr
    assert ensemble_factor(float("nan"), 0.70, 0.95, 0.0) == 1.0


def test_dispersion_ensemble_froth_orientation():
    """froth_pct yön: SON gözlemde yüksek spread/DSPX + DÜŞÜK COR1M → froth_pct yüksek; tersi → düşük."""
    import numpy as np
    from modules.dispersion_ensemble import froth_pct_series
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    base = pd.Series(np.linspace(10, 20, 120), index=idx)
    # froth SONU: spread & dspx tırmanır (yüksek=froth), COR1M çöker (düşük=froth) → froth_pct→~1
    fp_hi = froth_pct_series(20 - base, base, base, win=60, min_periods=20)   # cor=20-base (son=min)
    # anti-froth SONU: hepsi ters
    fp_lo = froth_pct_series(base, 20 - base, 20 - base, win=60, min_periods=20)
    assert fp_hi.dropna().iloc[-1] > 0.9        # üç bileşen de froth → yüksek
    assert fp_lo.dropna().iloc[-1] < 0.1        # üç bileşen de sakin → düşük


def test_dispersion_ensemble_overlay_active():
    """Overlay ON (default config): dispersion_ensemble aktif + pozisyon ≤ tide_dir (trim-only) + [0,1]."""
    from config import load_config
    import run
    d = run.build_decision(load_config())
    assert "dispersion_ensemble" in d["active_overlays"]
    assert d["position_target"] <= float(d["tide_dir"]) + 1e-9      # trim-only (asla > tide)
    assert 0.0 <= d["position_target"] <= 1.0


def test_dispersion_ensemble_fail_closed(monkeypatch):
    """Fail-closed: kaynak fetch patlarsa → fail_safe_block (sessiz factor=1.0 YOK); disabled → nötr."""
    from modules import dispersion_ensemble as DE
    CFG = {"overlays": {"dispersion_ensemble": {"enabled": True, "lo": 0.70, "hi": 0.95, "floor": 0.0}}}
    def _boom(*a, **k): raise RuntimeError("CBOE down")
    monkeypatch.setattr(DE, "_fetch_cboe", _boom)
    r = DE.evaluate(CFG)
    assert r["available"] is False and r.get("fail_safe_block") is True
    # disabled → nötr factor 1.0 (asla bloke)
    r2 = DE.evaluate({"overlays": {"dispersion_ensemble": {"enabled": False}}})
    assert r2["available"] is False and r2["factor"] == 1.0 and r2["reason"] == "disabled"


def test_gex_shield_factor():
    """shield_factor: zg≥−thr → 1.0, derinleştikçe floor'a iner, [floor,1] sınırlı, veri-yok → 1.0."""
    from modules.gex_shield import shield_factor
    assert shield_factor(0.0, 0.5, 1.0, 0.4) == 1.0               # z=0 → normal
    assert shield_factor(-1.0, 0.5, 1.0, 0.4) == 1.0             # z=−thr sınırı → henüz trim yok
    assert abs(shield_factor(-2.0, 0.5, 1.0, 0.4) - 0.5) < 1e-9   # z=−2 → 1−0.5·1 = 0.5
    assert shield_factor(-3.0, 0.5, 1.0, 0.4) == 0.4             # z=−3 → 1−0.5·2 = 0 → floor 0.4
    assert shield_factor(-9.0, 0.5, 1.0, 0.4) == 0.4             # derin → floor (clip)
    assert shield_factor(5.0, 0.5, 1.0, 0.4) == 1.0             # yüksek GEX → normal (asla >1)
    assert shield_factor(None, 0.5, 1.0, 0.4) == 1.0           # veri yok → nötr
    assert shield_factor(float("nan"), 0.5, 1.0, 0.4) == 1.0


def test_gex_shield_overlay_active():
    """Overlay ON: gex_shield aktif + nihai pozisyon ≤ tide_dir (trim-only) + [0,1]."""
    from config import load_config
    import run
    d = run.build_decision(load_config())
    assert "gex_shield" in d["active_overlays"]
    assert d["position_target"] <= float(d["tide_dir"]) + 1e-9
    assert 0.0 <= d["position_target"] <= 1.0


def test_gex_shield_stack_sharpe():
    """KİLİT: tide × COR1M-froth × GEX-shield, modül koduyla çapa SPX ≥ 1.60 / NDX ≥ 1.70 (@2019+ frozen)."""
    from pathlib import Path
    from modules.cor1m_froth import froth_factor_series
    from modules.gex_shield import gex_zscore, shield_factor_series
    scores, prices, vector, _ = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    DATA = Path(__file__).resolve().parents[1] / "data" / "cache"
    cor = pd.read_parquet(DATA / "corr_pc.parquet")["COR1M"].dropna()
    gex = pd.read_parquet(DATA / "squeeze_dix_gex.parquet")["gex"].dropna()
    froth = froth_factor_series(cor.reindex(idx, method="ffill"), 8, 11, 0.0)
    zg = gex_zscore(gex).reindex(idx, method="ffill")
    shield = shield_factor_series(zg, 0.5, 1.0, 0.4)
    pos = (tdir * froth * shield).reindex(idx)
    for a, lo in (("SPX", 1.60), ("NDX", 1.70)):
        sh = E.backtest_overlay(tdir, pos, prices[a], lag=1)["strat"]["sharpe"]
        assert sh >= lo, f"{a} stack Sharpe {sh:.3f} < {lo}"
