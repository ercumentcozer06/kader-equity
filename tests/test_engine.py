"""
test_engine — Katman 3 (engine + veri borusu) kilitleri. Görev 0 (IV-mid), 3 (VRP), 6b (veri kapısı).
Spine/overlay testleri ayrı (test_spine.py); buradakiler motor/borunun davranış kilitleri.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "screen"))


# ── GÖREV 0: mid-IV inversiyonu (tek kanonik kaynak) ──────────────────────────
def test_bsiv_roundtrip():
    """BS fiyatla → IV'yi geri çevir → aynı IV (±1e-3). İnversiyon foundation'ı."""
    from _bsiv import bs_price, implied_vol
    S, K, T = 100.0, 100.0, 30 / 365
    for true_iv in (0.10, 0.18, 0.35, 0.60):
        for right in ("C", "P"):
            px = bs_price(S, K, T, true_iv, right)
            rec = implied_vol(px, S, K, T, right)
            assert rec is not None and abs(rec - true_iv) < 1e-3, f"{right} iv={true_iv} → {rec}"


def test_bsiv_invalid_returns_none():
    """Geçersiz girdi → None (ASLA ham IV'ye düşmez, sessiz kirlilik yok)."""
    from _bsiv import implied_vol, mid_iv_from_row
    assert implied_vol(None, 100, 100, 0.1, "C") is None
    assert implied_vol(0.0, 100, 100, 0.1, "C") is None
    assert implied_vol(5.0, 100, 100, 0.0, "C") is None          # T=0
    # bid/ask yok ya da ≤0 → None
    assert mid_iv_from_row({"bid": None, "ask": 2.0}, 100, 100, 0.1, "C") is None
    assert mid_iv_from_row({"bid": 0.0, "ask": 2.0}, 100, 100, 0.1, "C") is None
    assert mid_iv_from_row({"bid": 1.9, "ask": 2.1}, 100, 100, 0.1, "C") is not None


def test_bsiv_put_call_consistency():
    """ATM put ve call aynı IV'den fiyatlanır → invert ikisinde de ~aynı IV verir."""
    from _bsiv import bs_price, implied_vol
    S, K, T, iv = 100.0, 100.0, 45 / 365, 0.22
    civ = implied_vol(bs_price(S, K, T, iv, "C"), S, K, T, "C")
    piv = implied_vol(bs_price(S, K, T, iv, "P"), S, K, T, "P")
    assert abs(civ - piv) < 2e-3


# ── GÖREV 1: ledger sinyal-PnL / ifade-PnL / drag ─────────────────────────────
def test_ledger_signal_pnl_mark(tmp_path, monkeypatch):
    """signal_pnl = position × endeks ertesi-gün getirisi (bilinen kapanışla)."""
    import pandas as pd
    from validation import ledger as L
    monkeypatch.setattr(L, "ledger_path", lambda: tmp_path / "fl.parquet")
    L.append_call({"as_of": "2026-01-05", "position_target": 1.0, "direction": "LONG"})
    L.append_call({"as_of": "2026-01-06", "position_target": 0.5, "direction": "LONG"})
    cl = pd.Series([100.0, 102.0, 99.0, 101.0],
                   index=pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]))
    df = L.mark_to_market(closes=cl).set_index("as_of")
    assert abs(float(df.loc["2026-01-05", "signal_pnl"]) - 0.02) < 1e-4            # 1.0×(102/100−1)
    assert abs(float(df.loc["2026-01-06", "signal_pnl"]) - 0.5 * (99 / 102 - 1)) < 1e-4


def test_ledger_expression_drag(tmp_path, monkeypatch):
    """drag = signal_pnl − expression_pnl (ikisi de NAV-oranı)."""
    import pandas as pd
    from validation import ledger as L
    monkeypatch.setattr(L, "ledger_path", lambda: tmp_path / "fl.parquet")
    L.append_call({"as_of": "2026-01-05", "position_target": 1.0, "direction": "LONG"})
    cl = pd.Series([100.0, 102.0], index=pd.to_datetime(["2026-01-05", "2026-01-06"]))
    L.mark_to_market(closes=cl)                                                     # signal_pnl 0.02
    df = L.record_expression("2026-01-05", 0.012).set_index("as_of")               # ifade NAV +%1.2
    assert abs(float(df.loc["2026-01-05", "expression_pnl"]) - 0.012) < 1e-9
    assert abs(float(df.loc["2026-01-05", "expression_drag"]) - (0.02 - 0.012)) < 1e-4


def test_friction_ratio_math():
    """friction/risk = (komisyon+slippage)/risk; küçük hesapta >%50 olduğunu kilitle."""
    from validation.friction import friction_table
    cfg = {"accounts": {"options": {"size_usd": 1000, "max_risk_per_trade_pct": 2.0}},
           "costs": {"commission_per_contract": None, "half_spread_per_contract": 1.5, "close_round_trip": True}}
    t = friction_table(cfg, contracts=1)
    assert t["risk_usd"] == 20.0 and t["fills"] == 4
    assert len(t["scenarios"]) == 2                                                 # komisyon null → 2 senaryo
    for s in t["scenarios"]:
        assert s["friction/risk_%"] > 50                                           # $20 risk'te friction yıkıcı


# ── GÖREV 6b: veri kalite kapısı ──────────────────────────────────────────────
def _clean():
    g = {"as_of": "2026-06-10", "spot": 737.0, "net_gex_bn": -4.0, "exp_move_1d": 7.5,
         "gex_flip": 747.0, "call_wall": 750.0, "put_wall": 720.0}
    s = {"as_of": "2026-06-10", "spot": 737.0, "surface": {"7d": {"atm_iv": 19.8}}}
    return g, s


def test_dataguard_clean_passes():
    from engine import dataguard as DG
    assert DG.validate(*_clean())["ok"] is True


def test_dataguard_catches_corruption():
    from engine import dataguard as DG
    g, s = _clean()
    # 1) spot bozuk
    g1 = {**g, "spot": 0.0}; s1 = {**s, "spot": 0.0}
    assert DG.validate(g1, s1)["ok"] is False
    # 2) ham-IV saçması (ATM-IV %3)
    s2 = {**s, "surface": {"7d": {"atm_iv": 3.0}}}
    assert DG.validate(g, s2)["ok"] is False
    # 3) flip spot'tan çok uzak (şema/hesap bozuk)
    g3 = {**g, "gex_flip": 1200.0}
    assert DG.validate(g3, s)["ok"] is False
    # 4) çapraz tutarsız: exp_move IV-türevliden çok sapık
    g4 = {**g, "exp_move_1d": 60.0}
    assert DG.validate(g4, s)["ok"] is False
    # 5) şema değişti (zorunlu anahtar yok)
    g5 = {k: v for k, v in g.items() if k != "net_gex_bn"}
    assert DG.validate(g5, s)["ok"] is False
    # 6) snapshot yok
    assert DG.validate(None, None)["ok"] is False


def test_dataguard_blocks_trade_in_brief(monkeypatch):
    """Bozuk snapshot → build_brief STAND-ASIDE + data_junk (trade üretmez)."""
    import engine.brief as B
    import engine.state as S

    def fake_state(cfg, ticker="SPY"):
        model = {"direction": "LONG", "position_target": 0.9, "tide_score": 7.0,
                 "call_status": "current", "overlays": {}}
        state = {"ticker": ticker, "spot": 737.0, "exp_move_1d": 7.5}
        meta = {"data_junk": True, "data_fails": ["ön ATM-IV aralık-dışı: 3.0%"],
                "stale": False, "index": "SPX", "mult": 10, "snapshot_as_of": "2026-06-10",
                "snapshot_age_days": 0, "model_call_status": "current"}
        return model, state, meta
    monkeypatch.setattr(S, "build_state", fake_state)
    b = B.build_brief("SPY")
    assert b["decision"]["vehicle"]["class"] == "stand_aside"
    assert b["trade"]["ticket"] is None
    assert b["risk"]["dollar_risk"] == 0.0


# ── GÖREV 3: VRP kapısı (prim-sat yalnız VRP zengin iken) ─────────────────────
def _pin_state(vrp):
    """long-gamma + duvara-yakın + sakin + froth-yok pin kurulumu; VRP parametrik."""
    return {"net_gex_bn": +1.0, "gex_flip": 99.0, "spot": 100.0, "exp_move_1d": 1.0,
            "call_wall": 100.3, "put_wall": 95.0, "ts_ratio": 0.90, "cor1m": 15.0, "vrp": vrp}


def _cfg():
    return {"sizing": {"no_trade_conviction_floor": 0.15},
            "accounts": {"options": {"level": "defined_risk"}, "futures_prop": {"enabled": False}},
            "vehicles": {"directional_no_prop": "cash_etf"}}


def test_vrp_gates_premium_sell():
    """Pin kurulumu: VRP zengin → prim-sat; VRP ucuz/None → prim SATMA (aşağı düş)."""
    from engine import decision as D
    cfg = _cfg()
    # VRP zengin (+5 > 2) → iron-condor (prim sat)
    reg = D.classify_regime(_pin_state(5.0))
    v = D.pick_vehicle("LONG", 0.9, reg, cfg)
    assert v["class"] == "sell_premium_condor"
    # VRP ucuz (+0.5 ≤ 2) → prim SATMA → directional_long'a düşer (flip-üstü, sakin)
    reg2 = D.classify_regime(_pin_state(0.5))
    v2 = D.pick_vehicle("LONG", 0.9, reg2, cfg)
    assert v2["class"] != "sell_premium_condor"
    # VRP bilinmiyor (None) → muhafazakâr, prim SATMA
    reg3 = D.classify_regime(_pin_state(None))
    v3 = D.pick_vehicle("LONG", 0.9, reg3, cfg)
    assert v3["class"] != "sell_premium_condor"


def test_vrp_classify_flags():
    from engine import decision as D
    assert D.classify_regime({"vrp": 5.0})["vrp_rich"] is True
    assert D.classify_regime({"vrp": -1.0})["vrp_cheap"] is True
    r = D.classify_regime({"vrp": None})
    assert r["vrp_rich"] is False and r["vrp_cheap"] is False


# ── GÖREV 6a: modül attribution erken-uyarı ───────────────────────────────────
def test_attribution_structure_and_score_extreme():
    """evaluate(): m9/m5/m2 için band + 21g-corr; skor-uç alarmı doğru tetiklenir."""
    from validation import attribution as A
    att = A.evaluate()
    assert att["horizon"] == 21 and "m9" in att["modules"]
    m9 = att["modules"]["m9"]
    for k in ("live_score", "score_band_p1_p99", "hist_corr_mean_21d", "hist_corr_mean_1d", "alarm"):
        assert k in m9
    # m9 canlı skoru 10.03 → normal bölge (tarihsel ~%80), alarm yok
    assert m9["alarm"] is False
    # skor-uç mantığı: bandın dışındaki skor extreme işaretlenir
    band = m9["score_band_p1_p99"]
    assert band[0] < m9["live_score"] < band[1]


def test_attribution_uses_slow_horizon():
    """21g (yavaş sinyal horizonu) ile 1g referans AYRI raporlanır — slow-signal dersi."""
    from validation import attribution as A
    assert A.HORIZON == 21                                                          # a priori, GÖREV 2 run-length
    att = A.evaluate()
    m9 = att["modules"]["m9"]
    assert m9["hist_corr_mean_21d"] != m9["hist_corr_mean_1d"]                      # iki horizon farklı bilgi


# ── GÖREV 2 (engine): min-DTE tabanı ──────────────────────────────────────────
def test_min_dte_floor():
    """intraday rejim YAPIYI seçer ama DTE 21'in altına inemez."""
    from engine import trade as TR
    cfg = {"accounts": {"options": {"min_dte": 21}}}
    state = {"spot": 100.0, "exp_move_1d": 1.0, "call_wall": 102.0, "put_wall": 98.0, "gex_flip": 99.0,
             "ticker": "SPY"}
    dec = {"horizon": "intraday", "vehicle": {"class": "call_debit", "instrument": "x"}}
    t = TR.construct(dec, state, cfg)
    assert t["ticket"]["dte"] >= 21                                                 # 2g intraday → 21'e floor


# ── GÖREV 2 (prop): FTMO faz simülasyonu kural-doğrulama ──────────────────────
def test_propsim_daily_limit_kills():
    """−%6 gün → günlük-limit (−%5) kesmeli; +%10 4-günde pass; yavaş −%11 → toplam-limit."""
    from backtest import prop_sim as PS
    import numpy as np
    # daily kill: gün-2'de -%6
    st, i, why = PS._sim_phase(np.array([0.03, -0.06, 0.0]), np.array([1, 1, 1]), 0, 0.10)
    assert st == "kill" and why == "daily"
    # pass: +%2.5×5 → eq 1.10, 4+ işlem günü
    st, i, why = PS._sim_phase(np.array([0.025]*6), np.array([1]*6), 0, 0.10)
    assert st == "pass"
    # total kill: -%2×8 yavaş → eq ≤ 0.90 (günlük-limiti aşmadan)
    st, i, why = PS._sim_phase(np.array([-0.02]*8), np.array([1]*8), 0, 0.10)
    assert st == "kill" and why == "total"


def test_propsim_min_trading_days():
    """+%10'a 2 günde ulaşsa bile min 4 işlem günü dolmadan PASS YOK."""
    from backtest import prop_sim as PS
    import numpy as np
    # gün-1'de +%11 (hedef aşıldı) ama tdays<4 → hemen pass etmemeli
    st, i, why = PS._sim_phase(np.array([0.11, 0.0, 0.0, 0.0, 0.0]), np.array([1, 1, 1, 1, 1]), 0, 0.10)
    assert st == "pass" and i >= 3                                                  # en erken 4. işlem günü (i=3)


# ── GÖREV 3: kural-1 max-3g koşulsuz giriş ────────────────────────────────────
def test_rule1_unconditional_after_maxdelay():
    """Sürekli negatif-γ'da giriş max 3g ertelenir, sonra KOŞULSUZ girer."""
    import pandas as pd
    from backtest import gex_playbook_v0 as G
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    tdir = pd.Series([0, 1, 1, 1, 1, 1], index=idx)
    zg = pd.Series([-2.0]*6, index=idx)                                            # hep negatif-γ
    out = G.apply_rule1(tdir, zg).values
    assert list(out) == [0, 0, 0, 0, 1, 1]                                          # 3g ertele (i1-3=0), i4 koşulsuz


def test_rule1_no_delay_in_normal_gamma():
    """Normal-γ'da giriş ERTELENMEZ."""
    import pandas as pd
    from backtest import gex_playbook_v0 as G
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    tdir = pd.Series([0, 1, 1], index=idx)
    zg = pd.Series([5.0, 5.0, 5.0], index=idx)
    assert list(G.apply_rule1(tdir, zg).values) == [0, 1, 1]


# ── Ş1-ek v1.2: iade-ayarlı net fee ───────────────────────────────────────────
def test_v12_net_fee_refund():
    """İade: geçen→net 0; p=0.5→1 kayıp-fee; p=0.96→fee×(0.04/0.96)."""
    from backtest import prop_sim_v12 as V
    assert V.net_fee(1.0) == 0.0
    assert abs(V.net_fee(0.5) - V.EVAL_FEE_EUR) < 1e-9                              # (1-.5)/.5 = 1
    assert abs(V.net_fee(0.96) - V.EVAL_FEE_EUR*(0.04/0.96)) < 1e-6


# ── Ş2b intraday sleeve: kural mantığı (sentetik; gerçek veri YOK) ─────────────
def test_s2b_r1_fade():
    """R1: pozitif-γ & spot ∈ (HVL, call_wall] → SHORT hedef HVL; aksi None."""
    from backtest import intraday_gex_v0 as I
    s = I.r1_fade(spot=101.0, hvl=100.0, call_wall=102.0, regime_positive_gamma=True)
    assert s and s["side"] == "SHORT" and s["target"] == 100.0 and s["stop"] > 102.0
    assert I.r1_fade(101.0, 100.0, 102.0, False) is None                            # negatif-γ → R1 yok
    assert I.r1_fade(99.0, 100.0, 102.0, True) is None                              # band-dışı (spot<HVL) → yok


def test_s2b_r2_momentum():
    """R2: negatif-γ & flip-altı kırılım → SHORT; aksi None."""
    from backtest import intraday_gex_v0 as I
    s = I.r2_momentum(spot=98.0, flip=100.0, regime_negative_gamma=True, broke_below_flip=True)
    assert s and s["side"] == "SHORT" and s["stop"] == 100.0
    assert I.r2_momentum(98.0, 100.0, False, True) is None                          # pozitif-γ → yok
    assert I.r2_momentum(101.0, 100.0, True, False) is None                         # kırılım yok → yok


def test_s2b_f1_and_friction_and_nodata():
    """F1 OPEX/CPI/FOMC bloklar; friction negatif drag; load_data veri-yok hatası."""
    from backtest import intraday_gex_v0 as I
    assert I.f1_no_trade(True, False, False) is True
    assert I.f1_no_trade(False, False, False) is False
    # F1 sinyali bloklar (CPI günü → None, kurulum olsa bile)
    bar = {"spot": 101.0}; lv = {"hvl": 100.0, "call_wall": 102.0, "net_gex": 1, "regime_z": 0.0}
    assert I.signal(bar, lv, {"is_cpi": True}) is None
    assert I.friction_return(100.0) < 0                                             # round-trip drag negatif
    import pytest as _pt
    with _pt.raises(NotImplementedError):
        I.load_data()                                                              # gerçek veri YOK


# ── ŞERİT-0 HOTFIX testleri ───────────────────────────────────────────────────
def test_h1_live_fail_falls_back_frozen_no_crash():
    """H1: canlı (bozuk repo) patlar → frozen'a düşer, CRASH YOK, STALE damgalı, _live_error var."""
    import copy
    from config import load_config
    from engine import state as S
    cfg = copy.deepcopy(load_config())                                             # base source=frozen (default)
    cfg.setdefault("macro", {})["repo_path"] = r"C:\NONEXISTENT_REPO_H1"
    model, st, meta = S.build_state(cfg, "SPY")                                     # canlı patlar → frozen fallback
    assert model.get("data_source") == "frozen" and "_live_error" in model
    assert meta["stale"] is True                                                   # frozen 05-22 → STALE (güvenli)


def test_h3_gex_failsafe_short_history(monkeypatch):
    """H3: <60 gözlem → z hep NaN → fail_safe_block (sessiz factor=1.0 ASLA)."""
    import pandas as pd
    from modules import gex_shield as G
    cfg = {"overlays": {"gex_shield": {"enabled": True, "k": 0.5, "thr": 1.0, "floor": 0.4, "win": 252}}}
    idx = pd.date_range("2020-01-01", periods=10)
    monkeypatch.setattr(G, "fetch_gex_live", lambda *a, **k: pd.Series(range(10), index=idx, dtype=float))
    r = G.evaluate(cfg)
    assert r.get("fail_safe_block") is True and r["factor"] == 1.0                  # blok (factor 1.0 ama no-trade)


def test_h3_gex_normal_full_history(monkeypatch):
    """H3: tam tarih + TAZE kuyruk → normal değerlendirme, blok YOK, z_stale False.
    (F3 denetim 2026-07-05: seri artık SON İŞLEM GÜNÜNDE bitmeli — eski 2021'de biten sabit index
    yeni kaynak-donma kapısını haklı olarak tetikler; o davranış test_audit_2026_07_05'te kilitli.)"""
    import numpy as np, pandas as pd
    from datetime import datetime, timezone
    from modules import gex_shield as G
    from modules.opex_calendar import _prior_trading_day
    cfg = {"overlays": {"gex_shield": {"enabled": True, "k": 0.5, "thr": 1.0, "floor": 0.4, "win": 252}}}
    end = _prior_trading_day(datetime.now(timezone.utc).date())
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=400)
    s = pd.Series(np.linspace(100, 120, 400) + np.sin(np.arange(400)) * 5, index=idx)
    monkeypatch.setattr(G, "fetch_gex_live", lambda *a, **k: s)
    r = G.evaluate(cfg)
    assert not r.get("fail_safe_block") and r["available"] is True and r.get("z_stale") is False


def test_h5_rr_degenerate_skip():
    """H5: dejenere genişlik (absürd-küçük em) → trade RR NA/skip; dataguard absürd em'i reddeder; normal em → bounded RR."""
    from engine import trade as TR
    from engine import dataguard as DG
    cfg = {"accounts": {"options": {"min_dte": 21}}}
    dec = {"horizon": "position", "vehicle": {"class": "directional_long", "instrument": "delta-one SPLG"}}
    # (a) absürd-küçük em → em5 minik → denom < eşik → ticket None (dev RR basılmaz)
    st = {"spot": 100.0, "exp_move_1d": 0.0001, "call_wall": 100.0, "put_wall": 100.0, "gex_flip": 99.0, "ticker": "SPY"}
    assert TR.construct(dec, st, cfg)["ticket"] is None
    # (b) dataguard absürd-küçük em'i upstream reddeder
    g = {"as_of": "2026-06-10", "spot": 100.0, "net_gex_bn": -1.0, "exp_move_1d": 0.0001,
         "gex_flip": 100.0, "call_wall": 101.0, "put_wall": 99.0}
    s = {"as_of": "2026-06-10", "spot": 100.0, "surface": {"7d": {"atm_iv": 15.0}}}
    assert DG.validate(g, s)["ok"] is False
    # (c) normal em → finite, bounded RR
    st2 = {**st, "exp_move_1d": 1.0, "call_wall": 103.0, "put_wall": 97.0}
    t2 = TR.construct(dec, st2, cfg)
    assert t2["ticket"] is not None and 0 < t2["ticket"]["rr"] < 100


def test_h7_qqq_no_silent_gamma_fallback(monkeypatch):
    """H7: gamma_qqq yoksa SPY'a düşüş AÇIK flag (sessiz değil)."""
    import run
    from engine import state as S
    monkeypatch.setattr(run, "build_decision",
                        lambda cfg: {"direction": "LONG", "position_target": 0.9, "tide_score": 7.0,
                                     "call_status": "current", "data_source": "frozen", "overlays": {}})
    def fake_latest(sub):
        if sub == "gamma_qqq":
            return None                                                            # QQQ gamma YOK
        if sub == "gamma_spy":
            return {"as_of": "2026-06-10", "spot": 730.0, "net_gex_bn": -4.0, "exp_move_1d": 7.0,
                    "gex_flip": 740.0, "call_wall": 745.0, "put_wall": 720.0}
        if sub.startswith("surface"):
            return {"as_of": "2026-06-10", "spot": 730.0, "surface": {"30d": {"atm_iv": 20.0}}}
        return None
    monkeypatch.setattr(S, "_latest", fake_latest)
    monkeypatch.setattr(S, "_realized_vol_ewma", lambda *a, **k: 14.0)
    _m, _s, meta = S.build_state({}, "QQQ")
    assert meta["gamma_fallback_to_spy"] is True                                   # sessiz değil — flag açık


# ── ŞERİT-1 v1.3: low-bazlı kill + Wilson ─────────────────────────────────────
def test_v13_low_based_kill_catches_intraday():
    """1.1: close düz ama intraday low −%6 (tam pozisyon) → low-bazlı KILL, close-bazlı görmüyor."""
    import numpy as np
    from backtest import prop_sim_v13 as V
    rets = np.zeros(6); lows = np.array([-0.06, 0, 0, 0, 0, 0]); opens = np.ones(6)
    st_low, _, why = V._sim(rets, lows, opens, 0.0, 0, 0.10, low_based=True)
    assert st_low == "kill" and why == "daily"
    st_close, _, _ = V._sim(rets, lows, opens, 0.0, 0, 0.10, low_based=False)
    assert st_close != "kill"                                                      # intraday-only → close görmez


def test_v13_wilson_lo():
    """1.3: Wilson alt sınır — 100/100 pass < 1.0 (0-kill ≠ 0-risk); 50/100 < 0.5."""
    from backtest import prop_sim_v13 as V
    assert 0.95 < V.wilson_lo(100, 100) < 1.0
    assert V.wilson_lo(50, 100) < 0.5
    import math
    _w00 = V.wilson_lo(0, 0)                       # Denetim 07-11 P3 ([35]): eski assert totolojikti
    assert _w00 is None or (isinstance(_w00, float) and (math.isnan(_w00) or _w00 == 0.0))  # n=0 -> nan/0/None, crash yok


# ── ŞERİT-3: ENTEGRASYON — decision tüm dalları / trade tüm yapılar / risk / brief end-to-end ──
def _icfg():
    return {"sizing": {"no_trade_conviction_floor": 0.15, "fractional_kelly": 0.5},
            "accounts": {"options": {"level": "defined_risk", "size_usd": 1000, "max_risk_per_trade_pct": 2.0},
                         "futures_prop": {"enabled": False}},
            "vehicles": {"directional_no_prop": "cash_etf"},
            "live_book": {"mode": "delta_one", "etf_map": {"SPY": "SPLG", "QQQ": "QQQM"}}}


def _istate(**kw):
    base = {"spot": 100.0, "exp_move_1d": 1.0, "net_gex_bn": 1.0, "gex_flip": 99.0,
            "call_wall": 103.0, "put_wall": 97.0, "ts_ratio": 0.90, "cor1m": 15.0, "vrp": 5.0, "ticker": "SPY"}
    base.update(kw)
    return base


def test_s3_decision_all_branches():
    """decision.pick_vehicle TÜM dallar (froth/short-gamma/pin/clean/stand-aside) doğru ifade."""
    from engine import decision as D
    cfg = _icfg()
    def veh(direction, conv, **st):
        reg = D.classify_regime(_istate(**st))
        return D.pick_vehicle(direction, conv, reg, cfg)["class"]
    assert veh("LONG", 0.9, cor1m=7.0) == "buy_convexity_put"                       # froth (COR1M<8)
    assert veh("LONG", 0.9, net_gex_bn=-1.0, spot=98.0, gex_flip=100.0) == "call_debit"   # kısa-γ/flip-altı
    assert veh("LONG", 0.9, net_gex_bn=1.0, spot=103.0, call_wall=103.2, ts_ratio=0.9, vrp=5.0) == "sell_premium_condor"  # pin+VRP-zengin
    assert veh("LONG", 0.9, net_gex_bn=1.0, spot=101.0, gex_flip=99.0, ts_ratio=0.9, cor1m=15.0) == "directional_long"    # temiz trend
    assert veh("FLAT", 0.9) == "stand_aside"                                        # model FLAT
    assert veh("LONG", 0.05) == "stand_aside"                                       # konviksiyon eşik-altı


def test_s3_trade_all_structures():
    """trade.construct 4 yapı + min-DTE floor + delta-one instrument."""
    from engine import trade as TR
    cfg = _icfg(); st = _istate()
    for klass, key in [("buy_convexity_put", "structure"), ("call_debit", "structure"),
                       ("sell_premium_condor", "structure"), ("directional_long", "side")]:
        dec = {"horizon": "swing", "vehicle": {"class": klass, "instrument": "delta-one SPLG"}}
        t = TR.construct(dec, st, cfg)["ticket"]
        assert t is not None and t["dte"] >= 21 and key in t                        # min-DTE + yapı alanı


def test_s3_risk_size_branches():
    """risk.size: options budget / directional shares / stand-aside 0."""
    from engine import risk as RK
    cfg = _icfg()
    # options defined-risk
    r1 = RK.size({"vehicle": {"class": "call_debit"}, "conviction": 0.9},
                 {"class": "call_debit", "ticket": {"structure": "call-debit-spread"}}, cfg)
    assert r1["dollar_risk"] > 0 and r1["dollar_risk"] <= 20.0                       # ≤ %2×$1000 bütçe
    # stand-aside
    r2 = RK.size({"vehicle": {"class": "stand_aside"}, "conviction": 0.0}, {"ticket": None}, cfg)
    assert r2["dollar_risk"] == 0.0


def test_s3_brief_end_to_end_nominal(monkeypatch):
    """brief.build_brief NOMİNAL yol (data_ok) → tüm anahtarlar + trade üretildi (data_junk YOLU DEĞİL)."""
    import engine.brief as B
    import engine.state as S
    def fake_state(cfg, ticker="SPY"):
        model = {"direction": "LONG", "position_target": 0.8, "tide_score": 7.0,
                 "call_status": "current", "overlays": {}, "overlay_block": False}
        state = _istate(ticker=ticker)
        meta = {"data_junk": False, "stale": False, "index": "SPX", "mult": 10,
                "snapshot_as_of": "2026-06-10", "snapshot_age_days": 0, "model_call_status": "current",
                "gamma_fallback_to_spy": False}
        return model, state, meta
    monkeypatch.setattr(S, "build_state", fake_state)
    b = B.build_brief("SPY")
    for k in ("computed_at", "ticker", "index", "model", "state", "meta", "decision", "trade", "risk", "live_book"):
        assert k in b
    assert b["decision"]["vehicle"]["class"] != "stand_aside" or b["decision"]["direction"] == "FLAT"
    assert b["live_book"]["mode"] == "delta_one"


# ── v1.3-final: restart/iade ekonomisi ────────────────────────────────────────
def test_v13final_restart_economics():
    """E[deneme]=1/p; E[net-fee]=€155·(E[deneme]−1) (geçen iade, kill ücret); E[takvim]=(E[deneme]−1)·medFail+medPass."""
    from backtest import prop_sim_v13_final as F
    r1 = F._restart(1.0, 300, 100)
    assert abs(r1["e_attempts"] - 1.0) < 1e-9 and r1["e_net_fee"] == 0.0                # %100 geçen → net €0
    r2 = F._restart(0.5, 300, 100)
    assert abs(r2["e_attempts"] - 2.0) < 1e-9 and abs(r2["e_net_fee"] - F.EVAL_FEE) < 1e-9
    assert abs(r2["e_cal_mo"] - (1 * 100 + 300) / F.MD) < 1e-6                          # (2−1)·medFail+medPass


# ── ŞERİT-B GO-LIVE: position_translator + prop_tracker + policy ──────────────
def test_b1_translate_lot_and_delta():
    """B1: lot = floor(equity×exp/(cs×spot)/step)×step (AŞAĞI yuvarla); delta-emir + coarse_flag."""
    from engine import position_translator as T
    # 10k, exposure 1.2, NDX spot 28440, cs 10, step 0.01 → lot=floor(12000/284400/0.01)*0.01=floor(4.22)*0.01=0.04
    r = T.translate(model_deploy=1.0, spot=28440.0, prev_close=28440.0, equity=10000.0, eval_pos=1.2,
                    contract_size=10, lot_step=0.01, current_lot=0.01, instrument="US100")
    assert r["lot_target"] == 0.04 and r["realized_exposure"] <= r["target_exposure"]   # aşağı-yuvarlama
    assert r["delta_lot"] == 0.03 and r["side"] == "AL"
    assert r["coarse_flag"] is True and r["exposure_step_pct"] > 15                      # A2: NDX cs=10 kaba
    # SPX cs=10 → ince (adım ~%7)
    r2 = T.translate(1.0, 7258.0, 7258.0, 10000.0, 1.0, 10, 0.01, instrument="US500")
    assert r2["coarse_flag"] is False


def test_b1_stop_and_limit_levels():
    """B1: felaket-stop = prev×(1−0.045/exp); günlük-limit = prev×(1−0.05/exp)."""
    from engine import position_translator as T
    r = T.translate(1.0, 100.0, 100.0, 10000.0, 1.0, 1, 0.01, use_stop=True)            # exp=1.0
    assert abs(r["stop_level"] - 95.5) < 0.01 and abs(r["daily_limit_level"] - 95.0) < 0.01


def test_b4_policy_violation():
    """B4: eval_pos pre-registered'dan saparsa policy-ihlali FLAG."""
    from engine import position_translator as T
    assert T.check_policy({"accounts": {"eval": {"eval_pos": 1.2, "eval_pos_registered": 1.2}}}) is None
    v = T.check_policy({"accounts": {"eval": {"eval_pos": 1.5, "eval_pos_registered": 1.2}}})
    assert v and v["violation"] is True


def test_b2_prop_tracker(tmp_path, monkeypatch):
    """B2: model_equity = prev×(1+pos×getiri); trading-days sayacı; totalDD mesafesi."""
    from engine import prop_tracker as PT
    monkeypatch.setattr(PT, "ledger_path", lambda acc: tmp_path / f"{acc}.parquet")
    PT.append_day("E1", "2026-06-11", "P1", day_index_return=0.02, position_exposure=1.0, lot=0.04)
    df = PT.append_day("E1", "2026-06-12", "P1", day_index_return=-0.01, position_exposure=1.0, lot=0.04)
    r = df.iloc[-1]
    assert abs(float(r["model_equity"]) - (1.02 * 0.99)) < 1e-6                          # bileşik
    assert int(r["trading_days"]) == 2
    s = PT.drag_summary("E1")
    assert s["rows"] == 2 and s["trading_days"] == 2


# ── ŞERİT-C: chain_guard QC + marketdata token-gate ──────────────────────────
def test_c2_chain_guard_schema_bounds():
    """C2: şema (eksik kolon/negatif OI) + bound (strike≤0) RED."""
    import pandas as pd
    from engine import chain_guard as CG
    good = pd.DataFrame({"date": ["2026-06-11"], "expiration": ["2026-06-19"], "strike": [100.0],
                         "right": ["C"], "open_interest": [50]})
    assert CG.validate_schema(good) == [] and CG.validate_bounds(good) == []
    assert CG.validate_schema(good.drop(columns=["open_interest"]))                     # eksik kolon → fail
    assert CG.validate_schema(good.assign(open_interest=[-5]))                           # negatif OI → fail
    assert CG.validate_bounds(good.assign(strike=[-1.0]))                                # strike≤0 → fail


def test_c2_opex_oi_collapse():
    """C2: 3.Cuma OPEX-sonrası OI ≥%70 düşerse PASS, düşmezse fail (PIT-sağlık)."""
    import pandas as pd
    from engine import chain_guard as CG
    # 2026-06-19 = 3. Cuma; ertesi işlem günü +3 (Pzt 06-22). OI 1000→100 = %90 düşüş → pass
    df = pd.DataFrame({"date": ["2026-06-19", "2026-06-22"], "expiration": ["2026-06-19", "2026-06-19"],
                       "strike": [100.0, 100.0], "right": ["C", "C"], "open_interest": [1000, 100]})
    assert CG.opex_oi_collapse(df)["ok"] is True
    df2 = df.assign(open_interest=[1000, 950])                                           # %5 düşüş → fail
    assert CG.opex_oi_collapse(df2)["ok"] is False


def test_c2_gex_cross_check():
    """C2: GEX işaret-uyumu ≥%90 + korr ≥0.9 → ok; aksi RED."""
    import numpy as np, pandas as pd
    from engine import chain_guard as CG
    idx = pd.date_range("2026-01-01", periods=40)
    a = pd.Series(np.sin(np.arange(40)), index=idx)
    assert CG.gex_cross_check(a, a * 1.1)["ok"] is True                                  # aynı işaret+korr
    assert CG.gex_cross_check(a, -a)["ok"] is False                                      # ters işaret


def test_c1_marketdata_needs_token(monkeypatch):
    """C1: token yoksa NET hata (sonuç uydurulmaz)."""
    import pytest as _pt
    from screen import marketdata_backfill as MD
    monkeypatch.delenv("MARKETDATA_TOKEN", raising=False)
    with _pt.raises(RuntimeError):
        MD._token()


def test_c_alpaca_needs_secret(monkeypatch):
    """Ş2B: Alpaca secret placeholder/eksikse NET hata (sonuç uydurulmaz)."""
    import pytest as _pt
    from screen import alpaca_bars_backfill as AB
    monkeypatch.setenv("APCA_API_KEY_ID", "PKxxx")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "REPLACE_ME_GERCEK_SECRET")
    with _pt.raises(RuntimeError):
        AB._client()
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with _pt.raises(RuntimeError):
        AB._client()
