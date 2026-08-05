"""ÜÇ-GÜN KURALI spec-testleri (deploy 2026-08-05, Emir kararı).

Kilitler: (1) bayrak mantığı ablation ile birebir (frozen seride flag/aktif-gün sayıları
three_down_rule_ablation_result.json'a eşit); (2) fail-closed no-op (eksik/bayat seri);
(3) NDX config-KAPALI (ablation-FAIL) — boost yalnız SPX'te; (4) boost yalnız target>0'da,
clip ≤ cap; (5) position_target ŞEMASI değişmedi (0..1, deira kırılmaz)."""
import copy

import numpy as np
import pandas as pd
import pytest

from modules import three_down_rule as TD


def _closes(vals, start="2026-01-05"):
    """Hafta-içi takvimli sentetik kapanış serisi (2026-01-05 = Pazartesi)."""
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)))


# ── (1) bayrak mantığı ─────────────────────────────────────────────────────────
def test_flag_three_consecutive_down_activates_window():
    # 100 101 100 99 98: son üç kapanış (101→100→99→98) ardışık aşağı → bugün bayrak, kalan 3
    s = _closes([100, 101, 100, 99, 98])
    snap = TD.three_down_snapshot(s, max_age_bd=None)
    assert snap["available"] and snap["flag_active"]
    assert snap["days_left"] == 3
    assert snap["last_flag_date"] == str(s.index[-1].date())


def test_two_down_is_not_a_flag():
    snap = TD.three_down_snapshot(_closes([100, 101, 100, 99]), max_age_bd=None)
    assert snap["available"] and not snap["flag_active"] and snap["days_left"] == 0


def test_window_expires_after_3_business_days():
    # bayrak t'de; t+1,t+2 aktif (kalan 2,1), t+3 pasif — ablation act3 (rolling 3, bayrak-günü dahil)
    base = [100, 99, 98, 97]                                  # bayrak son günde
    for extra, want_active, want_left in (([98], True, 2), ([98, 99], True, 1), ([98, 99, 100], False, 0)):
        snap = TD.three_down_snapshot(_closes(base + extra), max_age_bd=None)
        assert snap["flag_active"] is want_active, f"extra={extra}"
        assert snap["days_left"] == want_left, f"extra={extra}"


def test_flat_day_breaks_streak():
    # ablation notu: "ffill flat gun asagi sayilmaz" — aradaki değişimsiz gün seriyi keser
    snap = TD.three_down_snapshot(_closes([100, 99, 99, 98, 97]), max_age_bd=None)
    assert snap["available"] and not snap["flag_active"]


def test_holiday_gap_ffill_counts_as_flat(monkeypatch=None):
    # kapanış serisinde eksik hafta-içi gün (tatil) ffill edilir → chg==0 → streak KESİLİR (ablation birebir)
    idx = pd.bdate_range("2026-01-05", periods=5)
    s = pd.Series([100, 99, 98, 97, 96], index=idx).drop(idx[2])   # ortadaki işgünü tatil
    snap = TD.three_down_snapshot(s, max_age_bd=None)
    # ffill sonrası: -1, -1, 0(ffill), -1, -1 → hiçbir yerde 3 ardışık aşağı yok
    assert snap["available"] and not snap["flag_active"]


def test_frozen_parity_with_ablation_counts():
    """Frozen seride bayrak/aktif-gün sayıları = three_down_rule_ablation_result.json (birebir kilit)."""
    from spine import contract as C
    _scores, prices, _v, _p = C.read_frozen()
    expect = {"SPX": (147, 321), "NDX": (130, 284)}           # result.json flag_days / active3_days (full)
    for a, (nf, na) in expect.items():
        flag = TD._flag_series(prices[a])
        fac = TD.boost_factor_series(prices[a])
        assert int(flag.sum()) == nf, f"{a} bayrak sayısı ablation'dan saptı"
        assert int((fac > 1.0).sum()) == na, f"{a} aktif-gün sayısı ablation'dan saptı"


# ── (2) fail-closed no-op ──────────────────────────────────────────────────────
def test_fail_closed_missing_series():
    for bad in (None, pd.Series(dtype=float), _closes([100, 99, 98])):   # None / boş / çok kısa
        snap = TD.three_down_snapshot(bad, max_age_bd=None)
        assert snap["available"] is False and not snap["flag_active"]
        assert snap["reason"]                                  # reason alanı dolu (sessiz değil)


def test_fail_closed_stale_series():
    s = _closes([100, 99, 98, 97, 96], start="2026-01-05")     # son kapanış 2026-01-09 (Cuma)
    stale = TD.three_down_snapshot(s, max_age_bd=3, today="2026-01-19")  # 6 işgünü sonra → bayat
    assert stale["available"] is False and not stale["flag_active"]
    assert stale["reason"] and "no-op" in stale["reason"]
    fresh = TD.three_down_snapshot(s, max_age_bd=3, today="2026-01-12")  # 1 işgünü → taze
    assert fresh["available"] is True and fresh["flag_active"]


# ── (3)+(4)+(5) entegrasyon: NDX kapalı, boost yalnız target>0, clip, şema ────
def _decision(monkeypatch, spx_tail=None):
    """Frozen build_decision; istenirse frozen SPX kuyruğu monkeypatch'le bayrak-aktif hale getirilir."""
    import run as R
    from config import load_config
    from spine import contract as C
    if spx_tail is not None:
        orig = C.read_frozen

        def _patched():
            scores, prices, vector, prov = orig()
            p = prices.copy()
            p.loc[p.index[-4]:, "SPX"] = spx_tail              # son 4 kapanışı zorla (3-aşağı)
            return scores, p, vector, prov
        monkeypatch.setattr(C, "read_frozen", _patched)
        monkeypatch.setattr(R.C, "read_frozen", _patched, raising=False)
    cfg = copy.deepcopy(load_config())
    cfg.setdefault("spine", {})["source"] = "frozen"
    return R.build_decision(cfg)


def test_integration_fields_present_and_ndx_disabled(monkeypatch):
    d = _decision(monkeypatch)
    assert "three_down_active" in d and "three_down_boost_applied" in d and "three_down_rule" in d
    # NDX: config-kapalı → asla aktif/boost olmaz
    assert d["three_down_active"]["NDX"] is False
    assert d["three_down_boost_applied"]["NDX"] == 1.0
    assert (d["three_down_rule"]["NDX"].get("reason") or "").startswith("disabled")
    # şema: position_target 0..1 KORUNUR (boost asset_deploy katmanında; deira sleeve okuması kırılmaz)
    assert 0.0 <= d["position_target"] <= 1.0


def test_integration_boost_applies_only_to_positive_spx_target(monkeypatch):
    d = _decision(monkeypatch, spx_tail=[100.0, 99.0, 98.0, 97.0])   # frozen SPX kuyruğu → bayrak AKTİF
    assert d["three_down_active"]["SPX"] is True
    ad = d["asset_deploy"] or {}
    k2 = (float(d["supply_demand_derisk"]["trim_factor"])
          if (d.get("supply_demand_derisk") or {}).get("fired") and d.get("_sd_derisk_position_effect")
          else 1.0)
    base = min(1.0, d["position_target"]) * k2
    if base > 0:
        assert d["three_down_boost_applied"]["SPX"] == pytest.approx(1.25)
        # payload position_target 4-hane yuvarlanmış; asset_deploy ham deploy'dan hesaplanır → abs tolerans
        assert ad["SPX"] == pytest.approx(min(base * 1.25, 1.25), abs=2e-4)
        assert ad["SPX"] <= 1.25 + 1e-9                        # clip ≤ cap
    else:
        # target ≤ 0 (tide FLAT) → boost DOKUNMAZ (E3: yalnız-long)
        assert d["three_down_boost_applied"]["SPX"] == 1.0
        assert ad.get("SPX", 0.0) == pytest.approx(base)
    # NDX aynı koşuda etkisiz kalır
    assert d["three_down_boost_applied"]["NDX"] == 1.0


def test_integration_inactive_window_no_boost(monkeypatch):
    d = _decision(monkeypatch)                                 # frozen kuyruk: son bayrak as_of-3 → pasif
    assert d["three_down_active"]["SPX"] is False
    assert d["three_down_boost_applied"]["SPX"] == 1.0
    ad = d.get("asset_deploy") or {}
    if "SPX" in ad:
        assert ad["SPX"] <= 1.0 + 1e-9                         # boost yok → kaldıraçsız tavan korunur
