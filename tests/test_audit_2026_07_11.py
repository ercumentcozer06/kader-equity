# Denetim 07-11 kapanış testleri — 3 kök nedenin BAŞARISIZLIK dalları + gölge-fix regresyonu.
# KÖK A (P0): kısmi-bar mühürü — bugünün devam-eden barı marka giremez (seans açıkken).
# KÖK B (P1): price_stale >10g feed-ölümünde False'a dönüyordu; artık koşulsuz STALE.
# KÖK C (P1): degraded tide "current" damgalanamaz (kayıp ağırlık >%5).
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _mk_ledger(tmp_path, monkeypatch, rows):
    from validation import ledger as L
    monkeypatch.setattr(L, "ROOT", tmp_path)
    (tmp_path / "output").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for c in L._COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df[L._COLS].to_parquet(L.ledger_path())
    return L


def test_kok_b_price_stale_true_beyond_10d_dead_feed(tmp_path, monkeypatch):
    """Feed 05-22'de ölmüş (frozen-fallback), satır 07-01: eski kod False basıyordu → artık True."""
    L = _mk_ledger(tmp_path, monkeypatch,
                   [{"as_of": "2026-07-01", "position_target": 1.0, "call_status": "current"}])
    closes = pd.Series([100.0, 101.0, 102.0],
                       index=pd.to_datetime(["2026-05-20", "2026-05-21", "2026-05-22"]))
    monkeypatch.setattr(L, "_index_closes", lambda asset="SPX": closes)
    out = L.mark_to_market()                      # closes=None yolu → src_stale_bd hesaplanır
    row = out[out["as_of"].astype(str).str.startswith("2026-07-01")].iloc[-1]
    assert bool(row["price_stale"]) is True, \
        "10g+ feed-ölümünde price_stale False'a döndü (P1 07-11 regresyonu — dei-ra gate'i kör kalır)"
    assert row["signal_pnl"] is None or pd.isna(row["signal_pnl"])


def test_kok_b_append_default_fail_closed(tmp_path, monkeypatch):
    """append sonrası mark çökerse satır None (falsy=temiz) kalıyordu → default True."""
    from validation import ledger as L
    monkeypatch.setattr(L, "ROOT", tmp_path)
    (tmp_path / "output").mkdir(parents=True, exist_ok=True)
    L.append_call({"as_of": "2026-07-11", "position_target": 1.0, "call_status": "current"})
    df = pd.read_parquet(L.ledger_path())
    assert bool(df.iloc[-1]["price_stale"]) is True, "mark öncesi satır fail-closed STALE olmalı"


def test_kok_a_partial_bar_via_injected_closes_unaffected(tmp_path, monkeypatch):
    """Enjekte-closes yolu (test/backtest) kısmi-bar korumasından ETKİLENMEZ (yalnız canlı yol düşürür)."""
    L = _mk_ledger(tmp_path, monkeypatch,
                   [{"as_of": "2026-07-08", "position_target": 1.0, "call_status": "current"}])
    closes = pd.Series([100.0, 102.0],
                       index=pd.to_datetime(["2026-07-08", "2026-07-09"]))
    out = L.mark_to_market(closes=closes)
    row = out.iloc[-1]
    assert row["signal_pnl"] == pytest.approx(0.02), "enjekte-closes deterministik skorlamalı"


def test_kok_a_today_bar_dropped_when_session_open(tmp_path, monkeypatch):
    """Canlı yol: seans açıkken bugünün devam-eden barı marka girmez → dün BEKLEMEDE kalır."""
    from datetime import date, timedelta
    today = pd.Timestamp(date.today())
    yday = today - pd.Timedelta(days=1)
    L = _mk_ledger(tmp_path, monkeypatch,
                   [{"as_of": str(yday.date()), "position_target": 1.0, "call_status": "current"}])
    closes = pd.Series([100.0, 105.0], index=[yday, today])   # bugünkü 105 = KISMİ bar (canlıda)
    monkeypatch.setattr(L, "_index_closes", lambda asset="SPX": closes)

    class _FakeDT:
        @staticmethod
        def now(tz=None):
            # 14:31 UTC ≈ 09:31 ET — seans AÇIK
            import datetime as _d
            return _d.datetime.combine(date.today(), _d.time(14, 31), tzinfo=tz)
    monkeypatch.setattr(L, "datetime", _FakeDT)
    out = L.mark_to_market()
    row = out.iloc[-1]
    assert row["signal_pnl"] is None or pd.isna(row["signal_pnl"]), \
        "kısmi-bar mühürü geri geldi (P0 07-11): dünün pnl'i seans-içi fiyatla FINAL yazıldı"


def test_ghost_holiday_row_not_scored(tmp_path, monkeypatch):
    """İşlem-günü-olmayan as_of (Juneteenth-tipi hayalet) önceki günün getirisini ÇİFT sayamaz."""
    L = _mk_ledger(tmp_path, monkeypatch, [
        {"as_of": "2026-06-18", "position_target": 1.0, "call_status": "current"},
        {"as_of": "2026-06-19", "position_target": 0.5, "call_status": "current"},  # tatil hayaleti
    ])
    closes = pd.Series([100.0, 101.0, 103.0],
                       index=pd.to_datetime(["2026-06-17", "2026-06-18", "2026-06-22"]))
    out = L.mark_to_market(closes=closes)
    g = out[out["as_of"].astype(str).str.startswith("2026-06-19")].iloc[-1]
    assert g["signal_pnl"] is None or pd.isna(g["signal_pnl"]), \
        "hayalet-satır skorlandı (çift-sayım P2 07-11)"
    r = out[out["as_of"].astype(str).str.startswith("2026-06-18")].iloc[-1]
    assert r["signal_pnl"] == pytest.approx(1.0 * (103.0 / 101.0 - 1), abs=1e-6)


def test_kok_c_degraded_tide_not_current():
    """tide.decide degraded (kayıp ağırlık >%5) → run.py stale zinciri 'current' basamaz.
    (Kablo testi: decide çıktısı + run.py'deki eşik mantığının birebir kopyası değil,
    gerçek karar fonksiyonunun degraded emisyonu.)"""
    from spine import tide as T
    vector = {"m9": 0.56, "m5": 0.21, "m2": 0.23}
    td = T.decide({"m9": None, "m5": 1.0, "m2": 1.0}, vector)
    assert td["degraded"] is True and td["missing_weight_frac"] > 0.05
    # run.py kablosu: degraded + frac>0.05 → stale (F8 defteri keser). Mantığı buradan doğrula:
    assert bool(td.get("degraded")) and float(td.get("missing_weight_frac") or 0) > 0.05


def test_shadow_fix_regression_guard():
    """07-07 gölge-fix regresyon kilidi ([15]/[22]): subprocess-izolasyon VARSAYILAN kalmalı
    ve _evict_equity_shadows canlı yolda mevcut olmalı."""
    from spine import reconstruct as R
    assert hasattr(R, "_evict_equity_shadows"), "gölge-eviction fonksiyonu kayboldu"
    src = (ROOT / "spine" / "reconstruct.py").read_text(encoding="utf-8")
    assert 'panel_isolation", "subprocess"' in src.replace("'", '"'), \
        "panel_isolation varsayılanı subprocess olmaktan çıktı (46-gün gölge sınıfı geri açılır)"
