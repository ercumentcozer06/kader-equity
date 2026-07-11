"""
modules/cor1m_froth — FROTH-TOP DE-RISK overlay (kader-equity'nin İLK alfa modülü).

COR1M (CBOE 1-ay implied correlation) DÜŞÜK = single-stock call-froth / complacency = KONTRARİAN
BEARISH (SpotGamma/Kochuba). Düşük COR1M'de tide-long'u kıs (trim-only, rebound-safe; ASLA short/add).

Kanıt (screen/candidate_cor1m_froth + _robust + bucket_scan_tails):
  • 20yr bucket monotonik: COR1M<8 → fwd-21g SPX −1.5%/74%neg, NDX −3.2%/77%neg (tek negatif uç-kuyruk)
  • strict BH-FDR PASS: flat<9 +0.15/+0.19, flat<10 +0.16/+0.21 (P(v>b)~%100 ikisinde); 8-12 hepsi pozitif
  • 3 ayrı episode: 2024 (Tem-froth→Ağu VIX-60), 2025, 2026-Haz. CAVEAT: 2024+ rejimi (genç) → forward-watch.

FORM = düz ramp: COR1M≥hi → faktör 1.0 (normal); COR1M≤lo → floor (derin froth = de-risk); arası lineer.
factor ∈ [floor, 1]; nihai pozisyon = tide_dir × factor. Veri yoksa factor=1.0 (nötr, asla agresif).

EŞİK PROVENANCE (GÖREV 5 denetimi):
  • lo=8  : A PRİORİ (SpotGamma/Kochuba froth eşiği ~8), grid'de doğrulandı (candidate_*). DSR-muhasebesinde.
  • hi=11 : FITTED (verify_cor1m 5-ramp seçimi 8/11·8/10·7/11·8/12). DSR-muhasebesinde.
  • floor=0: FITTED (3-floor seçimi 0.0/0.2/0.3). DSR-muhasebesinde.
  Robust BÖLGE (8-12 hepsi pozitif), knife-edge DEĞİL → overfit düşük. DSR N=60 İYİMSER → dürüst ~0.96/0.98
  (finalize_stack N_TRIALS notu). lo/hi/floor config.yaml'da; DEĞER DEĞİŞMEDİ, yalnız etiketlendi.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

CBOE_COR1M = "https://cdn.cboe.com/api/global/us_indices/daily_prices/COR1M_History.csv"
CBOE_COR1M_QUOTE = "https://cdn.cboe.com/api/global/delayed_quotes/quotes/_COR1M.json"


def froth_factor(cor1m: float | None, lo: float = 8.0, hi: float = 11.0, floor: float = 0.0) -> float:
    """Tek-değer froth trim faktörü. COR1M yoksa 1.0 (nötr)."""
    if cor1m is None or (isinstance(cor1m, float) and np.isnan(cor1m)):
        return 1.0
    return float(np.clip((float(cor1m) - lo) / (hi - lo), floor, 1.0))


def froth_factor_series(cor1m: pd.Series, lo: float = 8.0, hi: float = 11.0, floor: float = 0.0) -> pd.Series:
    return ((cor1m - lo) / (hi - lo)).clip(floor, 1.0)


def fetch_cor1m_live(timeout: int = 20) -> pd.Series:
    """CBOE CDN'den günlük COR1M (free, abonelik gerektirmez)."""
    from modules._netutil import http_get_retry
    r = http_get_retry(CBOE_COR1M, timeout=timeout)     # 3-deneme backoff (geçici hıçkırık = bayatlık DEĞİL)
    df = pd.read_csv(io.StringIO(r.text))
    dcol = [c for c in df.columns if "date" in c.lower()][0]
    vcol = [c for c in df.columns if c != dcol][-1]
    s = pd.Series(pd.to_numeric(df[vcol], errors="coerce").values,
                  index=pd.to_datetime(df[dcol], errors="coerce")).dropna()
    return s.sort_index()


def fetch_cor1m_quote(timeout: int = 15) -> tuple[float, str]:
    """CBOE canlı gecikmeli quote (~15dk) — BUGÜNÜN baskısı. (price, as_of ISO-tarih) döndürür.

    Günlük COR1M_History.csv yalnız TAMAMLANMIŞ kapanışları taşır; seans-içi (ve CBOE'nin EOD
    dosyayı henüz basmadığı sabahlarda) bugünün değerini KAÇIRIR. Bu endpoint current_price +
    last_trade_time verir → model en güncel COR1M'yi görür, tatil-sınırı STALE-flip riski ölür.
    """
    from datetime import datetime as _dt
    from modules._netutil import http_get_retry
    r = http_get_retry(CBOE_COR1M_QUOTE, timeout=timeout)   # 3-deneme backoff: transient timeout → CSV'ye erken düşmesin
    d = r.json().get("data", {}) or {}
    px = d.get("current_price")
    if px is None:
        px = d.get("close")
    if px is None:
        raise ValueError("quote'ta current_price/close yok")
    ltt = str(d.get("last_trade_time", ""))[:10]
    if not ltt:
        raise ValueError("quote'ta last_trade_time yok")
    asof = _dt.fromisoformat(ltt).date().isoformat()
    return float(px), asof


def evaluate(cfg: dict) -> dict:
    """Canlı froth overlay. Döndürür factor (tide pozisyonunu çarpan) + bağlam. Flag OFF → factor 1.0."""
    o = ((cfg.get("overlays", {}) or {}).get("cor1m_froth", {}) or {})
    if not bool(o.get("enabled")):
        return {"available": False, "factor": 1.0, "reason": "disabled"}
    lo, hi, fl = float(o.get("lo", 8.0)), float(o.get("hi", 11.0)), float(o.get("floor", 0.0))
    # Freshness fix (2026-07-06, Emir): BİRİNCİL = CBOE canlı quote (bugünün current_price),
    # YEDEK = günlük CSV son kapanış. Günlük CSV bugünün baskısını kaçırıyordu (tatil-sınırında
    # age=4 → STALE-flip riski). Eşik/mimari AYNI, yalnız input tazelendi. Not: seans-içi quote
    # backtest'in kapanış-serisine göre biraz oynak olabilir; band geniş (8-11) + trim-only → güvenli.
    try:
        c, asof = fetch_cor1m_quote()
        src = "cboe_quote_live"
    except Exception as e_live:
        try:
            s = fetch_cor1m_live()
            c, asof = float(s.iloc[-1]), str(s.index[-1].date())
            src = "cboe_daily_csv_fallback"
        except Exception as e_csv:
            return {"available": False, "factor": 1.0,
                    "error": f"live={type(e_live).__name__}; csv={type(e_csv).__name__}: {str(e_csv)[:80]}"}
    # Audit 2026-06-19: as_of was recorded but NEVER gated — a stale COR1M would silently drive the
    # de-risk factor as if live. Treat a beyond-tolerance print like "no data" (factor 1.0, neutral,
    # consistent with the disabled/error path) and surface it, instead of hiding the staleness.
    from datetime import date as _date, datetime as _dt
    max_age = int(o.get("max_age_days", 4))   # daily series; ~4 cal-days spans a weekend/holiday
    age_days = (_date.today() - _dt.fromisoformat(asof).date()).days
    if age_days > max_age:
        return {"available": False, "factor": 1.0, "cor1m": round(c, 2), "as_of": asof,
                "source": src, "stale": True, "age_days": age_days,
                "reason": f"COR1M STALE: as_of {asof} ({age_days}g > {max_age}g) → de-risk OFF (factor 1.0)"}
    f = froth_factor(c, lo, hi, fl)
    return {"available": True, "cor1m": round(c, 2), "as_of": asof, "factor": round(f, 3),
            "source": src, "froth": bool(c < hi), "age_days": age_days,
            "reason": (f"COR1M {c:.1f} < {hi:.0f} → froth de-risk (factor {f:.2f})" if c < hi
                       else f"COR1M {c:.1f} ≥ {hi:.0f} → normal (factor 1.0)")}
