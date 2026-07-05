"""
modules/gex_shield — DEALER SHORT-GAMMA DRAWDOWN KALKANI (kader-equity'nin 2. overlay modülü).

GEX (SqueezeMetrics dealer gamma exposure) DERİN-DÜŞÜK = dealer SHORT gamma = hareket-amplifikasyon /
kırılgan rejim → tide-long'u kıs (trim-only, rebound-safe; ASLA short/add). Sinyal = z(GEX) trailing-252g.

Kanıt (screen/candidate_gex + finalize_stack):
  • standalone GEX strict-FDR ALFA DEĞİL: yön yok (düşük→yüksek monotonik, contrarian uç-kuyruk YOK;
    bucket_scan_tails: low-GEX +0.5, high-GEX complacency da pozitif). Tek başına ~+0.07-0.12/P85-95% (sub-FDR).
  • AMA STACK içinde KALKAN olarak ÖDÜYOR (kader-btc B8-precedent): tide × COR1M-froth × GEX-shield →
    SPX maxDD −17→−13% / NDX −20→−16%, CVaR ikisinde iyi, P(stack>tide) %100, DSR 0.985/0.994.
  • İki ortogonal katman (COR1M-froth options-froth + GEX dealer-flow) toplamda +0.22/+0.28 Sharpe
    AND −6/−7pp maxDD. GEX mekanistik olarak vol-LEVEL'dan farklı (pozisyon/akış) → vol ailesinin
    geçemediği kalkan barını geçer. Variance-reduction katmanı, alfa değil.

FORM (screen/finalize_stack ile BYTE-AYNI): zg = z(GEX, 252g/min60); zg ≤ −thr iken trim başlar:
  factor = (1 − k·clip(−zg − thr, 0, 3)).clip(floor, 1). zg ≥ −thr → 1.0 (normal). Veri yok → 1.0 (nötr).
  Default k=0.5 / thr=1.0 / floor=0.4 (max %60 trim; rebound-safe — ASLA full-flat/short).

EŞİK PROVENANCE (GÖREV 5 denetimi):
  • thr=1.0  : A PRİORİ (1σ z-cutoff), grid'de {0.5,1.0} tutuldu.
  • floor=0.4: A PRİORİ (rebound-safe %60-max-trim gerekçesi), grid'de {0.3,0.4} tutuldu.
  • k=0.5    : FITTED (candidate_gex k-grid {0.5,1.0} seçimi). DSR-muhasebesinde.
  • win=252  : A PRİORİ, OPTİMİZE EDİLMEDİ (standart 1-yıl trailing; hiç taranmadı).
  GEX standalone strict-FDR alfa DEĞİL (zaten in-stack KALKAN) → eşik-overfit riski düşük. DSR N=60 İYİMSER
  → dürüst ~SPX 0.96/NDX 0.98. config.yaml'da; DEĞER DEĞİŞMEDİ, yalnız etiketlendi.

CAVEAT: 2019+ m9-çağı tek-rejim; standalone sub-strict-FDR (in-stack kalkan). Finer gamma (gamma-flip-
  distance / vanna / charm) FORWARD-only — gamma_engine + collect_daily topluyor. SqueezeMetrics endpoint'i
  bayatlarsa/değişirse → factor 1.0 (nötr, asla agresif). frozen path z'yi data/cache parquet'inden hesaplar.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# F3 (denetim 2026-07-05): kaynak-donma kapısı eşiği — son geçerli GEX gözleminin NYSE-işlem-günü yaşı.
# Normal takvim: dünkü kapanış bugün CSV'de = 1 işlem günü; yayın hıçkırığı 2-3; >4 = ~1 hafta donmuş kaynak.
MAX_SRC_AGE_TD = 4

# SqueezeMetrics FREE günlük DIX/GEX CSV (~2011+; cols date/price/dix/gex). screen/fetch_squeeze_gex ile aynı.
SQUEEZE_URLS = (
    "https://squeezemetrics.com/monitor/static/DIX.csv",
    "https://squeezemetrics.com/monitor/static/DIX",
)


def shield_factor(zg: float | None, k: float = 0.5, thr: float = 1.0, floor: float = 0.4) -> float:
    """Tek-değer kalkan faktörü. zg = z(GEX). zg ≤ −thr → trim, derinleştikçe floor'a iner. zg yoksa 1.0 (nötr)."""
    if zg is None or (isinstance(zg, float) and np.isnan(zg)):
        return 1.0
    return float(np.clip(1.0 - k * np.clip(-float(zg) - thr, 0.0, 3.0), floor, 1.0))


def shield_factor_series(zg: pd.Series, k: float = 0.5, thr: float = 1.0, floor: float = 0.4) -> pd.Series:
    """Seri kalkan faktörü (finalize_stack ile byte-aynı: (1 − k·clip(−zg−thr,0,3)).clip(floor,1))."""
    return (1.0 - k * np.clip(-zg - thr, 0.0, 3.0)).clip(floor, 1.0)


def gex_zscore(gex: pd.Series, win: int = 252, min_periods: int = 60) -> pd.Series:
    """GEX'in trailing-252g z-skoru (candidate_gex/finalize_stack ile aynı rolling)."""
    gex = gex.dropna()
    m = gex.rolling(win, min_periods=min_periods).mean()
    s = gex.rolling(win, min_periods=min_periods).std()
    return (gex - m) / s


def fetch_gex_live(timeout: int = 30) -> pd.Series:
    """SqueezeMetrics CDN'den günlük GEX serisi (free, abonelik gerektirmez)."""
    import requests
    last = None
    for url in SQUEEZE_URLS:
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and "," in r.text[:200]:
                df = pd.read_csv(io.StringIO(r.text))
                dcol = [c for c in df.columns if "date" in c.lower()][0]
                gcol = [c for c in df.columns if c.lower() == "gex"][0]
                s = pd.Series(pd.to_numeric(df[gcol], errors="coerce").values,
                              index=pd.to_datetime(df[dcol], errors="coerce")).dropna()
                return s.sort_index()
        except Exception as e:  # noqa: BLE001 — bir sonraki URL'i dene
            last = e
    raise RuntimeError(f"SqueezeMetrics GEX çekilemedi (endpoint değişmiş/bloklu olabilir): {last}")


def evaluate(cfg: dict) -> dict:
    """Canlı GEX-shield. Döndürür factor (tide pozisyonunu çarpar) + bağlam. Flag OFF → factor 1.0."""
    o = ((cfg.get("overlays", {}) or {}).get("gex_shield", {}) or {})
    if not bool(o.get("enabled")):
        return {"available": False, "factor": 1.0, "reason": "disabled"}
    k, thr = float(o.get("k", 0.5)), float(o.get("thr", 1.0))
    fl, win = float(o.get("floor", 0.4)), int(o.get("win", 252))
    try:
        zs = gex_zscore(fetch_gex_live(), win)
    except Exception as e:  # noqa: BLE001
        return {"available": False, "factor": 1.0, "fail_safe_block": True,
                "error": f"{type(e).__name__}: {str(e)[:100]}"}
    # H3 fail-safe (a priori konvansiyon): son z NaN ise ≤5 işlem günü geçerli-z TAŞI + STALE; >5g → FAIL-LOUD
    # (brief no-trade). SESSİZ factor=1.0 (koruma-kapalı) ASLA. shield_factor MATH değişmez (frozen reproduce intact).
    valid = zs.dropna()
    if len(valid) == 0:
        return {"available": False, "factor": 1.0, "fail_safe_block": True, "error": "GEX z hiç geçerli değil (<min_periods)"}
    stale_obs = (len(zs) - 1) - zs.index.get_loc(valid.index[-1])
    if stale_obs > 5:
        return {"available": False, "factor": 1.0, "fail_safe_block": True,
                "error": f"GEX z {stale_obs} gözlem bayat (>5) → fail-loud (sessiz koruma-kapalı yok)"}
    # F3 (denetim 2026-07-05): KAYNAK-DONMA kapısı — stale_obs yalnız SERİ-İÇİ kuyruk-NaN sayar; CSV
    # TÜMÜYLE donarsa (endpoint ölü ama eski dosyayı servis ediyor) son satır GEÇERLİ z'dir → stale_obs=0,
    # kalkan sessizce 'canlı' görünür ve run.py DISARMED bile basmazdı (kader-btc GEX sınıf-hatası).
    # Takvim kapısı: son geçerli gözlemin bugüne NYSE-işlem-günü yaşı (opex_calendar tatil takvimi) >
    # MAX_SRC_AGE_TD ⇒ >5-obs bloğuyla AYNI fail-LOUD yol. Sinyal matematiği (shield_factor/z) DEĞİŞMEZ.
    from modules.opex_calendar import _holiday_np
    src_age_td = int(np.busday_count(np.datetime64(valid.index[-1].date(), "D"),
                                     np.datetime64(datetime.now(timezone.utc).date(), "D"),
                                     holidays=_holiday_np()))
    if src_age_td > MAX_SRC_AGE_TD:
        return {"available": False, "factor": 1.0, "fail_safe_block": True,
                "error": (f"GEX kaynağı DONMUŞ: son gözlem {valid.index[-1].date()} "
                          f"({src_age_td} işlem günü eski > {MAX_SRC_AGE_TD}) → fail-loud "
                          f"(bayat kalkan canlı diye servis edilmez)")}
    zg, asof, zstale = float(valid.iloc[-1]), str(valid.index[-1].date()), bool(stale_obs > 0)
    f = shield_factor(zg, k, thr, fl)
    return {"available": True, "gex_z": round(zg, 2), "as_of": asof, "factor": round(f, 3),
            "short_gamma": bool(zg < -thr), "z_stale": zstale, "z_stale_obs": int(stale_obs),
            "src_age_td": src_age_td,                       # F3: kaynak yaşı görünür (0-1 normal)
            "reason": (f"z(GEX) {zg:+.1f} < −{thr:.0f} → dealer short-gamma kalkan (factor {f:.2f})" if zg < -thr
                       else f"z(GEX) {zg:+.1f} ≥ −{thr:.0f} → normal (factor 1.0)")
                      + (f" [STALE z, {stale_obs}g taşındı]" if stale_obs > 0 else "")}
