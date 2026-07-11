"""
modules/dispersion_ensemble — 3-WAY DISPERSION-FROTH ENSEMBLE (cor1m_froth'un tam-konstitüent HALEFİ).

Tekil-hisse dispersion iki bağımsız CBOE ölçüsü + implied-corr üçlüsü → yüksek dispersion / düşük korelasyon
= single-name call-froth / complacency → tide-long'u kıs (trim-only, rebound-safe; ASLA short/add).
cor1m_froth'u SUPERSEDE eder — COR1M zaten 1/3 bileşen; ikisi birden açıksa ÇİFT-SAYIM olur (config: biri).

Bileşenler (hepsi 'yüksek = froth' yönüne çevrili, unit-free PIT trailing-756g percentile):
  • VIXEQ − VIX spread   : tekil-hisse cap-ağırlıklı 30g örtük vol − endeks vol (Cboe Constituent Vol)
  • DSPX                 : Cboe S&P500 Dispersion Index (ayrı metodoloji, aynı aile)
  • 1 − COR1M            : Cboe 1-ay implied correlation (DÜŞÜK corr = froth → 1−pct)
  froth_pct = mean(pit(spread), pit(DSPX), 1−pit(COR1M));  factor = ramp(froth_pct; lo,hi,floor).

NEDEN (deploy gerekçesi — MC forward-dağılım, mc_implied_distribution block-bootstrap 10k yol, 3-way vs canlı):
  • ΔmaxDD p50 +2.1pp DAHA SIĞ; P(3-way daha sığ)=%88-90 NDX / %84-87 SPX (blok 10/21/42 HEPSİNDE, alt-kuyruk≈0)
  • ΔSharpe non-inferior: P(≥canlı−0.1)=%93-97 NDX; SPX Sharpe-NÖTR ama maxDD yine daha iyi.
  DÜRÜST ETİKET: bu bir Sharpe-ALFASI DEĞİL (T4 çoklu-test/FWER-fail, T1 Sharpe param-hassas). PARAM-ROBUST bir
  maxDD/TAIL upgrade, Sharpe'tan feragatsiz → drawdown-kısıtlı book için meşru (Alpha Swing bariyer-amaç).
MEKANİK: NDX = Mag-7 ~%50 → tekil-isim dispersion'ın ta kendisi (SPX geniş) → sinyal NDX'te güçlü, ARTEFAKT DEĞİL.

CAVEAT (dürüst): VIXEQ/DSPX yeni ürün → geçmiş 2014-06+ (backfill'in tamamı); frozen-tide penceresi 2019+ tek-rejim.
  Kazanç 2022-26 konsantrasyon-rejimine yaslı (mega-cap dispersion yapısal olarak post-2020) — bu ürünün DOĞASI,
  metodoloji kusuru değil. Halef statüsü CONFIG-flag'li/geri-alınabilir; cor1m_froth kodu SİLİNMEDİ (toggle).

EŞİK PROVENANCE: lo=0.70 / hi=0.95 = A PRİORİ (üst-çeyrek→üst-yaprak froth-percentile bandı); T1-grid: maxDD
  faydası TÜM grid'de (robust), Sharpe param-hassas → eşik maxDD-motive seçildi, Sharpe-fit DEĞİL. floor=0.0
  yalnız derin-froth'ta (percentile≥0.95, ~yılda birkaç gün) tam-flat; band geniş + trim-only → rebound-safe.

Fail-closed (Bible #1: bayat ASLA canlı diye servis edilmez): 3 kaynaktan HERHANGİ biri fetch-fail / eşik-üstü
  bayat ⇒ available=False + stale/error (run.py position_overlay_block bloke eder). Sessiz factor=1.0 YOK.
"""
from __future__ import annotations

import io
from datetime import date as _date, datetime as _dt

import numpy as np
import pandas as pd

CBOE = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"


# ────────────────────────── saf sinyal matematiği (backtest ile byte-aynı) ──────────────────────────
def _pit_series(s: pd.Series, win: int = 756, min_periods: int = 252) -> pd.Series:
    """Trailing-win PIT percentile (son değerin kendi penceresi içindeki rank'i; look-ahead YOK)."""
    return s.rolling(win, min_periods=min_periods).apply(
        lambda w: pd.Series(w).rank(pct=True).iloc[-1], raw=False)


def froth_pct_series(cor1m: pd.Series, spread: pd.Series, dspx: pd.Series,
                     win: int = 756, min_periods: int = 252) -> pd.Series:
    """3-way froth percentile serisi (yüksek = froth). Ortak indekste hizalar."""
    fc = 1.0 - _pit_series(cor1m, win, min_periods)     # düşük COR1M = froth
    fs = _pit_series(spread, win, min_periods)           # yüksek spread = froth
    fd = _pit_series(dspx, win, min_periods)             # yüksek DSPX = froth
    df = pd.concat([fc.rename("cor"), fs.rename("spr"), fd.rename("dsp")], axis=1)
    return df.mean(axis=1, skipna=True).rename("froth_pct")   # eşit-ağırlık (FIT YOK)


def ensemble_factor(froth_pct: float | None, lo: float = 0.70, hi: float = 0.95, floor: float = 0.0) -> float:
    """Tek-değer trim faktörü. froth_pct≤lo→1.0 (normal); ≥hi→floor (derin froth); arası lineer. Yoksa 1.0."""
    if froth_pct is None or (isinstance(froth_pct, float) and np.isnan(froth_pct)):
        return 1.0
    return float(np.clip((hi - float(froth_pct)) / (hi - lo), floor, 1.0))


def ensemble_factor_series(froth_pct: pd.Series, lo: float = 0.70, hi: float = 0.95, floor: float = 0.0) -> pd.Series:
    return ((hi - froth_pct) / (hi - lo)).clip(floor, 1.0)


# ────────────────────────────────── canlı fetch (CBOE CDN, retry) ──────────────────────────────────
def _fetch_cboe(sym: str, timeout: int = 25) -> pd.Series:
    from modules._netutil import http_get_retry
    r = http_get_retry(CBOE.format(sym), timeout=timeout)
    df = pd.read_csv(io.StringIO(r.text))
    dcol = [c for c in df.columns if "date" in c.lower()][0]
    # Denetim 07-11 P2 ([21]): kolon POZISYONLA ('son cogunlukla-sayisal') seciliyordu + deger
    # sinir-kontrolu yoktu — CBOE semasi kayarsa (yeni kolon eklenirse) sessizce YANLIS seri
    # okunurdu. Once isimle (close/last) dene; sinama bandi disi = fail-loud.
    num = [c for c in df.columns if c != dcol and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8]
    named = [c for c in num if c.strip().lower() in ("close", "last", "settlement", sym.lower())]
    col = named[0] if named else num[-1]
    s = pd.Series(pd.to_numeric(df[col], errors="coerce").values,
                  index=pd.to_datetime(df[dcol], errors="coerce"), name=sym).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    bad = s[(s < 1) | (s > 500)]                    # vol-endeksi makul bandi
    if len(bad) > max(3, 0.01 * len(s)):
        raise ValueError(f"CBOE {sym} kolon '{col}' band-disi ({len(bad)} gozlem) — sema kaymis olabilir")
    return s.drop(bad.index) if len(bad) else s


def _busday_age(last: pd.Timestamp) -> int:
    from modules.opex_calendar import _holiday_np
    return int(np.busday_count(np.datetime64(last.date(), "D"),
                               np.datetime64(_date.today(), "D"), holidays=_holiday_np()))


def evaluate(cfg: dict) -> dict:
    """Canlı 3-way dispersion-froth. factor (tide pozisyonunu çarpar) + bağlam. Flag OFF → factor 1.0.

    Fail-closed: 3 kaynaktan biri fetch-fail ⇒ fail_safe_block; biri eşik-üstü bayat ⇒ stale=True.
    """
    o = ((cfg.get("overlays", {}) or {}).get("dispersion_ensemble", {}) or {})
    if not bool(o.get("enabled")):
        return {"available": False, "factor": 1.0, "reason": "disabled"}
    lo, hi, fl = float(o.get("lo", 0.70)), float(o.get("hi", 0.95)), float(o.get("floor", 0.0))
    win, mp = int(o.get("win", 756)), int(o.get("min_periods", 252))
    max_age = int(o.get("max_age_days", 4))

    # 1) üç kaynağı çek (VIXEQ, DSPX, VIX CBOE'den; COR1M cor1m_froth ile aynı endpoint)
    try:
        vixeq = _fetch_cboe("VIXEQ")
        dspx = _fetch_cboe("DSPX")
        vix = _fetch_cboe("VIX")
        from modules.cor1m_froth import fetch_cor1m_live
        cor = fetch_cor1m_live()
    except Exception as e:  # noqa: BLE001
        return {"available": False, "factor": 1.0, "fail_safe_block": True,
                "error": f"dispersion fetch fail — {type(e).__name__}: {str(e)[:100]}"}

    spread = (vixeq.rename("v").to_frame().join(vix.rename("x"), how="inner"))
    spread = (spread["v"] - spread["x"]).dropna()
    if min(len(spread), len(dspx), len(cor)) < mp:
        return {"available": False, "factor": 1.0, "fail_safe_block": True,
                "error": f"dispersion geçmişi < min_periods ({mp}) — percentile kurulamaz"}

    # 2) FRESHNESS — en bayat kaynak bağlayıcı (Bible #1: bayat canlı diye servis EDİLMEZ)
    ages = {"spread": _busday_age(spread.index[-1]), "dspx": _busday_age(dspx.index[-1]),
            "cor1m": _busday_age(cor.index[-1])}
    binding = max(ages, key=ages.get)
    if ages[binding] > max_age:
        asof = {"spread": spread.index[-1], "dspx": dspx.index[-1], "cor1m": cor.index[-1]}[binding]
        return {"available": False, "factor": 1.0, "stale": True, "age_days": ages[binding],
                "reason": f"dispersion STALE: {binding} as_of {asof.date()} "
                          f"({ages[binding]} işgünü > {max_age}) → de-risk OFF (factor 1.0)"}

    # 3) bugünün froth_pct'i = üç bileşenin trailing-win percentile ortalaması (backtest ile aynı math)
    froth = froth_pct_series(cor, spread, dspx, win, mp)
    fp = float(froth.dropna().iloc[-1]) if froth.notna().any() else None
    comp = {"spread_pct": round(float(_pit_series(spread, win, mp).dropna().iloc[-1]), 3),
            "dspx_pct": round(float(_pit_series(dspx, win, mp).dropna().iloc[-1]), 3),
            "cor1m_inv_pct": round(float((1.0 - _pit_series(cor, win, mp)).dropna().iloc[-1]), 3)}
    f = ensemble_factor(fp, lo, hi, fl)
    return {"available": True, "factor": round(f, 3), "froth_pct": round(fp, 3) if fp is not None else None,
            "components": comp, "spread": round(float(spread.iloc[-1]), 2),
            "as_of": str(spread.index[-1].date()), "age_days": ages[binding],
            "froth": bool(fp is not None and fp >= lo),
            "reason": (f"froth_pct {fp:.2f} ≥ {lo:.2f} → dispersion de-risk (factor {f:.2f}; "
                       f"spr{comp['spread_pct']:.2f}/dspx{comp['dspx_pct']:.2f}/¬cor{comp['cor1m_inv_pct']:.2f})"
                       if (fp is not None and fp >= lo)
                       else f"froth_pct {fp:.2f} < {lo:.2f} → normal (factor 1.0)")}
