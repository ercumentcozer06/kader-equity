"""es_basis_unwind — ES equity-futures-basis MANIA-UNWIND trim (overlay #4; DEPLOY 2026-08-04, EMIR KARARI).

Tez (Conks "equity repo" 2026-07): ES cash-futures basis'in implied-funding spread'i (bps over o/n FF)
= dealer bilanco-kirasi = kaldirac-talebinin tek public fiyati. Spread trailing-2y ZENGIN bolgeden
(pctl504 > p_thr) HIZLA donerken (10g-degisim z < dz_thr) = kaldirac-UNWIND nowcast -> tide-long FLAT.

Kanit sicili (Desktop/backtesting/es_basis_lab, 2026-08-04): tide-ustu P(v>b) 88/96 -> PIT-uzatilmis
(forward_ledger tide, Jun-Aug'26 dahil) 91/97; CANLI-stack (tide x dispersion x GEX) ustu SPX +1.61->+1.63
(P88) / NDX +1.82->+1.86 (P96), DSR200 ve boot-p5 her katmanda yukselir, maxDD/CVaR notr; tek Jun'26
tetigi 22-Jun (spread +121bp, dz -3.46) -> T+1 SPX -1.44% / NDX -3.22% (tam tez-isabeti).
STRICT-FDR ALTI (SPX P<95) -> Sharpe-gate'i EMIR KARARI asti ("deploy et", 2026-08-04). Sinif =
kucuk-alfa trim (kuyruk-kalkani DEGIL). Binary flat (floor=0), trim-only, tetik-nadir (~%1 gun),
rebound-safe (washout/negatif-spread bolgesine DOKUNMAZ — o bolge tide/GEX isi).

Veri: data/cache/es_basis_daily.parquet — Barchart dolmus-kontrat arsivi tabani (2009+; kolonlar
spread_bps, dy, dff, ...). evaluate() her koste bugunun spread'ini canli hesaplar (yfinance ES=F + ^GSPC
son ortak kapanis + takvim-DTE [F1 = expiry>7g, tabanla ayni pre-registered konvansiyon] + taban-son
dy/DFF) ve parquet'e APPEND eder -> taban kendiliginden uzar. FOMC faiz degisiminde dff taban-son degeri
bayatlar; es_basis_lab build'leri + kopya ile tazelenir. Son veri > max_age_bd isgunu -> factor 1.0 +
available=False (position_overlay_block fail-closed tepe-kapisi bloke eder; staleness doktrini).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARQ = ROOT / "data" / "cache" / "es_basis_daily.parquet"

P_THR, DZ_THR, PCTL_WIN, PCTL_MINP, DZ_N, Z_WIN, Z_MINP, FLOOR = 0.75, -1.5, 504, 252, 10, 252, 126, 0.0
SMOOTH_N, SMOOTH_MINP = 5, 3          # spread_5dm = rolling(5, min3).median  (lab build ile BIREBIR)


def spread_signal_series(spread_bps: pd.Series, pctl_win: int = PCTL_WIN, pctl_minp: int = PCTL_MINP,
                         dz_n: int = DZ_N, z_win: int = Z_WIN, z_minp: int = Z_MINP) -> pd.DataFrame:
    """Ham gunluk spread (bps) -> (spread_5dm, pctl, dz). Trailing-only; look-ahead yok."""
    sp = spread_bps.dropna().sort_index()
    s5 = sp.rolling(SMOOTH_N, min_periods=SMOOTH_MINP).median()
    d = s5.diff(dz_n)
    dz = (d - d.rolling(z_win, min_periods=z_minp).mean()) / d.rolling(z_win, min_periods=z_minp).std()
    pctl = s5.rolling(pctl_win, min_periods=pctl_minp).apply(lambda w: (w < w.iloc[-1]).mean())
    return pd.DataFrame({"spread_5dm": s5, "pctl": pctl, "dz": dz})


def unwind_factor(pctl_v, dz_v, p_thr: float = P_THR, dz_thr: float = DZ_THR, floor: float = FLOOR) -> float:
    """Tekil factor: zengin-bolgeden hizli donus -> floor (flat); aksi/veri-yok -> 1.0 (notr)."""
    if pctl_v is None or dz_v is None or pd.isna(pctl_v) or pd.isna(dz_v):
        return 1.0
    return float(floor) if (float(pctl_v) > p_thr and float(dz_v) < dz_thr) else 1.0


def unwind_factor_series(spread_bps: pd.Series, p_thr: float = P_THR, dz_thr: float = DZ_THR,
                         floor: float = FLOOR, **kw) -> pd.Series:
    """Tarihsel factor serisi (backtest/test tek-kaynak). Sinyal-NaN gunler notr 1.0."""
    sig = spread_signal_series(spread_bps, **kw)
    trig = (sig["pctl"] > p_thr) & (sig["dz"] < dz_thr)
    return pd.Series(np.where(trig, float(floor), 1.0), index=sig.index)


def _third_friday(y: int, m: int) -> pd.Timestamp:
    d = pd.Timestamp(y, m, 1)
    return pd.date_range(d, d + pd.offsets.MonthEnd(0), freq="W-FRI")[2]


def _should_write(d: pd.Timestamp, base: pd.DataFrame) -> bool:
    """Canli gun d parquet'e yazilmali mi? GECMIS gune asla dokunulmaz; bugunku satir TABAN-SETTLE ise
    (F1 dolu = Barchart kontrat-settle, kesin deger) EZILMEZ; canli-append satiri (F1 NaN) GUN-ICI
    GUNCELLENIR (son kosu kazanir — piyasa acikken ES/^GSPC es-zamansizligi anlik degeri +-50-100bp
    oynatabilir; sinyal katmani 5dm-MEDYAN bu gurultuyu emer, periyodik es_basis_lab rebuild'i
    intraday gunleri kontrat-settle ile degistirir)."""
    mx = base.index.max()
    if d < mx:
        return False
    if d == mx and ("F1" in base.columns) and not pd.isna(base.loc[mx, "F1"]):
        return False
    return True


def _live_spread_today(base: pd.DataFrame):
    """Bugunun ham spread'i (bps) — yfinance ES=F + ^GSPC son ORTAK kapanis; taban-son dy/dff.
    Basari: (date, spread_bps) | yazilmamali/hata: None."""
    import yfinance as yf
    es = yf.download("ES=F", period="5d", auto_adjust=False, progress=False)["Close"]
    gs = yf.download("^GSPC", period="5d", auto_adjust=False, progress=False)["Close"]
    if isinstance(es, pd.DataFrame):
        es = es.iloc[:, 0]
    if isinstance(gs, pd.DataFrame):
        gs = gs.iloc[:, 0]
    j = pd.concat({"f": es, "s": gs}, axis=1).dropna()
    if not len(j) or not _should_write(j.index[-1], base):
        return None
    d = j.index[-1]
    exps = [_third_friday(y, m) for y in (d.year, d.year + 1) for m in (3, 6, 9, 12)]
    exp = min(e for e in exps if (e - d).days > 7)
    dy = float(base["dy"].dropna().iloc[-1])
    dff = float(base["dff"].dropna().iloc[-1])
    spread = ((float(j["f"].iloc[-1]) / float(j["s"].iloc[-1]) - 1.0)
              * 365.0 / (exp - d).days + dy - dff) * 1e4
    return d, float(spread)


def evaluate(cfg: dict) -> dict:
    """Canli degerlendirme: parquet + bugunku canli spread (append, self-extending) -> factor.
    Fail-soft hesap / fail-closed bayatlik: veri > max_age_bd isgunu -> factor 1.0 + available=False."""
    ov = ((cfg.get("overlays", {}) or {}).get("es_basis_unwind", {}) or {})
    p_thr = float(ov.get("p_thr", P_THR)); dz_thr = float(ov.get("dz_thr", DZ_THR))
    floor = float(ov.get("floor", FLOOR)); max_age = int(ov.get("max_age_bd", 4))
    out = {"factor": 1.0, "available": False, "spread_bps": None, "pctl": None, "dz": None,
           "as_of": None, "age_bd": None, "src": None, "warning": None}
    try:
        if not PARQ.exists():
            out["warning"] = "es_basis_daily.parquet YOK — overlay notr + UNAVAILABLE (fail-closed)"
            return out
        base = pd.read_parquet(PARQ)
        src = "taban"
        try:
            live = _live_spread_today(base)
            if live is not None:
                d, sbps = live
                row = pd.DataFrame({c: [np.nan] for c in base.columns}, index=[d])
                row.loc[d, "spread_bps"] = sbps
                for c in ("dy", "dff"):
                    row.loc[d, c] = float(base[c].dropna().iloc[-1])
                base = pd.concat([base[base.index < d], row]).sort_index()
                try:
                    base.to_parquet(PARQ)          # self-extending taban; yazim-hatasi olumcul degil
                except Exception:
                    pass
                src = "canli-append"
        except Exception as e:
            out["warning"] = f"canli uc atlandi ({type(e).__name__}) — taban-son degerle devam"
        sig = spread_signal_series(base["spread_bps"],
                                   pctl_win=int(ov.get("pctl_win", PCTL_WIN)),
                                   dz_n=int(ov.get("dz_n", DZ_N)), z_win=int(ov.get("z_win", Z_WIN)))
        last = sig.dropna(subset=["spread_5dm"]).iloc[-1]
        as_of = sig.dropna(subset=["spread_5dm"]).index[-1]
        age_bd = int(np.busday_count(as_of.date(), pd.Timestamp.now().date()))
        out.update({"spread_bps": round(float(last["spread_5dm"]), 1),
                    "pctl": None if pd.isna(last["pctl"]) else round(float(last["pctl"]), 3),
                    "dz": None if pd.isna(last["dz"]) else round(float(last["dz"]), 2),
                    "as_of": str(as_of.date()), "age_bd": age_bd, "src": src})
        if age_bd > max_age:
            out["warning"] = (f"es-basis verisi {age_bd} isgunu bayat (>{max_age}) — factor NOTR ama "
                              f"available=False (fail-closed tepe-kapi bloke eder)")
            return out
        out["available"] = True
        out["factor"] = unwind_factor(last["pctl"], last["dz"], p_thr, dz_thr, floor)
        return out
    except Exception as e:
        out["warning"] = f"es_basis_unwind hesap hatasi: {type(e).__name__}: {e}"
        return out
