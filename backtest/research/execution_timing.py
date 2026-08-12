# -*- coding: utf-8 -*-
"""ICRA ZAMANLAMASI LABI (2026-08-11, Emir onayi — Adim 1).

SORU: kader-equity cagrisi hangi ANDA pozisyona girilirse backtest'teki getiriyi verir, ve
erken/gec girmenin OLCULEN maliyeti nedir?

KONVANSIYON (backtest/engine.py'den birebir dogrulandi):
    fwd_ret: ret[t] = close[t+1]/close[t] - 1
    backtest_dir(lag=1): pos_used[t] = tdir[t-1]
    => sinyal as_of=S  ->  PnL = tdir[S] x (close[S+2]/close[S+1] - 1)
    yani: S+1 KAPANISINDA pozisyonda ol, S+2 kapanisina kadar tut.

DUZELTME (2026-08-11, ilk pas kendi hatam): ilk surum senaryolari "open[S+1]->close[S+2]" gibi
1.5 GUNLUK pencerelerle kurmustu -> ARDISIK sinyaller S+1 seansini IKI KEZ sayiyordu (cakisan
pencere). Ureyen %594 getiri SISKINDI ve kiyas elma-armut oluyordu (1 gun vs 1.5 gun tutus).
Dogru kiyas: hepsi SUREKLI, 1-GUNLUK, CAKISMAYAN — degisen tek sey REBALANS ANI.

Dort senaryo (ayni sinyal serisi, yalniz rebalans ani/penceresi farkli):
    BASE  (sozlesme)   : pos[t-1] x (close[t+1]/close[t] - 1)    <- engine'in birebir konvansiyonu
    ERKEN (acilista)   : pos[t-1] x (open[t+1]/open[t]  - 1)     <- bir seans once rebalans
    GEC   (bir gun ge.): pos[t-1] x (close[t+2]/close[t+1] - 1)  <- bir gun geciktirilmis
    SEANS (gece FLAT)  : pos[t-1] x (close[t+1]/open[t+1] - 1)   <- AYRI strateji: geceyi hic tutma

ILERI-BAKIS YOK: pos[t-1] sinyali t-1 kapanisinda tamamlanan veriyle uretilir; ERKEN'in girisi
open[t] o kapanistan SONRA gelir. (kader-equity run_daily t-1 gunu kosar.)
Istatistik: fark-Sharpe'lari blok-bootstrap CI ile.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EQ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EQ))

from spine import contract as C, tide as T   # noqa: E402

SEED, N_BOOT, BLOCK = 7, 5000, 21
ANN = np.sqrt(252.0)


def ohlc(sym: str, start="2018-12-01") -> pd.DataFrame:
    import yfinance as yf
    h = yf.Ticker(sym).history(start=start, auto_adjust=False)
    h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
    return h[["Open", "Close"]].dropna()


def sharpe(r):
    r = np.asarray(r)[np.isfinite(r)]
    sd = r.std(ddof=1)
    return float(ANN * r.mean() / sd) if sd > 0 else float("nan")


def block_boot(x, fn, n=N_BOOT, block=BLOCK, seed=SEED):
    rng = np.random.default_rng(seed)
    T_ = len(x)
    nb = int(np.ceil(T_ / block))
    out = np.empty(n)
    for i in range(n):
        st = rng.integers(0, max(1, T_ - block), size=nb)
        idx = np.concatenate([np.arange(s, s + block) for s in st])[:T_]
        out[i] = fn(x[idx])
    return float(np.percentile(out, 5)), float(np.percentile(out, 95)), float((out > 0).mean())


def stack_positions():
    scores, prices, vector, _ = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector)).astype(float)
    from modules import dispersion_ensemble as DE, gex_shield as GS, es_basis_unwind as EB
    cache, idx = EQ / "data" / "cache", tdir.index
    disp = pd.read_parquet(cache / "dispersion.parquet")
    corr = pd.read_parquet(cache / "corr_pc.parquet")["COR1M"].dropna()
    fp = DE.froth_pct_series(corr, disp["spread"].dropna(), disp["dspx"].dropna(), 756, 252)
    f1 = fp.reindex(idx, method="ffill").map(
        lambda v: DE.ensemble_factor(v, 0.70, 0.95, 0.0) if pd.notna(v) else 1.0)
    gex = pd.read_parquet(cache / "squeeze_dix_gex.parquet")["gex"].dropna()
    f2 = GS.shield_factor_series(GS.gex_zscore(gex, 252).reindex(idx, method="ffill"),
                                 0.5, 1.0, 0.4).fillna(1.0)
    bp = cache / "es_basis_daily.parquet"
    f3 = (EB.unwind_factor_series(pd.read_parquet(bp)["spread_bps"], p_thr=0.75, dz_thr=-1.5,
                                  floor=0.0).reindex(idx, method="ffill").fillna(1.0)
          if bp.exists() else pd.Series(1.0, index=idx))
    return {"spine (kilitli capa)": tdir, "tam stack": (tdir * f1 * f2 * f3).clip(0, 1)}, prices


def scenarios(px: pd.DataFrame) -> pd.DataFrame:
    """Satir t = pos[t-1]'in kazandigi getiri. HEPSI 1 gunluk + CAKISMAYAN (surekli strateji)."""
    c, o = px["Close"], px["Open"]
    return pd.DataFrame({
        "base":  c.shift(-1) / c - 1.0,               # close[t] -> close[t+1]   (engine konvansiyonu)
        "erken": o.shift(-1) / o - 1.0,               # open[t]  -> open[t+1]    (bir seans once rebalans)
        "gec":   c.shift(-2) / c.shift(-1) - 1.0,     # close[t+1] -> close[t+2] (bir gun gec)
        "seans": c / o - 1.0,                         # open[t]  -> close[t]     (geceyi TUTMA)
    }).dropna()


def main():
    pos_map, frozen = stack_positions()
    print("=" * 96)
    print("  ICRA ZAMANLAMASI — ayni sinyal, farkli giris ani")
    print("  BASE=kapanista rebalans | ERKEN=acilista | GEC=bir gun gec | SEANS=geceyi tutma (ayri strateji)")
    print("=" * 96)
    for asset, sym in (("SPX", "^GSPC"), ("NDX", "^NDX")):
        sc = scenarios(ohlc(sym))
        print(f"\n### {asset} ({sym})")
        for label, pos in pos_map.items():
            idx = pos.index.intersection(sc.index)
            p = pos.reindex(idx).shift(1).values          # pos[t-1] (engine lag=1 ile birebir)
            s = sc.reindex(idx)
            m = np.isfinite(p) & s.notna().all(axis=1).values
            p, s = p[m], s[m]
            r = {k: p * s[k].values for k in ("base", "erken", "gec", "seans")}
            print(f"\n  -- {label}  (n={len(p)}, expo %{100*p.mean():.0f})")
            for k in ("base", "erken", "gec", "seans"):
                tot = float(np.prod(1 + r[k]) - 1)
                print(f"     {k:<6}: Sharpe {sharpe(r[k]):+.3f}   toplam getiri %{100*tot:>7.0f}")
            for k in ("erken", "gec", "seans"):
                d = r[k] - r["base"]
                X = np.column_stack([r[k], r["base"]])
                lo, hi, pp = block_boot(X, lambda z: sharpe(z[:, 0]) - sharpe(z[:, 1]))
                print(f"     {k} - base: dSharpe {sharpe(r[k])-sharpe(r['base']):+.3f}  "
                      f"CI[{lo:+.3f},{hi:+.3f}]  P(iyilesme)={pp:.2f}")
    print("\n" + "=" * 96)
    print("  KARAR: dSharpe negatif + CI sifiri kapsamiyorsa o giris ani PARA KAYBETTIRIR.")
    print("=" * 96)


if __name__ == "__main__":
    main()
