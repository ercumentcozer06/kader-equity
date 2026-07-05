"""
ADVERSARIAL VERIFY — candidate 'sizing' (continuous tide-magnitude conviction vs binary LONG/FLAT).
Try to REFUTE finding (recommendation=exclude, edge_real=no). Real python, real numbers.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\admin\Downloads\kader-equity")
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from spine import contract as C, tide as T          # noqa
from backtest import engine as E                     # noqa
from screen._util import load_price_csv, paired_win_prob  # noqa

DESK = Path(r"C:\Users\admin\Desktop\backtesting")
PRICES = {"SPX": "SPX_daily.csv", "NDX": "NASDAQ_daily.csv"}
ANN = np.sqrt(252)


def sh(r):
    r = np.asarray(r); r = r[np.isfinite(r)]
    return float(r.mean()/r.std()*ANN) if (len(r) > 20 and r.std() > 0) else float("nan")
def sortino(r):
    r = np.asarray(r); r = r[np.isfinite(r)]; d = r[r < 0]
    return float(r.mean()/d.std()*ANN) if (len(d) > 5 and d.std() > 0) else float("nan")
def dd(r):
    r = np.asarray(r); r = r[np.isfinite(r)]
    eq = np.cumprod(1+r); return float((eq/np.maximum.accumulate(eq)-1).min())
def cvar(r, q=0.05):
    r = np.asarray(r); r = r[np.isfinite(r)]
    k = max(1, int(q*len(r))); return float(np.sort(r)[:k].mean())
def calmar(r):
    r = np.asarray(r); r = r[np.isfinite(r)]
    eq = np.cumprod(1+r); tot = eq[-1]; n = len(r); ann = tot**(252/n)-1
    d = abs(dd(r)); return float(ann/d) if d > 0 else float("nan")


def pnl_from_pos(pos: pd.Series, close: pd.Series, lag=1, slip=0.0) -> pd.Series:
    idx = pos.index
    ret = E.fwd_ret(close, idx).values
    p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    sret = p * ret
    if slip:
        turn = np.abs(np.diff(np.concatenate([[0.0], p])))
        sret = sret - turn*slip
    return pd.Series(sret, index=idx).dropna()


def expo(pos: pd.Series, lag=1) -> float:
    p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return float(np.nanmean(p))


def report(tag, r):
    return (f"{tag:<26} Sh {sh(r):+.3f}  Sort {sortino(r):+.3f}  maxDD {100*dd(r):+.1f}%  "
            f"Calmar {calmar(r):+.2f}  CVaR {100*cvar(r):+.3f}%")


def main():
    scores, prices_frozen, vector, prov = C.read_frozen()
    ts = T.tide_score_series(scores, vector)
    tdir = T.tide_dir_series(ts)
    idx = tdir.index
    print(f"frozen window n={len(idx)}  {idx.min().date()}..{idx.max().date()}")
    print(f"tide_score stats: mean {ts.mean():.3f} std {ts.std():.3f} "
          f"min {ts.min():.2f} max {ts.max():.2f}  frac>0 {(ts>0).mean():.3f}")

    closes = {a: load_price_csv(DESK / fn) for a, fn in PRICES.items()}

    # ---- BINARY BASE (reproduce anchor) ----
    print("\n" + "="*100)
    print("  REPRO BINARY BASE (claim: SPX Sh 1.425 maxDD -18.9% expo 0.77; NDX 1.489 maxDD -22.5%)")
    print("="*100)
    base_pnl = {}
    for a in PRICES:
        r = pnl_from_pos(tdir, closes[a])
        base_pnl[a] = r
        print(f"  {a}: {report('BINARY', r)}  expo {expo(tdir):.3f}")

    # ---- KEY MECHANISTIC TEST: does tide MAGNITUDE rank-predict fwd return WHEN LONG? ----
    print("\n" + "="*100)
    print("  CORE: does tide_score MAGNITUDE rank-predict next-day fwd return (within LONG days)?")
    print("  If YES -> conviction edge exists (finding WRONG). If ~0 -> no edge (finding HOLDS).")
    print("="*100)
    for a in PRICES:
        ret = E.fwd_ret(closes[a], idx)
        # align signal[t] -> ret[t] (ret already next-day fwd; +1d exec lag means pos=signal[t-1])
        # we test predictive content of ts[t] for ret over t->t+1 with the SAME lag convention as engine:
        # engine: pos[t]=tdir[t-1]; pnl[t]=pos[t]*ret[t]. So the score that ACTS on day t is ts[t-1].
        s_lag = ts.shift(1)
        long_mask = (tdir.shift(1) > 0)
        df = pd.concat([s_lag.rename("s"), ret.rename("r"), long_mask.rename("L")], axis=1).dropna()
        dlong = df[df["L"]]
        # spearman of score-magnitude vs fwd ret, only on days we are LONG
        sp = dlong["s"].corr(dlong["r"], method="spearman")
        pe = dlong["s"].corr(dlong["r"], method="pearson")
        # bucket fwd ret by score quintile (within long)
        q = pd.qcut(dlong["s"], 5, labels=False, duplicates="drop")
        bucket = dlong.groupby(q)["r"].mean()*252*100
        print(f"  {a}: spearman(score,fwdret|LONG)={sp:+.4f}  pearson={pe:+.4f}  n={len(dlong)}")
        print(f"      score-quintile ann fwd ret (low->high tide): "
              + "  ".join(f"{v:+.1f}%" for v in bucket.values))

    # ---- (1) RAW PIT sizing ramps (no look-ahead): expanding-z + score-ramp ----
    print("\n" + "="*100)
    print("  (1) RAW PIT sizing (expanding-z ramp, score-ramp[0,8]) — claim: every transform LOWERS Sharpe,")
    print("      maxDD/CVaR fall ONLY because expo collapses (pure de-risk)")
    print("="*100)
    # PIT expanding z of tide score (min 252 obs)
    ez_mean = ts.expanding(252).mean()
    ez_std = ts.expanding(252).std()
    z = ((ts - ez_mean) / ez_std)
    raw_variants = {
        "z-ramp[0,2]":   z.clip(0, 2)/2.0 * (ts > 0),
        "z-ramp[-1,2]":  ((z+1)/3).clip(0, 1) * (ts > 0),
        "score-ramp[0,8]": (ts.clip(0, 8)/8.0) * (ts > 0),
        "score-ramp[0,4]": (ts.clip(0, 4)/4.0) * (ts > 0),
        "tanh(z)":       (np.tanh(z).clip(0, 1)) * (ts > 0),
    }
    for a in PRICES:
        print(f"  -- {a} (base {report('', base_pnl[a]).strip()}) expo {expo(tdir):.2f}")
        for lbl, vfac in raw_variants.items():
            pos = (vfac.reindex(idx).fillna(0.0)).clip(0, 1)
            r = pnl_from_pos(pos, closes[a])
            print(f"     {report(lbl, r)}  expo {expo(pos):.2f}")

    # ---- (2) EXPOSURE-MATCHED conviction shape (rescale to base expo 0.77, leverage allowed) ----
    print("\n" + "="*100)
    print("  (2) EXPOSURE-MATCHED to binary expo (fair conviction-SHAPE test; leverage allowed)")
    print("      claim: Sharpe STILL drops AND maxDD/CVaR get WORSE (conviction-leverage concentrates risk)")
    print("="*100)
    base_expo = expo(tdir)
    for a in PRICES:
        print(f"  -- {a} (base Sh {sh(base_pnl[a]):+.3f} maxDD {100*dd(base_pnl[a]):+.1f}% expo {base_expo:.2f})")
        for lbl, vfac in raw_variants.items():
            pos0 = (vfac.reindex(idx).fillna(0.0)).clip(0, 1)
            e0 = expo(pos0)
            if e0 <= 0:
                continue
            pos = pos0 * (base_expo / e0)   # rescale avg expo to match binary (allows >1)
            r = pnl_from_pos(pos, closes[a])
            print(f"     {report(lbl, r)}  expo {expo(pos):.2f}")

    # ---- (3) ERA-SPLIT ----
    print("\n" + "="*100)
    print("  (3) ERA-SPLIT (binary vs best exposure-matched ramp). claim: binary ties/beats EVERY era")
    print("="*100)
    eras = [("2019", "2019-01-01", "2019-12-31"),
            ("2020-21", "2020-01-01", "2021-12-31"),
            ("2022", "2022-01-01", "2022-12-31"),
            ("2023-24", "2023-01-01", "2024-12-31"),
            ("2025-26", "2025-01-01", "2026-12-31")]
    # use exposure-matched score-ramp[0,8] as representative conviction
    for a in PRICES:
        print(f"  -- {a}")
        vfac = raw_variants["score-ramp[0,8]"]
        pos0 = (vfac.reindex(idx).fillna(0.0)).clip(0, 1)
        pos_m = pos0 * (base_expo / max(expo(pos0), 1e-9))
        rb = base_pnl[a]; rv = pnl_from_pos(pos_m, closes[a])
        for nm, s, e in eras:
            wb = rb[(rb.index >= s) & (rb.index <= e)]
            wv = rv[(rv.index >= s) & (rv.index <= e)]
            print(f"     {nm:<9} BIN Sh {sh(wb):+.2f}  RAMP Sh {sh(wv):+.2f}  dSh {sh(wv)-sh(wb):+.2f}")

    # ---- (4) MILDEST tilt (floor 0.9) ----
    print("\n" + "="*100)
    print("  (4) MILDEST tilt — floor f, ramp between f..1 on z (claim: still loses, monotone worse as f drops)")
    print("="*100)
    for a in PRICES:
        print(f"  -- {a} (base Sh {sh(base_pnl[a]):+.3f} Calmar {calmar(base_pnl[a]):+.2f})")
        for floor in (0.9, 0.7, 0.5):
            tilt = (floor + (1-floor)*((z+1)/3).clip(0, 1))
            pos = (tdir * tilt.reindex(idx).fillna(1.0)).clip(0, 1)
            r = pnl_from_pos(pos, closes[a])
            print(f"     floor {floor}: {report('', r).strip()}  expo {expo(pos):.2f}")

    # ---- (5) LOOK-AHEAD full-sample z (cheating) — claim it ALSO loses ----
    print("\n" + "="*100)
    print("  (5) FULL-SAMPLE-z sizing (LOOK-AHEAD, cheating) — claim: ALSO loses (so PIT-vs-LA not the cause)")
    print("="*100)
    zfull = (ts - ts.mean())/ts.std()
    for a in PRICES:
        for lbl, vfac in {"LA z-ramp[0,2]": zfull.clip(0, 2)/2*(ts > 0),
                          "LA tanh(z)": np.tanh(zfull).clip(0, 1)*(ts > 0)}.items():
            pos = (vfac.reindex(idx).fillna(0.0)).clip(0, 1)
            # exposure-match too
            pos_m = pos * (base_expo / max(expo(pos), 1e-9))
            r = pnl_from_pos(pos_m, closes[a])
            print(f"  {a} {lbl} (expo-matched): {report('', r).strip()}")

    # ---- (6) PAIRED BOOTSTRAP P(ramp Sh > binary Sh) ----
    print("\n" + "="*100)
    print("  (6) PAIRED block-bootstrap P(exposure-matched ramp Sharpe > binary Sharpe)")
    print("="*100)
    for a in PRICES:
        vfac = raw_variants["score-ramp[0,8]"]
        pos0 = (vfac.reindex(idx).fillna(0.0)).clip(0, 1)
        pos_m = pos0 * (base_expo / max(expo(pos0), 1e-9))
        rv = pnl_from_pos(pos_m, closes[a])
        p = paired_win_prob(base_pnl[a], rv)
        print(f"  {a}: P(ramp>binary)={p:.3f}  (want >0.5 to favor sizing; <<0.5 confirms exclude)")

    # ---- (7) SLIPPAGE 5bps ----
    print("\n" + "="*100)
    print("  (7) SLIPPAGE 5bps (claim: widens gap; binary turns less, ramp turns daily)")
    print("="*100)
    for a in PRICES:
        rb = pnl_from_pos(tdir, closes[a], slip=0.0005)
        vfac = raw_variants["score-ramp[0,8]"]
        pos0 = (vfac.reindex(idx).fillna(0.0)).clip(0, 1)
        rv = pnl_from_pos(pos0, closes[a], slip=0.0005)
        print(f"  {a}: BIN Sh {sh(rb):+.3f}  RAW-RAMP Sh {sh(rv):+.3f}")

    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
