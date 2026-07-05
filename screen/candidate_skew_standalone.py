"""
screen/candidate_skew_standalone — does regime-relative SKEW predict forward DRAWDOWN/CVaR
(a risk signal), and is it already absorbed by COR1M-froth (both options-complacency)?

1) Forward 21d MIN-return (drawdown proxy) + CVaR by raw_z / ortho_z / d21 quintile, FULL history + era-split.
2) Correlation/overlap with COR1M (the VALIDATED options-complacency signal): does skew-z add
   orthogonal info, or is it the same froth axis (then it's redundant, not additive)?
3) Honest verdict on extreme-tail asymmetry with proper neg-rate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from screen._util import load_price_csv                 # noqa: E402
from screen.candidate_skew_v2 import build_signals, load_skew   # noqa: E402

DESK = Path(r"C:\Users\admin\Desktop\backtesting")


def fwd_min(close, idx, n):
    cb = close.reindex(idx, method="ffill")
    out = {}
    arr = cb.values
    di = {d: i for i, d in enumerate(cb.index)}
    res = pd.Series(index=idx, dtype=float)
    for d in idx:
        i = di.get(d)
        if i is None or i + n >= len(arr):
            continue
        window = arr[i + 1:i + n + 1]
        res[d] = window.min() / arr[i] - 1.0   # worst close-to-close over next n days
    return res


def fwd_n(close, idx, n):
    cb = close.reindex(idx, method="ffill")
    return (cb.shift(-n) / cb - 1).reindex(idx)


def main():
    sk = load_skew()
    spx = load_price_csv(DESK / "SPX_daily.csv")
    sig, _ = build_signals(sk, spx)
    idx_uni = sk.index.intersection(spx.index)

    # ── 1) forward DRAWDOWN (21d worst dip) by signal quintile ──
    print("=" * 100)
    print("  FORWARD 21d WORST-DIP (drawdown proxy) by signal quintile — is high skew a RISK warning?")
    print("=" * 100)
    fmin = fwd_min(spx, idx_uni, 21)
    for key in ("raw_z", "ortho_z", "d21"):
        s = sig[key].reindex(idx_uni).dropna()
        qs = s.quantile([0, .2, .4, .6, .8, 1.0]).values
        print(f"\n  [{key}]  quintile -> mean forward worst-dip (more negative = deeper expected drawdown):")
        for i in range(5):
            lo, hi = qs[i], qs[i + 1]
            m = (s >= lo) & (s < hi) if i < 4 else (s >= lo) & (s <= hi)
            sub = fmin.reindex(s.index[m]).dropna()
            print(f"    Q{i+1} [{lo:>7.2f},{hi:>7.2f})  n{len(sub):>5}  mean-dip {100*sub.mean():+.2f}%  "
                  f"p5-dip {100*np.percentile(sub,5):+.2f}%  %dip<-5% {100*(sub<-0.05).mean():.0f}%")

    # ── 2) overlap with COR1M (the validated complacency signal) ──
    print("\n" + "=" * 100)
    print("  OVERLAP with COR1M-froth (validated). Same axis => redundant; orthogonal => potential add.")
    print("=" * 100)
    cor = pd.read_parquet(ROOT / "data" / "cache" / "corr_pc.parquet")["COR1M"].dropna()
    common = sk.index.intersection(cor.index)
    for key in ("raw", "raw_z", "ortho_z", "d21"):
        a = sig[key].reindex(common)
        b = cor.reindex(common)
        df = pd.concat([a, b], axis=1).dropna()
        sp = df.corr(method="spearman").iloc[0, 1]
        print(f"    Spearman(skew[{key}], COR1M) = {sp:+.2f}")
    # also vs VIX level (is skew-z just VIX in disguise?)
    vix = pd.read_parquet(ROOT / "data" / "cache" / "vixcls.parquet").iloc[:, 0].dropna()
    for key in ("raw", "raw_z", "ortho_z", "d21"):
        a = sig[key]; b = vix
        df = pd.concat([a, b], axis=1).dropna()
        print(f"    Spearman(skew[{key}], VIX)   = {df.corr(method='spearman').iloc[0,1]:+.2f}")

    # ── 3) extreme-tail asymmetry: high raw_z vs unconditional, era-split ──
    print("\n" + "=" * 100)
    print("  ERA-SPLIT: high regime-relative skew (raw_z>1.5) forward 21d return vs unconditional")
    print("=" * 100)
    eras = [("2007-2009", "2007-01-01", "2009-12-31"), ("2010-2015", "2010-01-01", "2015-12-31"),
            ("2016-2019", "2016-01-01", "2019-12-31"), ("2020-2021", "2020-01-01", "2021-12-31"),
            ("2022", "2022-01-01", "2022-12-31"), ("2023-2026", "2023-01-01", "2026-06-08")]
    f21 = fwd_n(spx, idx_uni, 21)
    rawz = sig["raw_z"].reindex(idx_uni)
    for name, s0, e0 in eras:
        m = (idx_uni >= pd.Timestamp(s0)) & (idx_uni <= pd.Timestamp(e0))
        sub_all = f21[m].dropna()
        hi = (rawz > 1.5) & m
        sub_hi = f21[hi].dropna()
        if len(sub_hi) < 10:
            print(f"  {name:<12} n_hi<10 (skip)  uncond {100*sub_all.mean():+.2f}%")
            continue
        print(f"  {name:<12} uncond {100*sub_all.mean():+.2f}% ({100*(sub_all<0).mean():.0f}%neg) | "
              f"rawz>1.5 {100*sub_hi.mean():+.2f}% ({100*(sub_hi<0).mean():.0f}%neg, n{len(sub_hi)}) | "
              f"diff {100*(sub_hi.mean()-sub_all.mean()):+.2f}%")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
