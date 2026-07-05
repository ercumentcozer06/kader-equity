"""
screen/candidate_skew_incr — SKEW incremental-over-TIDE (2019+) strict-FDR + multi-regime era-split.

Promising leads from the bucket pass:
  - raw_z HIGH (regime-relative elevated skew)  -> elevated tail/CVaR, mild fwd weakness  => RISK-CEILING (trim long)
  - dSKEW (5/10/21) RISING fast                 -> mild fwd weakness; FALLING -> bounce    => trim-on-rising
  - ortho_z extreme HIGH                        -> worst CVaR                               => trim
We test these as TIDE-overlays (cut/trim the tide-long), the deployment shape, with paired
block-bootstrap + Benjamini-Hochberg over {SPX,NDX}. Then ERA-SPLIT each promising overlay.
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

from spine import contract as C, tide as T              # noqa: E402
from backtest import engine as E                         # noqa: E402
from screen._util import load_price_csv, paired_win_prob, fdr_bh   # noqa: E402
from screen.candidate_skew_v2 import build_signals, load_skew      # noqa: E402

DESK = Path(r"C:\Users\admin\Desktop\backtesting")
PRICES = {"SPX": "SPX_daily.csv", "NDX": "NASDAQ_daily.csv"}


def _sh(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")


def _dd(r):
    eq = (1 + r.dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min())


def _cvar(r, q=0.05):
    r = r.dropna().values
    if len(r) < 30:
        return float("nan")
    k = max(1, int(q * len(r)))
    return float(np.sort(r)[:k].mean())


def _ep(r, s, e):
    w = r[(r.index >= s) & (r.index <= e)]
    return _sh(w)


def strat_ret(pos, close, lag=1):
    idx = pos.index
    ret = E.fwd_ret(close, idx).values
    p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def main():
    sk = load_skew()
    spx = load_price_csv(DESK / PRICES["SPX"])
    ndx = load_price_csv(DESK / PRICES["NDX"])
    sig, _ = build_signals(sk, spx)

    scores, prices, vector, prov = C.read_frozen()
    ts = T.tide_score_series(scores, vector)
    tdir = T.tide_dir_series(ts)
    idx = tdir.index
    print("TIDE window:", idx.min(), "->", idx.max(), "n=", len(idx))

    # align signals to tide index (ffill last known skew-derived value)
    A = {k: sig[k].reindex(idx, method="ffill") for k in sig}

    # ── overlay variants (RISK-CEILING shapes: cut/trim the tide-long when skew "rich/rising") ──
    rawz = A["raw_z"]
    orthoz = A["ortho_z"]
    d10 = A["d10"]
    d21 = A["d21"]
    variants = {
        # regime-relative HIGH skew -> trim
        "trim rawz>1.5 (50%)":   (1.0 - 0.5 * (rawz > 1.5).astype(float)).clip(0.5, 1.0),
        "flat rawz>1.5":         (rawz <= 1.5).astype(float),
        "trim rawz>1.0 soft":    (1.0 - 0.5 * np.clip((rawz - 1.0) / 1.0, 0, 1)),
        # orthogonalized z high -> trim (worst CVaR bucket)
        "trim orthoz>1.5 (50%)": (1.0 - 0.5 * (orthoz > 1.5).astype(float)).clip(0.5, 1.0),
        # rising skew velocity -> trim
        "trim d10>+5 (50%)":     (1.0 - 0.5 * (d10 > 5).astype(float)).clip(0.5, 1.0),
        "trim d21>+5 (50%)":     (1.0 - 0.5 * (d21 > 5).astype(float)).clip(0.5, 1.0),
        # combo: elevated AND rising
        "trim rawz>1 & d10>0":   (1.0 - 0.5 * ((rawz > 1.0) & (d10 > 0)).astype(float)).clip(0.5, 1.0),
    }

    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    bsh = {a: _sh(bases[a]) for a in bases}
    bdd = {a: _dd(bases[a]) for a in bases}
    bcv = {a: _cvar(bases[a]) for a in bases}
    print("=" * 116)
    print(f"  INCREMENTAL over TIDE (2019+) — base SPX Sh {bsh['SPX']:+.3f} dd {100*bdd['SPX']:+.0f}% cvar {100*bcv['SPX']:+.2f}%"
          f" | NDX Sh {bsh['NDX']:+.3f} dd {100*bdd['NDX']:+.0f}%")
    print("=" * 116)
    print(f"  {'variant':<24}{'SPX ΔSh':>9}{'SPX P':>7}{'SPX dd':>8}{'SPX cvar':>9}"
          f"{'NDX ΔSh':>9}{'NDX P':>7}{'expo':>7}{'FDR':>6}")
    keep = {}
    for label, vfac in variants.items():
        vfac = vfac.reindex(idx, method="ffill").fillna(1.0).clip(0, 1)
        res = {}
        for a in ("SPX", "NDX"):
            v = strat_ret((tdir * vfac).reindex(idx), prices[a])
            res[a] = {"P": paired_win_prob(bases[a], v), "dSh": _sh(v) - bsh[a],
                      "dd": _dd(v), "cvar": _cvar(v), "v": v}
        passed = fdr_bh({a: 1.0 - res[a]["P"] for a in res if res[a]["P"] is not None}, alpha=0.05)
        both = "PASS" if all(passed.get(a, False) for a in ("SPX", "NDX")) else "—"
        expo = float(vfac.reindex(idx).mean())
        print(f"  {label:<24}{res['SPX']['dSh']:>+9.2f}{res['SPX']['P']:>7.0%}{100*res['SPX']['dd']:>+7.0f}%"
              f"{100*res['SPX']['cvar']:>+8.2f}%{res['NDX']['dSh']:>+9.2f}{res['NDX']['P']:>7.0%}{expo:>7.2f}{both:>6}")
        keep[label] = res

    # ── ERA-SPLIT the two best-CVaR overlays ──
    print("\n" + "=" * 116)
    print("  ERA-SPLIT (multi-regime) — base vs overlay Sharpe by sub-period (SPX). Robust = positive across eras.")
    print("=" * 116)
    eras = [("2019-2020 (covid)", "2019-01-01", "2020-12-31"),
            ("2021 (melt-up)", "2021-01-01", "2021-12-31"),
            ("2022 (bear)", "2022-01-01", "2022-12-31"),
            ("2023-2024 (recovery)", "2023-01-01", "2024-12-31"),
            ("2025-2026 (recent)", "2025-01-01", "2026-06-08")]
    pick = [l for l in ("trim rawz>1.5 (50%)", "trim orthoz>1.5 (50%)", "trim d21>+5 (50%)", "trim rawz>1 & d10>0") if l in keep]
    hdr = f"  {'era':<22}{'base Sh':>9}"
    for l in pick:
        hdr += f"{l[:14]:>16}"
    print(hdr)
    for name, s, e in eras:
        line = f"  {name:<22}{_ep(bases['SPX'], s, e):>+9.2f}"
        for l in pick:
            line += f"{_ep(keep[l]['SPX']['v'], s, e):>+16.2f}"
        print(line)

    print("\n  2026-YTD SPX (base vs overlays):")
    print(f"    base {_ep(bases['SPX'],'2026-01-01','2026-06-08'):+.2f}", end="")
    for l in pick:
        print(f" | {l[:16]} {_ep(keep[l]['SPX']['v'],'2026-01-01','2026-06-08'):+.2f}", end="")
    print()
    print("=" * 116)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
