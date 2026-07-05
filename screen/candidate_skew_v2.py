"""
screen/candidate_skew_v2 — CBOE SKEW tested CORRECTLY (Emir 2026-06-10).

Raw SKEW level is return-confounded: equity declines -> realized downside -> OTM puts bid up ->
skew steepens. So raw level partly REFLECTS recent returns (mean-reversion artifact), not a clean
forward signal. We may have rejected it for the wrong reason. Test:
  (a) RETURN/RV-ORTHOGONALIZED residual: regress SKEW on contemporaneous + lagged SPX returns
      and realized vol; the RESIDUAL = "skew rich/cheap vs what returns explain".
  (b) CHANGES / velocity: dSKEW over 5/10/21d (a change is return-neutral-ish by construction).
Forward 21d returns + tail (CVaR/maxDD), BOTH orientations, EXTREME tails, multi-regime era-split,
incremental-over-tide strict-FDR. Compare each to the RAW level (did orthogonalize/change add edge?).
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


def fwd_n(close, idx, n):
    cb = close.reindex(idx, method="ffill")
    return (cb.shift(-n) / cb - 1).reindex(idx)


def load_skew():
    sk = pd.read_parquet(ROOT / "data" / "cache" / "skew_vvix.parquet")["SKEW"].dropna()
    return sk


def build_signals(sk, spx):
    """Return dict of skew-derived signal series, all on SKEW's daily index."""
    idx = sk.index
    # SPX daily log returns aligned to skew dates
    sp = spx.reindex(idx, method="ffill")
    r1 = np.log(sp / sp.shift(1))
    # realized vol (21d) of SPX
    rv21 = r1.rolling(21).std() * np.sqrt(252)
    # past cumulative returns (return-confounding drivers of skew)
    r5 = sp / sp.shift(5) - 1
    r21 = sp / sp.shift(21) - 1

    sig = {}
    # (0) RAW level (baseline)
    sig["raw"] = sk.copy()

    # (a) ORTHOGONALIZED residual: regress SKEW ~ const + r1 + r5 + r21 + rv21 (contemp+recent return/RV)
    X = pd.concat([r1.rename("r1"), r5.rename("r5"), r21.rename("r21"), rv21.rename("rv21")], axis=1)
    df = pd.concat([sk.rename("y"), X], axis=1).dropna()
    Xm = np.column_stack([np.ones(len(df)), df["r1"], df["r5"], df["r21"], df["rv21"]])
    beta, *_ = np.linalg.lstsq(Xm, df["y"].values, rcond=None)
    resid = df["y"].values - Xm @ beta
    sig["ortho_resid"] = pd.Series(resid, index=df.index)
    # z-score of residual over expanding-ish 252d window (so "rich/cheap" is regime-relative, no look-ahead via rolling)
    rs = sig["ortho_resid"]
    sig["ortho_z"] = ((rs - rs.rolling(252, min_periods=63).mean()) / rs.rolling(252, min_periods=63).std())

    # (b) CHANGES / velocity
    sig["d5"] = sk - sk.shift(5)
    sig["d10"] = sk - sk.shift(10)
    sig["d21"] = sk - sk.shift(21)
    # z of raw level (for reference, regime-relative level)
    sig["raw_z"] = ((sk - sk.rolling(252, min_periods=63).mean()) / sk.rolling(252, min_periods=63).std())
    return sig, beta


def bucket_table(name, sig, close_map, idx_universe, qedges=(0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    """Quintile buckets of `sig` -> forward 21d mean return + %neg + CVaR for SPX & NDX."""
    print("=" * 104)
    print(f"  BUCKET — {name}: quintiles -> forward 21d return (both assets). High vs low = which orientation?")
    print("=" * 104)
    print(f"  {'quintile':<12}{'thr_lo':>9}{'thr_hi':>9}{'n':>6}"
          f"{'SPX f21':>10}{'SPX%neg':>9}{'SPX cvar':>10}{'NDX f21':>10}{'NDX%neg':>9}")
    s = sig.reindex(idx_universe).dropna()
    qs = s.quantile(list(qedges)).values
    f21 = {a: fwd_n(close_map[a], s.index, 21) for a in close_map}
    for i in range(len(qedges) - 1):
        lo, hi = qs[i], qs[i + 1]
        if i < len(qedges) - 2:
            m = (s >= lo) & (s < hi)
        else:
            m = (s >= lo) & (s <= hi)
        sub_idx = s.index[m]
        row = [f"Q{i+1}", lo, hi, int(len(sub_idx))]
        for a in ("SPX", "NDX"):
            v = f21[a].reindex(sub_idx).dropna()
            row += [v.mean(), (v < 0).mean(), _cvar(v)] if a == "SPX" else [v.mean(), (v < 0).mean()]
        print(f"  {row[0]:<12}{row[1]:>9.2f}{row[2]:>9.2f}{row[3]:>6}"
              f"{100*row[4]:>+9.1f}%{100*row[5]:>8.0f}%{100*row[6]:>+9.2f}%{100*row[7]:>+9.1f}%{100*row[8]:>8.0f}%")
    return


def extreme_tail(name, sig, close_map, idx_universe, p_lo=0.05, p_hi=0.95):
    """Compare the EXTREME tails (top/bottom 5%) forward 21d return."""
    s = sig.reindex(idx_universe).dropna()
    lo, hi = s.quantile(p_lo), s.quantile(p_hi)
    print(f"  [{name}] extreme tails: bottom5%<{lo:.2f}  top5%>{hi:.2f}")
    for a in ("SPX", "NDX"):
        f21 = fwd_n(close_map[a], s.index, 21)
        vb = f21.reindex(s.index[s <= lo]).dropna()
        vt = f21.reindex(s.index[s >= hi]).dropna()
        print(f"    {a}: bottom5% f21 {100*vb.mean():+.2f}% (n{len(vb)}, {100*(vb<0).mean():.0f}%neg) | "
              f"top5% f21 {100*vt.mean():+.2f}% (n{len(vt)}, {100*(vt<0).mean():.0f}%neg) | "
              f"spread {100*(vt.mean()-vb.mean()):+.2f}%")


def main():
    sk = load_skew()
    spx = load_price_csv(DESK / PRICES["SPX"])
    ndx = load_price_csv(DESK / PRICES["NDX"])
    close_map = {"SPX": spx, "NDX": ndx}
    sig, beta = build_signals(sk, spx)
    print("ORTHO regression beta [const, r1, r5, r21, rv21]:", np.round(beta, 2))
    print(f"  -> sign check: declines (neg r) should RAISE skew, so r-betas expected negative.\n")

    # universe = skew dates that have price coverage
    idx_uni = sk.index.intersection(spx.index)
    for key in ("raw", "raw_z", "ortho_resid", "ortho_z", "d5", "d10", "d21"):
        bucket_table(key, sig[key], close_map, idx_uni)
        print()

    print("=" * 104)
    print("  EXTREME-TAIL (top/bottom 5%) forward 21d — does the tail carry a directional edge?")
    print("=" * 104)
    for key in ("raw", "ortho_resid", "ortho_z", "d5", "d10", "d21"):
        extreme_tail(key, sig[key], close_map, idx_uni)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
