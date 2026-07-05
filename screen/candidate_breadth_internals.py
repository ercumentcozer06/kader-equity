"""
screen/candidate_breadth_internals — GERÇEK breadth (% above 200d, A/D). TAM çerçeve:
bucket iki-yön + çoklu horizon + DIVERGENCE (fiyat-tepe & breadth-zayıf = distribution) + THRUST (long-add).
S&P breadth her iki endekse (market-wide). strict BH-FDR {SPX,NDX}.
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


def _sh(r): r = r.dropna(); return float(r.mean()/r.std()*np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")
def _dd(r): eq = (1+r.dropna()).cumprod(); return float((eq/eq.cummax()-1).min())


def strat_ret(pos, close, lag=1):
    idx = pos.index; ret = E.fwd_ret(close, idx).values; p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def main():
    bi = pd.read_parquet(ROOT / "data" / "cache" / "breadth_internals.parquet")
    p200 = bi["pct_above_200d"].dropna()

    # ── 1) BUCKET: %>200d → fwd 21/63g (yön-agnostik + uç) ──
    print("=" * 96)
    print("  1) % above 200d → forward getiri (5-kova + uç). Düşük=washout, Yüksek=broad. Şekil?")
    print("=" * 96)
    print(f"  {'asset':<7}{'h':>4}  {'Q1(zayıf)':>10}{'Q2':>7}{'Q3':>7}{'Q4':>7}{'Q5(güçlü)':>10}   {'bot10%':>8}{'top10%':>8}")
    for a, fn in PRICES.items():
        close = load_price_csv(DESK / fn)
        idx = p200.index.intersection(close.index)
        sv = p200.reindex(idx); cb = close.reindex(idx, method="ffill")
        for h in (21, 63):
            fh = (cb.shift(-h)/cb - 1)
            q = pd.qcut(sv, 5, labels=False, duplicates="drop")
            b = [100*fh[q == i].mean() for i in range(5)]
            lo10, hi90 = sv.quantile([.10, .90])
            t = [100*fh[sv <= lo10].mean(), 100*fh[sv >= hi90].mean()]
            print(f"  {a:<7}{h:>4}  {b[0]:>+10.1f}{b[1]:>+7.1f}{b[2]:>+7.1f}{b[3]:>+7.1f}{b[4]:>+10.1f}   {t[0]:>+8.1f}{t[1]:>+8.1f}")

    # ── 2) INCREMENTAL over TIDE (2019+) ──
    print("\n" + "=" * 96)
    print("  2) INCREMENTAL over TIDE (2019+): low/high/divergence/thrust. STRICT BH-FDR {SPX,NDX}")
    print("=" * 96)
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    p2 = p200.reindex(idx, method="ffill")
    p2_ma = p2.rolling(63).mean()
    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    print(f"  base SPX {_sh(bases['SPX']):+.3f}/NDX {_sh(bases['NDX']):+.3f}")
    print(f"  {'kural':<30}{'SPX ΔSh':>9}{'SPX P':>7}{'NDX ΔSh':>9}{'NDX P':>7}{'SPX dd':>8}{'FDR':>6}")

    def run_rule(facfn, label):
        res = {}
        for a in ("SPX", "NDX"):
            close = prices[a]
            prox = (close / close.rolling(126, min_periods=60).max()).reindex(idx, method="ffill")
            fac = pd.Series(facfn(a, prox), index=idx)
            v = strat_ret((tdir * fac).reindex(idx), close)
            res[a] = (paired_win_prob(bases[a], v), _sh(v) - _sh(bases[a]), _dd(v))
        passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
        both = "PASS" if all(passed.get(a, False) for a in ("SPX", "NDX")) else "—"
        print(f"  {label:<30}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}"
              f"{100*res['SPX'][2]:>+7.0f}%{both:>6}")

    run_rule(lambda a, px: np.where(p2 < 40, 0.5, 1.0), "düşük breadth<40 → trim")
    run_rule(lambda a, px: np.where(p2 > 80, 0.5, 1.0), "yüksek breadth>80 → trim")
    run_rule(lambda a, px: np.where((px > 0.98) & (p2 < p2_ma), 0.4, 1.0), "DIVERGENCE: tepe & breadth↓ → trim")
    run_rule(lambda a, px: np.where((px > 0.98) & (p2 < p2_ma) & (p2 < 60), 0.3, 1.0), "DIVERGENCE+zayıf(<60) → trim")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
