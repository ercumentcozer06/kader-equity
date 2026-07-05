"""
screen/candidate_breadth — konsantrasyon breadth (RSP/SPY→SPX, QQEW/QQQ→NDX). TAM çerçeve (ders uygulanmış):
bucket iki-yön + çoklu horizon (21/63g) + uç-kuyruk + incremental iki-yön, strict BH-FDR.
Daralan breadth (equal-w underperform) = mega-cap konsantrasyon = kırılganlık adayı. Yön-agnostik bucket karar verir.
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


def z(s, win=252): return (s - s.rolling(win, min_periods=60).mean()) / s.rolling(win, min_periods=60).std()
def _sh(r): r = r.dropna(); return float(r.mean()/r.std()*np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")


def strat_ret(pos, close, lag=1):
    idx = pos.index; ret = E.fwd_ret(close, idx).values; p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def main():
    br = pd.read_parquet(ROOT / "data" / "cache" / "breadth.parquet")
    # breadth = relatif konsantrasyon (6ay MA'ya göre, detrend): negatif = daralma
    sig = {"SPX": z(br["RSP_SPY"] / br["RSP_SPY"].rolling(126, min_periods=40).mean()).dropna(),
           "NDX": z(br["QQEW_QQQ"] / br["QQEW_QQQ"].rolling(126, min_periods=40).mean()).dropna()}

    # ── 1) BUCKET yön-agnostik + çoklu horizon ──
    print("=" * 96)
    print("  1) BUCKET — z(relatif breadth) quintile → fwd 21/63g. (düşük=daralma/konsantrasyon)")
    print("=" * 96)
    print(f"  {'asset':<7}{'h':>4}  {'Q1(dar)':>9}{'Q2':>7}{'Q3':>7}{'Q4':>7}{'Q5(geniş)':>10}   {'bot5%':>7}{'top5%':>7}")
    for a, fn in PRICES.items():
        close = load_price_csv(DESK / fn)
        s = sig[a]; idx = s.index.intersection(close.index)
        sv = s.reindex(idx); cb = close.reindex(idx, method="ffill")
        for h in (21, 63):
            fh = (cb.shift(-h) / cb - 1)
            try:
                q = pd.qcut(sv, 5, labels=False, duplicates="drop")
            except ValueError:
                continue
            b = [100*fh[q == i].mean() for i in range(5)]
            p5, p95 = sv.quantile([.05, .95])
            t = [100*fh[sv <= p5].mean(), 100*fh[sv >= p95].mean()]
            print(f"  {a:<7}{h:>4}  {b[0]:>+9.1f}{b[1]:>+7.1f}{b[2]:>+7.1f}{b[3]:>+7.1f}{b[4]:>+10.1f}   {t[0]:>+7.1f}{t[1]:>+7.1f}")

    # ── 2) INCREMENTAL over TIDE: iki yön + uç ──
    print("\n" + "=" * 96)
    print("  2) INCREMENTAL over TIDE (2019+): trim iki yönde + uç. STRICT BH-FDR {SPX,NDX}")
    print("=" * 96)
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    print(f"  base SPX {_sh(bases['SPX']):+.3f}/NDX {_sh(bases['NDX']):+.3f}")
    print(f"  {'kural':<26}{'SPX ΔSh':>9}{'SPX P':>7}{'NDX ΔSh':>9}{'NDX P':>7}{'FDR':>6}")

    def trim_low(s):  return (1 - 0.5*np.clip(-s - 1, 0, 3)).clip(0.4, 1)   # daralma(düşük) → trim
    def trim_high(s): return (1 - 0.5*np.clip(s - 1, 0, 3)).clip(0.4, 1)    # genişleme(yüksek) → trim
    def flat_low(s):  return (s >= -1.5).astype(float)                       # ekstrem daralma → flat
    for label, fnf in (("daralma→trim", trim_low), ("genişleme→trim", trim_high), ("ekstrem-daralma→flat", flat_low)):
        res = {}
        for a in ("SPX", "NDX"):
            fac = pd.Series(fnf(sig[a].reindex(idx, method="ffill")), index=idx)
            v = strat_ret((tdir * fac).reindex(idx), prices[a])
            res[a] = (paired_win_prob(bases[a], v), _sh(v) - _sh(bases[a]))
        passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
        both = "PASS" if all(passed.get(a, False) for a in ("SPX", "NDX")) else "—"
        print(f"  {label:<26}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}{both:>6}")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
