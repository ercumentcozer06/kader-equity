"""
screen/candidate_flows — AKIŞ test (elimdeki free veriyle, 'data-thin→red' demeden). ETF creation/redemption
(SPY shares, 2016-21 penceresi) + MMF (yavaş risk-off). Yön-agnostik bucket + incremental. Veri-kapsama dürüst.
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


def _sh(r): r = r.dropna(); return float(r.mean()/r.std()*np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")


def strat_ret(pos, close, lag=1):
    idx = pos.index; ret = E.fwd_ret(close, idx).values; p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def main():
    fl = pd.read_parquet(ROOT / "data" / "cache" / "flows.parquet")
    spx = load_price_csv(DESK / "SPX_daily.csv")

    # ── 1) ETF creation/redemption flow (SPY shares değişimi) → SPX fwd getiri, mevcut pencere ──
    print("=" * 88)
    print("  1) ETF flow (SPY shares Δ = para giriş/çıkış) → SPX fwd-21g, mevcut pencere (yön-agnostik)")
    print("=" * 88)
    sh = fl["SPY_shares"].dropna()
    flow = sh.pct_change()                              # >0 = creation = inflow
    flow = flow[flow.abs() > 0].reindex(sh.index).dropna()
    idx = flow.index.intersection(spx.index)
    if len(idx) > 100:
        cb = spx.reindex(idx, method="ffill"); f21 = (cb.shift(-21)/cb - 1)
        z = (flow - flow.rolling(26, min_periods=10).mean())/flow.rolling(26, min_periods=10).std()
        zz = z.reindex(idx).dropna()
        try:
            q = pd.qcut(zz, 4, labels=False, duplicates="drop")
            b = [100*f21.reindex(zz.index)[q == i].mean() for i in range(4)]
            print(f"  SPY-flow z-kova (Q1 çıkış..Q4 giriş): " + " ".join(f"Q{i+1}:{b[i]:+.1f}%" for i in range(4)))
            print(f"  IC(flow, fwd21): {zz.corr(f21.reindex(zz.index), method='spearman'):+.3f}   "
                  f"pencere {idx.min().date()}..{idx.max().date()} ({len(idx)} nokta) — 2021'de DURUYOR (free-veri limiti)")
        except ValueError:
            print("  (yetersiz)")
    else:
        print("  ETF-flow: pencere çok dar.")

    # ── 2) MMF (yavaş risk-off) over TIDE ──
    print("\n" + "=" * 88)
    print("  2) MMF (money-market assets, çeyreklik→ffill) risk-off overlay over TIDE (2019+). STRICT FDR")
    print("=" * 88)
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    tidx = tdir.index
    mmf = fl["MMF_assets"].dropna()
    mmf_g = mmf.pct_change(1)                           # çeyreklik büyüme
    zg = ((mmf_g - mmf_g.rolling(8, min_periods=4).mean())/mmf_g.rolling(8, min_periods=4).std()).reindex(tidx, method="ffill")
    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    print(f"  base SPX {_sh(bases['SPX']):+.3f}/NDX {_sh(bases['NDX']):+.3f}")
    print(f"  {'kural':<26}{'SPX ΔSh':>9}{'SPX P':>7}{'NDX ΔSh':>9}{'NDX P':>7}{'FDR':>6}")
    for label, sign in (("MMF↑(nakit-istif)→trim", +1), ("MMF↓(deploy)→trim", -1)):
        fac = (1 - 0.5*np.clip(sign*zg - 1, 0, 3)).clip(0.4, 1)
        res = {}
        for a in ("SPX", "NDX"):
            v = strat_ret((tdir * fac).reindex(tidx), prices[a])
            res[a] = (paired_win_prob(bases[a], v), _sh(v) - _sh(bases[a]))
        passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
        both = "PASS" if all(passed.get(a, False) for a in ("SPX", "NDX")) else "—"
        print(f"  {label:<26}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}{both:>6}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
