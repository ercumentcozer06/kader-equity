"""
screen/candidate_cot_div — son COT açısı: FLOW (pozisyon değişimi) + price-DIVERGENCE (distribution).
Klasik pro setup: fiyat 52h-tepede AMA specs tepede kalabalıklaşıyor / commercials satıyor = dağıtım → de-risk.
+ spec-net hızlı artış (specs piling in) = bearish. Incremental over tide, strict BH-FDR.
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
from screen._util import paired_win_prob, fdr_bh         # noqa: E402


def cot_index(net, win=156):
    lo = net.rolling(win, min_periods=52).min(); hi = net.rolling(win, min_periods=52).max()
    return ((net - lo) / (hi - lo) * 100).clip(0, 100)


def _sh(r): r = r.dropna(); return float(r.mean()/r.std()*np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")


def strat_ret(pos, close, lag=1):
    idx = pos.index; ret = E.fwd_ret(close, idx).values; p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def main():
    cot = pd.read_parquet(ROOT / "data" / "cache" / "cot_legacy.parquet")
    cot.index = cot.index + pd.Timedelta(days=3)
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    print(f"  base SPX {_sh(bases['SPX']):+.3f}/NDX {_sh(bases['NDX']):+.3f}  (STRICT BH-FDR {{SPX,NDX}})")
    print(f"  {'kural':<34}{'SPX ΔSh':>9}{'SPX P':>7}{'NDX ΔSh':>9}{'NDX P':>7}{'FDR':>6}")

    SP = {"SPX": ("ES_spec_net", "ES_comm_net"), "NDX": ("NQ_spec_net", "NQ_comm_net")}
    for label, mk in (
        ("divergence: tepe & spec-idx>70", "div"),
        ("spec-flow 13w↑ (z>1)", "flow"),
        ("tepe & comm satıyor (comm-idx<30)", "commdiv"),
    ):
        res = {}
        for a in ("SPX", "NDX"):
            spec_c, comm_c = SP[a]
            close = prices[a]
            prox = (close / close.rolling(252, min_periods=120).max())   # 1 = 52h tepe
            si = cot_index(cot[spec_c].dropna()).reindex(idx, method="ffill")
            ci_ = cot_index(cot[comm_c].dropna()).reindex(idx, method="ffill")
            px = prox.reindex(idx, method="ffill")
            if mk == "div":
                fac = np.where((px > 0.97) & (si > 70), 0.4, 1.0)
            elif mk == "flow":
                fl = cot[spec_c].diff(13).dropna()
                flz = ((fl - fl.rolling(156, min_periods=52).mean()) / fl.rolling(156, min_periods=52).std()).reindex(idx, method="ffill")
                fac = np.where(flz > 1.0, 0.5, 1.0)
            else:
                fac = np.where((px > 0.97) & (ci_ < 30), 0.4, 1.0)
            v = strat_ret((tdir * pd.Series(fac, index=idx)).reindex(idx), close)
            res[a] = (paired_win_prob(bases[a], v), _sh(v) - _sh(bases[a]), float(np.mean(np.asarray(fac) < 1)))
        passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
        both = "PASS" if all(passed.get(a, False) for a in ("SPX", "NDX")) else "—"
        print(f"  {label:<34}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}{both:>6}"
              f"   tetik~{100*res['SPX'][2]:.0f}%")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
