"""screen/candidate_cot_tail — COT EKSTREM kalabalıklık kuyruğu (COR1M dersi: quintile uçları yıkar).
flat/trim when z(COT-net) > thr (ekstrem net-long = kalabalık tepe). lev+am, ES→SPX/NQ→NDX, strict FDR."""
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


def z(s, win=156): return (s - s.rolling(win, min_periods=52).mean()) / s.rolling(win, min_periods=52).std()
def _sh(r): r = r.dropna(); return float(r.mean()/r.std()*np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")


def strat_ret(pos, close, lag=1):
    idx = pos.index; ret = E.fwd_ret(close, idx).values; p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def main():
    cot = pd.read_parquet(ROOT / "data" / "cache" / "cot_es_nq.parquet")
    cot.index = cot.index + pd.Timedelta(days=3)
    zc = {c: z(cot[c].dropna()) for c in cot.columns}
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    print(f"  base SPX {_sh(bases['SPX']):+.3f}/NDX {_sh(bases['NDX']):+.3f}.  flat when z(COT-net) > thr (ekstrem kalabalık)")
    print(f"  {'kind/thr':<16}{'SPX ΔSh':>9}{'SPX P':>7}{'NDX ΔSh':>9}{'NDX P':>7}{'tetik%':>8}{'FDR':>6}")
    for kind, (es_c, nq_c) in (("lev", ("ES_lev_net", "NQ_lev_net")), ("am", ("ES_am_net", "NQ_am_net"))):
        for thr in (1.0, 1.5, 2.0):
            res = {}; trig = []
            for a, sigc in (("SPX", es_c), ("NDX", nq_c)):
                zt = zc[sigc].reindex(idx, method="ffill")
                fac = (zt <= thr).astype(float)            # ekstrem net-long → FLAT
                v = strat_ret((tdir * fac).reindex(idx), prices[a])
                res[a] = (paired_win_prob(bases[a], v), _sh(v) - _sh(bases[a]))
                trig.append((zt > thr).mean())
            passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
            both = "PASS" if all(passed.get(a, False) for a in ("SPX", "NDX")) else "—"
            print(f"  {kind+' z>'+str(thr):<16}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}"
                  f"{100*np.mean(trig):>+7.0f}%{both:>6}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
