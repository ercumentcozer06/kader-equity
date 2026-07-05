"""screen/verify_cor1m — cor1m_froth modül formunu strict FDR'da doğrula + en robust (lo,hi,floor) seç."""
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
from modules.cor1m_froth import froth_factor_series      # noqa: E402


def _sh(r): r = r.dropna(); return float(r.mean()/r.std()*np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")
def _dd(r): eq = (1+r.dropna()).cumprod(); return float((eq/eq.cummax()-1).min())


def sr(pos, close, lag=1):
    idx = pos.index; ret = E.fwd_ret(close, idx).values; p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def main():
    cor = pd.read_parquet(ROOT / "data" / "cache" / "corr_pc.parquet")["COR1M"].dropna()
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    cor_t = cor.reindex(idx, method="ffill")
    bases = {a: sr(tdir, prices[a]) for a in ("SPX", "NDX")}
    print(f"  base: SPX {_sh(bases['SPX']):+.3f} / NDX {_sh(bases['NDX']):+.3f}   (FDR geç = ikisi de P≥95%)")
    print(f"  {'form (lo,hi,floor)':<22}{'SPX ΔSh':>9}{'SPX P':>7}{'NDX ΔSh':>9}{'NDX P':>7}{'SPX dd':>8}{'FDR':>6}")
    forms = [("ramp 8,11,0.0", 8, 11, 0.0), ("ramp 8,11,0.3", 8, 11, 0.3),
             ("ramp 8,10,0.0", 8, 10, 0.0), ("ramp 7,11,0.0", 7, 11, 0.0),
             ("ramp 8,12,0.2", 8, 12, 0.2)]
    best = None
    for label, lo, hi, fl in forms:
        fac = froth_factor_series(cor_t, lo, hi, fl)
        res = {}
        for a in ("SPX", "NDX"):
            v = sr((tdir * fac).reindex(idx), prices[a])
            res[a] = (paired_win_prob(bases[a], v), _sh(v) - _sh(bases[a]), _dd(v))
        passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
        both = all(passed.get(a, False) for a in ("SPX", "NDX"))
        print(f"  {label:<22}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}"
              f"{100*res['SPX'][2]:>+7.0f}%{('PASS' if both else '—'):>6}")
        if both and (best is None or res["SPX"][1] + res["NDX"][1] > best[1]):
            best = (label, res["SPX"][1] + res["NDX"][1], (lo, hi, fl))
    print("=" * 70)
    print(f"  SEÇİM: {best[0]} → config lo/hi/floor = {best[2]}" if best else "  [!] hiçbiri FDR geçmedi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
