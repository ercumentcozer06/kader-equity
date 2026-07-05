"""
screen/candidate_cor1m_robust — COR1M-froth robustluk: eşik taraması + tetik-gün sayısı + yıl-yıl.
flat<9'un +0.15'i robust bir BÖLGE mi yoksa knife-edge + birkaç episode mu? Dürüst kontrol.
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


def _sh(r): r = r.dropna(); return float(r.mean()/r.std()*np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")


def strat_ret(pos, close, lag=1):
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
    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}

    print("  EŞİK TARAMASI — variant = flat when COR1M<thr (2019+)")
    print(f"  {'thr':>5}{'tetik-gün':>11}{'SPX ΔSh':>9}{'SPX P':>7}{'NDX ΔSh':>9}{'NDX P':>7}{'FDR':>6}")
    for thr in (7, 8, 9, 10, 11, 12):
        vfac = (cor_t >= thr).astype(float)
        ntrig = int((cor_t < thr).sum())
        res = {}
        for a in ("SPX", "NDX"):
            v = strat_ret((tdir * vfac).reindex(idx), prices[a])
            res[a] = (paired_win_prob(bases[a], v), _sh(v) - _sh(bases[a]))
        passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
        both = "PASS" if all(passed.get(a, False) for a in ("SPX", "NDX")) else "—"
        print(f"  {thr:>5}{ntrig:>11}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}{both:>6}")

    # yıl-yıl: flat<9 hangi yıllardan geliyor (tek-episode mu?)
    print("\n  YIL-YIL (flat<9): tetik-gün + o yıl SPX katkısı (variant−base Sharpe, o yıl)")
    vfac = (cor_t >= 9).astype(float)
    vspx = strat_ret((tdir * vfac).reindex(idx), prices["SPX"])
    bspx = bases["SPX"]
    for yr in range(2019, 2027):
        a, b = pd.Timestamp(f"{yr}-01-01"), pd.Timestamp(f"{yr}-12-31")
        nt = int(((cor_t < 9) & (cor_t.index >= a) & (cor_t.index <= b)).sum())
        vy = vspx[(vspx.index >= a) & (vspx.index <= b)]
        by = bspx[(bspx.index >= a) & (bspx.index <= b)]
        if len(by) > 20:
            print(f"    {yr}: tetik {nt:>3}g   SPX Sharpe base {_sh(by):+.2f} → variant {_sh(vy):+.2f}  (Δ {_sh(vy)-_sh(by):+.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
