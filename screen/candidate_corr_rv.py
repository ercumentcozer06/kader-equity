"""
screen/candidate_corr_rv — implied correlation (COR1M/COR3M) + RV-ratio, strict FDR incremental.

Hipotez: YÜKSEK implied-corr = sistemik/makro-sürücülü, dispersion-ölü, beta-tek = hızlı-şok imzası →
tide-long'u kıs. COR-term (COR1M/COR3M>1) = yakın-uç sistemik stres. RV-ratio (rv5/rv20) = vol-genişleme.
Hepsi vol-ailesi (VIX ile co-move) → şüpheli; ama corr dispersyon ölçer, FARKLI olabilir → dürüst test.
STRICT BH-FDR {SPX,NDX} İKİSİ (= ikisi de P(v>b)≥95%).
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


def z(s, win=252): return (s - s.rolling(win, min_periods=60).mean()) / s.rolling(win, min_periods=60).std()
def _sh(r): r = r.dropna(); return float(r.mean()/r.std()*np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")
def _dd(r): eq = (1+r.dropna()).cumprod(); return float((eq/eq.cummax()-1).min())
def _ep(r, s, e): w = r[(r.index >= s) & (r.index <= e)]; return _sh(w)


def strat_ret(pos, close, lag=1):
    idx = pos.index; ret = E.fwd_ret(close, idx).values; p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def rv_ratio(close, idx):
    cb = close.reindex(idx, method="ffill")
    lr = np.log(cb / cb.shift(1))
    return (lr.rolling(5).std() / lr.rolling(20).std())


def main():
    cp = pd.read_parquet(ROOT / "data" / "cache" / "corr_pc.parquet")
    cor1m, cor3m = cp["COR1M"].dropna(), cp["COR3M"].dropna()
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index

    cor_z = z(cor1m).reindex(idx, method="ffill")
    cor_term = z(cor1m / cor3m.reindex(cor1m.index, method="ffill")).reindex(idx, method="ffill")
    rv = {a: z(rv_ratio(prices[a], idx)) for a in ("SPX", "NDX")}     # asset-specific

    def factor(stress_z): return (1.0 - 0.5 * np.clip(stress_z - 1.0, 0.0, 3.0)).clip(0.4, 1.0)
    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    sigs = {"COR1M-z": (cor_z, cor_z), "COR-term": (cor_term, cor_term),
            "RV-ratio": (rv["SPX"], rv["NDX"])}                       # (SPX-sinyal, NDX-sinyal)

    print(f"  base: SPX {_sh(bases['SPX']):+.3f} / NDX {_sh(bases['NDX']):+.3f}   (BH-FDR geç = İKİSİ de P(v>b)≥95%)")
    print("=" * 88)
    print(f"  {'sinyal':<12}{'SPX ΔSh':>9}{'SPX P':>7}{'NDX ΔSh':>9}{'NDX P':>7}{'SPX dd':>8}{'NDX dd':>8}{'COVIDrt(SPX)':>13}{'FDR':>6}")
    for name, (ssig, nsig) in sigs.items():
        res = {}
        for a, sig in (("SPX", ssig), ("NDX", nsig)):
            v = strat_ret((tdir * factor(sig.reindex(idx, method="ffill"))).reindex(idx), prices[a])
            res[a] = (paired_win_prob(bases[a], v), _sh(v) - _sh(bases[a]), _dd(v),
                      _ep(v, "2020-02-01", "2020-06-30"))
        passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
        both = "PASS" if all(passed.get(a, False) for a in ("SPX", "NDX")) else "—"
        rt = f"{res['SPX'][3]:+.1f}v{_ep(bases['SPX'],'2020-02-01','2020-06-30'):+.1f}"
        print(f"  {name:<12}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}"
              f"{100*res['SPX'][2]:>+7.0f}%{100*res['NDX'][2]:>+7.0f}%{rt:>13}{both:>6}")
    print("=" * 88)
    print("  PASS = strict BH-FDR ikisi de. (vol-ailesi: COR/RV VIX ile co-move → büyük ihtimal redundant.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
