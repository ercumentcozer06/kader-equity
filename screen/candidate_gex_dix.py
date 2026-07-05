"""
screen/candidate_gex_dix — GEX'i strict FDR üstüne taşıma denemesi: GEX+DIX KOMPOZİT fragility.

SqueezeMetrics mantığı: düşük GEX (short-gamma=kırılgan) + düşük DIX (zayıf dark-buying=dağıtım/destek-yok)
= en kırılgan setup. stress = mean(-z(GEX), -z(DIX)); yüksek = ikisi de düşük → tide-long'u soft-trim.
STRICT BH-FDR {SPX,NDX} İKİSİ. (vol kompoziti analojisi — kompozit-önce, strict.)
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
def _cv(r, q=0.05): r = r.dropna().values; k = max(1, int(q*len(r))); return float(np.sort(r)[:k].mean())
def _ep(r, s, e): w = r[(r.index >= s) & (r.index <= e)]; return _sh(w)


def strat_ret(pos, close, lag=1):
    idx = pos.index; ret = E.fwd_ret(close, idx).values; p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def main():
    sg = pd.read_parquet(ROOT / "data" / "cache" / "squeeze_dix_gex.parquet")
    stress = (-0.5 * z(sg["gex"]) - 0.5 * z(sg["dix"])).dropna()    # yüksek = GEX↓ & DIX↓ = kırılgan
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    st = stress.reindex(idx, method="ffill")

    def factor(k, thr, fl): return (1.0 - k * np.clip(st - thr, 0.0, 3.0)).clip(fl, 1.0)
    variants = {"PRIMARY k.5 thr1 fl.4": (0.5, 1.0, 0.4), "sens early thr0.5": (0.5, 0.5, 0.4),
                "sens strong k1.0": (1.0, 1.0, 0.3)}
    print(f"  GEX+DIX stress: {len(stress)} gün {stress.index.min().date()}..{stress.index.max().date()}")
    print("  STRICT BH-FDR {SPX,NDX} İKİSİ. variant = kompozit-kırılgan iken tide soft-trim (rebound-safe)")
    print("=" * 92)
    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    for a in ("SPX", "NDX"):
        b = bases[a]
        print(f"\n  [{a}]  base: Sharpe {_sh(b):+.3f}  maxDD {100*_dd(b):+.0f}%  CVaR {100*_cv(b):+.2f}%")
        print(f"    {'variant':<24}{'Sharpe':>8}{'ΔSh':>7}{'maxDD':>8}{'CVaR':>8}{'P(v>b)':>8}   {'COVID rt':>10}")
        for label, (k, thr, fl) in variants.items():
            v = strat_ret((tdir * factor(k, thr, fl)).reindex(idx), prices[a])
            rt = f"{_ep(v,'2020-02-01','2020-06-30'):+.1f}v{_ep(b,'2020-02-01','2020-06-30'):+.1f}"
            wp = paired_win_prob(b, v)
            print(f"    {label:<24}{_sh(v):>+8.3f}{_sh(v)-_sh(b):>+7.2f}{100*_dd(v):>+7.0f}%{100*_cv(v):>+7.2f}%"
                  f"{(f'{wp:.0%}' if wp is not None else 'n/a'):>8}   {rt:>10}")
    k, thr, fl = variants["PRIMARY k.5 thr1 fl.4"]
    wps = {a: paired_win_prob(bases[a], strat_ret((tdir*factor(k, thr, fl)).reindex(idx), prices[a]))
           for a in ("SPX", "NDX")}
    passed = fdr_bh({a: 1.0-w for a, w in wps.items() if w is not None}, alpha=0.05)
    both = all(passed.get(a, False) for a in ("SPX", "NDX"))
    print("\n" + "-" * 92)
    print(f"  STRICT BH-FDR (PRIMARY): P(v>b) {{{', '.join(f'{a}:{wps[a]:.0%}' for a in wps)}}}  → pass {passed}")
    print(f"  VERDICT: GEX+DIX kompozit {'→ LIVE ADAYI (ikisi de strict FDR geçti)' if both else '→ strict FDR geçemedi (shield-sekonderini geçiyor olabilir)'}")
    print("-" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
