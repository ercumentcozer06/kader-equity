"""
screen/candidate_reentry — FARKLI ÇERÇEVE: de-risk değil, LONG-ADD/re-entry. Tide ~%23 FLAT; o dönemde 0.
Tide-flat'ken rebound'u erken yakalayan sinyal = saf additive alfa (tide'ın kaçırdığı toparlanma).

1) FIRSAT: tide-flat günlerde forward getiri pozitif mi (kaçan rebound var mı)?
2) RE-ENTRY sinyalleri: flat'ken long-aç eğer [VIX çöküyor / GEX>0 / net-liq dönüyor / COR1M froth'tan çıkıyor].
   variant = (tide LONG) OR (flat & reentry). base=tide. strict BH-FDR {SPX,NDX}.
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

CACHE = ROOT / "data" / "cache"


def z(s, win=252): return (s - s.rolling(win, min_periods=60).mean()) / s.rolling(win, min_periods=60).std()
def _sh(r): r = r.dropna(); return float(r.mean()/r.std()*np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")
def _dd(r): eq = (1+r.dropna()).cumprod(); return float((eq/eq.cummax()-1).min())


def strat_ret(pos, close, lag=1):
    idx = pos.index; ret = E.fwd_ret(close, idx).values; p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def main():
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    flat = (tdir == 0)
    print(f"  tide FLAT günleri: {100*flat.mean():.0f}%  ({int(flat.sum())}/{len(flat)})")

    # ── 1) FIRSAT: flat-günlerde fwd getiri ──
    print("\n  1) tide-FLAT günlerde forward getiri (kaçan rebound var mı?)")
    for a in ("SPX", "NDX"):
        cb = prices[a].reindex(idx, method="ffill")
        for h in (5, 21):
            fh = (cb.shift(-h)/cb - 1)
            fl = fh[flat].dropna(); al = fh.dropna()
            print(f"    {a} h{h}: FLAT-gün ort {100*fl.mean():+.2f}% (%neg {100*(fl<0).mean():.0f})  vs tüm-gün {100*al.mean():+.2f}%")

    # ── 2) RE-ENTRY sinyalleri ──
    vix = pd.read_parquet(CACHE / "vol_surface.parquet")["vix"]
    gex = pd.read_parquet(CACHE / "squeeze_dix_gex.parquet")["gex"]
    cor = pd.read_parquet(CACHE / "corr_pc.parquet")["COR1M"]
    vix_d = vix.reindex(idx, method="ffill"); vix_ma = vix_d.rolling(20).mean()
    gex_d = gex.reindex(idx, method="ffill")
    cor_d = cor.reindex(idx, method="ffill")
    nlq = scores["m2"].reindex(idx, method="ffill") if "m2" in scores else None

    reentry = {
        "VIX çöküyor (5g↓ & <MA20)":  (vix_d.diff(5) < 0) & (vix_d < vix_ma),
        "GEX>0 (long-gamma döndü)":    (gex_d > 0),
        "COR1M>14 (froth bitti)":      (cor_d > 14),
        "net-liq(m2)>0 dönüyor":       ((nlq > 0) if nlq is not None else pd.Series(False, index=idx)),
    }
    print("\n  2) RE-ENTRY: flat'ken long-aç eğer sinyal. STRICT BH-FDR {SPX,NDX}")
    print(f"  base SPX {_sh(strat_ret(tdir, prices['SPX'])):+.3f}/NDX {_sh(strat_ret(tdir, prices['NDX'])):+.3f}")
    print(f"  {'re-entry kural':<28}{'SPX ΔSh':>9}{'SPX P':>7}{'SPX dd':>8}{'NDX ΔSh':>9}{'NDX P':>7}{'FDR':>6}")
    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    for label, sig in reentry.items():
        newpos = tdir.copy()
        newpos[flat & sig.reindex(idx).fillna(False)] = 1.0          # flat'ken sinyal varsa LONG
        res = {}
        for a in ("SPX", "NDX"):
            v = strat_ret(newpos, prices[a])
            res[a] = (paired_win_prob(bases[a], v), _sh(v) - _sh(bases[a]), _dd(v))
        passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
        both = "PASS" if all(passed.get(a, False) for a in ("SPX", "NDX")) else "—"
        print(f"  {label:<28}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{100*res['SPX'][2]:>+7.0f}%"
              f"{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}{both:>6}")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
