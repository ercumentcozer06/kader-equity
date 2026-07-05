"""
screen/candidate_gex_eng — GEX'i strict-FDR üstüne taşıma mühendisliği (Emir 2026-06-09).

BH-FDR {SPX,NDX} geçmek = büyük-p ≤ 0.05 = İKİSİ de P(v>b) ≥ %95. z-level SPX'i 88-93'te bırakıyor.
Aynı "kırılganken tide-long'u soft-trim" fikrinin DAHA KESKİN normalizasyonları (her biri AYRI hipotez,
overfit'e karşı hepsi raporlanır; geçen olursa yfinance forward-collector ile OOS teyit gerek):

  z      : trim ∝ relu(-z(GEX)-1)                          [referans = mevcut PRIMARY]
  pct    : trim ∝ relu(thr - pctRank₂₅₂(GEX))              [fat-tail'e robust]
  mom    : trim yalnız GEX düşük VE düşüyor (dGEX₂₀<0)      [whipsaw azalt → win-prob↑ hedef]
  regime : trim ∝ relu(-z(GEX/price)-1)                    [cap-trend'i price ile temizle]
hepsi rebound-safe (floor 0.4, asla full-flat).
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


def build_factors(sg: pd.DataFrame) -> dict:
    gex, price = sg["gex"], sg["price"]
    zg = z(gex)
    pct = gex.rolling(252, min_periods=60).apply(lambda w: (w < w[-1]).mean(), raw=True)   # 0..1 düşük=kırılgan
    mom = gex.diff(20)
    zgp = z(gex / price)
    F = {
        "z      ": (1.0 - 0.5 * np.clip(-zg - 1.0, 0, 3)).clip(0.4, 1.0),
        "pct    ": (1.0 - 0.6 * np.clip(0.25 - pct, 0, 0.25) / 0.25).clip(0.4, 1.0),
        "mom    ": (1.0 - 0.5 * np.clip(-zg - 0.5, 0, 3) * (mom < 0).astype(float)).clip(0.4, 1.0),
        "regime ": (1.0 - 0.5 * np.clip(-zgp - 1.0, 0, 3)).clip(0.4, 1.0),
    }
    return F


def main():
    sg = pd.read_parquet(ROOT / "data" / "cache" / "squeeze_dix_gex.parquet")
    F = build_factors(sg)
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    print(f"  base: SPX {_sh(bases['SPX']):+.3f} / NDX {_sh(bases['NDX']):+.3f}   (BH-FDR geç = İKİSİ de P(v>b)≥95%)")
    print("=" * 92)
    print(f"  {'form':<8}{'SPX ΔSh':>9}{'SPX P':>7}{'NDX ΔSh':>9}{'NDX P':>7}{'SPX dd':>8}{'NDX dd':>8}{'COVIDrt(SPX)':>13}{'  FDR':>7}")
    for name, fac in F.items():
        f = fac.reindex(idx, method="ffill")
        wps, dsh, dds, rts = {}, {}, {}, {}
        for a in ("SPX", "NDX"):
            v = strat_ret((tdir * f).reindex(idx), prices[a])
            wps[a] = paired_win_prob(bases[a], v)
            dsh[a] = _sh(v) - _sh(bases[a]); dds[a] = _dd(v)
            rts[a] = (_ep(v, "2020-02-01", "2020-06-30"), _ep(bases[a], "2020-02-01", "2020-06-30"))
        passed = fdr_bh({a: 1.0 - wps[a] for a in wps if wps[a] is not None}, alpha=0.05)
        both = "PASS" if all(passed.get(a, False) for a in ("SPX", "NDX")) else "—"
        rt_spx = f"{rts['SPX'][0]:+.1f}v{rts['SPX'][1]:+.1f}"
        print(f"  {name:<8}{dsh['SPX']:>+9.2f}{wps['SPX']:>7.0%}{dsh['NDX']:>+9.2f}{wps['NDX']:>7.0%}"
              f"{100*dds['SPX']:>+7.0f}%{100*dds['NDX']:>+7.0f}%{rt_spx:>13}{both:>7}")
    print("=" * 92)
    print("  PASS = strict BH-FDR ikisi de geçti. Geçen varsa OOS teyit (yfinance forward-collector) gerek.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
