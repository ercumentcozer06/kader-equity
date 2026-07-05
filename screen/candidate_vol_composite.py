"""
screen/candidate_vol_composite — wave-1 ADAY #1 (KOMPOZİT): equity vol-surface stres.

Kompozit z = mean( z(VIX/VIX3M), z(SKEW), z(VVIX) )   [hepsi YÜKSEK = stres; 252g trailing]
  • term-structure : backwardation (ön-uç panik)
  • SKEW           : kuyruk/put-talebi (crash-hedge)
  • VVIX           : vol-of-vol (belirsizlik)
put/call atlandı (CBOE temiz uzun-tarih free yayınlamıyor) — 3-bileşen.

AŞAMA 1 standalone (2007+): -kompozit → SPX/NQ forward IC + hit + Sharpe + 8 episode.
AŞAMA 2 incremental (2019+ tide): soft-trim tide-long'u kompozit-stres'le (rebound-safe form).
  STRICT BH-FDR: {SPX,NDX} İKİSİ de geçmeli (α=0.05). PRE-REG: primer ΔSharpe; sekonder maxDD/CVaR @Sh≥0.
  Geçerse vol ekseni LIVE adayı; geçmezse vol ekseni DÜŞER (Emir: kompozit-önce, strict).
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
from screen._util import load_price_csv, paired_win_prob, fdr_bh, bootstrap_ci   # noqa: E402

DESK = Path(r"C:\Users\admin\Desktop\backtesting")
PRICES = {"SPX": "SPX_daily.csv", "NDX": "NASDAQ_daily.csv"}
EP_TIDE = {"2020-COVID": ("2020-02-01", "2020-06-30"), "2022-bear": ("2022-01-01", "2022-12-31"),
           "2023-SVB": ("2023-03-01", "2023-05-31"), "2024-Aug": ("2024-07-15", "2024-09-15")}
EP_STD = {"2010-flash": ("2010-04-01", "2010-07-31"), "2011-USdg": ("2011-07-01", "2011-10-31"),
          "2015-Aug": ("2015-08-01", "2015-10-31"), "2018-Q4": ("2018-10-01", "2018-12-31")}


def z(s: pd.Series, win: int = 252) -> pd.Series:
    return (s - s.rolling(win, min_periods=60).mean()) / s.rolling(win, min_periods=60).std()


def strat_ret(pos: pd.Series, close: pd.Series, lag: int = 1) -> pd.Series:
    idx = pos.index
    ret = E.fwd_ret(close, idx).values
    p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def _sh(r): r = r.dropna(); return float(r.mean() / r.std() * np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")
def _dd(r): eq = (1 + r.dropna()).cumprod(); return float((eq / eq.cummax() - 1).min())
def _cv(r, q=0.05): r = r.dropna().values; k = max(1, int(q * len(r))); return float(np.sort(r)[:k].mean())
def _ep(r, s, e): w = r[(r.index >= s) & (r.index <= e)]; return _sh(w)


def build_composite() -> pd.Series:
    vs = pd.read_parquet(ROOT / "data" / "cache" / "vol_surface.parquet")
    sv = pd.read_parquet(ROOT / "data" / "cache" / "skew_vvix.parquet")
    comp = pd.DataFrame({"ts": z(vs["ts_ratio"]), "skew": z(sv["SKEW"]), "vvix": z(sv["VVIX"])})
    return comp.mean(axis=1, skipna=True).dropna().rename("vol_stress")   # yüksek = stres


def stage1(cz: pd.Series) -> None:
    print("\n" + "=" * 96)
    print("  AŞAMA 1 — STANDALONE (2007+):  sinyal = -kompozit_vol_stres  (stres → bearish)")
    print("=" * 96)
    print(f"  {'asset':<7}{'IC_1d':>8}{'IC_21d':>8}{'hit':>7}{'standalone Sh':>15}   {'episode Sh':<46}")
    sig = -cz
    for a, fn in PRICES.items():
        close = load_price_csv(DESK / fn)
        idx = sig.dropna().index.intersection(close.index)
        sg = sig.reindex(idx)
        f1 = E.fwd_ret(close, idx)
        cb = close.reindex(idx, method="ffill"); f21 = cb.shift(-21) / cb - 1
        ic1, ic21 = sg.corr(f1, method="spearman"), sg.corr(f21, method="spearman")
        hit = float((np.sign(sg) == np.sign(f1)).mean())
        pos = (sg > 0).astype(float); rr = strat_ret(pos, close)
        eps = "  ".join(f"{k}:{_ep(rr, s, e):+.1f}" for k, (s, e) in EP_STD.items())
        print(f"  {a:<7}{ic1:>+8.3f}{ic21:>+8.3f}{hit:>+7.2f}{_sh(rr):>+15.2f}   {eps}")


def factor(cz_idx: pd.Series, k: float, thr: float, floor: float, cap: float = 3.0) -> pd.Series:
    """Rebound-safe soft-trim: stres yükseldikçe tide-long'u kıs (asla full-flat değil → floor)."""
    return (1.0 - k * np.clip(cz_idx - thr, 0.0, cap)).clip(floor, 1.0)


def stage2(cz: pd.Series) -> None:
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    czt = cz.reindex(idx, method="ffill")

    variants = {                                          # (k, thr, floor)
        "PRIMARY k.5 thr1 fl.4": (0.5, 1.0, 0.4),
        "sens early thr0.5":     (0.5, 0.5, 0.4),
        "sens strong k1.0":      (1.0, 1.0, 0.3),
    }
    print("\n" + "=" * 96)
    print("  AŞAMA 2 — INCREMENTAL over TIDE (2019+):  variant = kompozit-stres ile soft-trim (rebound-safe)")
    print("  STRICT BH-FDR α=0.05: {SPX,NDX} İKİSİ de geçmeli. PRE-REG primer=ΔSharpe, sekonder=maxDD/CVaR@Sh≥0")
    print("=" * 96)

    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    for a in ("SPX", "NDX"):
        b = bases[a]
        print(f"\n  [{a}]  base tide: Sharpe {_sh(b):+.3f}  maxDD {100*_dd(b):+.0f}%  CVaR {100*_cv(b):+.2f}%")
        print(f"    {'variant':<24}{'Sharpe':>8}{'ΔSh':>7}{'maxDD':>8}{'CVaR':>8}{'P(v>b)':>8}   {'COVID rt':>10}{'  episodes(v)':<34}")
        for label, (k, thr, fl) in variants.items():
            vpos = tdir * factor(czt, k, thr, fl)
            v = strat_ret(vpos.reindex(idx), prices[a])
            eps = " ".join(f"{kk[:4]}:{_ep(v, s, e):+.1f}" for kk, (s, e) in EP_TIDE.items())
            rt = f"{_ep(v,'2020-02-01','2020-06-30'):+.1f}v{_ep(b,'2020-02-01','2020-06-30'):+.1f}"
            wp = paired_win_prob(b, v)
            print(f"    {label:<24}{_sh(v):>+8.3f}{_sh(v)-_sh(b):>+7.2f}{100*_dd(v):>+7.0f}%{100*_cv(v):>+7.2f}%"
                  f"{(f'{wp:.0%}' if wp is not None else 'n/a'):>8}   {rt:>10}  {eps}")

    # STRICT FDR verdict — PRIMARY variant, {SPX,NDX} family
    k, thr, fl = variants["PRIMARY k.5 thr1 fl.4"]
    wps = {}
    for a in ("SPX", "NDX"):
        v = strat_ret((tdir * factor(czt, k, thr, fl)).reindex(idx), prices[a])
        wps[a] = paired_win_prob(bases[a], v)
    pvals = {a: (1.0 - wp) for a, wp in wps.items() if wp is not None}
    passed = fdr_bh(pvals, alpha=0.05)
    print("\n" + "-" * 96)
    print(f"  STRICT BH-FDR (PRIMARY): P(v>b) {{{', '.join(f'{a}:{wps[a]:.0%}' for a in wps)}}}  "
          f"→ pass {passed}")
    both = all(passed.get(a, False) for a in ("SPX", "NDX"))
    print(f"  VERDICT: vol-surface kompozit {'→ LIVE ADAYI (ikisi de FDR geçti)' if both else '→ DÜŞER (FDR geçemedi; strict bar, Emir kararı)'}")
    print("-" * 96)


def main() -> int:
    cz = build_composite()
    print(f"  kompozit vol-stres: {len(cz)} gün {cz.index.min().date()}..{cz.index.max().date()}  "
          f"(bileşen: z(VIX/VIX3M)+z(SKEW)+z(VVIX))")
    stage1(cz)
    stage2(cz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
