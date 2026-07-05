"""
screen/candidate_gex — wave-1 ADAY: GEX (dealer gamma) + DIX, SqueezeMetrics FREE tarih (2011+).

GEX = dealer gamma exposure. DÜŞÜK/negatif GEX = dealer SHORT gamma = hareket-amplifikasyon/kırılgan;
YÜKSEK GEX = LONG gamma = pin/sakin. Mekanistik olarak vol-LEVEL'dan FARKLI (pozisyon/akış).

AŞAMA 1 standalone (2011+): z(GEX) her iki yönde → SPX/NQ forward IC + hit + Sharpe + episode. (DIX bonus.)
AŞAMA 2 incremental (2019+ tide): fragility-trim — GEX derin-düşükken tide-long'u kıs (rebound-safe).
  STRICT BH-FDR {SPX,NDX} İKİSİ + COVID round-trip. GEX vol-surface'in geçemediği bar'ı geçer mi?
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
PRICES = {"SPX": "SPX_daily.csv", "NDX": "NASDAQ_daily.csv"}
EP_TIDE = {"2020-COVID": ("2020-02-01", "2020-06-30"), "2022-bear": ("2022-01-01", "2022-12-31"),
           "2023-SVB": ("2023-03-01", "2023-05-31"), "2024-Aug": ("2024-07-15", "2024-09-15")}
EP_STD = {"2011-USdg": ("2011-07-01", "2011-10-31"), "2015-Aug": ("2015-08-01", "2015-10-31"),
          "2018-Q4": ("2018-10-01", "2018-12-31"), "2020-COVID": ("2020-02-01", "2020-06-30")}


def z(s, win=252): return (s - s.rolling(win, min_periods=60).mean()) / s.rolling(win, min_periods=60).std()


def strat_ret(pos, close, lag=1):
    idx = pos.index
    ret = E.fwd_ret(close, idx).values
    p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def _sh(r): r = r.dropna(); return float(r.mean()/r.std()*np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")
def _dd(r): eq = (1+r.dropna()).cumprod(); return float((eq/eq.cummax()-1).min())
def _cv(r, q=0.05): r = r.dropna().values; k = max(1, int(q*len(r))); return float(np.sort(r)[:k].mean())
def _ep(r, s, e): w = r[(r.index >= s) & (r.index <= e)]; return _sh(w)


def stage1(zg, dix):
    print("\n" + "=" * 96)
    print("  AŞAMA 1 — STANDALONE (2011+):  GEX yönü? (z(GEX) bullish mi bearish mi) + DIX bonus")
    print("=" * 96)
    print(f"  {'asset':<7}{'IC+zg_1d':>9}{'IC+zg_21d':>10}{'IC_DIX_21d':>11}{'hit(+zg)':>9}{'Sh(gex>0 long)':>15}   episodes")
    for a, fn in PRICES.items():
        close = load_price_csv(DESK / fn)
        idx = zg.dropna().index.intersection(close.index)
        sg = zg.reindex(idx)
        f1 = E.fwd_ret(close, idx)
        cb = close.reindex(idx, method="ffill"); f21 = cb.shift(-21)/cb - 1
        ic1 = sg.corr(f1, method="spearman"); ic21 = sg.corr(f21, method="spearman")
        icd = z(dix).reindex(idx).corr(f21, method="spearman")
        hit = float((np.sign(sg) == np.sign(f1)).mean())
        pos = (sg > 0).astype(float); rr = strat_ret(pos, close)     # long when GEX above-avg
        eps = "  ".join(f"{k}:{_ep(rr, s, e):+.1f}" for k, (s, e) in EP_STD.items())
        print(f"  {a:<7}{ic1:>+9.3f}{ic21:>+10.3f}{icd:>+11.3f}{hit:>+9.2f}{_sh(rr):>+15.2f}   {eps}")
    print("  (IC>0 = yüksek-GEX bullish/sonraki-getiri+ ; IC<0 = contrarian. DIX>0 = dark-buying bullish.)")


def factor_lowgex(zg_idx, k, thr, floor):
    """GEX derin-DÜŞÜKken (kırılgan/short-gamma) tide-long'u kıs (rebound-safe, asla full-flat değil)."""
    return (1.0 - k * np.clip(-zg_idx - thr, 0.0, 3.0)).clip(floor, 1.0)


def stage2(zg):
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    zt = zg.reindex(idx, method="ffill")
    variants = {"PRIMARY lowgex k.5 thr1 fl.4": (0.5, 1.0, 0.4),
                "sens early thr0.5": (0.5, 0.5, 0.4),
                "sens strong k1.0": (1.0, 1.0, 0.3)}
    print("\n" + "=" * 96)
    print("  AŞAMA 2 — INCREMENTAL over TIDE (2019+):  variant = GEX-düşük (short-gamma) iken soft-trim")
    print("  STRICT BH-FDR α=0.05: {SPX,NDX} İKİSİ de. PRE-REG primer=ΔSharpe, sekonder=maxDD/CVaR@Sh≥0")
    print("=" * 96)
    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    for a in ("SPX", "NDX"):
        b = bases[a]
        print(f"\n  [{a}]  base tide: Sharpe {_sh(b):+.3f}  maxDD {100*_dd(b):+.0f}%  CVaR {100*_cv(b):+.2f}%")
        print(f"    {'variant':<28}{'Sharpe':>8}{'ΔSh':>7}{'maxDD':>8}{'CVaR':>8}{'P(v>b)':>8}   {'COVID rt':>10}")
        for label, (k, thr, fl) in variants.items():
            v = strat_ret((tdir * factor_lowgex(zt, k, thr, fl)).reindex(idx), prices[a])
            rt = f"{_ep(v,'2020-02-01','2020-06-30'):+.1f}v{_ep(b,'2020-02-01','2020-06-30'):+.1f}"
            wp = paired_win_prob(b, v)
            print(f"    {label:<28}{_sh(v):>+8.3f}{_sh(v)-_sh(b):>+7.2f}{100*_dd(v):>+7.0f}%{100*_cv(v):>+7.2f}%"
                  f"{(f'{wp:.0%}' if wp is not None else 'n/a'):>8}   {rt:>10}")
    k, thr, fl = variants["PRIMARY lowgex k.5 thr1 fl.4"]
    wps = {a: paired_win_prob(bases[a], strat_ret((tdir*factor_lowgex(zt, k, thr, fl)).reindex(idx), prices[a]))
           for a in ("SPX", "NDX")}
    passed = fdr_bh({a: 1.0-w for a, w in wps.items() if w is not None}, alpha=0.05)
    both = all(passed.get(a, False) for a in ("SPX", "NDX"))
    print("\n" + "-" * 96)
    print(f"  STRICT BH-FDR (PRIMARY): P(v>b) {{{', '.join(f'{a}:{wps[a]:.0%}' for a in wps)}}}  → pass {passed}")
    print(f"  VERDICT: GEX-fragility-trim {'→ LIVE ADAYI (ikisi de FDR geçti)' if both else '→ bu form FDR geçemedi'}")
    print("-" * 96)


def main():
    sg = pd.read_parquet(ROOT / "data" / "cache" / "squeeze_dix_gex.parquet")
    gex, dix = sg["gex"], sg["dix"]
    zg = z(gex)
    print(f"  GEX: {len(gex)} gün {gex.index.min().date()}..{gex.index.max().date()}  "
          f"(negatif-GEX günleri: {100*(gex < 0).mean():.0f}%)")
    stage1(zg, dix)
    stage2(zg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
