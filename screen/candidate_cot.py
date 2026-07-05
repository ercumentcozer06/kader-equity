"""
screen/candidate_cot — COT positioning (ES→SPX, NQ→NDX): lev-money (hedge-fon) + asset-mgr (real-money).

DERS UYGULANIYOR (COR1M sign-hatası): önce yön-agnostik BUCKET (shape), sonra incremental'ı HER İKİ
yönde test (yüksek-net→trim VE düşük-net→trim), strict BH-FDR. Sign'ı veri seçsin, ben değil.
COT weekly (Tue as-of) → +3g publish-lag (Fri) → günlük ffill. z(net/OI, 156hafta).
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
SIG = {"SPX": ("ES_lev_net", "ES_am_net"), "NDX": ("NQ_lev_net", "NQ_am_net")}


def z(s, win=156): return (s - s.rolling(win, min_periods=52).mean()) / s.rolling(win, min_periods=52).std()
def _sh(r): r = r.dropna(); return float(r.mean()/r.std()*np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")
def _dd(r): eq = (1+r.dropna()).cumprod(); return float((eq/eq.cummax()-1).min())


def strat_ret(pos, close, lag=1):
    idx = pos.index; ret = E.fwd_ret(close, idx).values; p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def main():
    cot = pd.read_parquet(ROOT / "data" / "cache" / "cot_es_nq.parquet")
    cot.index = cot.index + pd.Timedelta(days=3)          # Tue as-of → Fri publish (PIT look-ahead-free)
    zc = {c: z(cot[c].dropna()) for c in cot.columns}

    # ── 1) BUCKET (yön-agnostik): z(COT) → forward-21g getiri, eşleşen endeks ──
    print("=" * 96)
    print("  1) BUCKET — z(COT net/OI) quintile → forward-21g getiri (eşleşen endeks). Shape neyi söylüyor?")
    print("=" * 96)
    print(f"  {'sinyal':<14}{'Q1(düşük)':>10}{'Q2':>7}{'Q3':>7}{'Q4':>7}{'Q5(yüksek)':>11}   okuma")
    for a, fn in PRICES.items():
        close = load_price_csv(DESK / fn)
        for sig in SIG[a]:
            s = zc[sig].dropna()
            idx = s.index.intersection(close.index)
            if len(idx) < 300:
                continue
            cb = close.reindex(idx, method="ffill"); f21 = (cb.shift(-21)/cb - 1)
            sv = s.reindex(idx)
            try:
                q = pd.qcut(sv, 5, labels=False, duplicates="drop")
            except ValueError:
                continue
            b = [100*f21[q == i].mean() for i in range(5)]
            read = ("YÜKSEK-net→bearish (kontrarian)" if b[4] < b[0]-0.5 else
                    "DÜŞÜK-net→bearish" if b[0] < b[4]-0.5 else "düz/karışık")
            print(f"  {sig:<14}{b[0]:>+10.1f}{b[1]:>+7.1f}{b[2]:>+7.1f}{b[3]:>+7.1f}{b[4]:>+11.1f}   {read}")

    # ── 2) INCREMENTAL over TIDE (2019+): HER İKİ yön, strict BH-FDR ──
    print("\n" + "=" * 96)
    print("  2) INCREMENTAL over TIDE (2019+): trim HER İKİ yönde test. STRICT BH-FDR {SPX,NDX} İKİSİ")
    print("=" * 96)
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    print(f"  base: SPX {_sh(bases['SPX']):+.3f} / NDX {_sh(bases['NDX']):+.3f}")
    print(f"  {'signal/yön':<22}{'SPX ΔSh':>9}{'SPX P':>7}{'NDX ΔSh':>9}{'NDX P':>7}{'FDR':>6}")

    def factor(zser, sign):  # sign=+1: yüksek-z→trim ; sign=-1: düşük-z→trim
        return (1.0 - 0.5*np.clip(sign*zser - 1.0, 0.0, 3.0)).clip(0.4, 1.0)

    # lev ve am için, ES→SPX / NQ→NDX eşleşmesiyle, iki yön
    for kind, (es_c, nq_c) in (("lev", ("ES_lev_net", "NQ_lev_net")), ("am", ("ES_am_net", "NQ_am_net"))):
        for sign, ynstr in ((+1, "yüksek→trim"), (-1, "düşük→trim")):
            res = {}
            for a, sigc in (("SPX", es_c), ("NDX", nq_c)):
                fac = factor(zc[sigc].reindex(idx, method="ffill"), sign)
                v = strat_ret((tdir * fac).reindex(idx), prices[a])
                res[a] = (paired_win_prob(bases[a], v), _sh(v) - _sh(bases[a]))
            passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
            both = "PASS" if all(passed.get(a, False) for a in ("SPX", "NDX")) else "—"
            print(f"  {kind+' '+ynstr:<22}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}{both:>6}")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
