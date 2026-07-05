"""
screen/candidate_fundamental — FUNDAMENTAL (#7): earnings-momentum + valuation, Shiller verisi (free).
Earnings-mom = trailing S&P earnings YoY büyüme (momentum). Valuation = P/E z (pahalı=kontrarian?).
PIT-lag +2ay (earnings raporlama). Yön-agnostik bucket + iki yön + stabilite + incremental strict FDR.
(Forward-EPS-revision S&P-Excel'i 403 blokladı → o ayrı; bu trailing-earnings + valuation.)
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


def z(s, win): return (s - s.rolling(win, min_periods=win//3).mean()) / s.rolling(win, min_periods=win//3).std()
def _sh(r): r = r.dropna(); return float(r.mean()/r.std()*np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")


def strat_ret(pos, close, lag=1):
    idx = pos.index; ret = E.fwd_ret(close, idx).values; p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def shiller():
    xl = pd.ExcelFile(ROOT / "data" / "cache" / "_shiller.bin")
    d = xl.parse("Data", header=7)
    d = d[pd.to_numeric(d["Date"], errors="coerce").notna()].copy()
    dt = pd.to_numeric(d["Date"])
    yr = dt.astype(int); mo = ((dt - yr) * 100).round().astype(int).clip(1, 12)
    d.index = pd.to_datetime(dict(year=yr, month=mo, day=1))
    P = pd.to_numeric(d["P"], errors="coerce"); Ee = pd.to_numeric(d["E"], errors="coerce")
    return P.dropna(), Ee.dropna()


def main():
    P, Ee = shiller()
    earn_mom = z(Ee.pct_change(12), 60)                   # trailing earnings YoY büyüme
    pe = (P / Ee); pe_z = z(pe, 120)                      # P/E valuation
    # PIT lag +2 ay (earnings raporlama)
    earn_mom.index = earn_mom.index + pd.DateOffset(months=2)
    pe_z.index = pe_z.index + pd.DateOffset(months=2)

    spx = load_price_csv(DESK / "SPX_daily.csv")
    print("=" * 92)
    print("  1) BUCKET — earnings-mom & P/E-z → SPX fwd-63g (yön-agnostik) + eski/yeni stabilite")
    print("=" * 92)
    old_m = None
    for name, sig in (("earn-mom", earn_mom), ("PE-valuation", pe_z)):
        s = sig.dropna(); idx = spx.index
        sv = s.reindex(idx, method="ffill"); cb = spx.reindex(idx, method="ffill")
        f63 = (cb.shift(-63)/cb - 1)
        common = sv.dropna().index.intersection(idx)
        sv2, f2 = sv.reindex(common), f63.reindex(common)
        oldm = common < pd.Timestamp("2000-01-01")
        try:
            q = pd.qcut(sv2, 5, labels=False, duplicates="drop")
            b = [100*f2[q == i].mean() for i in range(5)]
            bo = [100*f2[oldm][pd.qcut(sv2[oldm], 5, labels=False, duplicates="drop") == i].mean() for i in range(5)] if oldm.sum() > 200 else [np.nan]*5
            print(f"  {name:<14} Q1düşük..Q5yüksek (full): " + " ".join(f"{v:+.1f}" for v in b))
            print(f"  {'':14} eski(<2000):            " + " ".join(f"{v:+.1f}" for v in bo))
        except ValueError:
            print(f"  {name}: yetersiz")

    print("\n" + "=" * 92)
    print("  2) INCREMENTAL over TIDE (2019+): iki yön. STRICT BH-FDR {SPX,NDX}")
    print("=" * 92)
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    tidx = tdir.index
    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    print(f"  base SPX {_sh(bases['SPX']):+.3f}/NDX {_sh(bases['NDX']):+.3f}")
    print(f"  {'kural':<28}{'SPX ΔSh':>9}{'SPX P':>7}{'NDX ΔSh':>9}{'NDX P':>7}{'FDR':>6}")
    for nm, sig in (("earn-mom", earn_mom), ("PE-valuation", pe_z)):
        st = sig.reindex(tidx, method="ffill")
        for sgn, ys in ((+1, "yüksek→trim"), (-1, "düşük→trim")):
            fac = (1 - 0.5*np.clip(sgn*st - 1, 0, 3)).clip(0.4, 1)
            res = {}
            for a in ("SPX", "NDX"):
                v = strat_ret((tdir * fac).reindex(tidx), prices[a])
                res[a] = (paired_win_prob(bases[a], v), _sh(v) - _sh(bases[a]))
            passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
            both = "PASS" if all(passed.get(a, False) for a in ("SPX", "NDX")) else "—"
            print(f"  {nm+' '+ys:<28}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}{both:>6}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
