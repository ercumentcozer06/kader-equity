"""
screen/candidate_flows3 — ICI weekly equity flow RE-TEST tam pencerede (Emir haklı: monthly-refute yanlıştı).
ICI Excel (2007-2021 weekly) + datahub (2022-2026) birleşik → tam 2007-2026. Bucket (full + sub-period
stabilite) + incremental over tide, iki yön, strict FDR. Sinyal GERÇEKTEN robust mu yoksa 2022-fluke mu?
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
CACHE = ROOT / "data" / "cache"


def z(s, win=52): return (s - s.rolling(win, min_periods=20).mean()) / s.rolling(win, min_periods=20).std()
def _sh(r): r = r.dropna(); return float(r.mean()/r.std()*np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")


def strat_ret(pos, close, lag=1):
    idx = pos.index; ret = E.fwd_ret(close, idx).values; p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def load_full_ici():
    xl = pd.ExcelFile(CACHE / "_ici_full.xls")
    d = xl.parse("Weekly MF & ETF Public Report", header=4)
    dcol = [c for c in d.columns if "date" in str(c).lower()][0]
    ecol = [c for c in d.columns if str(c).strip().lower() == "equity"][0]
    old = pd.Series(pd.to_numeric(d[ecol], errors="coerce").values,
                    index=pd.to_datetime(d[dcol], errors="coerce")).dropna()
    new = pd.read_parquet(CACHE / "flows2.parquet")["ici_weekly_equity"].dropna()
    flow = pd.concat([old, new]).sort_index()
    return flow[~flow.index.duplicated(keep="last")]


def main():
    flow = load_full_ici()
    flow.index = flow.index + pd.Timedelta(days=6)         # PIT publish lag
    print(f"  ICI weekly equity flow (birleşik): {len(flow)} hafta {flow.index.min().date()}..{flow.index.max().date()}")
    fz = z(flow, 52)
    spx = load_price_csv(DESK / "SPX_daily.csv")

    # ── 1) STANDALONE BUCKET tam pencere + sub-period stabilite ──
    print("\n  1) BUCKET — z(equity flow) → SPX fwd-5g (kontrarian: yüksek-giriş→düşük getiri?) + stabilite")
    idx = fz.dropna().index
    cb = spx.reindex(idx, method="ffill"); f5 = (cb.shift(-5)/cb - 1)
    common = fz.reindex(idx).dropna().index.intersection(spx.index)
    sv, fv = fz.reindex(common), f5.reindex(common)
    for lbl, a, b in (("FULL 2008-26", "2008", "2026"), ("eski 2008-16", "2008", "2016"), ("yeni 2017-26", "2017", "2026")):
        mm = (common >= a) & (common <= b)
        if mm.sum() < 100:
            continue
        q = pd.qcut(sv[mm], 5, labels=False, duplicates="drop")
        bk = [100*fv[mm][q == i].mean() for i in range(5)]
        print(f"    {lbl:<13} Q1düşük-giriş..Q5yüksek-giriş: " + " ".join(f"{v:+.2f}" for v in bk))

    # ── 2) INCREMENTAL over TIDE (2019+) ──
    print("\n  2) INCREMENTAL over TIDE (2019+): iki yön, strict BH-FDR {SPX,NDX}")
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    tidx = tdir.index
    st = fz.reindex(tidx, method="ffill")
    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    print(f"  base SPX {_sh(bases['SPX']):+.3f}/NDX {_sh(bases['NDX']):+.3f}")
    print(f"  {'kural':<24}{'SPX ΔSh':>9}{'SPX P':>7}{'NDX ΔSh':>9}{'NDX P':>7}{'FDR':>6}")
    for sgn, ys in ((+1, "yüksek-giriş→trim"), (-1, "düşük-giriş→trim")):
        fac = (1 - 0.5*np.clip(sgn*st - 1, 0, 3)).clip(0.4, 1)
        res = {}
        for a in ("SPX", "NDX"):
            v = strat_ret((tdir * fac).reindex(tidx), prices[a])
            res[a] = (paired_win_prob(bases[a], v), _sh(v) - _sh(bases[a]))
        passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
        both = "PASS" if all(passed.get(a, False) for a in ("SPX", "NDX")) else "—"
        print(f"  {ys:<24}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}{both:>6}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
