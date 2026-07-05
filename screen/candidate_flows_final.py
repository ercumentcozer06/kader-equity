"""screen/candidate_flows_final — KESİN equity-flow testi, max-free pencere (ICI 2020-2026 stitched, 192hf).
+0.16/87% gerçek mi 2022-fluke mu? Bucket iki-yön + sub-period (2020-22 vs 23-26) + incremental strict FDR + bootstrap."""
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


def z(s, win=52): return (s - s.rolling(win, min_periods=20).mean()) / s.rolling(win, min_periods=20).std()
def _sh(r): r = r.dropna(); return float(r.mean()/r.std()*np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")


def strat_ret(pos, close, lag=1):
    idx = pos.index; ret = E.fwd_ret(close, idx).values; p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def main():
    flow = pd.read_parquet(ROOT / "data" / "cache" / "ici_full_weekly.parquet")["equity_flow"].dropna()
    flow.index = flow.index + pd.Timedelta(days=6)
    fz = z(flow, 52)
    spx = load_price_csv(DESK / "SPX_daily.csv")
    print(f"  ICI weekly equity flow: {len(flow)} hafta {flow.index.min().date()}..{flow.index.max().date()}")

    print("\n  1) BUCKET — z(flow) → SPX fwd-Hg; +giriş kontrarian mı momentum mu? + sub-period")
    for h in (5, 21):
        idx = fz.dropna().index; cb = spx.reindex(idx, method="ffill"); fh = (cb.shift(-h)/cb - 1)
        common = fz.reindex(idx).dropna().index.intersection(spx.index)
        sv, fv = fz.reindex(common), fh.reindex(common)
        for lbl, a, b in (("full", "2020", "2026"), ("2020-22", "2020", "2022"), ("2023-26", "2023", "2026")):
            mm = (common >= a) & (common <= b)
            if mm.sum() < 80:
                continue
            try:
                q = pd.qcut(sv[mm], 4, labels=False, duplicates="drop")
                bk = [100*fv[mm][q == i].mean() for i in range(4)]
                ic = sv[mm].corr(fv[mm], method="spearman")
                print(f"    h{h:<3}{lbl:<9} Q1düşük..Q4yüksek: " + " ".join(f"{v:+.2f}" for v in bk) + f"   IC {ic:+.2f}")
            except ValueError:
                pass

    print("\n  2) INCREMENTAL over TIDE (2019+): iki yön + bootstrap-CI + strict FDR")
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    tidx = tdir.index; st = fz.reindex(tidx, method="ffill")
    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    print(f"  base SPX {_sh(bases['SPX']):+.3f}/NDX {_sh(bases['NDX']):+.3f}")
    print(f"  {'kural':<22}{'SPX ΔSh':>9}{'SPX P':>7}{'SPX boot[p5,p95]':>18}{'NDX ΔSh':>9}{'NDX P':>7}{'FDR':>6}")
    for sgn, ys in ((+1, "yüksek-giriş→trim"), (-1, "düşük-giriş→trim")):
        fac = (1 - 0.5*np.clip(sgn*st - 1, 0, 3)).clip(0.4, 1)
        res = {}; ci = None
        for a in ("SPX", "NDX"):
            v = strat_ret((tdir * fac).reindex(tidx), prices[a])
            res[a] = (paired_win_prob(bases[a], v), _sh(v) - _sh(bases[a]))
            if a == "SPX":
                ci = bootstrap_ci(v)
        passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
        both = "PASS" if all(passed.get(a, False) for a in ("SPX", "NDX")) else "—"
        ci_str = f"[{ci['p5']},{ci['p95']}]"
        print(f"  {ys:<22}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{ci_str:>18}"
              f"{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}{both:>6}")
    print("=" * 96)
    print("  OKU: kontrarian gerçekse → +giriş bucket'ta DÜŞÜK getiri + İKİ sub-period'da + incremental FDR-PASS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
