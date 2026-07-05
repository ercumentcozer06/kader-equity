"""
screen/candidate_flows2 — GERÇEK flow test: FINRA margin debt (1997+ aylık) + ICI equity fund flows.
Tam çerçeve: yön-agnostik bucket + iki yön + uç-kuyruk + PIT-lag + strict BH-FDR + stabilite.

Mekanizma (yön-agnostik test edilir, varsaymıyorum):
  • margin-debt YoY büyüme: yüksek=kaldıraç/froth (tarihte 2000/2007/2021 tepeleri) → kontrarian?
  • ICI equity flow: aşırı giriş=euforya kontrarian? momentum?  → bucket karar verir
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


def z(s, win): return (s - s.rolling(win, min_periods=max(8, win//3)).mean()) / s.rolling(win, min_periods=max(8, win//3)).std()
def _sh(r): r = r.dropna(); return float(r.mean()/r.std()*np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")


def strat_ret(pos, close, lag=1):
    idx = pos.index; ret = E.fwd_ret(close, idx).values; p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def main():
    fl = pd.read_parquet(ROOT / "data" / "cache" / "flows2.parquet")
    md = fl["margin_debt"].dropna(); md = md[md > 0]
    md.index = md.index + pd.Timedelta(days=45)            # PIT: ay-sonu + ~3hafta publish
    md_g = z(md.pct_change(12), 36)                        # YoY büyüme z (3y pencere)
    ici = fl["ici_weekly_equity"].dropna()
    ici.index = ici.index + pd.Timedelta(days=6)           # PIT: hafta + publish
    ici_z = z(ici, 52)
    icim = fl["ici_monthly_equity"].dropna()
    icim.index = icim.index + pd.Timedelta(days=25)         # PIT: ay + publish
    icim_z = z(icim, 24)

    spx = load_price_csv(DESK / "SPX_daily.csv")

    # ── 1) BUCKET (yön-agnostik) ──
    print("=" * 92)
    print("  1) BUCKET — margin-debt-YoY-z & ICI-flow-z → SPX fwd getiri (yön-agnostik, full window)")
    print("=" * 92)
    for name, sig, h in (("margin_YoY", md_g, 21), ("ICI_flow", ici_z, 5)):
        s = sig.dropna(); idx = s.index.union(spx.index).sort_values()
        sv = s.reindex(idx, method="ffill"); cb = spx.reindex(idx, method="ffill")
        fh = (cb.shift(-h)/cb - 1)
        common = sv.dropna().index.intersection(spx.index)
        sv2, fh2 = sv.reindex(common), fh.reindex(common)
        try:
            q = pd.qcut(sv2, 5, labels=False, duplicates="drop")
            b = [100*fh2[q == i].mean() for i in range(5)]
            p10, p90 = sv2.quantile([.1, .9])
            print(f"  {name:<12}(h{h}) Q1düşük..Q5yüksek: " + " ".join(f"{v:+.1f}" for v in b) +
                  f"   |  bot10%:{100*fh2[sv2 <= p10].mean():+.1f}  top10%:{100*fh2[sv2 >= p90].mean():+.1f}  n={len(sv2)}")
        except ValueError:
            print(f"  {name}: yetersiz")

    # ── 2) INCREMENTAL over TIDE — iki yön, strict FDR ──
    print("\n" + "=" * 92)
    print("  2) INCREMENTAL over TIDE: iki yönde trim. STRICT BH-FDR {SPX,NDX}")
    print("=" * 92)
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    tidx = tdir.index
    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    print(f"  base SPX {_sh(bases['SPX']):+.3f}/NDX {_sh(bases['NDX']):+.3f}")
    print(f"  {'kural':<30}{'SPX ΔSh':>9}{'SPX P':>7}{'NDX ΔSh':>9}{'NDX P':>7}{'FDR':>6}")
    for nm, sig in (("margin-YoY", md_g), ("ICI-flow-W", ici_z), ("ICI-flow-M", icim_z)):
        st = sig.reindex(tidx, method="ffill")
        for sgn, ys in ((+1, "yüksek→trim"), (-1, "düşük→trim")):
            fac = (1 - 0.5*np.clip(sgn*st - 1, 0, 3)).clip(0.4, 1)
            res = {}
            for a in ("SPX", "NDX"):
                v = strat_ret((tdir * fac).reindex(tidx), prices[a])
                res[a] = (paired_win_prob(bases[a], v), _sh(v) - _sh(bases[a]))
            passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
            both = "PASS" if all(passed.get(a, False) for a in ("SPX", "NDX")) else "—"
            print(f"  {nm+' '+ys:<30}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}{both:>6}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
