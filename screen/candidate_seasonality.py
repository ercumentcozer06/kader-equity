"""
screen/candidate_seasonality — TAKVİM/SEZONALİTE. MAX self-skepticism (Emir): takvim = en yüksek overfit
riski → SADECE mekanizma-destekli pencereler + TÜM-TARİH stabilite (eski-vs-yeni dönem; gerçek etki kalıcı).

Pencereler (mekanizma):
  • turn-of-month (TOM): ay sonu+başı = pension/401k inflow = güçlü
  • post-OpEx week: 3.Cuma sonrası = gamma-unwind = zayıf
  • buyback blackout: çeyrek-sonu + earnings-sezonu = kurumsal bid çekilir = zayıf
  • month-of-year: Eylül-zayıf vb (bonus, overfit-şüpheli)

1) BUCKET: pencere-İÇİ vs DIŞI fwd-5g getiri, full-hist + eski/yeni stabilite. 2) INCREMENTAL: zayıf-pencerede
trim, strict BH-FDR {SPX,NDX}.
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


def _sh(r): r = r.dropna(); return float(r.mean()/r.std()*np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")


def calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=idx)
    g = df.groupby([idx.year, idx.month])
    df["dom_rank"] = g.cumcount() + 1
    df["dom_size"] = g.cumcount(ascending=False) + df["dom_rank"]    # size per month-group
    df["tom"] = (df["dom_rank"] <= 3) | (df["dom_rank"] >= df["dom_size"])           # ilk3 + son1
    # 3.Cuma + post-opex 5 işgünü
    is_opex = (idx.weekday == 4) & (idx.day >= 15) & (idx.day <= 21)
    opex_dates = idx[is_opex]
    post = pd.Series(False, index=idx)
    for od in opex_dates:
        post.loc[(idx > od) & (idx <= od + pd.Timedelta(days=9))] = True
    df["post_opex"] = post.values
    # buyback blackout proxy: çeyrek-sonu ayının 2.yarısı + earnings-ayının ilk ~22 günü
    m, d = idx.month, idx.day
    df["blackout"] = ((np.isin(m, [3, 6, 9, 12])) & (d >= 15)) | ((np.isin(m, [1, 4, 7, 10])) & (d <= 22))
    df["month"] = m
    return df


def strat_ret(pos, close, lag=1):
    idx = pos.index; ret = E.fwd_ret(close, idx).values; p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def main():
    # ── 1) BUCKET + stabilite ──
    print("=" * 96)
    print("  1) PENCERE-İÇİ fwd-5g getiri (mean) — full-hist + eski(<2013)/yeni(>=2013) stabilite")
    print("=" * 96)
    print(f"  {'asset':<7}{'pencere':<14}{'içi-full':>10}{'dışı-full':>10}{'içi-eski':>10}{'içi-yeni':>10}{'n-içi':>7}")
    for a, fn in PRICES.items():
        close = load_price_csv(DESK / fn)
        idx = close.index
        cal = calendar(idx)
        f5 = (close.shift(-5)/close - 1)
        for win in ("tom", "post_opex", "blackout"):
            m = cal[win].values
            inr, outr = f5[m].dropna(), f5[~m].dropna()
            old = f5[m & (idx < "2013-01-01")].dropna(); new = f5[m & (idx >= "2013-01-01")].dropna()
            print(f"  {a:<7}{win:<14}{100*inr.mean():>+9.2f}%{100*outr.mean():>+9.2f}%"
                  f"{100*old.mean():>+9.2f}%{100*new.mean():>+9.2f}%{len(inr):>7}")
        # ay-bazlı (bonus): en zayıf 3 ay
        mo = f5.groupby(idx.month).mean() * 100
        weak = mo.nsmallest(3)
        print(f"  {a:<7}{'en-zayıf-ay':<14}{'  '.join(f'{int(k)}:{v:+.2f}%' for k, v in weak.items())}")

    # ── 2) INCREMENTAL over TIDE (2019+): zayıf-pencerede trim ──
    print("\n" + "=" * 96)
    print("  2) INCREMENTAL over TIDE (2019+): zayıf-pencerede tide-long trim. STRICT BH-FDR {SPX,NDX}")
    print("=" * 96)
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    cal = calendar(idx)
    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    print(f"  base SPX {_sh(bases['SPX']):+.3f}/NDX {_sh(bases['NDX']):+.3f}")
    print(f"  {'kural':<28}{'SPX ΔSh':>9}{'SPX P':>7}{'NDX ΔSh':>9}{'NDX P':>7}{'FDR':>6}")
    rules = {
        "post-OpEx → trim.5": (cal["post_opex"].values, 0.5),
        "blackout → trim.5": (cal["blackout"].values, 0.5),
        "post-OpEx → flat": (cal["post_opex"].values, 0.0),
        "blackout+postOpEx → trim.5": ((cal["post_opex"] | cal["blackout"]).values, 0.5),
    }
    for label, (mask, lvl) in rules.items():
        fac = pd.Series(np.where(mask, lvl, 1.0), index=idx)
        res = {}
        for a in ("SPX", "NDX"):
            v = strat_ret((tdir * fac).reindex(idx), prices[a])
            res[a] = (paired_win_prob(bases[a], v), _sh(v) - _sh(bases[a]))
        passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
        both = "PASS" if all(passed.get(a, False) for a in ("SPX", "NDX")) else "—"
        print(f"  {label:<28}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}{both:>6}")
    print("=" * 96)
    print("  OKU: pencere-içi belirgin negatif + eski/yeni STABİL + incremental FDR-PASS = gerçek. Aksi = overfit/ölü.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
