"""
screen/candidate_seasonality2 — DERİN sezonalite (Emir: OpEx vs edge taşımaması imkansız). Önceki test
sığ+yanlış-çerçeveydi (pencere-gününden forward-5g ölçtüm). Şimdi DOĞRU: her takvim-pozisyonunun KENDİ
realize next-day getirisi + OpEx döngüsü günlük-granül (3.Cuma ±7 gün) + stabilite + incremental strict FDR.
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


def opex_offset(idx: pd.DatetimeIndex) -> np.ndarray:
    """Her gün için en yakın 3.Cuma'ya işaretli işgünü-offset (− önce / + sonra)."""
    is_op = (idx.weekday == 4) & (idx.day >= 15) & (idx.day <= 21)
    pos = np.where(is_op)[0]
    off = np.full(len(idx), 99)
    for i in range(len(idx)):
        if len(pos):
            j = pos[np.argmin(np.abs(pos - i))]
            off[i] = i - j
    return off


def strat_ret(pos, close, lag=1):
    idx = pos.index; ret = E.fwd_ret(close, idx).values; p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def main():
    spx = load_price_csv(DESK / "SPX_daily.csv")
    idx = spx.index
    nd = spx.pct_change().shift(-1)                        # next-day return (pozisyonun kazandığı)
    g = pd.Series(idx, index=idx).groupby([idx.year, idx.month])
    tdom = (g.cumcount() + 1).values
    off = opex_offset(idx)
    dow = idx.weekday

    print("=" * 92)
    print("  1) OpEx DÖNGÜSÜ — 3.Cuma'ya göre işgünü-offset → ort next-day getiri (bps) + eski/yeni stabilite")
    print("=" * 92)
    print(f"  {'offset':>7}{'ort-bps':>9}{'eski-bps':>9}{'yeni-bps':>9}{'n':>7}   (− önce / 0=3Cuma / + sonra)")
    old_m = idx < pd.Timestamp("2013-01-01")
    for o in range(-5, 6):
        m = off == o
        a = 1e4*nd[m].mean(); oe = 1e4*nd[m & old_m].mean(); ne = 1e4*nd[m & ~old_m].mean()
        print(f"  {o:>+7}{a:>+9.1f}{oe:>+9.1f}{ne:>+9.1f}{int(m.sum()):>7}")

    print("\n" + "=" * 92)
    print("  2) TRADING-DAY-OF-MONTH — ort next-day getiri (bps); TOM (1-3 + son) güçlü mü?")
    print("=" * 92)
    for lo, hi, lbl in ((1, 3, "ay-başı 1-3"), (4, 7, "4-7"), (8, 12, "orta 8-12"),
                        (13, 17, "13-17"), (18, 99, "ay-sonu 18+")):
        m = (tdom >= lo) & (tdom <= hi)
        print(f"  {lbl:<14}{1e4*nd[m].mean():>+7.1f} bps   eski {1e4*nd[m & old_m].mean():>+6.1f}  yeni {1e4*nd[m & ~old_m].mean():>+6.1f}  n={int(m.sum())}")
    print(f"  {'gün-of-week':<14}" + "  ".join(f"{['Pzt','Sal','Çar','Per','Cum'][d]}:{1e4*nd[dow == d].mean():+.0f}" for d in range(5)))

    # ── 3) INCREMENTAL over TIDE: revealed-zayıf pencerede trim ──
    print("\n" + "=" * 92)
    print("  3) INCREMENTAL over TIDE (2019+): revealed-zayıf takvim pencerelerinde trim. STRICT BH-FDR")
    print("=" * 92)
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    tidx = tdir.index
    toff = opex_offset(tidx)
    tg = pd.Series(tidx, index=tidx).groupby([tidx.year, tidx.month]).cumcount().values + 1
    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    print(f"  base SPX {_sh(bases['SPX']):+.3f}/NDX {_sh(bases['NDX']):+.3f}")
    print(f"  {'kural':<30}{'SPX ΔSh':>9}{'SPX P':>7}{'NDX ΔSh':>9}{'NDX P':>7}{'FDR':>6}")
    rules = {
        "post-OpEx +1..+4 → flat": np.isin(toff, [1, 2, 3, 4]),
        "post-OpEx +1..+5 → trim.5": np.isin(toff, [1, 2, 3, 4, 5]),
        "ay-ortası 13-17 → trim.5": (tg >= 13) & (tg <= 17),
        "OpEx-haftası 0..+4 → flat": np.isin(toff, [0, 1, 2, 3, 4]),
    }
    for label, mask in rules.items():
        lvl = 0.0 if "flat" in label else 0.5
        fac = pd.Series(np.where(mask, lvl, 1.0), index=tidx)
        res = {}
        for a in ("SPX", "NDX"):
            v = strat_ret((tdir * fac).reindex(tidx), prices[a])
            res[a] = (paired_win_prob(bases[a], v), _sh(v) - _sh(bases[a]))
        passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
        both = "PASS" if all(passed.get(a, False) for a in ("SPX", "NDX")) else "—"
        print(f"  {label:<30}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}{both:>6}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
