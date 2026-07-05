"""
screen/candidate_seasonality3 — son seasonality framing: calendar L/S sleeve (gerçek standalone edge)
+ tide'a BLEND (diversification). Overlay eklemiyordu; ama uncorrelated L/S sleeve blend'de katabilir.
Era-validate (overfit kontrol). Stable-strong: ay-sonu/Pazartesi/orta/OpEx−4; stable-weak: mid-month/OpEx-yakını.
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
from screen._util import load_price_csv                  # noqa: E402

DESK = Path(r"C:\Users\admin\Desktop\backtesting")


def _sh(r): r = r.dropna(); return float(r.mean()/r.std()*np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")
def _dd(r): eq = (1+r.dropna()).cumprod(); return float((eq/eq.cummax()-1).min())


def opex_offset(idx):
    is_op = (idx.weekday == 4) & (idx.day >= 15) & (idx.day <= 21)
    pos = np.where(is_op)[0]; off = np.full(len(idx), 99)
    for i in range(len(idx)):
        if len(pos):
            off[i] = i - pos[np.argmin(np.abs(pos - i))]
    return off


def cal_position(idx):
    """Calendar L/S: +1 stable-strong, −1 stable-weak, 0 else (pre-specified, era-stable pencerelerden)."""
    tg = (pd.Series(idx, index=idx).groupby([idx.year, idx.month]).cumcount() + 1).values
    off = opex_offset(idx); dow = idx.weekday
    strong = (tg >= 18) | (dow == 0) | ((tg >= 8) & (tg <= 12)) | (off == -4)
    weak = ((tg >= 13) & (tg <= 17)) | np.isin(off, [-3, -1, 0])
    return pd.Series(np.where(strong, 1.0, np.where(weak, -1.0, 0.0)), index=idx)


def main():
    spx = load_price_csv(DESK / "SPX_daily.csv")
    nd = spx.pct_change().shift(-1)
    cal = cal_position(spx.index)
    cret = (cal.shift(1) * nd).dropna()                   # +1g lag (sinyal[t]→t+1)

    print("  CALENDAR L/S STANDALONE (market-neutral seasonal premium):")
    for lbl, a, b in (("full 2000-26", "2000", "2026"), ("eski 2000-12", "2000", "2012"), ("yeni 2013-26", "2013", "2026")):
        w = cret[(cret.index >= a) & (cret.index <= b)]
        print(f"    {lbl:<14} Sharpe {_sh(w):+.2f}  maxDD {100*_dd(w):+.0f}%  expo {100*(cal != 0).mean():.0f}%")

    # tide ile blend
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    ret = E.fwd_ret(prices["SPX"], idx)
    tide_r = (tdir.shift(1).fillna(0) * ret).dropna()
    cal_r = (cal_position(idx).shift(1).fillna(0) * ret).reindex(tide_r.index).dropna()
    common = tide_r.index.intersection(cal_r.index)
    tr, cr = tide_r.reindex(common), cal_r.reindex(common)
    corr = float(tr.corr(cr))
    print(f"\n  tide vs calendar-L/S korelasyon (2019+): {corr:+.2f}")
    print(f"  {'blend':<22}{'Sharpe':>8}{'maxDD':>8}")
    print(f"  {'tide tek':<22}{_sh(tr):>+8.2f}{100*_dd(tr):>+7.0f}%")
    for w in (0.1, 0.2, 0.3):
        bl = (1-w)*tr + w*cr
        print(f"  {'tide+'+str(w)+'·cal':<22}{_sh(bl):>+8.2f}{100*_dd(bl):>+7.0f}%")
    print("\n  OKU: calendar-L/S standalone Sharpe pozitif+era-stabil + corr~0 + blend tide'ı geçer = gerçek katkı.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
