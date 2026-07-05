"""
backtest/a3_a4 — A3 (2-paralel yüksek-pos) + A4 (Stooq low çapraz-kontrol). Ölçüm, yeni DoF yok.
A3: SPX+NDX @ {1.0,1.2} low-bazlı joint (funded=≥1 geçer) → P[≥1], P≤6/12/18ay, E[net-fee], E[≥1-funded takvim].
A4: kill-kritik en kötü 50 excursion gününde yfinance-low vs Stooq-low (bedava, anahtarsız) sapması; >%0.2 sapan
    günleri listele + o günde kill-statüsü (pos 1.2 intraday −%5 dokunuşu) değişiyor mu.
  & <venv python> backtest/a3_a4.py
"""
from __future__ import annotations

import io
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

from backtest import prop_sim_v13 as V

EVAL_FEE = 155.0; MD = 21
STOOQ = {"NDX": "^ndx", "SPX": "^spx"}


def a3():
    print("=" * 96)
    print("  A3 — 2-PARALEL yüksek-pos (SPX+NDX low-bazlı, funded=≥1 geçer; restart/iade-dahil)")
    print("=" * 96)
    dS, dN = V._book("SPX"), V._book("NDX")
    print(f"  {'pos':>5}{'P[≥1]':>8}{'P≤6ay':>8}{'P≤12ay':>8}{'P≤18ay':>8}{'E[net-fee]€':>13}{'E[≥1-funded ay]':>17}"
          f"{'tek-NDX-pass':>14}")
    for ps in (1.0, 1.2):
        oS, _ = V.run_eval(dS, ps, low_based=True, low_col="low_exc")
        oN, _ = V.run_eval(dN, ps, low_based=True, low_col="low_exc")
        sS = pd.Series(oS, index=dS.index); sN = pd.Series(oN, index=dN.index)
        common = dS.index.intersection(dN.index)
        sS, sN = sS.reindex(common), sN.reindex(common)
        bv = (~sS.isna()) & (~sN.isna()); nv = int(bv.sum())
        joint = np.minimum(np.where(sS.isna(), np.inf, sS.values), np.where(sN.isna(), np.inf, sN.values))
        b = bv.values
        pj = float((np.isfinite(joint) & b).sum()) / nv if nv else float("nan")
        pX = {x: float(((joint <= x * MD) & b).sum()) / nv if nv else float("nan") for x in (6, 12, 18)}
        med_funded = float(np.median(joint[np.isfinite(joint) & b])) / MD if pj else float("nan")
        ndx_pass = float(np.isfinite(sN.values[b]).sum()) / nv
        ea = 1.0 / pj if pj else float("inf")
        p_single = ndx_pass
        net = EVAL_FEE * (2 * (ea - 1) + (2 * p_single * (1 - p_single) / pj if pj else 0))
        print(f"  {ps:>5.1f}{100*pj:>7.0f}%{100*pX[6]:>7.0f}%{100*pX[12]:>7.0f}%{100*pX[18]:>7.0f}%{net:>13.0f}"
              f"{med_funded:>17.1f}{100*ndx_pass:>13.0f}%")
    print("  NOT: SPX/NDX korele (aynı tide) → joint-süre ≈ tek-track; 2-paralel hız değil GÜVENİLİRLİK (P[≥1]) verir.\n")


def _stooq_low(book):
    import requests
    url = f"https://stooq.com/q/d/l/?s={STOOQ[book]}&i=d"
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    df = pd.read_csv(io.StringIO(r.text))
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")[["Low", "Close"]].sort_index()


def a4():
    print("=" * 96)
    print("  A4 — yfinance-low vs STOOQ-low çapraz-kontrol (kill-kritik en kötü 50 excursion günü; RİSK-H daraltma)")
    print("=" * 96)
    book = "NDX"
    df = V._book(book)                                    # yfinance low_exc (prev_close→low)
    import yfinance as yf
    yo = yf.download(V.SYM[book], start="2018-12-01", end="2026-05-23", progress=False, auto_adjust=True)
    if isinstance(yo.columns, pd.MultiIndex):
        yo.columns = yo.columns.get_level_values(0)
    yo.index = pd.to_datetime(yo.index).tz_localize(None)
    yf_low = yo["Low"]; yf_prevc = yo["Close"].shift(1)
    yf_exc = (yf_low / yf_prevc - 1).dropna()
    try:
        st = _stooq_low(book)
        st_prevc = st["Close"].shift(1)
        st_exc = (st["Low"] / st_prevc - 1).dropna()
    except Exception as e:
        print(f"  [!] Stooq çekilemedi ({type(e).__name__}: {str(e)[:60]}) → çapraz-kontrol atlandı, FLAG.")
        return
    worst = yf_exc.nsmallest(50).index
    devs = []
    for d in worst:
        if d in st_exc.index:
            dev = abs(yf_exc[d] - st_exc[d]) * 100
            devs.append((d, yf_exc[d] * 100, st_exc[d] * 100, dev))
    devs.sort(key=lambda x: -x[3])
    big = [x for x in devs if x[3] > 0.2]
    print(f"  worst-50 gün, Stooq'ta eşleşen {len(devs)} | sapma>%0.2 olan {len(big)} gün:")
    for d, y, s, dv in big[:10]:
        # pos 1.2'de intraday kill-eşiği: exposure×exc ≤ −%5 → exc ≤ −%5/(1.2×book_exp). Kaba: tam-long 1.2 → exc≤−4.17%
        yk = "KILL" if y <= -100*0.05/1.2 else "—"; sk = "KILL" if s <= -100*0.05/1.2 else "—"
        flip = " ⚠ STATÜ DEĞİŞİR" if yk != sk else ""
        print(f"    {d.date()}: yf {y:+.2f}% ({yk}) vs stooq {s:+.2f}% ({sk}) | sapma {dv:.2f}%{flip}")
    if not big:
        print("    → %0.2'den büyük sapan gün YOK → yfinance-low güvenilir (RİSK-H daralır).")
    print(f"  medyan sapma {np.median([x[3] for x in devs]):.3f}% | max {max(x[3] for x in devs):.2f}%" if devs else "")


def main():
    a3(); a4()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
