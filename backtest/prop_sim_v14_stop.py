"""
backtest/prop_sim_v14_stop — A1 FELAKET-STOPU simülasyonu (ölçüm/ops, yeni DoF YOK; Katman 1-2 FROZEN).
v13 makinesi + GÜNLÜK ops-level fail-safe stop. Stop = o günkü pozisyonla hesap-günü-kaybının −%4.5'e denk
geldiği endeks fiyatı. DÜRÜST DOLUM (iyimserlik yasak):
  • Açılış stop'un ALTINDA (gap-through) → dolum AÇILIŞTA (tam gap zararı); ≤−%5 ise yine KILL (stop gap'i korumaz).
  • Açılış üstte & low ≤ stop → dolum stop−%0.1 slippage; günün kalanı FLAT; ertesi gün normal sinyal.
  • Stop'a değmeyen gün → değişiklik yok.
Stop parametreleri (−%4.5, %0.1 slippage) = A PRİORİ ops/fail-safe (frozen stratejinin parçası DEĞİL).
WHIPSAW: stop-tetiklenen günlerin kaçı gün-sonu close'da −%5'ten İYİ kapanırdı (toparlayacak günü kesme payı).

  & <venv python> backtest/prop_sim_v14_stop.py
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

from backtest import prop_sim as PS
from backtest import prop_sim_v13 as V

STOP_LOSS = 0.045        # a priori ops fail-safe: hesap-günü −%4.5'te stop (FTMO −%5 limitinin altında)
SLIP = 0.001             # stop dolumunda %0.1 slippage (a priori)
POS = [0.6, 1.0, 1.2, 1.5]
EVAL_FEE = 155.0


def _book_ohlc(book):
    """pos (lag1 exposure) + open/low/close getirisi (prev_close'a göre). +1g lag."""
    pos, _p, idx = PS._stack_pos()
    def one(asset):
        o = V._ohlc(asset, idx)
        c = o["Close"].values; lo = o["Low"].values; op = o["Open"].values
        prevc = np.concatenate([[np.nan], c[:-1]])
        return op/prevc - 1, lo/prevc - 1, c/prevc - 1
    p = pos.astype(float).reindex(idx).values
    plag = np.concatenate([[0.0], p[:-1]])
    if book in ("SPX", "NDX"):
        oR, lR, cR = one(book)
        df = pd.DataFrame({"pos": plag, "open": oR, "low": lR, "close": cR}, index=idx).dropna()
    else:
        oS, lS, cS = one("SPX"); oN, lN, cN = one("NDX")
        df = pd.DataFrame({"pos": plag, "open": 0.5*np.nan_to_num(oS)+0.5*np.nan_to_num(oN),
                           "low": np.minimum(lS, lN),               # muhafazakâr: iki low eşzamanlı
                           "close": 0.5*np.nan_to_num(cS)+0.5*np.nan_to_num(cN)}, index=idx).dropna()
    return df


def _sim(opens, openr, lowr, closer, swap, start, target, use_stop):
    """Tek faz. use_stop=True → felaket-stop dolum modeli. (status, end_idx, reason, whipsaw_list)."""
    eq = 1.0; td = 0; i = start; n = len(opens); whip = []
    while i < n:
        prev = eq; op = opens[i]
        if op > 0:
            td += 1
        if op <= 0:
            eq = prev * (1.0 - swap * op)                          # flat (op=0 → 0)
        elif not use_stop:
            # v13 low-bazlı: intraday low −%5 kill
            eql = prev * (1.0 + op * lowr[i] - swap * op)
            if (prev - eql) >= PS.DAILY_LIM:
                return ("kill", i, "daily", whip)
            if eql <= (1.0 - PS.TOTAL_LIM):
                return ("kill", i, "total", whip)
            eq = prev * (1.0 + op * closer[i] - swap * op)
        else:
            stop_ret = -STOP_LOSS / op                            # bu endeks-getirisi −%4.5 hesap-kaybı tetikler
            o, l, c = openr[i], lowr[i], closer[i]
            if o <= stop_ret:                                     # GAP-THROUGH → açılışta dolum (tam gap)
                day = op * o
                eq = prev * (1.0 + day - swap * op)
                if (prev - eq) >= PS.DAILY_LIM:
                    return ("kill", i, "daily-gap", whip)
                if eq <= (1.0 - PS.TOTAL_LIM):
                    return ("kill", i, "total", whip)
            elif l <= stop_ret:                                  # INTRADAY TOUCH → stop−slippage, kalan flat
                eq = prev * (1.0 - STOP_LOSS - op * SLIP - swap * op)
                if eq <= (1.0 - PS.TOTAL_LIM):
                    return ("kill", i, "total", whip)
                whip.append(c > stop_ret)                         # close stop'tan iyi miydi? (whipsaw)
            else:                                                # değmedi → normal close
                eql = prev * (1.0 + op * l - swap * op)
                if (prev - eql) >= PS.DAILY_LIM:                 # stop yokken yine -%5 olamaz ama güvence
                    return ("kill", i, "daily", whip)
                if eql <= (1.0 - PS.TOTAL_LIM):
                    return ("kill", i, "total", whip)
                eq = prev * (1.0 + op * c - swap * op)
        if eq >= (1.0 + target) and td >= PS.MIN_DAYS:
            return ("pass", i, None, whip)
        i += 1
    return ("incomplete", i, None, whip)


def run(book, pos_scale, use_stop):
    df = _book_ohlc(book)
    sw = V.SWAP_YR / 252.0
    opens = pos_scale * df["pos"].values
    openr, lowr, closer = df["open"].values, df["low"].values, df["close"].values
    n = len(opens); passd, faild, kills, whip_all = [], [], 0, []
    for s in range(n):
        st1, e1, w1, wh1 = _sim(opens, openr, lowr, closer, sw, s, PS.P1_TARGET, use_stop)
        whip_all += wh1
        if st1 == "incomplete":
            continue
        if st1 == "kill":
            faild.append(e1 - s); kills += 1; continue
        st2, e2, w2, wh2 = _sim(opens, openr, lowr, closer, sw, e1+1, PS.P2_TARGET, use_stop)
        whip_all += wh2
        if st2 == "incomplete":
            continue
        if st2 == "kill":
            faild.append(e2 - s); kills += 1; continue
        passd.append(e2 - s)
    nv = len(passd) + len(faild)
    p = len(passd)/nv if nv else float("nan")
    return {"pass": p, "wilson": V.wilson_lo(len(passd), nv), "kill": kills,
            "medpass": float(np.median(passd)) if passd else None,
            "medfail": float(np.median(faild)) if faild else None,
            "whip_n": len(whip_all), "whip_pct": (100*np.mean(whip_all) if whip_all else 0.0),
            "e_attempts": (1/p if p else float("inf")), "e_net_fee": (EVAL_FEE*(1/p-1) if p else float("inf"))}


def funded_survival_stop(book, use_stop, horizon=252):
    """Funded@0.6 12-ay kill-yok (stoplu/stopsuz)."""
    df = _book_ohlc(book); sw = V.SWAP_YR/252.0
    opens = 0.6*df["pos"].values; openr, lowr, closer = df["open"].values, df["low"].values, df["close"].values
    n = len(opens); ok = tot = 0
    for s in range(n):
        if s+horizon > n:
            continue
        tot += 1; eq = 1.0; alive = True
        for j in range(s, s+horizon):
            op = opens[j]; prev = eq
            if op <= 0:
                eq = prev*(1-sw*op); continue
            if use_stop:
                sr = -STOP_LOSS/op; o, l, c = openr[j], lowr[j], closer[j]
                if o <= sr:
                    eq = prev*(1+op*o-sw*op)
                    if eq <= 1-PS.TOTAL_LIM: alive = False; break
                elif l <= sr:
                    eq = prev*(1-STOP_LOSS-op*SLIP-sw*op)
                    if eq <= 1-PS.TOTAL_LIM: alive = False; break
                else:
                    eq = prev*(1+op*c-sw*op)
            else:
                eql = prev*(1+op*lowr[j]-sw*op)
                if (prev-eql) >= PS.DAILY_LIM or eql <= 1-PS.TOTAL_LIM: alive = False; break
                eq = prev*(1+op*closer[j]-sw*op)
        ok += int(alive)
    return ok/tot if tot else float("nan")


def main():
    print("=" * 108)
    print("  A1 — FELAKET-STOP SİMÜLASYONU (ops fail-safe, −%4.5 stop + %0.1 slip; gap-through korunmaz). v1.3-final low-bazlı")
    print("=" * 108)
    print(f"  {'kitap':<7}{'pos':>5}{'STOP':>6}{'pass-LOW':>10}{'Wilson':>8}{'kill':>7}{'medPASS':>9}{'medFAIL':>9}"
          f"{'E[ay]':>7}{'E[fee]€':>9}")
    res = {}
    for book in ("NDX", "50/50", "SPX"):
        for ps in POS:
            for st in (False, True):
                r = run(book, ps, st); res[(book, ps, st)] = r
                print(f"  {book:<7}{ps:>5.1f}{('EVET' if st else 'HAYIR'):>6}{100*r['pass']:>9.0f}%{100*r['wilson']:>7.0f}%"
                      f"{r['kill']:>7}{(r['medpass'] or 0):>9.0f}{(r['medfail'] or 0):>9.0f}"
                      f"{(r['medpass'] or 0)/21:>7.1f}{r['e_net_fee']:>9.0f}")
        print()
    print("  WHIPSAW (stop-tetiklenen günlerin close'u −stop'tan İYİ olma payı = boşuna-kesme):")
    for book in ("NDX", "50/50"):
        for ps in (1.2, 1.5):
            r = res[(book, ps, True)]
            print(f"    {book}@{ps}: {r['whip_n']} stop-tetik, %{r['whip_pct']:.0f} aslında toparlardı")
    print("\n  FUNDED@0.6 12-ay kill-yok (stoplu vs stopsuz):")
    for book in ("NDX", "50/50", "SPX"):
        print(f"    {book}: stopsuz %{100*funded_survival_stop(book, False):.0f} | stoplu %{100*funded_survival_stop(book, True):.0f}")
    print("=" * 108)
    return res


if __name__ == "__main__":
    main()
