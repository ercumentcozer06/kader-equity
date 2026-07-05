"""
backtest/prop_sim_v13_final — EVAL FRONTIER (saf ölçüm-AGREGASYON; YENİ DoF YOK). v1.3 makinesini (low-bazlı
_sim + Wilson) kullanır; restart/iade ekonomisini ve funded-hayatta-kalımı ekler. Model/eşik DEĞİŞMEZ.

 (1) Tam grid low-bazlı: pos × kitap → pass-LOW (Wilson-CI), medyan PASS süresi, medyan FAIL süresi (kill'e
     kadar gün — erken mi ölüyor geç mi).
 (2) Restart-dahil beklenen takvim (ASIL KARAR KOLONU): E[deneme]=1/p; E[takvim]=(E[deneme]−1)·medFail+medPass;
     E[net-fee]=€155·(E[deneme]−1)  (iade: geçen €0, kill €155).
 (3) 2-paralel low-bazlı (SPX+NDX, H6 tarih-join): P[≥1 funded ≤6/12/18ay], E[net-fee] restart-dahil.
 (4) Funded-faz hayatta-kalım @0.6 low-bazlı: P[12ay kill-yok], P[ilk-payout(+%5)'e kill'siz ulaşma].
 (5) Tek karar tablosu + önerilen politika (sayıyla).

KISIT: in-sample tek-rejim + yfinance-OHLC (RİSK-I/H). +1g lag. Katman 1-2 FROZEN.
  & <venv python> backtest/prop_sim_v13_final.py
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

from backtest import prop_sim as PS                   # FTMO sabitleri
from backtest import prop_sim_v13 as V                # _book/_book5050/_sim/wilson_lo/SWAP_YR

EVAL_FEE = 155.0          # FTMO 10k 2-Step (iade-edilir; geçen→net 0)
FUNDED_POS = 0.6
PAYOUT = 0.05             # ilk-payout eşiği (funded fazda +%5)
MD = 21                   # ay ≈ 21 işlem günü
POS = [0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]
BOOKS = ["SPX", "NDX", "50/50"]


def run_eval_full(df, pos_scale, low_col="low_exc"):
    """pass-süreleri + FAIL-süreleri (kill'e kadar gün) + nv. low-bazlı."""
    rets = df["close_ret"].values
    lows = df[low_col].values if low_col in df else rets
    opens = pos_scale * df["pos"].values
    sw = V.SWAP_YR / 252.0
    n = len(rets); passd, faild = [], []
    for s in range(n):
        st1, e1, _ = V._sim(rets, lows, opens, sw, s, PS.P1_TARGET, True)
        if st1 == "incomplete":
            continue
        if st1 == "kill":
            faild.append(e1 - s); continue
        st2, e2, _ = V._sim(rets, lows, opens, sw, e1 + 1, PS.P2_TARGET, True)
        if st2 == "incomplete":
            continue
        if st2 == "kill":
            faild.append(e2 - s); continue
        passd.append(e2 - s)
    return passd, faild


def funded_survival(df, pos_scale=FUNDED_POS, low_col="low_exc", horizon=252):
    """Funded fazda (hedef yok, sadece −%5 günlük / −%10 toplam low-bazlı): P[12ay kill-yok],
    P[ilk-payout +%5'e kill'siz ulaşma]."""
    rets = df["close_ret"].values
    lows = df[low_col].values if low_col in df else rets
    opens = pos_scale * df["pos"].values
    sw = V.SWAP_YR / 252.0; n = len(rets)
    surv_n = surv_ok = 0
    reach_n = reach_ok = 0
    for s in range(n):
        # (a) 12-ay hayatta-kalım (tam horizon gereken başlangıçlar)
        if s + horizon <= n:
            surv_n += 1; eq = 1.0; ok = True
            for j in range(s, s + horizon):
                op = opens[j]; prev = eq
                eql = prev * (1 + op * lows[j] - sw * op)
                if (prev - eql) >= PS.DAILY_LIM or eql <= (1 - PS.TOTAL_LIM):
                    ok = False; break
                eq = prev * (1 + op * rets[j] - sw * op)
            surv_ok += int(ok)
        # (b) +%5'e kill'siz ulaşma (herhangi horizon, kill ya da payout'a kadar)
        eq = 1.0; res = None
        for j in range(s, n):
            op = opens[j]; prev = eq
            eql = prev * (1 + op * lows[j] - sw * op)
            if (prev - eql) >= PS.DAILY_LIM or eql <= (1 - PS.TOTAL_LIM):
                res = False; break
            eq = prev * (1 + op * rets[j] - sw * op)
            if eq >= 1 + PAYOUT:
                res = True; break
        if res is not None:
            reach_n += 1; reach_ok += int(res)
    return {"p_survive_12mo": surv_ok / surv_n if surv_n else float("nan"),
            "p_reach_payout_killfree": reach_ok / reach_n if reach_n else float("nan")}


def _restart(p, med_pass, med_fail):
    """Tek-track restart: E[deneme]=1/p, E[takvim-gün], E[net-fee]=155·(E[deneme]−1)."""
    if not p or p <= 0:
        return {"e_attempts": float("inf"), "e_cal_mo": float("inf"), "e_net_fee": float("inf")}
    ea = 1.0 / p
    cal = (ea - 1) * (med_fail or 0) + (med_pass or 0)
    return {"e_attempts": ea, "e_cal_mo": cal / MD, "e_net_fee": EVAL_FEE * (ea - 1)}


def main():
    print("=" * 110)
    print("  prop_sim v1.3-FINAL — EVAL FRONTIER (low-bazlı, restart/iade-dahil; yeni DoF yok). Katman1-2 FROZEN.")
    print("=" * 110)
    dS, dN = V._book("SPX"), V._book("NDX")
    books = {"SPX": (dS, "low_exc"), "NDX": (dN, "low_exc"), "50/50": (V._book5050(dS, dN), "low_exc_up")}

    # (1)+(2) tam grid + restart takvim
    print(f"\n  [1+2] FRONTIER — pos × kitap → pass-LOW(Wilson) | medyan PASS/FAIL gün | RESTART: E[takvim] E[net-fee]")
    print(f"      {'kitap':<7}{'pos':>5}{'passLOW':>8}{'Wilson':>8}{'medPASS':>8}{'medFAIL':>8}"
          f"{'E[deneme]':>10}{'E[takvim-ay]':>13}{'E[net-fee]€':>12}")
    grid = {}
    for bk, (df, lc) in books.items():
        for ps in POS:
            pd_, fd_ = run_eval_full(df, ps, lc)
            nv = len(pd_) + len(fd_)
            p = len(pd_) / nv if nv else float("nan")
            mp = float(np.median(pd_)) if pd_ else None
            mf = float(np.median(fd_)) if fd_ else None
            wl = V.wilson_lo(len(pd_), nv)
            rs = _restart(p, mp, mf)
            grid[(bk, ps)] = {"p": p, "wl": wl, "medpass": mp, "medfail": mf, **rs}
            print(f"      {bk:<7}{ps:>5.1f}{100*p:>7.0f}%{100*wl:>7.0f}%{(mp or 0):>8.0f}{(mf or 0):>8.0f}"
                  f"{rs['e_attempts']:>10.2f}{rs['e_cal_mo']:>13.1f}{rs['e_net_fee']:>12.0f}")
        print()

    # (3) 2-paralel low-bazlı (SPX+NDX joint, tarih-join)
    print(f"  [3] 2-PARALEL low-bazlı (SPX+NDX, funded=≥1 geçer) — restart/iade dahil")
    print(f"      {'pos':>5}{'P[≥1]':>8}{'P≤6ay':>8}{'P≤12ay':>8}{'P≤18ay':>8}{'E[net-fee]€':>13}")
    for ps in [0.6, 0.8, 1.0]:
        psS, fsS = run_eval_full(dS, ps, "low_exc")
        psN, fsN = run_eval_full(dN, ps, "low_exc")
        # joint: tarih-index hizalı pass-gün dizileri (run_eval → (out, kills); index = df.index)
        oS, _kS = V.run_eval(dS, ps, low_based=True, low_col="low_exc")
        oN, _kN = V.run_eval(dN, ps, low_based=True, low_col="low_exc")
        sS = pd.Series(oS, index=dS.index); sN = pd.Series(oN, index=dN.index)
        common = dS.index.intersection(dN.index); sS, sN = sS.reindex(common), sN.reindex(common)
        bv = (~sS.isna()) & (~sN.isna()); nv = int(bv.sum())
        joint = np.minimum(np.where(sS.isna(), np.inf, sS.values), np.where(sN.isna(), np.inf, sN.values))
        pj = float((np.isfinite(joint) & bv.values).sum()) / nv if nv else float("nan")
        pX = {x: float(((joint <= x * MD) & bv.values).sum()) / nv if nv else float("nan") for x in (6, 12, 18)}
        # restart net-fee (2 hesap/dalga): 2·(E[dalga]−1) başarısız + başarı-dalgasında ~P(tek-geçer) kill-fee
        ea = 1.0 / pj if pj else float("inf")
        # başarı-dalgasında tam-1-geçer payı (kaba): P(exactly1|≥1) ≈ 2p(1-p)/pj  (p=tek-track pass)
        p_single = len(psS) / max(1, len(psS) + len(fsS))
        p_exactly1 = (2 * p_single * (1 - p_single)) / pj if pj else 0.0
        net = EVAL_FEE * (2 * (ea - 1) + p_exactly1)
        print(f"      {ps:>5.1f}{100*pj:>7.0f}%{100*pX[6]:>7.0f}%{100*pX[12]:>7.0f}%{100*pX[18]:>7.0f}%{net:>13.0f}")
    print()

    # (4) funded hayatta-kalım @0.6
    print(f"  [4] FUNDED HAYATTA-KALIM @{FUNDED_POS} (low-bazlı; funded'ın 0.6 kısıtı yeniden-doğrulama)")
    print(f"      {'kitap':<7}{'P[12ay kill-yok]':>18}{'P[+%5 payout kill-siz]':>24}")
    for bk, (df, lc) in books.items():
        fs = funded_survival(df, FUNDED_POS, lc)
        print(f"      {bk:<7}{100*fs['p_survive_12mo']:>17.0f}%{100*fs['p_reach_payout_killfree']:>23.0f}%")

    # (5) öneri — restart-takvim + funded-survival en iyi denge
    print("\n" + "=" * 110)
    best = min(((bk, ps, g) for (bk, ps), g in grid.items() if ps <= 0.7 and g["wl"] > 0.85),
               key=lambda t: t[2]["e_cal_mo"], default=None)
    if best:
        bk, ps, g = best
        print(f"  ÖNERİ (sayıyla): {bk} @ {ps} — pass-LOW %{100*g['p']:.0f} (Wilson %{100*g['wl']:.0f}), "
              f"E[takvim] {g['e_cal_mo']:.1f} ay, E[net-fee] €{g['e_net_fee']:.0f} (iade-dahil).")
    print("  ASIL KARAR KOLONU = E[takvim-ay] × E[net-fee]: düşük-pos güvenli+ucuz-fee ama yavaş; yüksek-pos hızlı")
    print("  görünür ama low-bazlı pass↓ → E[deneme]↑ → net-fee↑ (iade geçeni kurtarır, kill'i değil). Karar Emir'in.")
    return grid


if __name__ == "__main__":
    main()
