"""
backtest/prop_sim_v12 — Ş1-ek (v1.2). prop_sim primitiflerini kullanır (Katman 1-2 + prop_sim FROZEN).
Ekler: (1) eval-pos taramasına KALDIRAÇ {1.2, 1.5} (FTMO Swing 1:30 cap → sığar); (2) ASİMETRİK politika =
eval'i agresif (yüksek pos, sadece fee riski) geç, funded'ı 0.6'da tut (payout-koruma); (3) fee=€155 İADE-EDİLİR
(ilk payout → geçenler net €0, sadece kill'ler kayıp); (4) 2-PARALEL eval (SPX-only + NDX-only, GERÇEK joint —
aynı tarihsel patika korelasyonu) → P[≥1 funded ≤ ay X] + E[toplam fee]; (5) funded'a-beklenen-takvim tablosu.

YÖNTEM: prop_sim ile aynı rolling-start exhaustive; her başlangıç için outcome = geçiş-günü / kill(inf) /
incomplete(nan). P[≤X] = (geçen & gün≤X·21) / geçerli-başlangıç. 2-paralel joint = başlangıç-başına min(SPX,NDX).
Aynı kısıtlar (RİSK-A in-sample tek-rejim/örtüşen-pencere; RİSK-B close-to-close intraday-kill az-tahmin).

  & <venv python> backtest/prop_sim_v12.py
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

from backtest import prop_sim as PS                   # noqa: E402

EVAL_FEE_EUR = 155.0      # FTMO 10k 2-Step (web-doğrulandı 2026-06, jptradingcapital) — İADE-EDİLİR (ilk payout)
FUNDED_POS = 0.6          # asimetrik: funded'da güvenli 0.6 (payout-koruma) [a priori, GÖREV 2/4]
EVAL_POS = [0.6, 0.8, 1.0, 1.2, 1.5]
X_MONTHS = [6, 12, 18, 24]
MD = 21                   # ay≈21 işlem günü
BOOKS = ["SPX", "NDX", "50/50"]


def run_eval(book: str, eval_pos: float):
    """Her başlangıç için outcome (geçiş-günü / inf=kill / nan=incomplete) + index."""
    df = PS.book_returns(book)
    sw = PS.SWAP_YR / 252.0
    rets = eval_pos * df["ret1x"].values - sw * eval_pos * df["pos"].values
    opens = eval_pos * df["pos"].values
    n = len(rets); out = np.full(n, np.nan)
    for s in range(n):
        st1, e1, _ = PS._sim_phase(rets, opens, s, PS.P1_TARGET)
        if st1 == "incomplete":
            continue
        if st1 == "kill":
            out[s] = np.inf; continue
        st2, e2, _ = PS._sim_phase(rets, opens, e1+1, PS.P2_TARGET)
        if st2 == "incomplete":
            continue
        out[s] = np.inf if st2 == "kill" else (e2 - s)
    return out, df.index


def _stats(out):
    valid = ~np.isnan(out)
    nv = int(valid.sum())
    passed = np.isfinite(out) & valid                  # sonlu = geçti (inf=kill)
    npass = int(passed.sum())
    pp = npass / nv if nv else float("nan")
    med = float(np.median(out[passed])) if npass else None
    pX = {x: (float((out[passed] <= x*MD).sum()) / nv if nv else float("nan")) for x in X_MONTHS}
    return {"n_valid": nv, "pass": pp, "median_days": med, "pX": pX, "out": out, "passed": passed}


def net_fee(p):
    """İade-ayarlı net fee (tek-track, ilk-geçişe-dek retry): geçenler iade → net = fee×(1-p)/p (kill'ler)."""
    return EVAL_FEE_EUR * (1-p)/p if p and p < 1 else (0.0 if p >= 1 else float("inf"))


def main():
    print("=" * 104)
    print("  Ş1-ek v1.2 — FTMO Swing 2-Step: KALDIRAÇ + ASİMETRİK(eval-agresif/funded@0.6) + İADE-FEE + 2-PARALEL")
    print(f"  fee €{EVAL_FEE_EUR:.0f} İADE-EDİLİR (geçen→net €0); funded@{FUNDED_POS}; KISIT: in-sample/close-to-close (RİSK-A/B)")
    print("=" * 104)

    # 1) genişletilmiş tarama (kaldıraç dahil)
    print(f"\n  [1] GENİŞLETİLMİŞ TARAMA (kaldıraç {{1.2,1.5}} dahil)")
    print(f"      {'kitap':<7}{'pos':>5}{'pass%':>7}{'med-gün':>9}{'P≤12ay':>8}{'P≤24ay':>8}{'kill?':>7}")
    stats = {}                                          # NDX-only zaten BOOKS içinde (SPX/NDX/50-50)
    for book in BOOKS:
        for ps in EVAL_POS:
            out, idx = run_eval(book, ps)
            st = _stats(out); stats[(book, ps)] = (st, idx)
            killed = int((np.isinf(out)).sum())
            print(f"      {book:<7}{ps:>5.1f}{100*st['pass']:>6.0f}%{(st['median_days'] or 0):>9.0f}"
                  f"{100*st['pX'][12]:>7.0f}%{100*st['pX'][24]:>7.0f}%{('var' if killed else 'yok'):>7}")
        print()

    # 2) asimetrik takvim: eval-pos {1.0,1.2,1.5}, funded@0.6 payout + net-fee
    print(f"  [2] ASİMETRİK — eval agresif, funded@{FUNDED_POS} (payout-koruma). funded yıllık@0.6 → payout$(10k,%80)")
    print(f"      {'kitap':<7}{'eval-pos':>9}{'pass%':>7}{'med-ay':>8}{'P≤12ay':>8}{'E[net-fee]€':>12}{'funded$/yıl':>12}")
    for book in BOOKS:
        ann06 = PS._annual_book_return(book, FUNDED_POS)
        payout = PS.FUNDED_USD * ann06 * PS.PROFIT_SPLIT
        for ps in [1.0, 1.2, 1.5]:
            st = stats[(book, ps)][0]
            nf = net_fee(st["pass"])
            print(f"      {book:<7}{ps:>9.1f}{100*st['pass']:>6.0f}%{(st['median_days'] or 0)/MD:>8.1f}"
                  f"{100*st['pX'][12]:>7.0f}%{nf:>12.0f}{payout:>12,.0f}")
        print()

    # 3) 2-paralel eval (SPX-only + NDX-only, GERÇEK joint patika) — funded = en az biri geçer
    print(f"  [3] 2-PARALEL EVAL (SPX-only + NDX-only, aynı patika joint; funded = ≥1 geçer)")
    print(f"      {'eval-pos':>9}{'P[≥1 funded]':>14}{'P≤6ay':>8}{'P≤12ay':>8}{'P≤18ay':>8}{'P≤24ay':>8}{'gross-fee€':>11}{'net-fee€':>10}")
    for ps in [0.6, 1.0, 1.2, 1.5]:
        oS, iS = run_eval("SPX", ps); oN, iN = run_eval("NDX", ps)
        # H6 (KRİTİK): pozisyon-hizalama (son-M-satır) DEĞİL → TARİH-index inner-join. Eksik-gün politikası: inner
        # (yalnız iki kitapta da var olan başlangıç günleri eşleşir; misalign imkansız). Düşen-gün sayısı raporlanır.
        sS = pd.Series(oS, index=iS); sN = pd.Series(oN, index=iN)
        common = iS.intersection(iN)
        dropped = (len(iS) - len(common)) + (len(iN) - len(common))
        sS, sN = sS.reindex(common), sN.reindex(common)
        both_valid = (~sS.isna()) & (~sN.isna())
        nv = int(both_valid.sum())
        joint = np.minimum(np.where(sS.isna(), np.inf, sS.values), np.where(sN.isna(), np.inf, sN.values))
        bv = both_valid.values
        p1 = float((np.isfinite(joint) & bv).sum())/nv if nv else float("nan")
        pX = {x: float(((joint <= x*MD) & bv).sum())/nv if nv else float("nan") for x in X_MONTHS}
        # fee: 2 hesap × €155 = gross €310/dalga; geçenler iade. net ≈ (1-p1)/p1 dalga × 2fee − iadeler ~ kabaca:
        gross = 2*EVAL_FEE_EUR
        net = 2*EVAL_FEE_EUR*(1-p1)/p1 if p1 and p1 < 1 else 0.0   # retry-to-≥1, geçen-dalgada iade ~kalan kill fee
        print(f"      {ps:>9.1f}{100*p1:>13.0f}%{100*pX[6]:>7.0f}%{100*pX[12]:>7.0f}%{100*pX[18]:>7.0f}%"
              f"{100*pX[24]:>7.0f}%{gross:>11.0f}{net:>10.0f}")
    print("\n" + "=" * 104)
    print("  OKU: yüksek eval-pos → hızlı funded (med-ay↓) ama pass%↓ → daha çok kill-fee. İADE → geçen ucuz.")
    print("  asimetrik: eval'de hız al (sadece fee riski), funded'da 0.6 ile payout'u koru. 2-paralel: NDX+SPX")
    print("  joint funded olasılığını yükseltir (örtüşmeyen kill'ler). Karar masada — go/no-go Emir'in.")
    return 0


if __name__ == "__main__":
    main()
