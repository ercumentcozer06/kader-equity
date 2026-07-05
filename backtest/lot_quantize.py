"""
backtest/lot_quantize — A2 LOT-KUANTİZASYON gerçeği (ops/granülarite; yeni DoF yok). Sürekli hedef-exposure'ı
ulaşılabilir lot-grid'ine AŞAĞI-yuvarlayınca (−%5 marjı korunur) ne kayboluyor.

lot = floor(equity × hedef_exposure / (contract_size × spot) / step) × step
realized_exposure = lot × contract_size × spot / equity
→ exposure-ADIMI (0.01 lot başına) = step × contract_size × spot / equity = spot/100000 (cs=10, eq=10k).
   NDX spot ~28440 → %28/adım (KABA!), SPX ~7258 → %7/adım. cs GERÇEK değeri platform-teyitli (FLAG).

Çıktı: kuantize vs sürekli — exposure-adımı, RMS sapma, ifade-edilebilir-trim-kademesi, kuantize pass-LOW,
drift kaybı. contract_size {1,10} duyarlılığı (gerçek bilinmiyor → FLAG). US100 vs US500.
  & <venv python> backtest/lot_quantize.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from backtest import prop_sim as PS
from backtest import prop_sim_v13 as V

EQUITY = 10000.0
STEP = 0.01
SPOT = {"NDX": 28440.0, "SPX": 7258.0}      # ~güncel (granülarite için referans; FLAG)


def quantize_exposure(target, spot, contract_size, equity=EQUITY, step=STEP):
    """Sürekli hedef-exposure dizisini ulaşılabilir lot-grid'ine AŞAĞI-yuvarla → gerçekleşen exposure."""
    lot = np.floor(equity * target / (contract_size * spot) / step) * step
    return lot * contract_size * spot / equity


def _passlow_quant(book, pos_scale, contract_size, spot):
    """Kuantize exposure ile low-bazlı pass-LOW (v13 _sim ile aynı mekanik)."""
    df = V._book(book)
    lc = "low_exc_up" if book == "50/50" else "low_exc"
    tgt = pos_scale * df["pos"].values
    q = quantize_exposure(tgt, spot, contract_size)
    rets, lows = df["close_ret"].values, (df[lc].values if lc in df else df["close_ret"].values)
    sw = V.SWAP_YR / 252.0; n = len(rets); passd, faild = 0, 0
    for s in range(n):
        st1, e1, _ = V._sim(rets, lows, q, sw, s, PS.P1_TARGET, True)
        if st1 == "incomplete":
            continue
        if st1 == "kill":
            faild += 1; continue
        st2, _, _ = V._sim(rets, lows, q, sw, e1 + 1, PS.P2_TARGET, True)
        if st2 == "incomplete":
            continue
        if st2 == "kill":
            faild += 1
        elif st2 == "pass":
            passd += 1
    nv = passd + faild
    return (passd / nv if nv else float("nan")), nv


def main():
    print("=" * 100)
    print("  A2 — LOT-KUANTİZASYON (10k hesap, step 0.01; contract_size GERÇEK=platform-teyitli FLAG)")
    print("=" * 100)
    for cs in (10, 1):
        print(f"\n  contract_size = {cs}  {'(default FLAG)' if cs==10 else '(alternatif — MT5 indeks sık 1)'}")
        print(f"    {'enstrüman':<10}{'spot':>8}{'exposure-adımı':>16}{'≈kademe(0..1.2)':>16}")
        for inst, book in (("US100/NDX", "NDX"), ("US500/SPX", "SPX")):
            spot = SPOT[book]; step_exp = STEP * cs * spot / EQUITY
            flag = "  ← RED (kaba)" if step_exp > 0.15 else ""
            print(f"    {inst:<10}{spot:>8.0f}{100*step_exp:>15.1f}%{1.2/step_exp:>15.1f}{flag}")
    # RMS sapma + kuantize pass (default cs=10) — seçilmiş configler
    print(f"\n  KUANTİZE vs SÜREKLİ (contract_size=10 default): RMS-exposure-sapma + pass-LOW")
    print(f"    {'config':<16}{'RMS-sapma':>11}{'sürekli-pass':>14}{'kuantize-pass':>15}")
    # NOT: 50/50 = iki ayrı enstrüman, her biri AYRI kuantize (granülarite per-leg) → tek-spot RMS anlamsız, atlandı.
    for book, ps in [("NDX", 1.2), ("SPX", 1.0), ("NDX", 0.7), ("SPX", 0.6)]:
        df = V._book(book); spot = SPOT[book]
        tgt = ps * df["pos"].values
        q = quantize_exposure(tgt, spot, 10)
        rms = float(np.sqrt(np.nanmean((q - tgt) ** 2)))
        cont = V._stats(V.run_eval(df, ps, low_based=True, low_col=("low_exc_up" if book == "50/50" else "low_exc"))[0])["pass"]
        qp, _ = _passlow_quant(book, ps, 10, spot)
        print(f"    {book+'@'+str(ps):<16}{rms:>11.3f}{100*cont:>13.0f}%{100*qp:>14.0f}%")
    print("\n" + "=" * 100)
    print("  OKU: cs=10'da NDX exposure-adımı %28 (KABA) → froth/shield trimleri + 1.2 hedefi ifade EDİLEMEZ;")
    print("  SPX %7 (4× ince). cs GERÇEĞİ platformdan teyit ŞART (cs=1 ise NDX %2.8 → sorun yok). Emir eval-kitap")
    print("  seçimini buna göre yapmalı: cs büyükse US500 daha ince ifade eder. (cs FLAG — alım günü kesinleşir.)")
    return 0


if __name__ == "__main__":
    main()
