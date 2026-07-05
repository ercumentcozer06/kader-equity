"""
backtest/disentangle — spine_diagnostic TERS bulgusunu confound'lardan ayıkla.
Bulgu: naive-+γ → momentum, naive-−γ → mean-revert (ders-kitabının TERSİ, M1+M2 tutarlı). İki confound:
  (1) PİYASA TRENDİ: yükseliş → call-wall kırılır → "momentum" trend-artefaktı olabilir.
  (2) VOL-REJİMİ: yüksek-vol→revert / düşük-vol→trend BİLİNEN etki; gamma sadece vol-proxy'si olabilir (eklemiyor).
Temiz, trend-nötr, wall-bağımsız sinyal kullanır: gap→intraday devam/dönüş (M1'in tradeable hali).
  pos_gamma_inv : +γ → gap-yönünde (momentum), −γ → gap-tersine (revert)   [TERS bulgu]
  pos_vol       : düşük-vol → momentum, yüksek-vol → revert                  [vol-only rakip]
KONTROLLER: (A) intraday drift-demean (trend çıkar), (B) gamma vs vol (gamma vol-ötesinde ekliyor mu),
            (C) train(ilk150)/holdout(son70). Maliyet 1.5bps/gün round-trip.
  & <venv python> backtest/disentangle.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backtest"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from spine_diagnostic import build_panel       # noqa: E402

COST = 0.00015


def sharpe(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean() / x.std() * np.sqrt(252) if len(x) > 2 and x.std() > 0 else 0.0


def _line(tag, pnl):
    pnl = np.asarray(pnl, float); pnl = pnl[~np.isnan(pnl)]
    net = pnl - COST
    return (f"    {tag:<26} Sharpe {sharpe(pnl):+.2f} (net {sharpe(net):+.2f})  "
            f"ort {1e4*pnl.mean():+.1f}bps/gün  isabet %{100*(pnl>0).mean():.0f}  n{len(pnl)}")


def run(sym):
    p = build_panel(sym).dropna(subset=["gap", "intraday", "flip", "atm_iv"]).reset_index(drop=True)
    gap = np.sign(p["gap"].values)
    intr = p["intraday"].values
    intr_dm = intr - np.nanmean(intr)                 # (A) trend-demean: ortalama intraday drift'i çıkar
    reg = p["regime"].values                          # net_gex işareti
    iv = p["atm_iv"].values
    volhigh = iv > np.median(iv)                      # PIT vol-rejimi (implied-vol seviyesi)

    pos = {
        "gamma_inv (+γ mom/−γ rev)": np.where(reg > 0, gap, -gap),   # TERS bulgu
        "gamma_txt (ders-kitabı)": np.where(reg > 0, -gap, gap),     # naive (NO-GO bekleniyor)
        "vol_only (lo mom/hi rev)": np.where(volhigh, -gap, gap),    # vol-rakip
        "hep momentum": gap,
        "hep revert": -gap,
    }

    print("=" * 98)
    print(f"  {sym} — DISENTANGLE  (n={len(p)})   [pnl = pozisyon × intraday-getiri]")
    print("=" * 98)
    print("  (1) TAM ÖRNEK — ham getiri:")
    for k, ps in pos.items():
        print(_line(k, ps * intr))

    print("\n  (A) TREND KONTROLÜ — intraday drift-demean (trend-artefaktı çıkarıldı):")
    for k in ("gamma_inv (+γ mom/−γ rev)", "vol_only (lo mom/hi rev)", "hep momentum"):
        print(_line(k, pos[k] * intr_dm))

    print("\n  (B) GAMMA vs VOL — gamma, vol-ötesinde yön bilgisi ekliyor mu?")
    pg = pos["gamma_inv (+γ mom/−γ rev)"]; pv = pos["vol_only (lo mom/hi rev)"]
    agree = (pg == pv).mean()
    print(f"    pozisyon örtüşmesi gamma_inv vs vol_only: %{100*agree:.0f}  "
          f"({'çoğunlukla aynı → gamma ≈ vol-proxy' if agree > 0.7 else 'ayrışıyor → gamma bağımsız olabilir'})")
    # vol-kovası İÇİNDE gamma hâlâ ayırıyor mu (trend-demean getiri ile)
    for lab, mask in (("düşük-vol", ~volhigh), ("yüksek-vol", volhigh)):
        sub_g = np.where(reg[mask] > 0, gap[mask], -gap[mask]) * intr_dm[mask]
        print(f"    {lab} kovası içinde gamma_inv:  Sharpe {sharpe(sub_g):+.2f} net {sharpe(sub_g-COST):+.2f} (n{mask.sum()})")

    print("\n  (C) TRAIN(ilk150) / HOLDOUT(son70) — gamma_inv (ham + demean):")
    for lab, r in (("TRAIN", slice(0, 150)), ("HOLDOUT", slice(150, None))):
        g = pos["gamma_inv (+γ mom/−γ rev)"][r]
        print(f"    {lab:<8} ham Sharpe {sharpe(g*intr[r]):+.2f} net {sharpe(g*intr[r]-COST):+.2f}  |  "
              f"demean Sharpe {sharpe(g*intr_dm[r]):+.2f} net {sharpe(g*intr_dm[r]-COST):+.2f}  (n{len(g)})")
    return p


def main():
    for sym in ("SPY", "QQQ"):
        run(sym); print()
    print("  OKU: gamma_inv demean-sonrası ve holdout'ta ve vol-kovası-içinde POZİTİF kalırsa → gerçek gamma-yön edge'i.")
    print("  Demean-sonrası çöküyorsa = TREND artefaktı; vol_only'yi geçemiyorsa = sadece vol-proxy (gamma eklemez).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
