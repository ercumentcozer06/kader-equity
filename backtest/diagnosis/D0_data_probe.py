"""
backtest/diagnosis/D0_data_probe — chain/level içerik gerçeği (TEŞHİS-ONLY, P&L üretmez).
(b) mid/bid=0 sorusu: md_*.parquet'te ham bid/ask YOK, yalnız 'mid' var mı? mid<=0 / NaN kaç satır?
(d) per-day expiry sayısı (tek-vade teyidi) + DTE.
(e) PIT: level index = D-EOD; D+1 session ayrı.
IV-drop teyidi: build_level_series mantığını taklit, kaç kontrat implied_vol=None ile DÜŞÜYOR.
  & <venv python> backtest/diagnosis/D0_data_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "screen"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from _bsiv import implied_vol  # noqa: E402

BAND = 0.15


def main() -> int:
    for sym in ("spy", "qqq"):
        ch = pd.read_parquet(ROOT / "data" / "historical_chains" / f"md_{sym}.parquet")
        print("=" * 80)
        print(f"  md_{sym}.parquet — {len(ch):,} satır, kolonlar={list(ch.columns)}")
        print("=" * 80)
        # (b) mid/bid sorusu
        has_bidask = ("bid" in ch.columns) and ("ask" in ch.columns)
        print(f"  ham bid/ask kolonu VAR mı? {has_bidask}  → MID kaynağı: {'bid/ask' if has_bidask else 'md_parquet.mid (MarketData server-mid)'}")
        mid = pd.to_numeric(ch["mid"], errors="coerce")
        print(f"  mid: NaN {mid.isna().sum():,}  ≤0 {int((mid <= 0).sum()):,}  >0 {int((mid > 0).sum()):,}  "
              f"(min {np.nanmin(mid.values):.4f}, med {np.nanmedian(mid.values):.4f})")
        # penny-mid (<=0.05) sayısı — instabilite riski
        penny = ((mid > 0) & (mid <= 0.05)).sum()
        print(f"  penny-mid (0<mid≤0.05): {int(penny):,} = %{100*penny/max(int((mid>0).sum()),1):.1f} (instabilite adayı)")
        # iv/delta kolonu boş mu (görev: BOŞ deniyor)
        for c in ("iv", "delta"):
            if c in ch.columns:
                col = pd.to_numeric(ch[c], errors="coerce")
                print(f"  kolon '{c}': dolu {int(col.notna().sum()):,} / {len(ch):,}  (BOŞ teyidi)")

        # (d) per-day expiry sayısı
        per_day_exp = ch.groupby("date")["expiration"].nunique()
        print(f"  per-day expiry nunique: min {per_day_exp.min()} med {int(per_day_exp.median())} max {per_day_exp.max()}  "
              f"(tek-vade teyidi: med={int(per_day_exp.median())})")
        print(f"  gün sayısı {ch['date'].nunique()}, tarih {ch['date'].min()} → {ch['date'].max()}")

        # IV-drop simülasyonu: build_level_series._levels_for_day mantığı (spot olmadan band-only IV-fail oranı)
        # spot proxy: ATM ≈ medyan-strike yerine, her gün strike-medyanı kullan (yalnız fail-oranı için, seviye DEĞİL)
        drop_no_oi = drop_mid = drop_iv = kept = 0
        for d, g in ch.groupby("date"):
            exp0 = pd.Timestamp(g["expiration"].iloc[0])
            dte = (exp0 - pd.Timestamp(d)).days
            T = max(dte, 0.5) / 365.0
            S = float(pd.to_numeric(g["strike"], errors="coerce").median())  # spot proxy (fail-oranı için)
            for _, r in g.iterrows():
                K = pd.to_numeric(r["strike"], errors="coerce")
                oi = pd.to_numeric(r["open_interest"], errors="coerce")
                m = pd.to_numeric(r["mid"], errors="coerce")
                right = r["right"]
                if pd.isna(K) or pd.isna(oi):
                    drop_no_oi += 1; continue
                if pd.isna(m) or m <= 0:
                    drop_mid += 1; continue
                if abs(K / S - 1) > BAND:
                    continue
                iv = implied_vol(float(m), S, float(K), T, right)
                if not iv or iv <= 0:
                    drop_iv += 1; continue
                kept += 1
        tot = drop_no_oi + drop_mid + drop_iv + kept
        print(f"  IV-pipeline (spot=strike-medyan PROXY, yalnız fail-oranı): "
              f"drop[no-oi {drop_no_oi}, mid≤0 {drop_mid}, iv-None {drop_iv}] kept {kept} / band-içi-toplam~{tot}")
        print(f"    → implied_vol=None DROP oranı (band-içi): %{100*drop_iv/max(drop_iv+kept,1):.1f}")
        print()

    # (e) PIT: level index = D, session = D+1 — örnek
    print("=" * 80)
    print("  (e) PIT örnek: level_series_spy.index[:3] (D-EOD) ve build_panel D→N (D+1) mantığı kanıtı")
    ls = pd.read_parquet(ROOT / "data" / "cache" / "level_series_spy.parquet")
    print(f"    level index ilk 3: {[str(x.date()) for x in ls.index[:3]]}")
    print(f"    level index son 3: {[str(x.date()) for x in ls.index[-3:]]}")
    print("    spine_diagnostic.build_panel: her D için nxt=[s>D] → N=nxt[0]; c0=rth[D].c, o1/h1/l1/c1=rth[N] → D+1 seansı.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
