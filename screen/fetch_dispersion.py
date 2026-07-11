"""
screen/fetch_dispersion — 3-way dispersion-froth omurgası (dispersion_ensemble overlay'inin veri katmanı).

CBOE CDN (free, abonelik yok): VIXEQ (S&P500 Constituent Volatility = tekil-hisse cap-ağırlıklı 30g örtük vol),
DSPX (S&P500 Dispersion Index), VIX (endeks örtük vol). spread = VIXEQ − VIX = tekil−endeks = dispersion /
ters-implied-corr. Hepsi 2014-06+ (VIXEQ/DSPX backfill). COR1M zaten corr_pc.parquet'te (fetch_corr_pc).

dispersion_ensemble frozen-yol bu parquet'ten okur (ağsız); live-yol modules/dispersion_ensemble canlı fetch eder.
PIT: market-close indeksleri → lag 0 (engine +1g uygular). Read-only, kader-macro import YOK.

Tazeleme: `python -m screen.fetch_dispersion` (KaderEquity otomasyonuna eklendi).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CACHE = ROOT / "data" / "cache"
CBOE = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"
UA = {"User-Agent": "Mozilla/5.0"}


def _fetch(sym: str) -> pd.Series:
    r = requests.get(CBOE.format(sym), timeout=30, headers=UA)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    dcol = [c for c in df.columns if "date" in c.lower()][0]
    num = [c for c in df.columns if c != dcol and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8]
    s = pd.Series(pd.to_numeric(df[num[-1]], errors="coerce").values,
                  index=pd.to_datetime(df[dcol], errors="coerce"), name=sym).dropna()
    return s[~s.index.duplicated(keep="last")].sort_index()


def main() -> int:
    out = {}
    for sym in ("VIXEQ", "DSPX", "VIX"):
        try:
            out[sym.lower()] = _fetch(sym)
            s = out[sym.lower()]
            print(f"  CBOE {sym:<6}: {len(s)} gün {s.index.min().date()}..{s.index.max().date()}  son={s.iloc[-1]:.2f}")
        except Exception as e:  # noqa: BLE001
            print(f"  CBOE {sym:<6}: FETCH HATA — {type(e).__name__}: {str(e)[:80]}")
    if "vixeq" not in out or "vix" not in out:
        print("  [!] VIXEQ/VIX eksik — dispersion omurgası kurulamadı.")
        return 1
    df = pd.DataFrame(out).dropna(subset=["vixeq", "vix"])
    df["spread"] = df["vixeq"] - df["vix"]            # tekil − endeks = dispersion (yüksek = froth)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE / "dispersion.parquet")
    print(f"\n  dispersion: {len(df)} gün {df.index.min().date()}..{df.index.max().date()}  "
          f"spread[min={df['spread'].min():.1f} son={df['spread'].iloc[-1]:.1f} max={df['spread'].max():.1f}]")
    print(f"  saved -> {CACHE / 'dispersion.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
