"""
screen/fetch_vol — wave-1 vol-surface backbone (Faz 1). VIX term-structure: VIX (1M) vs VIX3M.

ts_ratio = VIX / VIX3M:  >1 BACKWARDATION (stres, ön-uç panik) / <1 CONTANGO (sakin). En direkt
hızlı-vol-şok imzası (COVID-boşluğunun adayı). FRED: VIXCLS (1990+), VXVCLS=VIX3M (2007+) — m5=MOVE
faiz-vol'ü ile çift-saymaz (bu equity-vol). + VXN (VXNCLS, Nasdaq-100 vol) NDX-özel.
PIT: market-close indeksleri → lag 0 (engine +1g uygular). Read-only, kader-macro import YOK.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(Path(r"C:\Users\admin\Downloads\kader-macro") / ".env")     # FRED key
KEY = os.environ.get("FRED_API_KEY")
CACHE = ROOT / "data" / "cache"


def fred(sid: str, start: str = "2006-01-01") -> pd.Series:
    r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                     params={"series_id": sid, "api_key": KEY, "file_type": "json",
                             "observation_start": start}, timeout=30)
    r.raise_for_status()
    o = r.json().get("observations", [])
    return pd.Series({pd.Timestamp(x["date"]): float(x["value"])
                      for x in o if x["value"] not in (".", "")}, name=sid).sort_index()


def main() -> int:
    if not KEY:
        print("  [!] FRED_API_KEY yok (kader-macro/.env) — fetch atlandı.")
        return 2
    series = {}
    for sid in ("VIXCLS", "VXVCLS", "VXNCLS"):          # VIX(1M), VIX3M, VXN(NDX-vol)
        try:
            series[sid] = fred(sid)
            print(f"  {sid:<8}: {len(series[sid])} gün {series[sid].index.min().date()}..{series[sid].index.max().date()}")
        except Exception as e:
            print(f"  {sid:<8}: FETCH HATA — {type(e).__name__}: {str(e)[:80]} (endpoint api.stlouisfed.org)")
    if "VIXCLS" not in series or "VXVCLS" not in series:
        print("  [!] VIX/VIX3M eksik — term-structure kurulamadı.")
        return 1

    df = pd.DataFrame({"vix": series["VIXCLS"], "vix3m": series["VXVCLS"]})
    if "VXNCLS" in series:
        df["vxn"] = series["VXNCLS"]
    df = df.dropna(subset=["vix", "vix3m"])
    df["ts_ratio"] = df["vix"] / df["vix3m"]            # >1 backwardation (stres) / <1 contango (sakin)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE / "vol_surface.parquet")
    print(f"\n  vol_surface: {len(df)} gün {df.index.min().date()}..{df.index.max().date()}  "
          f"(backwardation günleri ts>1: {100*(df['ts_ratio'] > 1).mean():.0f}%)")
    print(df.tail(3).round(3).to_string())
    print(f"  saved -> {CACHE / 'vol_surface.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
