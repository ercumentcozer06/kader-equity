"""screen/fetch_squeeze_gex — SqueezeMetrics FREE tarihsel DIX + GEX CSV (~2011+). GEX'i BACKTEST etmek için."""
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
URLS = [
    "https://squeezemetrics.com/monitor/static/DIX.csv",
    "https://squeezemetrics.com/monitor/static/DIX",
]


def main() -> int:
    text = None
    for url in URLS:
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and "," in r.text[:200]:
                text = r.text
                print(f"  OK {url}")
                break
            print(f"  {url} → HTTP {r.status_code}")
        except Exception as e:
            print(f"  {url} → {type(e).__name__}: {str(e)[:70]}  (endpoint squeezemetrics.com)")
    if text is None:
        print("  [!] SqueezeMetrics CSV çekilemedi (endpoint değişmiş/bloklu olabilir).")
        return 1

    df = pd.read_csv(io.StringIO(text))
    print(f"  kolonlar: {list(df.columns)}  satır: {len(df)}")
    dcol = [c for c in df.columns if "date" in c.lower()][0]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).set_index(dcol).sort_index()
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE / "squeeze_dix_gex.parquet")
    print(f"  aralık: {df.index.min().date()}..{df.index.max().date()}")
    print(df.tail(3).to_string())
    print(f"  saved -> {CACHE / 'squeeze_dix_gex.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
