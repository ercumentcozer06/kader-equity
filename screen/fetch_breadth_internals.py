"""
screen/fetch_breadth_internals — GERÇEK breadth: % above 200d/50d, S&P 500 bileşenlerinden hesaplanır
($SPXA200R'ın kendisi; yfinance index'i 404 verdiği için bileşenden kuruyoruz). Advance-decline proxy de.

S&P 500 listesi Wikipedia'dan (CURRENT list → survivorship caveat; breadth large-cap dominant, kabul edilebilir).
Batch yf.download closes 2017+. data/cache/breadth_internals.parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CACHE = ROOT / "data" / "cache"


def main() -> int:
    import yfinance as yf
    try:
        import io

        import requests
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
        tbl = pd.read_html(io.StringIO(html))[0]
        syms = [s.replace(".", "-") for s in tbl["Symbol"].astype(str).tolist()]
        print(f"  S&P 500 listesi: {len(syms)} sembol")
    except Exception as e:
        print(f"  [!] liste çekilemedi: {type(e).__name__}: {str(e)[:80]}")
        return 1

    try:
        data = yf.download(syms, start="2017-01-01", auto_adjust=True, progress=False, threads=True)["Close"]
    except Exception as e:
        print(f"  [!] batch download hata: {type(e).__name__}: {str(e)[:80]}  (endpoint query*.finance.yahoo.com)")
        return 1
    data = data.dropna(axis=1, how="all")
    data.index = pd.to_datetime(data.index).tz_localize(None)
    cov = data.notna().mean(axis=1).iloc[-1]
    print(f"  fiyat matrisi: {data.shape[0]} gün × {data.shape[1]} isim  {data.index.min().date()}..{data.index.max().date()}  (son-gün coverage {cov:.0%})")

    ma200 = data.rolling(200, min_periods=150).mean()
    ma50 = data.rolling(50, min_periods=40).mean()
    pct200 = (data > ma200).sum(axis=1) / data.notna().sum(axis=1) * 100
    pct50 = (data > ma50).sum(axis=1) / data.notna().sum(axis=1) * 100
    # advance-decline proxy: günlük yükselen-azalan isim farkı (kümülatif)
    chg = data.pct_change()
    adv = (chg > 0).sum(axis=1); dec = (chg < 0).sum(axis=1)
    ad_line = (adv - dec).cumsum()

    out = pd.DataFrame({"pct_above_200d": pct200, "pct_above_50d": pct50, "ad_line": ad_line}).dropna()
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE / "breadth_internals.parquet")
    print(f"  son: %>200d {pct200.iloc[-1]:.0f}  %>50d {pct50.iloc[-1]:.0f}")
    print(f"  saved -> {CACHE / 'breadth_internals.parquet'}  ({out.shape[0]} gün, {list(out.columns)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
