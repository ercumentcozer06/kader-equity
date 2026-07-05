"""
screen/fetch_breadth — breadth/internals (atladığım eksen). RSP/SPY (equal-vs-cap = konsantrasyon),
QQEW/QQQ, + % above 200d endeksleri (^S5TH/^NDTH/^MMTH) + advance-decline (^ADD) yfinance dener.
Daralan breadth (RSP↓SPY) = mega-cap konsantrasyon = kırılganlık adayı. data/cache/breadth.parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CACHE = ROOT / "data" / "cache"
TICKERS = ["RSP", "SPY", "QQEW", "QQQ", "^S5TH", "^NDTH", "^MMTH", "^ADD", "^ADDN"]


def main() -> int:
    import yfinance as yf
    out = {}
    for t in TICKERS:
        try:
            h = yf.Ticker(t).history(period="max")["Close"].dropna()
            if len(h) > 200:
                h.index = h.index.tz_localize(None)
                out[t.replace("^", "")] = h
                print(f"  {t:<7}: {len(h)} gün {h.index.min().date()}..{h.index.max().date()}  son {h.iloc[-1]:.2f}")
            else:
                print(f"  {t:<7}: {len(h)} gün (yetersiz/yok)")
        except Exception as e:
            print(f"  {t:<7}: {type(e).__name__}")
    if not out:
        print("  [!] hiçbiri çekilemedi.")
        return 1
    df = pd.DataFrame(out)
    # türev: konsantrasyon oranları
    if "RSP" in df and "SPY" in df:
        df["RSP_SPY"] = df["RSP"] / df["SPY"]
    if "QQEW" in df and "QQQ" in df:
        df["QQEW_QQQ"] = df["QQEW"] / df["QQQ"]
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE / "breadth.parquet")
    print(f"\n  saved -> {CACHE / 'breadth.parquet'}  kolonlar={list(df.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
