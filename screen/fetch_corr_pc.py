"""screen/fetch_corr_pc — implied correlation (CBOE COR1M/COR3M/ICJ) + put/call (yfinance ^CPC/^CPCE)."""
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
CBOE_SYMS = ["COR1M", "COR3M", "COR30D", "ICJ", "KCJ", "JCJ"]      # implied correlation aileleri
YF_PC = ["^CPC", "^CPCE", "^PCALL"]                                # put/call


def _parse(text, name):
    df = pd.read_csv(io.StringIO(text))
    dcol = [c for c in df.columns if "date" in c.lower()][0]
    num = [c for c in df.columns if c != dcol and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8]
    s = pd.Series(pd.to_numeric(df[num[-1]], errors="coerce").values,
                  index=pd.to_datetime(df[dcol], errors="coerce"), name=name).dropna()
    return s[~s.index.duplicated(keep="last")].sort_index()


def main() -> int:
    out = {}
    for sym in CBOE_SYMS:
        try:
            r = requests.get(CBOE.format(sym), timeout=25, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and "," in r.text[:200]:
                s = _parse(r.text, sym)
                out[sym] = s
                print(f"  CBOE {sym:<7}: {len(s)} gün {s.index.min().date()}..{s.index.max().date()}  son={s.iloc[-1]:.2f}")
            else:
                print(f"  CBOE {sym:<7}: HTTP {r.status_code}")
        except Exception as e:
            print(f"  CBOE {sym:<7}: {type(e).__name__}: {str(e)[:50]}")
    # put/call via yfinance
    try:
        import yfinance as yf
        for sym in YF_PC:
            try:
                h = yf.Ticker(sym).history(period="max")["Close"].dropna()
                if len(h) > 200:
                    h.index = h.index.tz_localize(None)
                    out[sym] = h.rename(sym)
                    print(f"  yf   {sym:<7}: {len(h)} gün {h.index.min().date()}..{h.index.max().date()}  son={h.iloc[-1]:.2f}")
                else:
                    print(f"  yf   {sym:<7}: {len(h)} gün (yetersiz)")
            except Exception as e:
                print(f"  yf   {sym:<7}: {type(e).__name__}")
    except Exception as e:
        print(f"  yfinance yok: {e}")

    if not out:
        print("  [!] hiçbiri çekilemedi.")
        return 1
    CACHE.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out)
    df.to_parquet(CACHE / "corr_pc.parquet")
    print(f"\n  saved -> {CACHE / 'corr_pc.parquet'}  kolonlar={list(df.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
