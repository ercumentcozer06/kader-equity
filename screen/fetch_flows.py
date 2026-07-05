"""
screen/fetch_flows — AKIŞ sinyalleri (free). SPY/QQQ shares-outstanding = ETF creation/redemption
(gerçek para giriş-çıkışı: paylar artıyorsa para giriyor). yfinance get_shares_full. + FRED money-market
fund assets (risk-off proxy: MMF↑ = nakit-istif). data/cache/flows.parquet.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dotenv import load_dotenv                           # noqa: E402
load_dotenv(Path(r"C:\Users\admin\Downloads\kader-macro") / ".env")
KEY = os.environ.get("FRED_API_KEY")
CACHE = ROOT / "data" / "cache"


def _fred(sid, start="2010-01-01"):
    r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                     params={"series_id": sid, "api_key": KEY, "file_type": "json", "observation_start": start},
                     timeout=30)
    o = r.json().get("observations", [])
    return pd.Series({pd.Timestamp(x["date"]): float(x["value"]) for x in o if x["value"] not in (".", "")}).sort_index()


def main() -> int:
    import yfinance as yf
    out = {}
    for tk in ("SPY", "QQQ"):
        try:
            s = yf.Ticker(tk).get_shares_full(start="2015-01-01")
            if s is not None and len(s) > 100:
                s.index = pd.to_datetime(s.index).tz_localize(None)
                s = s[~s.index.duplicated(keep="last")].sort_index()
                out[f"{tk}_shares"] = s
                print(f"  {tk} shares: {len(s)} nokta {s.index.min().date()}..{s.index.max().date()}  son {s.iloc[-1]/1e6:.0f}M")
            else:
                print(f"  {tk} shares: yetersiz/yok")
        except Exception as e:
            print(f"  {tk} shares: {type(e).__name__}: {str(e)[:60]}")
    # money market fund assets (FRED haftalık) — risk-off proxy
    for sid in ("WMMFNS", "MMMFFAQ027S"):
        try:
            s = _fred(sid)
            if len(s) > 50:
                out["MMF_assets"] = s
                print(f"  MMF ({sid}): {len(s)} nokta {s.index.min().date()}..{s.index.max().date()}")
                break
        except Exception as e:
            print(f"  MMF {sid}: {type(e).__name__}")
    if not out:
        print("  [!] hiçbir akış serisi çekilemedi.")
        return 1
    df = pd.DataFrame(out)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE / "flows.parquet")
    print(f"\n  saved -> {CACHE / 'flows.parquet'}  ({list(df.columns)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
