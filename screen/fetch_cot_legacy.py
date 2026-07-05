"""
screen/fetch_cot_legacy — LEGACY COT (pro'ların klasiği): Commercials (smart money) + Non-commercials (specs).
Socrata 6dca-aqww. ES+NQ. comm/noncomm net + OI. Tarih ~1997+ (uzun). data/cache/cot_legacy.parquet.
"""
from __future__ import annotations

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
LEG = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
NAMES = {
    "ES": ["E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE", "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE",
           "S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE", "S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE"],
    "NQ": ["NASDAQ-100 STOCK INDEX (MINI) - CHICAGO MERCANTILE EXCHANGE", "NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE",
           "NASDAQ-100 Consolidated - CHICAGO MERCANTILE EXCHANGE", "NASDAQ-100 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE"],
}
F = ["report_date_as_yyyy_mm_dd", "open_interest_all", "comm_positions_long_all", "comm_positions_short_all",
     "noncomm_positions_long_all", "noncomm_positions_short_all"]


def _fetch(names):
    inlist = ",".join("'" + n.replace("'", "''") + "'" for n in names)
    p = {"$where": f"market_and_exchange_names in ({inlist})", "$select": ",".join(F),
         "$order": "report_date_as_yyyy_mm_dd", "$limit": "40000"}
    r = requests.get(LEG, params=p, timeout=40, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    for c in F[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["date", "open_interest_all"]).groupby("date", as_index=False).last()
    return df.set_index("date").sort_index()


def main():
    out = {}
    for tag, names in NAMES.items():
        try:
            df = _fetch(names)
            if df.empty:
                print(f"  {tag}: boş"); continue
            oi = df["open_interest_all"].replace(0, pd.NA)
            out[f"{tag}_comm_net"] = (df["comm_positions_long_all"] - df["comm_positions_short_all"]) / oi
            out[f"{tag}_spec_net"] = (df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]) / oi
            print(f"  {tag}: {len(df)} hafta {df.index.min().date()}..{df.index.max().date()}  "
                  f"comm-net {out[f'{tag}_comm_net'].iloc[-1]:+.3f}  spec-net {out[f'{tag}_spec_net'].iloc[-1]:+.3f}")
        except Exception as e:
            print(f"  {tag}: HATA {type(e).__name__}: {str(e)[:80]}")
    if not out:
        return 1
    pd.DataFrame(out).to_parquet(CACHE / "cot_legacy.parquet")
    print(f"\n  saved -> {CACHE / 'cot_legacy.parquet'}  ({list(out)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
