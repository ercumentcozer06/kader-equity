"""
screen/fetch_cot — CFTC COT (Traders in Financial Futures) ES + NQ. Free Socrata, key-yok.

Sinyal adayı: lev-money net (hedge-fon = hızlı para) + asset-mgr net (real-money), (long−short)/OI = kalabalıklık.
Tue as-of, Fri publish → exec-lag +3g. data/cache/cot_es_nq.parquet. Çoklu kontrat-ismi → tarih-dedup (max OI).
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
TFF = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
NAMES = {
    "ES": ["E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE", "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE",
           "S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE"],
    "NQ": ["NASDAQ-100 STOCK INDEX (MINI) - CHICAGO MERCANTILE EXCHANGE", "NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE",
           "NASDAQ-100 Consolidated - CHICAGO MERCANTILE EXCHANGE", "NASDAQ-100 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE"],
}
FIELDS = ["report_date_as_yyyy_mm_dd", "open_interest_all",
          "lev_money_positions_long", "lev_money_positions_short",
          "asset_mgr_positions_long", "asset_mgr_positions_short"]


def _fetch(names: list[str]) -> pd.DataFrame:
    inlist = ",".join("'" + n.replace("'", "''") + "'" for n in names)
    params = {"$where": f"market_and_exchange_names in ({inlist})",
              "$select": ",".join(FIELDS + ["market_and_exchange_names"]),
              "$order": "report_date_as_yyyy_mm_dd", "$limit": "30000"}
    r = requests.get(TFF, params=params, timeout=40, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    for c in FIELDS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["date", "open_interest_all"]).groupby("date", as_index=False).last()  # tarih başına max-OI
    return df.set_index("date").sort_index()


def main() -> int:
    out = {}
    for tag, names in NAMES.items():
        try:
            df = _fetch(names)
            if df.empty:
                print(f"  {tag}: boş")
                continue
            oi = df["open_interest_all"].replace(0, pd.NA)
            out[f"{tag}_lev_net"] = (df["lev_money_positions_long"] - df["lev_money_positions_short"]) / oi
            out[f"{tag}_am_net"] = (df["asset_mgr_positions_long"] - df["asset_mgr_positions_short"]) / oi
            print(f"  {tag}: {len(df)} hafta {df.index.min().date()}..{df.index.max().date()}  "
                  f"son lev-net/OI {out[f'{tag}_lev_net'].iloc[-1]:+.3f}  am-net/OI {out[f'{tag}_am_net'].iloc[-1]:+.3f}")
        except Exception as e:
            print(f"  {tag}: FETCH HATA — {type(e).__name__}: {str(e)[:90]}")
    if not out:
        print("  [!] COT çekilemedi.")
        return 1
    df = pd.DataFrame(out)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE / "cot_es_nq.parquet")
    print(f"\n  saved -> {CACHE / 'cot_es_nq.parquet'}  kolonlar={list(df.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
