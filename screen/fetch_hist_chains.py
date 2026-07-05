"""
screen/fetch_hist_chains — DoltHub `post-no-preference/options` FREE keyless SQL API'sinden tarihsel
vol-surface verisi çeker (FAZ 0 veri temeli). SPY/QQQ 2019-02-09+.

  • volatility_history → iv_current + hv_current → **VRP = iv−hv** (varyans risk primi) + IV-level tarihi (hazır seri).
  • option_chain → per-strike IV+greeks+delta → 25d-skew(RR)/term/ATM-IV (delta kolonu ile; ayrı adımda).

SINIR: DoltHub'da **open interest YOK** → GEX/gamma-flip/wall/max-pain için OptionsDX (2010+, OI'li) gerekir.
Çıktı: data/historical_chains/dolt_volhist_<sym>.parquet. Network-gerekli RESEARCH fetcher (runtime değil).
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
API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
OUT = ROOT / "data" / "historical_chains"


def query(sql: str, timeout: int = 80) -> list[dict]:
    r = requests.get(API, params={"q": sql}, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    return r.json().get("rows", [])


def fetch_all(sql_base: str, page: int = 1000) -> list[dict]:
    """OFFSET ile sayfalayarak tüm satırları çeker (DoltHub API sayfa-limitli)."""
    rows, off = [], 0
    while True:
        chunk = query(f"{sql_base} LIMIT {page} OFFSET {off}")
        if not chunk:
            break
        rows += chunk
        off += page
        if len(chunk) < page:
            break
    return rows


def fetch_volhist(sym: str) -> pd.DataFrame:
    rows = fetch_all(f"SELECT * FROM volatility_history WHERE act_symbol='{sym}' ORDER BY `date`")
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    for c in df.columns:
        if c != "act_symbol" and not c.endswith("_date") and c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("date").sort_index()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for sym in ("SPY", "QQQ"):
        df = fetch_volhist(sym)
        if df.empty:
            print(f"  {sym}: veri yok"); continue
        df.to_parquet(OUT / f"dolt_volhist_{sym}.parquet")
        vrp = (df["iv_current"] - df["hv_current"]).dropna()
        print(f"\n  {sym}: {len(df)} gün  {df.index.min().date()}..{df.index.max().date()}  → {OUT/f'dolt_volhist_{sym}.parquet'}")
        print(f"    VRP=iv−hv: mean {vrp.mean():+.3f} (varyans risk primi POZİTİF olmalı), "
              f"std {vrp.std():.3f}, son {vrp.iloc[-1]:+.3f}")
        # stres-dönemi sanity: IV bu tarihlerde sıçramalı
        for d in ("2020-03-16", "2022-06-13", "2025-04-07"):
            try:
                iv = df["iv_current"].asof(pd.Timestamp(d))
                print(f"    iv_current @ {d}: {iv:.3f}")
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
