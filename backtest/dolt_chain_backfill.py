"""
backtest/dolt_chain_backfill — DoltHub post-no-preference/options'tan GERCEK bid/ask zincir backfill'i (BEDAVA).

Kaynak: https://www.dolthub.com/repositories/post-no-preference/options (SQL API, auth yok, public).
option_chain: date, act_symbol, expiration, strike, call_put, bid, ask, vol(IV), greeks — OI YOK.
Kadans (probe): 2024 Pzt/Crs/Cum (haftada 3), ~2025+ her islem gunu. Bu backfill Alpaca-hacim
penceresiyle ayni araligi ceker (2024-01-18→2026-06-08), tarih basina TEK on-aylik expiry
(alpaca_chain_backfill.expiry_for — md metodolojisi), strike bandi = gunun spotu ±%16.

RESUMABLE: gun bazinda (bos gunler done_dates.csv'ye islenir, tekrar sorgulanmaz).
→ data/historical_chains/dolt_chain_{sym}.parquet  (date, expiration, strike, right, bid, ask, mid)
  & <kader-macro venv python> backtest/dolt_chain_backfill.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from build_level_series import _daily_spot                 # noqa: E402
from alpaca_chain_backfill import expiry_for, START, END   # noqa: E402

BASE = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
OUT = ROOT / "data" / "historical_chains"
SYMS = ["SPY", "QQQ"]
PAUSE = 0.5


def q(sql: str):
    """Tek sorgu → (rows, capped). Hata → None (cagiran atlar, resume telafi eder)."""
    url = BASE + "?q=" + urllib.parse.quote(sql)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                j = json.load(r)
            st = j.get("query_execution_status")
            if st in ("Success", "RowLimit"):
                return j["rows"], st == "RowLimit"
            return None, False                             # deadline vb. → gun atlanir
        except Exception:
            time.sleep(2.0 * (attempt + 1))
    return None, False


def fetch_day(sym: str, D: date, E: date, k_lo: float, k_hi: float, depth: int = 0):
    """Gun zinciri; RowLimit'te strike bandini ikiye bolerek in (cap ~200-1000 bilinmiyor)."""
    sql = (f"SELECT `strike`,`call_put`,`bid`,`ask` FROM `option_chain` "
           f"WHERE `act_symbol`='{sym}' AND `date`='{D}' AND `expiration`='{E}' "
           f"AND `strike` BETWEEN {k_lo:.0f} AND {k_hi:.0f} ORDER BY `strike`")
    rows, capped = q(sql)
    if rows is None:
        return None
    if capped and depth < 4:
        mid_k = (k_lo + k_hi) / 2
        a = fetch_day(sym, D, E, k_lo, mid_k, depth + 1)
        time.sleep(PAUSE)
        b = fetch_day(sym, D, E, mid_k + 0.01, k_hi, depth + 1)
        return (a or []) + (b or [])
    return rows


def backfill(sym: str) -> None:
    spot = _daily_spot(sym)
    cal = set(spot.index)
    days = [d for d in sorted(cal) if START <= d <= END]
    p = OUT / f"dolt_chain_{sym.lower()}.parquet"
    pdone = OUT / f"dolt_chain_{sym.lower()}_done_dates.csv"
    done = set(pd.read_csv(pdone)["date"].astype(str)) if pdone.exists() else set()
    parts = [pd.read_parquet(p)] if p.exists() else []
    todo = [d for d in days if str(d) not in done]
    print(f"{sym}: {len(days)} gun, {len(done)} hazir, {len(todo)} sorgulanacak")
    n_hit = 0
    for i, D in enumerate(todo):
        S = float(spot[D])
        rows = fetch_day(sym, D, expiry_for(D, cal), S * 0.84, S * 1.16)
        if rows is None:
            continue                                       # gecici hata — done'a YAZMA, resume alsin
        if rows:
            df = pd.DataFrame(rows)
            df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
            df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
            out = pd.DataFrame({
                "date": pd.Timestamp(D), "expiration": pd.Timestamp(expiry_for(D, cal)),
                "strike": pd.to_numeric(df["strike"]), "right": df["call_put"].str[0].str.upper(),
                "bid": df["bid"], "ask": df["ask"], "mid": (df["bid"] + df["ask"]) / 2.0})
            parts.append(out)
            n_hit += 1
        done.add(str(D))
        if (i + 1) % 25 == 0 or i == len(todo) - 1:
            if parts:
                merged = pd.concat(parts, ignore_index=True).drop_duplicates(
                    subset=["date", "expiration", "strike", "right"], keep="last")
                merged.to_parquet(p)
                parts = [merged]
            pd.DataFrame({"date": sorted(done)}).to_csv(pdone, index=False)
            print(f"  {sym}: {i + 1}/{len(todo)} gun islendi ({n_hit} dolu)")
        time.sleep(PAUSE)
    if p.exists():
        fin = pd.read_parquet(p)
        per = fin.groupby("date").size()
        print(f"{sym} CENSUS: {len(fin):,} satir, {per.index.min().date()}→{per.index.max().date()}, "
              f"{len(per)} dolu gun / {len(days)} islem gunu, kontrat/gun medyan {int(per.median())}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for sym in SYMS:
        backfill(sym)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
