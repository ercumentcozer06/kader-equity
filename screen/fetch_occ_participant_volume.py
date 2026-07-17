"""Archive OCC's official daily participant-type option volume (last 24 months).

This is not signed order flow: OCC reports Customer/Firm/Market-Maker volume by
call/put and exchange, not buy/sell or open/close.  It is retained as an
options-influence / participant-composition feature with that limitation.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "option_research" / "occ_participant_volume"
OUT = ROOT / "data" / "option_research" / "occ_participant_daily.parquet"
URL = "https://marketdata.theocc.com/volume-query"
SYMBOLS = {"SPY": "OSTK", "QQQ": "OSTK", "SPX": "OIND", "NDX": "OIND"}
UA = {"User-Agent": "KaderEquity research archive; contact=local account owner"}


def _fetch(symbol: str, kind: str, day: date) -> tuple[str, date, str, str]:
    path = RAW / symbol / f"{day}.csv.gz"
    if path.exists():
        return symbol, day, "exists", str(path)
    params = {
        "reportDate": day.strftime("%Y%m%d"), "format": "csv", "volumeQueryType": "O",
        "symbolType": "U", "symbol": symbol, "reportType": "D", "accountType": "ALL",
        "productKind": kind, "porc": "BOTH",
    }
    r = requests.get(URL, params=params, headers=UA, timeout=30)
    r.raise_for_status()
    text = r.content.decode("utf-8-sig", "replace")
    if not text.startswith("quantity,"):
        raise ValueError(f"unexpected OCC response: {text[:100]!r}")
    if len([line for line in text.splitlines() if line.strip()]) <= 1:
        raise FileNotFoundError("OCC report not published or no rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(r.content, compresslevel=6, mtime=0))
    return symbol, day, "ok", str(path)


def _dates(start: date, end: date) -> list[date]:
    p = ROOT / "data" / "historical_bars" / "alpaca_spy_1m.parquet"
    if p.exists():
        b = pd.read_parquet(p).reset_index()
        col = "timestamp" if "timestamp" in b else b.columns[1]
        ds = pd.to_datetime(b[col], utc=True).dt.tz_convert("America/New_York").dt.date.unique()
        return sorted(d for d in ds if start <= d <= end)
    return [x.date() for x in pd.bdate_range(start, end)]


def materialise() -> pd.DataFrame:
    rows = []
    for path in RAW.glob("*/*.csv.gz"):
        with gzip.open(path, "rt", encoding="utf-8-sig") as fh:
            # OCC rows currently carry a trailing empty field not present in
            # the header; index_col=False prevents pandas from shifting the
            # numeric quantity into the index.
            d = pd.read_csv(io.StringIO(fh.read()), index_col=False)
        if d.empty:
            continue
        d["quantity"] = pd.to_numeric(d["quantity"], errors="coerce").fillna(0)
        symbol, day = path.parent.name, date.fromisoformat(path.stem.replace(".csv", ""))
        agg = d.groupby(["actype", "porc"]).quantity.sum()
        row = {"date": pd.Timestamp(day), "symbol": symbol, "n_exchange_rows": len(d)}
        names = {"C": "customer", "F": "firm", "M": "market_maker"}
        rights = {"C": "call", "P": "put"}
        for a, an in names.items():
            for cp, cn in rights.items():
                row[f"{an}_{cn}_volume"] = float(agg.get((a, cp), 0.0))
        row["customer_put_call"] = ((row["customer_put_volume"] + 1) /
                                    (row["customer_call_volume"] + 1))
        total = sum(v for k, v in row.items() if k.endswith("_volume"))
        row["market_maker_volume_share"] = ((row["market_maker_call_volume"] + row["market_maker_put_volume"])
                                             / total if total else None)
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.drop_duplicates(["date", "symbol"], keep="last").sort_values(["date", "symbol"])
        OUT.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(OUT, index=False)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("daily", "backfill"), default="daily", nargs="?")
    p.add_argument("--start")
    p.add_argument("--end")
    a = p.parse_args(argv)
    end = date.fromisoformat(a.end) if a.end else date.today() - timedelta(days=1)
    default_start = end - timedelta(days=720) if a.mode == "backfill" else end - timedelta(days=14)
    start = date.fromisoformat(a.start) if a.start else default_start
    days = _dates(start, end)
    counts = {"ok": 0, "exists": 0, "pending": 0, "error": 0}
    # OCC documents this as a batch-processing endpoint. Six workers keeps the
    # backfill practical without turning it into an aggressive scraper.
    jobs = [(s, k, day) for day in days for s, k in SYMBOLS.items()]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(_fetch, *job): job for job in jobs}
        for fut in as_completed(futs):
            symbol, _, day = futs[fut]
            try:
                _, _, status, _ = fut.result(); counts[status] += 1
            except FileNotFoundError:
                counts["pending"] += 1
            except Exception as exc:
                counts["error"] += 1
                print(f"{day} {symbol}: {type(exc).__name__}: {str(exc)[:120]}")
    out = materialise()
    print(json.dumps({"start": str(start), "end": str(end), "days": len(days),
                      "requests": counts, "rows": len(out), "output": str(OUT)}))
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
