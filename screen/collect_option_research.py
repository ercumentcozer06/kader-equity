"""Authorized append-only option research collector (Alpaca Basic/indicative).

SPY and QQQ snapshots contain quotes, last trades, IV, vendor first Greeks and
contract open interest.  A delayed daily bar query adds cumulative volume for
near-spot short-dated contracts.  Higher-order Greeks are calculated locally.

Examples (run with the kader-macro venv):
  python screen/collect_option_research.py intraday
  python screen/collect_option_research.py eod --force
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
from screen.option_research_greeks import all_greeks  # noqa: E402

OCC = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")
SYMBOLS = ("SPY", "QQQ")
NY = "America/New_York"
OUT = ROOT / "data" / "option_research"
LEDGER = OUT / "capture_ledger.csv"
LEDGER_FIELDS = [
    "fetch_ts_utc", "mode", "symbol", "spot", "n_rows", "n_oi", "n_iv",
    "n_quotes", "n_volume", "min_expiry", "max_expiry", "path", "status",
]


def _credentials() -> tuple[str, str]:
    key, secret = os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret or secret.startswith("REPLACE_ME"):
        raise RuntimeError("Alpaca credentials missing from kader-equity/.env")
    return key, secret


def _clients():
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient
    key, secret = _credentials()
    return (OptionHistoricalDataClient(key, secret),
            StockHistoricalDataClient(key, secret),
            TradingClient(key, secret, paper=True))


def _append_ledger(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    with LEDGER.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LEDGER_FIELDS})


def _market_open(trading, now: datetime) -> bool:
    from alpaca.trading.requests import GetCalendarRequest
    local = pd.Timestamp(now).tz_convert(NY)
    cal = trading.get_calendar(GetCalendarRequest(start=local.date(), end=local.date()))
    if not cal:
        return False
    c = cal[0]
    op = pd.Timestamp(c.open)
    cl = pd.Timestamp(c.close)
    if op.tzinfo is None:
        op = op.tz_localize(NY)
    if cl.tzinfo is None:
        cl = cl.tz_localize(NY)
    return op <= local <= cl + pd.Timedelta(minutes=5)


def _eod_ready(trading, now: datetime) -> bool:
    from alpaca.trading.requests import GetCalendarRequest
    local = pd.Timestamp(now).tz_convert(NY)
    cal = trading.get_calendar(GetCalendarRequest(start=local.date(), end=local.date()))
    if not cal:
        return False
    cl = pd.Timestamp(cal[0].close)
    if cl.tzinfo is None:
        cl = cl.tz_localize(NY)
    return local >= cl + pd.Timedelta(minutes=20)


def _spot(stock, symbol: str) -> float:
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockLatestTradeRequest
    ans = stock.get_stock_latest_trade(StockLatestTradeRequest(
        symbol_or_symbols=symbol, feed=DataFeed.IEX))
    trade = ans[symbol]
    px = float(trade.price)
    if not np.isfinite(px) or px <= 0:
        raise ValueError(f"invalid {symbol} spot {px}")
    return px


def _contracts(trading, symbol: str, lo: float, hi: float, end_date: date) -> dict:
    from alpaca.trading.requests import GetOptionContractsRequest
    token = None
    out = {}
    while True:
        req = GetOptionContractsRequest(
            underlying_symbols=[symbol], expiration_date_gte=date.today(),
            expiration_date_lte=end_date, strike_price_gte=str(lo),
            strike_price_lte=str(hi), limit=10000, page_token=token)
        resp = trading.get_option_contracts(req)
        for x in resp.option_contracts:
            out[x.symbol] = x
        token = resp.next_page_token
        if not token:
            break
    return out


def _volume_bars(client, symbols: list[str], now: datetime) -> dict[str, dict]:
    """15m-delayed cumulative day bars for the short-dated near-spot subset."""
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame
    end = now - timedelta(minutes=16)
    if end.date() != now.date():
        return {}
    start = datetime.combine(now.date(), dtime.min, tzinfo=timezone.utc)
    out: dict[str, dict] = {}
    for i in range(0, len(symbols), 100):
        try:
            bars = client.get_option_bars(OptionBarsRequest(
                symbol_or_symbols=symbols[i:i + 100], timeframe=TimeFrame.Day,
                start=start, end=end)).df
            if bars is None or not len(bars):
                continue
            for (sym, _), r in bars.iterrows():
                out[str(sym)] = {f"day_{k}": float(r[k]) for k in
                                 ("open", "high", "low", "close", "volume", "trade_count", "vwap")}
        except Exception as exc:
            print(f"volume batch {i // 100}: {type(exc).__name__}: {str(exc)[:100]}")
        time.sleep(0.03)
    return out


def _normalise(symbol: str, spot: float, snaps: dict, contracts: dict,
               volumes: dict, fetched: datetime) -> pd.DataFrame:
    rows = []
    for name, snap in snaps.items():
        m = OCC.match(name)
        if not m:
            continue
        _, ymd, cp, strike8 = m.groups()
        expiry = datetime.strptime(ymd, "%y%m%d").date()
        strike = int(strike8) / 1000.0
        quote, trade, vg = snap.latest_quote, snap.latest_trade, snap.greeks
        meta = contracts.get(name)
        row = {
            "fetch_ts_utc": fetched, "source": "alpaca_indicative",
            "underlying": symbol, "underlying_spot": spot, "option_symbol": name,
            "expiration": pd.Timestamp(expiry), "right": cp, "strike": strike,
            "quote_ts": getattr(quote, "timestamp", None),
            "bid": getattr(quote, "bid_price", np.nan), "bid_size": getattr(quote, "bid_size", np.nan),
            "ask": getattr(quote, "ask_price", np.nan), "ask_size": getattr(quote, "ask_size", np.nan),
            "trade_ts": getattr(trade, "timestamp", None),
            "last": getattr(trade, "price", np.nan), "last_size": getattr(trade, "size", np.nan),
            "iv": snap.implied_volatility,
            "vendor_delta": getattr(vg, "delta", np.nan), "vendor_gamma": getattr(vg, "gamma", np.nan),
            "vendor_vega": getattr(vg, "vega", np.nan), "vendor_theta": getattr(vg, "theta", np.nan),
            "vendor_rho": getattr(vg, "rho", np.nan),
            "open_interest": getattr(meta, "open_interest", None),
            "open_interest_date": getattr(meta, "open_interest_date", None),
            "contract_close": getattr(meta, "close_price", None),
            "contract_close_date": getattr(meta, "close_price_date", None),
        }
        row.update(volumes.get(name, {}))
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    exp_close = pd.to_datetime(df["expiration"].dt.strftime("%Y-%m-%d") + " 16:00").dt.tz_localize(NY).dt.tz_convert("UTC")
    df["t_years"] = np.maximum((exp_close - pd.Timestamp(fetched)).dt.total_seconds() /
                                (365.0 * 86400.0), 1.0 / (365.0 * 24.0 * 60.0))
    g = all_greeks(df["underlying_spot"].to_numpy(), df["strike"].to_numpy(),
                   df["t_years"].to_numpy(), pd.to_numeric(df["iv"], errors="coerce").to_numpy(),
                   df["right"].eq("C").to_numpy())
    for key, value in g.items():
        df[key] = value
    df["mid"] = (pd.to_numeric(df["bid"], errors="coerce") +
                 pd.to_numeric(df["ask"], errors="coerce")) / 2.0
    df["quote_age_seconds"] = (pd.Timestamp(fetched) - pd.to_datetime(df["quote_ts"], utc=True)).dt.total_seconds()
    return df


def collect(mode: str, symbols: list[str], dte: int, band: float, force: bool) -> int:
    now = datetime.now(timezone.utc)
    local = pd.Timestamp(now).tz_convert(NY)
    # Scheduled every five minutes for DST safety. Avoid even opening API
    # clients outside a coarse NY-session window; the exchange calendar below
    # handles holidays and half-days precisely.
    if mode == "intraday" and not force:
        if local.weekday() >= 5 or not (dtime(9, 20) <= local.time() <= dtime(16, 10)):
            return 0
    option, stock, trading = _clients()
    if mode == "intraday" and not force and not _market_open(trading, now):
        print("market closed; intraday capture skipped")
        return 0
    if mode == "eod" and not force and not _eod_ready(trading, now):
        print("no completed US session today; EOD capture skipped")
        return 0
    base = OUT / ("alpaca_intraday" if mode == "intraday" else "alpaca_eod")
    ok = True
    for symbol in symbols:
        row = {"fetch_ts_utc": now.isoformat(), "mode": mode, "symbol": symbol}
        try:
            spot = _spot(stock, symbol)
            lo, hi = spot * (1.0 - band), spot * (1.0 + band)
            end_date = now.date() + timedelta(days=dte)
            from alpaca.data.requests import OptionChainRequest
            snaps = option.get_option_chain(OptionChainRequest(
                underlying_symbol=symbol, expiration_date_gte=now.date(),
                expiration_date_lte=end_date, strike_price_gte=lo, strike_price_lte=hi))
            contracts = _contracts(trading, symbol, lo, hi, end_date)
            near = []
            for name in snaps:
                m = OCC.match(name)
                if m:
                    exp = datetime.strptime(m.group(2), "%y%m%d").date()
                    strike = int(m.group(4)) / 1000.0
                    if (exp - now.date()).days <= 7 and abs(strike / spot - 1.0) <= 0.07:
                        near.append(name)
            volumes = _volume_bars(option, near, now)
            captured = datetime.now(timezone.utc)
            df = _normalise(symbol, spot, snaps, contracts, volumes, captured)
            if len(df) < 100:
                raise ValueError(f"chain too small: {len(df)}")
            day_dir = base / captured.strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            path = day_dir / f"{captured:%H%M%S}_{symbol}.parquet"
            df.to_parquet(path, index=False)
            row.update({
                "fetch_ts_utc": captured.isoformat(),
                "spot": spot, "n_rows": len(df), "n_oi": int(df.open_interest.notna().sum()),
                "n_iv": int(df.iv.notna().sum()), "n_quotes": int(df.bid.notna().sum()),
                "n_volume": int(df.get("day_volume", pd.Series(dtype=float)).notna().sum()),
                "min_expiry": str(df.expiration.min().date()), "max_expiry": str(df.expiration.max().date()),
                "path": str(path), "status": "ok",
            })
            print(json.dumps(row, default=str))
        except Exception as exc:
            ok = False
            row["status"] = f"error:{type(exc).__name__}:{str(exc)[:180]}"
            print(json.dumps(row, default=str))
        _append_ledger(row)
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("intraday", "eod"))
    p.add_argument("--symbols", nargs="+", default=list(SYMBOLS), choices=SYMBOLS)
    p.add_argument("--dte", type=int)
    p.add_argument("--band", type=float)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    dte = a.dte if a.dte is not None else (45 if a.mode == "intraday" else 365)
    band = a.band if a.band is not None else (0.12 if a.mode == "intraday" else 0.35)
    return collect(a.mode, a.symbols, dte, band, a.force)


if __name__ == "__main__":
    raise SystemExit(main())
