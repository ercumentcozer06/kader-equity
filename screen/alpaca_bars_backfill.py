"""
screen/alpaca_bars_backfill — Ş2B 1-dk SPY+QQQ bar backfill (Alpaca free, IEX feed). intraday GEX sleeve'in
fiyat-bacağı (OI-bacağı = MarketData backfill, ayrı). .env'den APCA_API_KEY_ID/SECRET (alpaca-py).
SECRET placeholder/eksikse NET hata (uydurma yok).

  probe()    : IEX 1-dk bar kaç yıl geriye gidiyor — derinlik (birkaç istek).
  backfill() : start..end aylık-chunk, RESUMABLE → data/historical_bars/alpaca_<tick>_1m.parquet.

  & <kader-macro venv python> screen/alpaca_bars_backfill.py probe
  & <kader-macro venv python> screen/alpaca_bars_backfill.py backfill
"""
from __future__ import annotations

import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
OUT = ROOT / "data" / "historical_bars"
SYMS = ["SPY", "QQQ"]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _client():
    kid = os.environ.get("APCA_API_KEY_ID")
    sec = os.environ.get("APCA_API_SECRET_KEY")
    if not kid or not sec or sec.startswith("REPLACE_ME"):
        raise RuntimeError(
            "APCA_API_SECRET_KEY eksik/placeholder (.env). Emir gerçek secret string'i .env'e yapıştırmalı "
            "(mesajda '(tam metin)' placeholder geldi). Secret gelene dek backfill SONUÇ ÜRETMEZ.")
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(kid, sec)


def _fetch(client, syms, start, end):
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    req = StockBarsRequest(symbol_or_symbols=syms, timeframe=TimeFrame.Minute,
                           start=start, end=end, feed=DataFeed.IEX)
    df = client.get_stock_bars(req).df
    return df if df is not None and len(df) else None


def probe() -> int:
    client = _client()
    print("Ş2B PROBE — Alpaca IEX 1-dk bar derinliği (SPY):")
    today = date.today()
    for back_y in (0, 1, 2, 3, 5, 7):
        y = today.year - back_y
        s = datetime(y, 1, 2, tzinfo=timezone.utc); e = datetime(y, 1, 3, tzinfo=timezone.utc)
        try:
            df = _fetch(client, ["SPY"], s, e)
            n = 0 if df is None else (len(df.loc["SPY"]) if "SPY" in df.index.get_level_values(0) else len(df))
            print(f"  {y}-01-02: {'BAR VAR ('+str(n)+' dk)' if n else 'YOK/erişilemez'}")
        except Exception as ex:
            print(f"  {y}: HATA {type(ex).__name__}: {str(ex)[:70]}")
        time.sleep(0.3)
    print("  → En eski bar-var yılı = IEX-free derinliği (Ş2A ajanı '7+ yıl' demişti, burada KESİN ölç).")
    return 0


def backfill(start: str = "2020-09-01", end: str = None) -> int:    # IEX-free derinlik ~2020-12+/2021 (probe)
    client = _client()
    OUT.mkdir(parents=True, exist_ok=True)
    end = end or (date.today()).isoformat()
    for sym in SYMS:
        p = OUT / f"alpaca_{sym.lower()}_1m.parquet"
        have_months = set()
        parts = []
        if p.exists():
            ex = pd.read_parquet(p); parts.append(ex)
            have_months = set(pd.to_datetime(ex.index.get_level_values(-1)).strftime("%Y-%m")) if len(ex) else set()
        months = pd.date_range(start, end, freq="MS")
        # Merely seeing a month in the parquet does not mean that month is
        # complete. The old resumable logic skipped a current month forever after
        # its first partial fetch (the local files therefore stopped at
        # 2026-06-10). Re-fetch the final two calendar months on every run and
        # de-duplicate; this is only two API calls per symbol and also repairs a
        # truncated previous month after the calendar rolls forward.
        refresh_months = set(m.strftime("%Y-%m") for m in months[-2:])
        n_new = 0
        for m in months:
            mk = m.strftime("%Y-%m")
            if mk in have_months and mk not in refresh_months:
                continue
            s = m.tz_localize("UTC"); e = (m + pd.offsets.MonthEnd(1)).tz_localize("UTC")
            try:
                df = _fetch(client, [sym], s, e)
            except Exception as ex:
                print(f"  {sym} {mk}: HATA {type(ex).__name__}: {str(ex)[:60]} — atla")
                continue
            if df is not None and len(df):
                parts.append(df if sym not in df.index.get_level_values(0) else df)
                n_new += 1
                if n_new % 12 == 0:
                    print(f"  {sym}: {n_new} ay çekildi...")
            time.sleep(0.35)                                  # 200/dk free limit — saygılı
        if parts:
            out = pd.concat(parts).sort_index()
            out = out[~out.index.duplicated(keep="last")]
            out.to_parquet(p)
            print(f"  {sym}: +{n_new} ay → {p} (toplam {len(out):,} bar)")
    print("  resume: tekrar çalıştır (kaldığı aydan). Bittiğinde derinlik raporu için: probe.")
    return 0


def main(argv=None) -> int:
    cmd = (argv or sys.argv[1:] or ["probe"])[0]
    return probe() if cmd == "probe" else backfill()


if __name__ == "__main__":
    raise SystemExit(main())
