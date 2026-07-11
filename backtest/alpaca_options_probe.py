"""
backtest/alpaca_options_probe — Alpaca free-plan OPSIYON verisi derinlik/kalite probe'u (chain_lab adayi).

Soru: Alpaca'nin bedava (Basic/indicative) planindaki tarihsel opsiyon verisi chain_lab'in
fiyat/IV bacagini besler mi? Olculen:
  A) zincir snapshot alanlari (OI / greeks / IV / quote var mi)
  B) tarihsel gunluk bar derinligi — SURESI GECMIS SPY kontratlari uzerinden (arsiv sinirini bulur)
  C) tarihsel kotasyon (bid/ask → mid) mevcut mu, spread makul mu

Anahtar: .env APCA_API_KEY_ID/SECRET (alpaca_bars_backfill ile ayni). SECRET yoksa NET hata.
Strike secimi: yerel alpaca_spy_1m.parquet'ten (expiry-~30g spot → 5'e yuvarla) — ek istek yok.

  & <kader-macro venv python> backtest/alpaca_options_probe.py
"""
from __future__ import annotations

import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Surelidolmus SPY expiry'leri (aylik/ceyreklik) — arsiv-siniri icin 2024-oncesi de dahil.
EXPIRIES = ["2023-09-15", "2023-12-15",
            "2024-01-19", "2024-02-16", "2024-03-15", "2024-06-21", "2024-09-20", "2024-12-20",
            "2025-03-21", "2025-06-20", "2025-09-19", "2025-12-19",
            "2026-03-20", "2026-05-15"]
PAUSE = 0.35  # 200/dk free limit — saygili


def _client():
    kid = os.environ.get("APCA_API_KEY_ID")
    sec = os.environ.get("APCA_API_SECRET_KEY")
    if not kid or not sec or sec.startswith("REPLACE_ME"):
        raise RuntimeError("APCA_API_SECRET_KEY eksik/placeholder (.env) — probe SONUC URETMEZ.")
    from alpaca.data.historical.option import OptionHistoricalDataClient
    return OptionHistoricalDataClient(kid, sec)


def _spot_near(bars_ts, bars_close, target: date) -> float | None:
    """expiry-35..25g penceresindeki son 1-dk kapanis (yerel parquet, istek yok)."""
    lo = pd.Timestamp(target - timedelta(days=35), tz="UTC")
    hi = pd.Timestamp(target - timedelta(days=25), tz="UTC")
    m = (bars_ts >= lo) & (bars_ts <= hi)
    if not m.any():
        return None
    return float(bars_close[m][-1])


def _occ(expiry: date, strike: float) -> str:
    return f"SPY{expiry:%y%m%d}C{int(round(strike * 1000)):08d}"


def main() -> int:
    from alpaca.data.requests import OptionBarsRequest, OptionChainRequest
    from alpaca.data.timeframe import TimeFrame
    import alpaca.data.requests as _rq

    client = _client()
    print("ALPACA OPSIYON PROBE — free plan, chain_lab adayligi\n")

    # ── A) Zincir snapshot (dar filtre: onumuzdeki cuma) — alan envanteri ──
    today = date.today()
    fri = today + timedelta(days=(4 - today.weekday()) % 7 or 7)
    try:
        ch = client.get_option_chain(OptionChainRequest(
            underlying_symbol="SPY", expiration_date=fri))
        print(f"A) ZINCIR SNAPSHOT (SPY, expiry {fri}): {len(ch)} kontrat")
        if ch:
            k, snap = next(iter(ch.items()))
            has = {f: getattr(snap, f, None) is not None
                   for f in ("latest_quote", "latest_trade", "greeks", "implied_volatility")}
            print(f"   ornek {k}: " + ", ".join(f"{f}={'VAR' if v else 'yok'}" for f, v in has.items()))
            if getattr(snap, "latest_quote", None) is not None:
                q = snap.latest_quote
                print(f"   quote: bid={q.bid_price} ask={q.ask_price} (mid hesaplanabilir)")
    except Exception as ex:
        print(f"A) ZINCIR: HATA {type(ex).__name__}: {str(ex)[:100]}")
    time.sleep(PAUSE)

    # ── B) Tarihsel gunluk bar derinligi — suresi gecmis kontratlar ──
    p = ROOT / "data" / "historical_bars" / "alpaca_spy_1m.parquet"
    bars = pd.read_parquet(p)
    bts = pd.to_datetime(bars.index.get_level_values(-1))
    bcl = bars["close"].to_numpy()
    print("\nB) TARIHSEL BAR DERINLIGI (gunluk bar, expiry-60g → expiry, ATM call):")
    ok_first = None
    for es in EXPIRIES:
        e = date.fromisoformat(es)
        spot = _spot_near(bts, bcl, e)
        if spot is None:
            print(f"   {es}: yerel spot yok — atla")
            continue
        sym = _occ(e, round(spot / 5) * 5)
        s = datetime(e.year, e.month, e.day, tzinfo=timezone.utc) - timedelta(days=60)
        t = datetime(e.year, e.month, e.day, tzinfo=timezone.utc)
        try:
            df = client.get_option_bars(OptionBarsRequest(
                symbol_or_symbols=sym, timeframe=TimeFrame.Day, start=s, end=t)).df
            if df is None or not len(df):
                print(f"   {es} {sym}: 0 bar — ARSIV DISI")
            else:
                d0 = df.index.get_level_values(-1).min().date()
                d1 = df.index.get_level_values(-1).max().date()
                vol = int(df["volume"].sum())
                print(f"   {es} {sym}: {len(df)} bar ({d0}→{d1}), hacim {vol:,}")
                ok_first = ok_first or es
        except Exception as ex:
            print(f"   {es} {sym}: HATA {type(ex).__name__}: {str(ex)[:80]}")
        time.sleep(PAUSE)
    print(f"   → arsiv en erken bar-VAR expiry: {ok_first or 'YOK'}")

    # ── C) Tarihsel KOTASYON (mid bacagi) — 1 kontrat, ogleden 1 dk ──
    print("\nC) TARIHSEL KOTASYON (bid/ask→mid):")
    if not hasattr(_rq, "OptionQuotesRequest"):
        print("   OptionQuotesRequest bu alpaca-py surumunde YOK — mid tarihsel kotasyondan cikmaz;")
        print("   alternatif = gunluk bar close (trade-bazli) veya trades. (SDK guncellemesi denenir.)")
    else:
        e = date.fromisoformat("2025-06-20")
        spot = _spot_near(bts, bcl, e)
        sym = _occ(e, round(spot / 5) * 5)
        s = datetime(2025, 6, 2, 16, 0, tzinfo=timezone.utc)  # 12:00 ET
        try:
            q = client.get_option_quotes(_rq.OptionQuotesRequest(
                symbol_or_symbols=sym, start=s, end=s + timedelta(minutes=1), limit=10)).df
            if q is None or not len(q):
                print(f"   {sym}: 0 kotasyon (free planda tarihsel quote kapali olabilir)")
            else:
                r = q.iloc[0]
                bid, ask = float(r["bid_price"]), float(r["ask_price"])
                mid = (bid + ask) / 2
                print(f"   {sym}: {len(q)} kotasyon; ornek bid={bid} ask={ask} mid={mid:.3f} "
                      f"spread={(ask - bid):.3f} ({(ask - bid) / mid * 100:.1f}% mid)")
        except Exception as ex:
            print(f"   {sym}: HATA {type(ex).__name__}: {str(ex)[:100]}")

    print("\nVERDICT girdileri yukarida — yorum probe'da degil, raporda.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
