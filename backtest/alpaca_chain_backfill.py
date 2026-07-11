"""
backtest/alpaca_chain_backfill — Alpaca opsiyon GUNLUK bar'larindan tarihsel zincir backfill'i (2024-01-18+).

Amac: chain_lab panelini 2024'e geriye uzatmak. MarketData zinciri (md_*.parquet) ile AYNI metodoloji:
tarih basina TEK on-aylik expiry (3.Cuma, tatilde Persembe'ye kayar; DTE 0-25 dongusu). Fark (PROXY):
  mid  yerine bar CLOSE (son islem — gercek mid TARIHSEL OLARAK YOK, probe: quotes ucu 404)
  OI   yerine bar VOLUME (Alpaca OI vermiyor)
Ham kolonlar ham adiyla saklanir (close/volume/vwap) — 'mid/oi' diye YENIDEN ADLANDIRILMAZ; esleme
level-serisi kurulurken acikca yapilir (build tarafinda PROXY etiketiyle). Kontrat evreni: expiry-donemi
spot araligi ±%16, $1 adim, C+P (BAND=0.15 filtresinin ustkumesi).

RESUMABLE: expiry bazinda → data/historical_chains/alpaca_chain_{sym}.parquet
  & <kader-macro venv python> backtest/alpaca_chain_backfill.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv(ROOT / ".env")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from build_level_series import _daily_spot     # noqa: E402  (spot+takvim tek kaynak)

OUT = ROOT / "data" / "historical_chains"
START = date(2024, 1, 18)                      # Alpaca opsiyon-bar arsivi burada basliyor (probe B)
END = date(2026, 6, 8)                         # md level_series sonu — ortusme dogrulamasi icin ayni uc
SYMS = ["SPY", "QQQ"]
PAUSE = 0.3
BATCH = 90


def _client():
    kid = os.environ.get("APCA_API_KEY_ID")
    sec = os.environ.get("APCA_API_SECRET_KEY")
    if not kid or not sec or sec.startswith("REPLACE_ME"):
        raise RuntimeError("APCA_API_SECRET_KEY eksik/placeholder (.env).")
    from alpaca.data.historical.option import OptionHistoricalDataClient
    return OptionHistoricalDataClient(kid, sec)


def _third_friday(y: int, m: int) -> date:
    d = date(y, m, 15)
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d


def expiry_for(D: date, cal: set) -> date:
    """On aylik expiry (md kuralinin kopyasi): 3.Cuma; takvimde yoksa (tatil) 1 gun geri; E<D ise sonraki ay."""
    y, m = D.year, D.month
    for _ in range(2):
        E = _third_friday(y, m)
        if cal and E <= max(cal) and E not in cal:
            E -= timedelta(days=1)             # Good Friday / Juneteenth → Persembe expiry (md'de dow=3 var)
        if E >= D:
            return E
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    raise RuntimeError(f"expiry bulunamadi: {D}")


def _occ(sym: str, E: date, right: str, K: int) -> str:
    return f"{sym}{E:%y%m%d}{right}{K * 1000:08d}"


def backfill(sym: str, client) -> None:
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame

    spot = _daily_spot(sym)
    cal = set(spot.index)
    days = [d for d in sorted(cal) if START <= d <= END]
    by_exp: dict[date, list[date]] = {}
    for D in days:
        by_exp.setdefault(expiry_for(D, cal), []).append(D)

    p = OUT / f"alpaca_chain_{sym.lower()}.parquet"
    parts, done = [], set()
    if p.exists():
        ex = pd.read_parquet(p)
        parts.append(ex)
        done = set(pd.to_datetime(ex["expiration"]).dt.date.unique())

    todo = [E for E in sorted(by_exp) if E not in done]
    print(f"{sym}: {len(by_exp)} expiry ({len(done)} hazir, {len(todo)} cekilecek), {len(days)} islem gunu")
    for E in todo:
        ds = by_exp[E]
        lo, hi = min(float(spot[d]) for d in ds), max(float(spot[d]) for d in ds)
        ks = range(int(lo * 0.84), int(hi * 1.16) + 2)
        symbols = [_occ(sym, E, r, k) for k in ks for r in ("C", "P")]
        s = datetime(ds[0].year, ds[0].month, ds[0].day, tzinfo=timezone.utc)
        t = datetime(E.year, E.month, E.day, tzinfo=timezone.utc) + timedelta(days=1)
        frames = []
        for i in range(0, len(symbols), BATCH):
            try:
                df = client.get_option_bars(OptionBarsRequest(
                    symbol_or_symbols=symbols[i:i + BATCH], timeframe=TimeFrame.Day, start=s, end=t)).df
                if df is not None and len(df):
                    frames.append(df.reset_index())
            except Exception as exn:
                print(f"  {sym} {E} batch{i // BATCH}: HATA {type(exn).__name__}: {str(exn)[:70]}")
            time.sleep(PAUSE)
        if frames:
            raw = pd.concat(frames, ignore_index=True)
            ts = pd.to_datetime(raw["timestamp"]).dt.tz_convert("America/New_York")
            n = len(sym)
            out = pd.DataFrame({
                "date": pd.to_datetime(ts.dt.date),
                "expiration": pd.Timestamp(E),
                "strike": raw["symbol"].str[n + 7:].astype(int) / 1000.0,
                "right": raw["symbol"].str[n + 6],
                "close": raw["close"].astype(float),
                "volume": raw["volume"].astype(float),
                "vwap": raw.get("vwap", pd.Series([float("nan")] * len(raw))).astype(float),
            })
            parts.append(out)
            print(f"  {sym} {E}: {len(out)} kontrat-gun ({len(ds)} islem gunu, {len(symbols)} sembol denendi)")
        else:
            print(f"  {sym} {E}: 0 satir — BOS")
        merged = pd.concat(parts, ignore_index=True).drop_duplicates(
            subset=["date", "expiration", "strike", "right"], keep="last")
        merged.to_parquet(p)                    # her expiry'de yaz → resumable
        parts = [merged]

    # veri sayimi (census) — sonuçtan once kapsam
    fin = pd.read_parquet(p)
    per_day = fin.groupby("date").size()
    print(f"{sym} CENSUS: {len(fin):,} satir, {per_day.index.min().date()}→{per_day.index.max().date()}, "
          f"{len(per_day)}/{len(days)} gun, kontrat/gun medyan {int(per_day.median())}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    client = _client()
    for sym in SYMS:
        backfill(sym, client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
