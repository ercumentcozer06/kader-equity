"""screen/fetch_0dte_flow — SPY 0DTE (aynı-gün vadeli) AKIŞ paneli, Alpaca options bars'tan (BEDAVA).

Neden: 0DTE çağında OI-bazlı GEX yapısal olarak kör (SPX kısa-vade hacim payı 2017 %38 → 2025 %79.5);
doğru ölçü stok değil AKIŞ. DDOI paralı; Alpaca bars bedava ve 2024-02-02'ye kadar gidiyor.

Ne çeker (gün başına 2 çağrı):
  • 1Day bars  → per-strike gün toplamı: v (kontrat hacmi), vw (VWAP), n (trade sayısı), OHLC
  • 1Hour bars → aynı alanların gün-içi profili (açılış/kapanış akış asimetrisi)
Sembol: SPY{yymmdd}{C|P}{strike*1000:08d}, bant = o günün SPY kapanışı ±%6 (1$ adım).

SINIR (dosyaya not düşülür): historical NBBO quote hiçbir bedava kaynakta YOK (Alpaca /options/quotes → 404,
marketdata.app EOD-only, CBOE/Databento/IVol paralı) → gerçek Lee-Ready İMKÂNSIZ. Bu panel İŞARETSİZ akıştır;
tick-rule istenirse ayrı /options/trades tape'inden ve "quote-doğrulanmamış" etiketiyle kurulur.

  & <kader-macro venv python> screen/fetch_0dte_flow.py [--start 2024-02-02] [--end YYYY-MM-DD]
→ data/option_research/zdte/spy_0dte_daily.parquet   (per-strike, gün)
  data/option_research/zdte/spy_0dte_hourly.parquet  (per-strike, saat)
  data/option_research/zdte/_ledger.csv              (gün, durum, kontrat, hacim — PIT/kapsama denetimi)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUT = ROOT / "data" / "option_research" / "zdte"
BASE = "https://data.alpaca.markets/v1beta1/options/bars"
HDR = {"APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"],
       "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"]}
FIRST_DAY = "2024-02-02"        # Alpaca opsiyon geçmişinin başı (2024-01-05 ve öncesi boş — probe edildi)
BAND = 0.06                     # spot ±%6; 0DTE hacmi bu bandın dışında ihmal edilebilir
PAUSE = 0.35                    # free tier ~200 req/dk


def _bars(syms: list[str], day: str, timeframe: str) -> list[tuple]:
    """Tek gün + tek timeframe için tüm sayfaları topla."""
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    out, token, guard = [], None, 0
    while True:
        p = {"symbols": ",".join(syms), "start": day, "end": end,
             "timeframe": timeframe, "limit": 10000}
        if token:
            p["page_token"] = token
        r = requests.get(BASE, headers=HDR, params=p, timeout=60)
        r.raise_for_status()
        j = r.json()
        for s, bars in (j.get("bars") or {}).items():
            for b in bars:
                out.append((day, s, b["t"], b["o"], b["h"], b["l"], b["c"], b["v"], b["vw"], b["n"]))
        token = j.get("next_page_token")
        guard += 1
        if not token or guard > 20:
            break
        time.sleep(PAUSE)
    return out


COLS = ["date", "sym", "t", "o", "h", "l", "c", "v", "vw", "n"]


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["cp"] = df["sym"].str[9]
    df["strike"] = df["sym"].str[10:].astype(int) / 1000.0
    df["premium"] = df["vw"] * df["v"] * 100.0        # $ akış
    df["avg_lot"] = df["v"] / df["n"].clip(lower=1)   # retail/kurumsal ayak izi
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=FIRST_DAY)
    ap.add_argument("--end", default=None)
    ap.add_argument("--no-hourly", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    import yfinance as yf
    end = a.end or (pd.Timestamp.today() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    spy = yf.download("SPY", start="2024-01-01", end=(pd.Timestamp(end) + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
                      progress=False, auto_adjust=False)["Close"]
    spy = spy.iloc[:, 0] if hasattr(spy, "columns") else spy
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    days = [d for d in spy.index if pd.Timestamp(a.start) <= d <= pd.Timestamp(end)]

    # devam-edilebilir: mevcut parquet'teki günleri atla
    fd, fh = OUT / "spy_0dte_daily.parquet", OUT / "spy_0dte_hourly.parquet"
    done: set[str] = set()
    if fd.exists():
        done = set(pd.read_parquet(fd, columns=["date"])["date"].unique())
    todo = [d for d in days if d.strftime("%Y-%m-%d") not in done]
    print(f"islem gunu {len(days)}  zaten var {len(done)}  cekilecek {len(todo)}")

    daily, hourly, ledger = [], [], []
    t0 = time.time()
    for i, d in enumerate(todo, 1):
        ds = d.strftime("%Y-%m-%d")
        S = float(spy.loc[d])
        lo, hi = int(round(S * (1 - BAND))), int(round(S * (1 + BAND)))
        syms = [f"SPY{d.strftime('%y%m%d')}{cp}{k*1000:08d}" for k in range(lo, hi + 1) for cp in "CP"]
        try:
            rows = _bars(syms, ds, "1Day")
            time.sleep(PAUSE)
            hrows = [] if a.no_hourly else _bars(syms, ds, "1Hour")
            time.sleep(PAUSE)
        except Exception as e:
            ledger.append((ds, S, "ERROR", 0, 0, f"{type(e).__name__}: {str(e)[:60]}"))
            print(f"  [!] {ds} {type(e).__name__} {str(e)[:60]}")
            continue
        daily += rows
        hourly += hrows
        vol = sum(r[7] for r in rows)
        ledger.append((ds, S, "ok" if rows else "EMPTY", len({r[1] for r in rows}), int(vol), ""))
        if i % 25 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  {ds}  kontrat={len({r[1] for r in rows}):3d} hacim={int(vol):>9,}  "
                  f"gecen={el/60:.1f}dk  kalan~{el/i*(len(todo)-i)/60:.1f}dk", flush=True)

    def _merge(new_rows, path, tf):
        if not new_rows:
            return 0
        df = _enrich(pd.DataFrame(new_rows, columns=COLS))
        if path.exists():
            df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
        df = df.drop_duplicates(["date", "sym", "t"]).sort_values(["date", "sym", "t"])
        df.attrs["note"] = "ISARETSIZ akis; historical NBBO quote bedava yok -> Lee-Ready imkansiz"
        df.to_parquet(path, index=False)
        print(f"  -> {path.name}: {len(df):,} satir ({tf})")
        return len(df)

    _merge(daily, fd, "1Day")
    _merge(hourly, fh, "1Hour")
    if ledger:
        lg = pd.DataFrame(ledger, columns=["date", "spot", "status", "n_contracts", "volume", "err"])
        lp = OUT / "_ledger.csv"
        if lp.exists():
            lg = pd.concat([pd.read_csv(lp), lg], ignore_index=True).drop_duplicates("date", keep="last")
        lg.sort_values("date").to_csv(lp, index=False)
    (OUT / "_fetch_ts.txt").write_text(datetime.now(timezone.utc).isoformat())
    print(f"BITTI  {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
