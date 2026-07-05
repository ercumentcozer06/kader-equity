"""
backtest/remeasure/R0_backfill — FAZ-R / R0: ENSTRÜMAN ONARIMI ham-veri yakalama (zaman-kritik).
D-FAZ enstrüman kırıktı (backtest gamma$'ın %10-11'ini görüyor; kök=marketdata_backfill.py:42 expiration omit).
Bu script TÜM-EXPIRY full-chain'i 4 sembol (SPY,QQQ,SPX,NDX) × D-FAZ'ın 243 tarihi için çeker, HAM yanıtı diske
gzip-cache eder (provenance; eski tek-expiry md_*.parquet SİLİNMEZ). EN-ESKİ-ÖNCE (free 1-yıl rolling pencere
Haziran-2025'i düşürmek üzere). Idempotent + resume-able (dosya varsa atla; kredi biterse dur, reset sonrası devam).
Cap YOK (spec). Kredi: ~57/gün (SPY9+QQQ9+SPX23+NDX16); 10k/gün limit → ~165 gün/gün-bütçe; ~2 günde tamam.
Her dosyaya _fetch_ts (PIT ledger). R1 bu ham veriden iki level-serisi (LIVE-MATCH N_EXP=5 + FULL-SURFACE) kurar.
  & <venv python> backtest/remeasure/R0_backfill.py
→ data/raw_chains/{SPY,QQQ,SPX,NDX}/<date>.json.gz
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

TOK = os.environ["MARKETDATA_TOKEN"]
BASE = "https://api.marketdata.app/v1/options/chain"
SYMS = ["SPY", "QQQ", "SPX", "NDX"]                 # ETF + index (③ havuz için)
RAW = ROOT / "data" / "raw_chains"
SAFETY = 200                                        # kalan kredi bu eşiğin altına inince dur (tampon)


def fetch(sym: str, d: str):
    r = requests.get(f"{BASE}/{sym}/", params={"date": d, "expiration": "all", "token": TOK}, timeout=120)
    rem = int(r.headers.get("X-Api-Ratelimit-Remaining", "-1"))
    if r.status_code not in (200, 203):
        return None, rem, r.status_code
    return r.json(), rem, r.status_code


def _save(sym, d, j):
    p = RAW / sym / f"{d}.json.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")                          # atomik yazım (yarım-dosya yarış-koşulu yok)
    obj = {"_sym": sym, "_date": d, "_fetch_ts": datetime.now(timezone.utc).isoformat(),
           "_endpoint": "options/chain expiration=all", "resp": j}
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f)
    tmp.replace(p)


def main():
    import concurrent.futures as cf
    import threading
    from datetime import date, timedelta
    # CLI: [SYM1,SYM2] [--workers N]  (FAZ-R wave-0: sembol-başına paralel fetcher)
    args = sys.argv[1:]
    syms = SYMS
    workers = 8
    for a in args:
        if a.startswith("--workers"):
            workers = int(a.split("=")[1]) if "=" in a else 2
        elif not a.startswith("--"):
            syms = [s.strip().upper() for s in a.split(",")]
    md = pd.read_parquet(ROOT / "data" / "historical_chains" / "md_spy.parquet")
    dates = [d.date().isoformat() for d in sorted(pd.to_datetime(md["date"].unique()))]    # EN-ESKİ-ÖNCE
    # RC0 uzatma: son-md-tarihinden bugün-1'e iş günleri (06-09/06-10 → RC2.8 canlı-uyum overlap'i; panel-DIŞI)
    ext = [d.date().isoformat() for d in pd.bdate_range(pd.Timestamp(dates[-1]) + timedelta(days=1),
                                                        pd.Timestamp(date.today()) - timedelta(days=1))]
    dates = dates + ext
    work = [(d, sym) for d in dates for sym in syms if not (RAW / sym / f"{d}.json.gz").exists()]
    have = len(dates) * len(syms) - len(work)
    st = {"rem": 99999, "stop": False, "fetched": 0, "warn": 0}
    lock = threading.Lock()
    t0 = time.time()
    print(f"PARALEL backfill [{','.join(syms)}]: {len(work)} iş kaldı ({have} zaten var), {workers} worker, EN-ESKİ-ÖNCE.")

    def worker(d, sym):
        if st["stop"]:
            return
        with lock:
            if st["rem"] != -1 and st["rem"] < SAFETY:
                st["stop"] = True
                return
        try:
            j, rem, code = fetch(sym, d)
        except Exception:
            with lock:
                st["warn"] += 1
            return
        with lock:
            if rem != -1:
                st["rem"] = min(st["rem"], rem)
        if code in (402, 429):
            with lock:
                st["stop"] = True
            return
        if j is None:
            with lock:
                st["warn"] += 1
            return
        _save(sym, d, j)
        with lock:
            st["fetched"] += 1
            if st["fetched"] % 25 == 0:
                print(f"  ...{st['fetched']}/{len(work)} | remaining≈{st['rem']} | son {sym} {d} | {time.time()-t0:.0f}s")

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(worker, d, sym) for (d, sym) in work]
        for _ in cf.as_completed(futs):
            pass
    tag = "STOP (kredi/limit düşük)" if st["stop"] else "TAMAM"
    print(f"{tag}: fetched={st['fetched']}, warn={st['warn']}, remaining≈{st['rem']}, süre={time.time()-t0:.0f}s")
    print("  resume: tekrar çalıştır (idempotent, kalan günlerden devam).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
