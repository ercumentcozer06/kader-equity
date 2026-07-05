"""
screen/marketdata_backfill — C1: MarketData.app free token ile SPY+QQQ tarihsel chain OI backfill.
DERİNLİK ÖLÇÜLDÜ (probe): tam 365 gün (free sınırı; 400g→402). intraday GEX sleeve'in OI-bacağı (fiyat-bacağı=
Alpaca bar, ayrı). Bar×OI örtüşmesi ~1 yıl = Ş2B backtest penceresi.

RATE LIMIT 100/gün → 252g×2 sembol=504 istek → ~5 günde tamamlanır (GLOBAL sayaç ≤95/run, RESUMABLE, EN YENİ
tarihten geriye). run_daily'ye bağlanabilir (her gün otomatik ~95 çek). TOKEN: MARKETDATA_TOKEN (.env).
chain_guard şeması: date,expiration,strike,right,open_interest[,iv,delta,mid]. data/historical_chains/md_<tick>.parquet.

  & <venv python> screen/marketdata_backfill.py probe       (derinlik)
  & <venv python> screen/marketdata_backfill.py backfill     (günlük ~95 çek; 5 gün tekrarla / run_daily)
"""
from __future__ import annotations

import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
OUT = ROOT / "data" / "historical_chains"
BASE = "https://api.marketdata.app/v1/options/chain"
SYMS = ["SPY", "QQQ"]
DAILY_CAP = 95           # 100/gün free limit'e saygı (global, tüm semboller)
DEPTH_DAYS = 363         # free OI derinliği ~365g (402 sınırından güvenli geri)


def _token() -> str:
    tok = os.environ.get("MARKETDATA_TOKEN")
    if not tok:
        raise RuntimeError("MARKETDATA_TOKEN yok (.env) → C1 çalışamaz (uydurma yok).")
    return tok


def _fetch_chain(sym: str, d: str, token: str) -> pd.DataFrame | None:
    import requests
    r = requests.get(f"{BASE}/{sym}/", params={"date": d, "token": token}, timeout=30)
    if r.status_code == 402:                                   # kredi/derinlik sınırı
        return "LIMIT"
    if r.status_code not in (200, 203):
        return None
    j = r.json()
    if j.get("s") != "ok":
        return None
    n = len(j.get("strike", []))

    def col(k):
        return j.get(k, [None] * n)
    exp = col("expiration")                                    # unix epoch → tarih
    side = col("side")
    rows = {"date": d, "expiration": [pd.to_datetime(x, unit="s").date().isoformat() if x else None for x in exp],
            "strike": col("strike"), "right": [(str(s).upper()[0] if s else None) for s in side],
            "open_interest": col("openInterest"), "iv": col("iv"), "delta": col("delta"), "mid": col("mid")}
    return pd.DataFrame(rows)


def probe() -> int:
    tok = _token()
    print("C1 PROBE — MarketData SPY OI derinliği:")
    today = date.today()
    for back in (3, 90, 180, 365, 400, 730):
        d = today - timedelta(days=back)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        ch = _fetch_chain("SPY", d.isoformat(), tok)
        n = (ch["open_interest"].notna().sum() if isinstance(ch, pd.DataFrame) else 0)
        is_limit = isinstance(ch, str) and ch == "LIMIT"
        tag = " ← 402 SINIR" if is_limit else (" ← boş" if n == 0 else "")
        print(f"  {back:>4}g ({d}): {'OI '+str(int(n)) if isinstance(ch, pd.DataFrame) else ch}{tag}")
        time.sleep(0.4)
    print("  → free derinlik ~365g (1 yıl); 400g+ 402. Bar×OI örtüşmesi ~1yıl.")
    return 0


def _have_dates(sym):
    p = OUT / f"md_{sym.lower()}.parquet"
    return (set(pd.read_parquet(p)["date"].astype(str)), p) if p.exists() else (set(), p)


def backfill() -> int:
    """EN YENİ tarihten geriye, GLOBAL ≤95/run, RESUMABLE. 5 gün tekrarla → 1 yıl tamamlanır."""
    tok = _token()
    OUT.mkdir(parents=True, exist_ok=True)
    today = date.today()
    # iş günleri: bugün−3 (free 24s-gecikme → son ~2 gün yok) → 363g geriye (en yeni önce)
    days = [d.date() for d in pd.bdate_range(today - timedelta(days=DEPTH_DAYS), today - timedelta(days=3))][::-1]
    used = 0
    stop = False
    consec402 = 0                                              # üst üste 402 = derinlik sınırı (tekil 402 = atla)
    new_rows = {s: [] for s in SYMS}
    for d in days:
        if stop:
            break
        ds = d.isoformat()
        for sym in SYMS:
            have, _ = _have_dates(sym)
            if ds in have:
                continue
            if used >= DAILY_CAP:
                stop = True
                break
            ch = _fetch_chain(sym, ds, tok)
            used += 1
            if isinstance(ch, str) and ch == "LIMIT":
                consec402 += 1
                if consec402 > 10:                            # >5 tarih × 2 sembol → derinlik sınırı, dur
                    print(f"  {sym} {ds}: üst üste 402 → derinlik sınırı, dur."); stop = True; break
                continue                                      # tekil 402 (gecikme/tatil) → atla
            consec402 = 0
            if isinstance(ch, pd.DataFrame) and len(ch):
                new_rows[sym].append(ch)
            time.sleep(0.4)
    for sym in SYMS:
        if not new_rows[sym]:
            continue
        have, p = _have_dates(sym)
        parts = ([pd.read_parquet(p)] if p.exists() else []) + new_rows[sym]
        out = pd.concat(parts, ignore_index=True).drop_duplicates(["date", "expiration", "strike", "right"])
        out.to_parquet(p)
        print(f"  {sym}: +{sum(len(x) for x in new_rows[sym]):,} satır ({len(new_rows[sym])} yeni gün) → {p} "
              f"(toplam {out['date'].nunique()} gün)")
    print(f"  bu run: {used} istek kullanıldı (cap {DAILY_CAP}). RESUME: tekrar çalıştır (kaldığı günden). "
          f"~5 günde 1-yıl tamam → chain_guard QC → yeşilse Ş2B.")
    return 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    cmd = (argv or sys.argv[1:] or ["probe"])[0]
    return probe() if cmd == "probe" else backfill()


if __name__ == "__main__":
    raise SystemExit(main())
