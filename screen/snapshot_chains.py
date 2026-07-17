"""
screen/snapshot_chains.py — legacy MarketData collector / scheduled-task compatibility entry point.

IMPORTANT (2026-07-16): the Windows task still points at this filename because
its ACL is administrator-owned. The ``__main__`` block delegates to
``snapshot_cboe_chains.py`` so all four full chains are archived without the
MarketData free-credit bottleneck. The functions below remain available only
for historical MarketData backfill compatibility.

RC0.3 GÜNLÜK SNAPSHOT JOB (FAZ-R sonrası da yaşar).

Her çağrıda config.SYMS (4 sembol) için EN-YENİ-MEVCUT işlem-gününü çeker:
bugün-1 işlem-gününden geriye; 402 (veya başka hata) → bir işlem-günü daha geri; max MAX_TRIES deneme.
expiration=all full-chain → data/raw_chains/<sym>/<date>.json.gz — R0_backfill._save ile AYNI format
(_sym,_date,_fetch_ts,_endpoint,resp; atomik tmp→replace; dosya varsa atla): _save fonksiyonunun KENDİSİ
import edilip kullanılır (tek kaynak, format kopyalanmaz). Her fetch-denemesi data/raw_chains/pit_ledger.csv'ye
satır ekler (sym,date,fetch_ts_utc,n_contracts,status). Token .env'den (ASLA ekrana basılmaz).

ZAMANLAMA (D4 hükmü, backtest/DIAGNOSIS.md "D4"): OI[D] OCC gece-batch → ertesi sabah ~09:30-09:45 ET
yayınlanır, D-akşamı bile 402; updated damgaları 16:00 ET EOD-settle. → job günlük 18:00 yerel
(Istanbul ≈ 11:00 ET), OI-yayınından güvenli marjla SONRA.

  & C:/Users/admin/Downloads/kader-macro/.venv/Scripts/python.exe screen/snapshot_chains.py

Çıkış: stdout'a JSON özet (config_sha dahil). Exit 0 = tüm semboller ok/exists; 1 = en az bir fail (fail-loud).
Ham cache append-only: hiçbir dosya silinmez/üzerine yazılmaz (varsa atla).
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                    # kader-equity kökü
sys.path.insert(0, str(ROOT / "backtest" / "remeasure"))

import config                                                  # TEK-GERÇEK-KAYNAK (semboller/yollar/SAFETY)
import R0_backfill as r0                                       # fetch + _save AYNI format (tek kaynak; .env'i kendi yükler)

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

LEDGER = config.RAW_DIR / "pit_ledger.csv"
LEDGER_COLS = ["sym", "date", "fetch_ts_utc", "n_contracts", "status"]
MAX_TRIES = 3                                                  # RC0.3 spec: bugün-1'den geriye max 3 işlem-günü


def _ledger_append(sym: str, d: str, n: int, status: str) -> list[str]:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    row = [sym, d, datetime.now(timezone.utc).isoformat(), str(n), status]
    with open(LEDGER, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LEDGER_COLS)
        w.writerow(row)
    return row


def _candidate_dates() -> list[str]:
    """Bugün-1'den geriye MAX_TRIES işlem günü, EN-YENİ-ÖNCE."""
    end = pd.Timestamp(date.today()) - timedelta(days=1)
    bdays = pd.bdate_range(end=end, periods=MAX_TRIES)
    return [d.date().isoformat() for d in reversed(bdays)]


def run_symbol(sym: str, dates: list[str], state: dict) -> dict:
    for d in dates:
        p = config.RAW_DIR / sym / f"{d}.json.gz"
        if p.exists():                                         # dosya varsa atla (append-only, idempotent)
            row = _ledger_append(sym, d, -1, "exists")
            return {"sym": sym, "date": d, "status": "exists", "path": str(p), "ledger_row": row}
        try:
            j, rem, code = r0.fetch(sym, d)                    # R0 ile AYNI endpoint + expiration=all
        except Exception as e:
            _ledger_append(sym, d, 0, f"error:{type(e).__name__}")
            continue
        if rem != -1:
            state["rem"] = min(state.get("rem", 10**9), rem)
            if rem < config.SAFETY_CREDITS:
                state["low_credit"] = True
        if code in (200, 203) and j is not None:
            r0._save(sym, d, j)                                # AYNI format: _sym,_date,_fetch_ts,_endpoint,resp; tmp→replace
            n = len(j.get("optionSymbol") or [])
            row = _ledger_append(sym, d, n, "ok")
            return {"sym": sym, "date": d, "status": "ok", "n_contracts": n, "path": str(p), "ledger_row": row}
        _ledger_append(sym, d, 0, f"http_{code}")              # 402 → bir işlem-günü daha geri (max MAX_TRIES)
        if state.get("low_credit"):
            break
    return {"sym": sym, "status": "fail", "tried": dates}


def main() -> int:
    dates = _candidate_dates()
    state: dict = {}
    results = []
    for sym in config.SYMS:
        if state.get("low_credit"):
            results.append({"sym": sym, "status": "skipped_low_credit"})
            continue
        results.append(run_symbol(sym, dates, state))
    ok = all(r["status"] in ("ok", "exists") for r in results)
    summary = {
        "job": "RC0.3 snapshot_chains",
        "run_ts_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_dates": dates,
        "results": results,
        "credits_remaining_approx": state.get("rem"),
        "ledger": str(LEDGER),
        "config_sha": config.config_sha(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    # The MarketData.app free-credit budget cannot capture all four full
    # chains every day (SPY consumed the allowance and the remaining symbols
    # drifted stale).  Keep this legacy task entry point so the existing
    # Windows task does not need administrator-only ACL changes, but route
    # scheduled runs to the complete CBOE archive used by the live GEX engine.
    try:
        from screen.snapshot_cboe_chains import main as cboe_main
    except ImportError:
        from snapshot_cboe_chains import main as cboe_main
    raise SystemExit(cboe_main())
