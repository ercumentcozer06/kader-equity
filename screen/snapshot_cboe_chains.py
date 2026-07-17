"""Daily append-only CBOE option-chain archive for forward GEX research.

The former scheduled collector used MarketData.app.  Its free daily credit
budget can fetch roughly one full chain, so SPY advanced while QQQ/SPX/NDX
silently fell behind.  CBOE's delayed-quotes endpoint is already the live GEX
engine's source and exposes OI, IV and Greeks for the complete listed chain.

Each successful run stores the unmodified vendor response under
``data/raw_chains_cboe/<symbol>/<vendor-date>.json.gz`` and appends a QC row to
``pit_ledger.csv``.  Existing daily files are validated and never overwritten.
One symbol failing does not prevent the remaining symbols from being captured;
the process exits non-zero unless all four archives are healthy.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    from screen._cboe_lib import fetch
except ImportError:
    from _cboe_lib import fetch


SYMBOLS = {"SPY": "SPY", "QQQ": "QQQ", "SPX": "_SPX", "NDX": "_NDX"}
OUT = ROOT / "data" / "raw_chains_cboe"
LEDGER = OUT / "pit_ledger.csv"
RUNS = OUT / "runs"
LEDGER_COLS = [
    "symbol", "source_symbol", "as_of", "fetch_ts_utc", "n_contracts",
    "n_with_oi", "n_with_iv", "n_with_gamma", "sha256", "status", "path",
]


def _vendor_date(payload: dict) -> str:
    raw = str(payload.get("timestamp") or "")
    try:
        return datetime.strptime(raw[:10], "%Y-%m-%d").date().isoformat()
    except ValueError:
        return datetime.now(timezone.utc).date().isoformat()


def _quality(payload: dict) -> dict[str, int]:
    data = payload.get("data", payload)
    options = data.get("options") or []
    if not isinstance(options, list) or len(options) < 1000:
        raise ValueError(f"chain too small: {len(options) if isinstance(options, list) else 'not-list'}")
    spot = data.get("current_price")
    if spot in (None, "") or float(spot) <= 0:
        raise ValueError(f"invalid spot: {spot}")
    return {
        "n_contracts": len(options),
        "n_with_oi": sum(o.get("open_interest") not in (None, "") for o in options),
        "n_with_iv": sum(o.get("iv") not in (None, "") for o in options),
        "n_with_gamma": sum(o.get("gamma") not in (None, "") for o in options),
    }


def _encode(record: dict) -> tuple[bytes, str]:
    raw = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return gzip.compress(raw, compresslevel=6, mtime=0), hashlib.sha256(raw).hexdigest()


def _read_existing(path: Path) -> tuple[dict, str]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        record = json.load(fh)
    payload = record.get("response")
    if not isinstance(payload, dict):
        raise ValueError("existing archive has no response object")
    qc = _quality(payload)
    raw = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return qc, hashlib.sha256(raw).hexdigest()


def _append_ledger(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    with LEDGER.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LEDGER_COLS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LEDGER_COLS})


def configure_session(session: str) -> None:
    """Select a physically separate archive for the post-close EOD capture."""
    global OUT, LEDGER, RUNS
    if session not in {"intraday", "eod"}:
        raise ValueError(f"unknown capture session: {session}")
    name = "raw_chains_cboe" if session == "intraday" else "raw_chains_cboe_eod"
    OUT = ROOT / "data" / name
    LEDGER = OUT / "pit_ledger.csv"
    RUNS = OUT / "runs"


def collect(symbol: str, source_symbol: str) -> dict:
    fetched_at = datetime.now(timezone.utc).isoformat()
    try:
        payload = fetch(source_symbol, timeout=60)
        qc = _quality(payload)
        as_of = _vendor_date(payload)
        path = OUT / symbol / f"{as_of}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing_qc, digest = _read_existing(path)
            status = "exists_valid"
            qc = existing_qc
        else:
            record = {
                "symbol": symbol,
                "source_symbol": source_symbol,
                "as_of": as_of,
                "fetch_ts_utc": fetched_at,
                "endpoint": f"https://cdn.cboe.com/api/global/delayed_quotes/options/{source_symbol}.json",
                "response": payload,
            }
            blob, digest = _encode(record)
            tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
            tmp.write_bytes(blob)
            os.replace(tmp, path)
            status = "ok"
        row = {
            "symbol": symbol, "source_symbol": source_symbol, "as_of": as_of,
            "fetch_ts_utc": fetched_at, **qc, "sha256": digest,
            "status": status, "path": str(path),
        }
    except Exception as exc:  # fail-loud after giving every symbol a chance
        row = {
            "symbol": symbol, "source_symbol": source_symbol, "as_of": "",
            "fetch_ts_utc": fetched_at, "n_contracts": 0, "n_with_oi": 0,
            "n_with_iv": 0, "n_with_gamma": 0, "sha256": "",
            "status": f"error:{type(exc).__name__}:{str(exc)[:160]}", "path": "",
        }
    _append_ledger(row)
    return row


def main() -> int:
    started = datetime.now(timezone.utc)
    results = [collect(symbol, source) for symbol, source in SYMBOLS.items()]
    ok = all(r["status"] in {"ok", "exists_valid"} for r in results)
    summary = {
        "job": "daily_cboe_full_chain_archive",
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "results": results,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    run_path = RUNS / f"{started:%Y%m%dT%H%M%SZ}.json"
    run_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    session = "eod" if "--session" in sys.argv and "eod" in sys.argv else "intraday"
    configure_session(session)
    raise SystemExit(main())
