"""Post-close capture and prospective GEX validation pipeline.

Run at 00:20 Europe/Istanbul. During U.S. daylight time this is 17:20 ET; during
standard time it is 16:20 ET, so the regular session is complete in both cases.
Friday EOD occurs on Saturday local time, hence the ET-date guard.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LEDGER = ROOT / "output" / "gex_forward_eod_levels.parquet"


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=600)
    if p.returncode:
        raise RuntimeError(f"command failed rc={p.returncode}: {' '.join(cmd)}\n{p.stderr[-1000:]}")
    if p.stdout:
        print(p.stdout[-800:])


def latest_snapshot(sym: str) -> dict:
    folder = ROOT / "data" / "cache" / f"gamma_{sym.lower()}"
    files = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no gamma snapshot for {sym}")
    return json.loads(files[-1].read_text(encoding="utf-8"))


def main() -> int:
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:
        print(f"skip: ET weekend {now_et.date()}")
        return 0
    session_date = pd.Timestamp(now_et.date())
    run([PY, str(ROOT / "screen" / "snapshot_cboe_chains.py"), "--session", "eod"])
    rows = []
    for sym in ("SPY", "QQQ"):
        run([PY, str(ROOT / "screen" / "gamma_engine.py"), sym])
        s = latest_snapshot(sym)
        rows.append({
            "as_of": session_date, "ticker": sym,
            "capture_ts_utc": datetime.now(timezone.utc).isoformat(),
            "underlying": s.get("underlying"), "spot": s.get("spot"),
            "put_wall": s.get("put_wall"), "call_wall": s.get("call_wall"),
            "ghost": s.get("ghost"), "gamma_flip": s.get("gex_flip"),
            "max_pain": s.get("max_pain"), "hvl": s.get("hvl"),
            "atm_iv_30d": s.get("atm_iv_30d"), "regime": s.get("regime"),
            "source": "cboe_post_close",
        })
    new = pd.DataFrame(rows)
    if LEDGER.exists():
        old = pd.read_parquet(LEDGER)
        new = pd.concat([old, new], ignore_index=True)
    new["as_of"] = pd.to_datetime(new["as_of"])
    # capture_ts_utc: yeni satirlar .isoformat() ile STRING uretir, eski defter datetime64[UTC] tutar
    # -> concat'ta object-mix -> pyarrow to_parquet coker (ArrowTypeError). Yazmadan once tek-tipe zorla.
    new["capture_ts_utc"] = pd.to_datetime(new["capture_ts_utc"], utc=True, errors="coerce")
    new = new.sort_values(["as_of", "ticker", "capture_ts_utc"]).drop_duplicates(
        ["as_of", "ticker"], keep="last")
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(LEDGER, index=False)
    print(f"EOD level ledger: {len(new)} rows -> {LEDGER}")

    # Refresh complete price paths, then score every EOD level for which the next
    # session is now present. The bar collector re-fetches the final two months.
    run([PY, str(ROOT / "screen" / "alpaca_bars_backfill.py"), "backfill"])
    run([PY, str(ROOT / "backtest" / "gex_master" / "forward_validation.py")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
