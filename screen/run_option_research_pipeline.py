"""Scheduled wrapper for capture + dynamic feature materialisation."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from screen.collect_option_research import collect  # noqa: E402
from backtest.gex_master.build_dynamic_option_features import main as build_features  # noqa: E402
from screen.export_authorized_gex_forward import main as export_forward  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("intraday", "eod"))
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    started = datetime.now(timezone.utc)
    try:
        rc = collect(a.mode, ["SPY", "QQQ"], 45 if a.mode == "intraday" else 365,
                     0.12 if a.mode == "intraday" else 0.35, a.force)
        if rc == 0:
            rc = build_features()
        if rc == 0 and a.mode == "eod":
            rc = export_forward()
        if rc == 0 and a.mode == "eod":
            from screen.alpaca_bars_backfill import backfill
            from backtest.gex_master.forward_validation import main as validate_forward
            rc = backfill()
            if rc == 0:
                rc = validate_forward()
        status = "ok" if rc == 0 else f"exit_{rc}"
    except Exception as exc:
        rc, status = 1, f"error:{type(exc).__name__}:{str(exc)[:240]}"
    record = {"started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
              "mode": a.mode, "status": status}
    log = ROOT / "output" / "option_research_pipeline.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    print(json.dumps(record))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
