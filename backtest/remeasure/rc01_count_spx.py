"""
backtest/remeasure/rc01_count_spx.py — RC0.1 [SPX] sayım + eksik-tarih raporu (TEŞHİS-ONLY, fetch yok).
Beklenen tarih listesi = md_spy.parquet tarihleri + sonrasındaki iş günleri (bugün-1'e kadar)
(R0_backfill.py:72-77 ile birebir aynı mantık). data/raw_chains/SPX/ ile karşılaştırır.
  & <venv python> backtest/remeasure/rc01_count_spx.py [SYM]
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main():
    sym = (sys.argv[1] if len(sys.argv) > 1 else "SPX").upper()
    assert sym in config.SYMS, f"bilinmeyen sembol {sym}"
    md = pd.read_parquet(config.ROOT / "data" / "historical_chains" / "md_spy.parquet")
    md_dates = [d.date().isoformat() for d in sorted(pd.to_datetime(md["date"].unique()))]
    ext = [d.date().isoformat() for d in pd.bdate_range(pd.Timestamp(md_dates[-1]) + timedelta(days=1),
                                                        pd.Timestamp(date.today()) - timedelta(days=1))]
    expected = md_dates + ext
    raw = config.RAW_DIR / sym
    have = sorted(p.name[:-len(".json.gz")] for p in raw.glob("*.json.gz")) if raw.exists() else []
    missing = sorted(set(expected) - set(have))
    extra = sorted(set(have) - set(expected))
    out = {
        "config_sha": config.config_sha(),
        "sym": sym,
        "n_expected": len(expected),
        "n_md_dates": len(md_dates),
        "n_ext_bdays": len(ext),
        "ext_range": [ext[0], ext[-1]] if ext else None,
        "n_have": len(have),
        "n_missing": len(missing),
        "missing": missing,
        "n_extra_not_expected": len(extra),
        "extra_not_expected": extra,
        "expected_first": expected[0],
        "expected_last": expected[-1],
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
