"""
backtest/remeasure/RC01_verify_ndx.py — RC0.1 NDX backfill doğrulama (TEŞHİS-ONLY).
Beklenen tarih listesi = md_spy.parquet tarihleri + sonrasındaki iş günleri (bugün-1'e kadar)
(R0_backfill.py main() ile birebir aynı mantık). data/raw_chains/<SYM>/ dosya sayısını sayar,
EKSİK tarihleri listeler. Sabitler config.py'den; çıktı JSON + config_sha.
  & <venv python> backtest/remeasure/RC01_verify_ndx.py [SYM=NDX]
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def main():
    sym = (sys.argv[1].strip().upper() if len(sys.argv) > 1 else "NDX")
    assert sym in config.SYMS, f"bilinmeyen sembol: {sym}"
    md = pd.read_parquet(config.ROOT / "data" / "historical_chains" / "md_spy.parquet")
    dates = [d.date().isoformat() for d in sorted(pd.to_datetime(md["date"].unique()))]
    ext = [d.date().isoformat() for d in pd.bdate_range(pd.Timestamp(dates[-1]) + timedelta(days=1),
                                                        pd.Timestamp(date.today()) - timedelta(days=1))]
    expected = dates + ext
    raw_dir = config.RAW_DIR / sym
    have = sorted(p.name[:-len(".json.gz")] for p in raw_dir.glob("*.json.gz")) if raw_dir.exists() else []
    missing = [d for d in expected if d not in set(have)]
    extra = [d for d in have if d not in set(expected)]
    out = {
        "sym": sym,
        "expected_n": len(expected),
        "expected_range": [expected[0], expected[-1]],
        "md_dates_n": len(dates),
        "ext_bdays": ext,
        "files_n": len(have),
        "missing_n": len(missing),
        "missing": missing,
        "extra_files": extra,
        "config_sha": config.config_sha(),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
