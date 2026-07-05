"""
backtest/remeasure/RC0_check_coverage.py — RC0.1 TEŞHİS: ham-cache kapsama sayımı (READ-ONLY).
Beklenen tarih listesi = data/historical_chains/md_spy.parquet tarihleri + sonrasındaki iş günleri
(bugün-1'e kadar; R0_backfill.py:72-77 ile BİREBİR aynı kurulum). data/raw_chains/<SYM>/ ile kıyaslar,
eksik tarihleri listeler. Hiçbir şey yazmaz/silmez (append-only cache'e dokunmaz).
  & <venv python> backtest/remeasure/RC0_check_coverage.py QQQ
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402  (TEK-GERÇEK-KAYNAK)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def expected_dates() -> list[str]:
    md = pd.read_parquet(config.ROOT / "data" / "historical_chains" / "md_spy.parquet")
    dates = [d.date().isoformat() for d in sorted(pd.to_datetime(md["date"].unique()))]
    ext = [d.date().isoformat() for d in pd.bdate_range(pd.Timestamp(dates[-1]) + timedelta(days=1),
                                                        pd.Timestamp(date.today()) - timedelta(days=1))]
    return dates + ext


def main() -> int:
    sym = sys.argv[1].upper() if len(sys.argv) > 1 else "QQQ"
    assert sym in config.SYMS, f"bilinmeyen sembol: {sym}"
    exp = expected_dates()
    raw = config.RAW_DIR / sym
    have = sorted(p.name.removesuffix(".json.gz") for p in raw.glob("*.json.gz")) if raw.exists() else []
    missing = [d for d in exp if d not in set(have)]
    extra = [d for d in have if d not in set(exp)]
    out = {
        "config_sha": config.config_sha(),
        "sym": sym,
        "expected_n": len(exp),
        "expected_first": exp[0],
        "expected_last": exp[-1],
        "have_n": len(have),
        "missing_n": len(missing),
        "missing": missing,
        "extra_n": len(extra),
        "extra": extra,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
