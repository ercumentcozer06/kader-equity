"""Deterministic health audit for the option-research data lake."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "option_research"
OUT = ROOT / "output" / "option_research_data_audit.json"


def main() -> int:
    checks: list[dict] = []

    def check(name: str, passed: bool, value, threshold: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "value": value,
                       "threshold": threshold})

    for mode in ("alpaca_intraday", "alpaca_eod"):
        for symbol in ("SPY", "QQQ"):
            files = sorted((DATA / mode).glob(f"*/*_{symbol}.parquet"))
            check(f"{mode}.{symbol}.exists", bool(files), len(files), ">=1 file")
            if not files:
                continue
            d = pd.read_parquet(files[-1])
            check(f"{mode}.{symbol}.oi_coverage",
                  d.open_interest.notna().mean() >= .50,
                  round(float(d.open_interest.notna().mean()), 4), ">=0.50")
            check(f"{mode}.{symbol}.iv_coverage", d.iv.notna().mean() >= .70,
                  round(float(d.iv.notna().mean()), 4), ">=0.70")
            check(f"{mode}.{symbol}.quote_coverage",
                  d.quote_ts.notna().mean() >= .90,
                  round(float(d.quote_ts.notna().mean()), 4), ">=0.90")

    occ_path = DATA / "occ_participant_daily.parquet"
    check("occ.exists", occ_path.exists(), str(occ_path), "file exists")
    if occ_path.exists():
        occ = pd.read_parquet(occ_path)
        counts = occ.groupby("symbol").size().to_dict()
        check("occ.symbols", set(counts) == {"SPY", "QQQ", "SPX", "NDX"},
              counts, "all four products")
        check("occ.unique_keys", not occ.duplicated(["date", "symbol"]).any(),
              int(occ.duplicated(["date", "symbol"]).sum()), "0 duplicates")

    feat_path = DATA / "dynamic_surface_features.parquet"
    check("features.exists", feat_path.exists(), str(feat_path), "file exists")
    if feat_path.exists():
        f = pd.read_parquet(feat_path)
        required = {"atm_iv_30d_pct", "gamma_flip", "net_gex_dollars_per_1pct",
                    "vanna_shares_per_1volpt", "charm_shares_per_day"}
        check("features.schema", required.issubset(f.columns),
              sorted(required - set(f.columns)), "no missing required columns")
        check("features.atm_iv", f.atm_iv_30d_pct.notna().all(),
              int(f.atm_iv_30d_pct.isna().sum()), "0 missing")

    failed = [x for x in checks if not x["passed"]]
    report = {"audited_utc": datetime.now(timezone.utc).isoformat(),
              "status": "ok" if not failed else "failed",
              "checks": checks, "failed": failed}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": len(checks),
                      "failed": len(failed), "path": str(OUT)}))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
