"""Export authorized Alpaca EOD surface rows into the frozen forward ledger."""
from __future__ import annotations

from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "option_research" / "dynamic_surface_features.parquet"
LEDGER = ROOT / "output" / "gex_forward_eod_levels.parquet"


def main() -> int:
    if not FEATURES.exists():
        print("no dynamic surface features")
        return 0
    f = pd.read_parquet(FEATURES)
    f["capture_ts_utc"] = pd.to_datetime(f.fetch_ts_utc, utc=True)
    local = f.capture_ts_utc.dt.tz_convert("America/New_York")
    # Forced development captures made before the close must never masquerade
    # as EOD levels. Scheduled EOD capture runs around 17:30 ET.
    mask = f.source_path.astype(str).str.contains("alpaca_eod") & (local.dt.time >= time(16, 20))
    f = f[mask].copy()
    if f.empty:
        print("no completed authorized EOD capture yet")
        return 0
    f["as_of"] = local[mask].dt.tz_localize(None).dt.normalize().to_numpy()
    new = pd.DataFrame({
        "as_of": f.as_of, "ticker": f.symbol, "capture_ts_utc": f.capture_ts_utc,
        "underlying": f.symbol, "spot": f.spot, "put_wall": f.put_wall,
        "call_wall": f.call_wall, "ghost": np.nan, "gamma_flip": f.gamma_flip,
        "max_pain": np.nan, "hvl": np.nan, "atm_iv_30d": f.atm_iv_30d_pct,
        "regime": np.where(f.net_gex_dollars_per_1pct < 0, "SHORT_GAMMA", "LONG_GAMMA"),
        "source": "alpaca_indicative_authorized",
    })
    if LEDGER.exists():
        new = pd.concat([pd.read_parquet(LEDGER), new], ignore_index=True)
    new["as_of"] = pd.to_datetime(new.as_of)
    new = new.sort_values(["as_of", "ticker", "capture_ts_utc"]).drop_duplicates(
        ["as_of", "ticker"], keep="last")
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(LEDGER, index=False)
    print(f"authorized EOD forward ledger: {len(new)} rows -> {LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
