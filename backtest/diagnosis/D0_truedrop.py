"""
backtest/diagnosis/D0_truedrop — GERÇEK build_level_series spot'u (Alpaca close) ile IV-drop oranı.
D0_data_probe strike-medyan PROXY kullandı (drop'u şişirdi). Burada gerçek _daily_spot + gerçek band.
TEŞHİS-ONLY; build() çağırMAZ (parquet'i yeniden-üretmez), yalnız fail-oranını sayar.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "screen")); sys.path.insert(0, str(ROOT / "backtest"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from _bsiv import implied_vol            # noqa: E402
from build_level_series import _daily_spot, BAND  # noqa: E402


def main() -> int:
    for sym in ("SPY", "QQQ"):
        ch = pd.read_parquet(ROOT / "data" / "historical_chains" / f"md_{sym.lower()}.parquet")
        spot = _daily_spot(sym)
        band_in = iv_none = kept = penny_kept = days = days_skipped_lt4 = 0
        for d, g in ch.groupby("date"):
            dd = pd.Timestamp(d).date()
            if dd not in spot.index:
                continue
            S = float(spot[dd]); days += 1
            kept_day = 0
            for _, r in g.iterrows():
                K, oi, mid, right = r["strike"], r["open_interest"], r["mid"], r["right"]
                if K is None or oi is None or oi != oi or mid is None or mid <= 0:
                    continue
                K = float(K)
                if abs(K / S - 1) > BAND:
                    continue
                band_in += 1
                iv = implied_vol(float(mid), S, K, T=max((pd.Timestamp(g["expiration"].iloc[0]) - pd.Timestamp(d)).days, 0.5) / 365.0, right=right)
                if not iv or iv <= 0:
                    iv_none += 1; continue
                kept += 1; kept_day += 1
                if 0 < mid <= 0.05:
                    penny_kept += 1
            if kept_day < 4:
                days_skipped_lt4 += 1
        print(f"  {sym}: gün {days}, band-içi {band_in:,}  →  IV-None DROP {iv_none:,} (%{100*iv_none/max(band_in,1):.1f})  "
              f"kept {kept:,}")
        print(f"        kept-içinde penny-mid(≤0.05) {penny_kept:,} = %{100*penny_kept/max(kept,1):.1f}  "
              f"|  <4-strike yüzünden düşen gün {days_skipped_lt4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
