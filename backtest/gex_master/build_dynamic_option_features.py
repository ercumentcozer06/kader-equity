"""Build one point-in-time surface/regime row per authorized Alpaca capture."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from screen.option_research_greeks import all_greeks  # noqa: E402

RAW = ROOT / "data" / "option_research"
OUT = RAW / "dynamic_surface_features.parquet"
FEATURE_SCHEMA_VERSION = 2


def _adv20(symbol: str, ts: pd.Timestamp) -> float:
    p = ROOT / "data" / "historical_bars" / f"alpaca_{symbol.lower()}_1m.parquet"
    b = pd.read_parquet(p).reset_index()
    tcol = "timestamp" if "timestamp" in b else b.columns[1]
    b["date"] = pd.to_datetime(b[tcol], utc=True).dt.tz_convert("America/New_York").dt.date
    prior = b[b["date"] < ts.tz_convert("America/New_York").date()]
    daily = prior.groupby("date").agg(close=("close", "last"), volume=("volume", "sum"))
    return float((daily.close * daily.volume).tail(20).mean())


def _flip(d: pd.DataFrame, spot: float) -> float:
    x = d.dropna(subset=["open_interest", "iv", "strike", "t_years"]).copy()
    for col in ("open_interest", "iv", "strike", "t_years"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["open_interest", "iv", "strike", "t_years"])
    x = x[x.open_interest > 0]
    if len(x) < 50:
        return np.nan
    sign = np.where(x.right.eq("C"), 1.0, -1.0)
    grid = np.linspace(spot * 0.85, spot * 1.15, 121)
    vals = []
    for s in grid:
        gamma = all_greeks(
            np.full(len(x), s), x.strike.to_numpy(float), x.t_years.to_numpy(float),
            x.iv.to_numpy(float), x.right.eq("C").to_numpy())["bs_gamma"]
        vals.append(np.nansum(sign * gamma * x.open_interest.to_numpy(float) * 100 * s * s * 0.01))
    vals = np.asarray(vals)
    candidates = []
    for i in range(len(grid) - 1):
        if vals[i] == 0 or vals[i] * vals[i + 1] < 0:
            den = vals[i + 1] - vals[i]
            candidates.append(grid[i] if den == 0 else grid[i] - vals[i] * (grid[i + 1] - grid[i]) / den)
    return float(min(candidates, key=lambda z: abs(z - spot))) if candidates else np.nan


def surface_row(d: pd.DataFrame, source_path: str = "") -> dict:
    d = d.copy()
    ts = pd.to_datetime(d.fetch_ts_utc.iloc[0], utc=True)
    symbol, spot = str(d.underlying.iloc[0]), float(d.underlying_spot.iloc[0])
    oi = pd.to_numeric(d.open_interest, errors="coerce").fillna(0).to_numpy(float)
    sign = np.where(d.right.eq("C"), 1.0, -1.0)
    gamma = pd.to_numeric(d.vendor_gamma, errors="coerce").to_numpy(float)
    gd = sign * gamma * oi * 100 * spot * spot * 0.01
    abs_gd = np.abs(gd)
    d["signed_gex"] = gd
    call_by = d[d.right.eq("C")].groupby("strike").signed_gex.sum()
    put_by = d[d.right.eq("P")].groupby("strike").signed_gex.sum()
    call_wall = float(call_by.idxmax()) if len(call_by) else np.nan
    put_wall = float(put_by.idxmin()) if len(put_by) else np.nan
    adv = _adv20(symbol, ts)
    vanna = pd.to_numeric(d.vanna_per_1vol, errors="coerce").to_numpy(float)
    charm = pd.to_numeric(d.charm_per_year, errors="coerce").to_numpy(float)
    vomma = pd.to_numeric(d.vomma_per_1vol2, errors="coerce").to_numpy(float)
    speed = pd.to_numeric(d.speed, errors="coerce").to_numpy(float)
    color = pd.to_numeric(d.color_per_year, errors="coerce").to_numpy(float)
    zomma = pd.to_numeric(d.zomma_per_1vol, errors="coerce").to_numpy(float)
    ultima = pd.to_numeric(d.ultima_per_1vol3, errors="coerce").to_numpy(float)
    exp = pd.to_datetime(d.expiration).dt.date
    dte = np.array([(x - ts.tz_convert("America/New_York").date()).days for x in exp])
    atm30 = pd.to_numeric(d.iv, errors="coerce")[(dte >= 20) & (dte <= 45) &
                                                  (np.abs(d.strike / spot - 1.0) <= 0.01)]
    row = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "fetch_ts_utc": ts, "symbol": symbol, "source_path": source_path,
        "spot": spot, "n_contracts": len(d), "oi_coverage": float((oi > 0).mean()),
        "iv_coverage": float(pd.Series(d.iv).notna().mean()), "total_oi": float(np.nansum(oi)),
        "net_gex_dollars_per_1pct": float(np.nansum(gd)),
        "gross_gex_dollars_per_1pct": float(np.nansum(abs_gd)),
        "gex_to_adv20": float(np.nansum(gd) / adv) if adv > 0 else np.nan,
        "gross_gex_to_adv20": float(np.nansum(abs_gd) / adv) if adv > 0 else np.nan,
        "call_wall": call_wall, "put_wall": put_wall,
        "atm_iv_30d_pct": float(atm30.median() * 100.0) if len(atm30.dropna()) else np.nan,
        "call_wall_concentration": float(call_by.max() / np.nansum(abs_gd)) if len(call_by) and np.nansum(abs_gd) else np.nan,
        "put_wall_concentration": float(abs(put_by.min()) / np.nansum(abs_gd)) if len(put_by) and np.nansum(abs_gd) else np.nan,
        "gamma_flip": _flip(d, spot),
        "vanna_shares_per_1volpt": float(np.nansum(sign * vanna * oi * 100 * 0.01)),
        "charm_shares_per_day": float(np.nansum(sign * charm * oi * 100 / 365.0)),
        "vomma_signed_per_1volpt2": float(np.nansum(sign * vomma * oi * 100 * 0.0001)),
        "speed_signed": float(np.nansum(sign * speed * oi * 100)),
        "color_signed_per_day": float(np.nansum(sign * color * oi * 100 / 365.0)),
        "zomma_signed_per_1volpt": float(np.nansum(sign * zomma * oi * 100 * 0.01)),
        "ultima_signed_per_1volpt3": float(np.nansum(sign * ultima * oi * 100 * 0.000001)),
        "short_dated_volume_observed": float(pd.to_numeric(d.get("day_volume"), errors="coerce").sum()),
    }
    for label, mask in {
        "0dte": dte == 0, "1dte": dte == 1, "2_5dte": (dte >= 2) & (dte <= 5),
        "6_30dte": (dte >= 6) & (dte <= 30), "31plus": dte >= 31,
    }.items():
        row[f"net_gex_{label}"] = float(np.nansum(gd[mask]))
        row[f"gross_gex_{label}"] = float(np.nansum(abs_gd[mask]))
    return row


def main() -> int:
    files = sorted((RAW / "alpaca_intraday").glob("*/*.parquet")) + sorted((RAW / "alpaca_eod").glob("*/*.parquet"))
    old = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame()
    # Feature definitions evolve while the prospective schema is being frozen.
    # Rebuild all captures when a required column is absent instead of silently
    # carrying stale rows that the forward exporter cannot use.
    required = {"atm_iv_30d_pct", "feature_schema_version"}
    stale_schema = (not required.issubset(old.columns) or
                    not old["feature_schema_version"].eq(FEATURE_SCHEMA_VERSION).all()) if len(old) else False
    if stale_schema:
        print(f"schema changed; rebuilding {len(old)} existing rows")
        old = pd.DataFrame()
    done = set(old.source_path.astype(str)) if len(old) and "source_path" in old else set()
    rows = []
    for p in files:
        rel = str(p.relative_to(ROOT))
        if rel in done:
            continue
        rows.append(surface_row(pd.read_parquet(p), rel))
    if rows:
        out = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
        out = out.drop_duplicates(["fetch_ts_utc", "symbol"], keep="last").sort_values(["fetch_ts_utc", "symbol"])
        for sym, g in out.groupby("symbol"):
            idx = g.index
            out.loc[idx, "spot_return_since_capture"] = g.spot.pct_change().to_numpy()
            out.loc[idx, "call_wall_move"] = g.call_wall.diff().to_numpy()
            out.loc[idx, "put_wall_move"] = g.put_wall.diff().to_numpy()
            out.loc[idx, "net_gex_change"] = g.net_gex_dollars_per_1pct.diff().to_numpy()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(OUT, index=False)
    final = pd.read_parquet(OUT) if OUT.exists() else old
    print(f"dynamic features: +{len(rows)} -> {len(final)} rows at {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
