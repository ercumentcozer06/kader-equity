"""
gen_snapshot — FREEZE the equities-tide spine snapshot (BUILD TOOL, one-time; NOT runtime).

This is the ONLY place kader-equity imports kader-macro. It reproduces the 4000-clone sweep
winner (kader-macro/backtest/revalidation/sweep4000.py) and bakes a network-free frozen snapshot
the runtime reads. Verified anchor: SPX 1.43 / NQ 1.49 @2019+ (raw-m2, 8-module, +1d exec-lag).

What it freezes into spine/frozen/:
  • module_scores.parquet  — baked DAILY module-score panel (m0..m11; m2 RAW net-liq, m3 auctions +
                             m6 Moody's Baa-Aaa injected by sweep4000.build_module_matrix)
  • prices.parquet         — SPX + NDX daily closes (Desktop/backtesting CSVs)
  • vector.json            — the frozen winner weight vector (argmax FULL-2019+ SPX Sharpe)
  • provenance.json        — anchor Sharpe/maxDD + window + recipe + locks (raw-m2, 8-module)

Run with the kader-macro venv (has FRED key + cache + scipy/pyarrow):
  & "C:\\Users\\admin\\Downloads\\kader-macro\\.venv\\Scripts\\python.exe" \\
        "C:\\Users\\admin\\Downloads\\kader-equity\\spine\\gen_snapshot.py"
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── kader-macro importable ONLY for this build tool (runtime never imports it) ──
KMR = Path(r"C:\Users\admin\Downloads\kader-macro")
sys.path.insert(0, str(KMR))

import yaml                                                       # noqa: E402
from dotenv import load_dotenv                                    # noqa: E402

from backtest import data_loader as dl                           # noqa: E402
from backtest.revalidation import oos_judge as J                 # noqa: E402
from backtest.revalidation.sweep4000 import (                    # noqa: E402
    build_module_matrix, make_vectors, MODS, _fwd_ret, _sharpe_rows, _metrics_row)

FROZEN = Path(__file__).resolve().parent / "frozen"
ASSETS = [("SPX", "SPX_daily.csv"), ("NDX", "NASDAQ_daily.csv")]
ANCHOR_MIN_SHARPE = 1.40            # raw-winner must clear this (smart-m2 collapses to 1.02 -> reject)
# canonical vector documented in kader-equity/ARCHITECTURE.md (sanity cross-check, not a hard gate)
CANON = {"m9": 0.563, "m5": 0.214, "m2": 0.118, "m0": 0.061, "m3": 0.025, "m6": 0.01, "m8": 0.006, "m4": 0.002}


def _lag1(pos: np.ndarray) -> np.ndarray:
    """+1 day execution lag (signal[t] traded at t+1 close); look-ahead-free."""
    return np.concatenate([[0.0], pos[:-1]])


def main() -> int:
    load_dotenv(J.ROOT / ".env")                                  # FRED key for m6/m3 injection
    cfg = yaml.safe_load((J.ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg.setdefault("fred", {})["cache_min_seconds"] = 10 ** 9     # prefer cache (reproducible)

    # ── 1) baked DAILY module-score panel (RAW m2 — NO SMART_M2 env) ──
    M = build_module_matrix(cfg)                                  # m3/m6 injected, m2 raw
    idx = M.index
    cov = {c: f"{M[c].notna().mean():.0%}" for c in MODS}
    print(f"  panel: {M.shape[0]} days  {idx.min().date()}..{idx.max().date()}")
    print(f"  coverage: {cov}")

    # ── 2) prices + forward returns ──
    closes = {a: dl.load_price_csv(str(J.DESK / f)) for a, f in ASSETS}
    rets = {a: _fwd_ret(closes[a], idx) for a, _ in ASSETS}

    # ── 3) reproduce the sweep winner (argmax FULL-2019+ SPX Sharpe over 4000 deterministic vectors) ──
    W = make_vectors(4000)
    Mf = np.nan_to_num(M.values, nan=0.0)                         # absent module = 0 vote
    pos_all = (W @ Mf.T > 0).astype(float)
    pos_all = np.concatenate([np.zeros((pos_all.shape[0], 1)), pos_all[:, :-1]], axis=1)  # +1d lag
    full_spx = _sharpe_rows(pos_all * rets["SPX"])
    iw = int(np.nanargmax(full_spx))
    w = W[iw]
    vec = {MODS[j]: round(float(w[j]), 6) for j in range(len(MODS))}
    nz = sorted([(k, v) for k, v in vec.items() if abs(v) > 1e-9], key=lambda t: -abs(t[1]))
    print(f"  winner (|w|>0): {[(k, round(v, 3)) for k, v in nz]}")

    # ── 4) anchor metrics (single winner vector) ──
    comp = (w @ Mf.T)
    pos = _lag1((comp > 0).astype(float))
    anchor = {}
    for a, _ in ASSETS:
        sh, dd, pnl = _metrics_row(pos * rets[a])
        anchor[a] = {"sharpe": sh, "maxdd": dd, "pnl": pnl}
    print(f"  anchor: SPX {anchor['SPX']}  NDX {anchor['NDX']}")

    # ── 5) gates: raw-winner must clear the bar; vector must resemble the canonical sweep winner ──
    if not (anchor["SPX"]["sharpe"] and anchor["SPX"]["sharpe"] >= ANCHOR_MIN_SHARPE):
        print(f"  [ABORT] SPX anchor {anchor['SPX']['sharpe']} < {ANCHOR_MIN_SHARPE} — not the raw-winner base. "
              f"(smart-m2 drift? check SMART_M2 env / m2 column).")
        return 2
    drift = max(abs(vec.get(k, 0.0) - cv) for k, cv in CANON.items())
    print(f"  vector drift vs ARCHITECTURE.md canonical: {drift:.4f} "
          f"({'OK' if drift < 0.05 else 'WARN — panel data moved; review'})")

    # ── 6) freeze ──
    FROZEN.mkdir(parents=True, exist_ok=True)
    M.to_parquet(FROZEN / "module_scores.parquet")
    px = pd.DataFrame({a: closes[a].reindex(idx, method="ffill") for a, _ in ASSETS}, index=idx)
    px.to_parquet(FROZEN / "prices.parquet")
    (FROZEN / "vector.json").write_text(json.dumps(vec, indent=2), encoding="utf-8")
    prov = {
        "model": "kader-equity-tide",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recipe": "sweep4000 winner (argmax FULL-2019+ SPX Sharpe); RAW net-liq m2 (NOT smart-RRP); +1d exec-lag",
        "source_panel": "_pit_signals_brakes_full.csv + m3(live auctions) + m6(Moody's Baa-Aaa DBAA-DAAA z) injected",
        "window": {"start": str(idx.min().date()), "end": str(idx.max().date()), "n_days": int(len(idx))},
        "modules_used": [k for k, _ in nz],
        "vector": vec,
        "anchor": anchor,
        "locks": {"raw_m2": True, "n_modules_nonzero": len(nz), "anchor_min_sharpe": ANCHOR_MIN_SHARPE},
        "note": ("Parity proven (_tide_liveparity): smart-RRP m2 -> 1.02/1.19 (REJECT), top-4 -> 1.16/1.28 "
                 "(REJECT). Keep RAW m2 + all 8 modules. m9-era single-regime: forward realistic ~1.0-1.3."),
    }
    (FROZEN / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(f"  [OK] froze -> {FROZEN}  (module_scores, prices, vector.json, provenance.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
