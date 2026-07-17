"""PDH/PDL liquidity-sweep amendment requested after the canonical unblinding.

This is deliberately a separate exploratory family. It does not overwrite the
frozen P6 acceptance/rejection results.
"""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results"
spec = importlib.util.spec_from_file_location("canonical_playbook", HERE / "canonical_playbook.py")
C = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(C)

SWEEP_EM = .05       # meaningful breach beyond the prior-day level
RECLAIM_EM = .02     # close back inside by this amount
WICK_STOP_EM = .05   # invalidation beyond the sweep extreme


def first_sweep(b5: pd.DataFrame, level: float, em: float, lower: bool):
    prev = float(b5["o"].iloc[0])
    for ts, b in b5.iterrows():
        if (ts + pd.Timedelta(minutes=5)).time() > C.CUTOFF:
            break
        if lower:
            approached_inside = prev >= level
            swept = float(b["l"]) <= level - SWEEP_EM * em
            reclaimed = float(b["c"]) >= level + RECLAIM_EM * em
        else:
            approached_inside = prev <= level
            swept = float(b["h"]) >= level + SWEEP_EM * em
            reclaimed = float(b["c"]) <= level - RECLAIM_EM * em
        if approached_inside and swept and reclaimed:
            return ts, b
        prev = float(b["c"])
    return None


def collect() -> pd.DataFrame:
    rows = []
    for sym in ("SPY", "QQQ"):
        panel = pd.read_parquet(ROOT / "backtest" / "scenario_engine" / f"panel_{sym.lower()}.parquet")
        mins = C.se_panel.rth_minutes(sym)
        days = {pd.Timestamp(d): g.sort_index() for d, g in mins.groupby("date") if len(g) >= 350}
        for D, r in panel.iterrows():
            D, N = pd.Timestamp(D), pd.Timestamp(r["N"])
            prev, day = days.get(D), days.get(N)
            if prev is None or day is None or pd.isna(r["em1"]):
                continue
            em = float(r["em1"]); pdh, pdl = float(prev["h"].max()), float(prev["l"].min())
            b5 = C.bars5(day)
            for setup, level, lower, side in (
                ("PDL_sweep_reclaim_long", pdl, True, +1),
                ("PDH_sweep_reject_short", pdh, False, -1),
            ):
                ev = first_sweep(b5, level, em, lower)
                if not ev: continue
                ts, bar = ev; fill = C.fill_after_bar(day, ts)
                if not fill: continue
                et, entry = fill
                stop = (float(bar["l"]) - WICK_STOP_EM * em) if side > 0 else (float(bar["h"]) + WICK_STOP_EM * em)
                risk = side * (entry - stop)
                if risk <= 0: continue
                levels = {"pdh": pdh, "pdl": pdl}
                policies = ("1R", "2R", "3R", "EOD", "pdh") if side > 0 else ("1R", "2R", "3R", "EOD", "pdl")
                for pol in policies:
                    try: target = C.target_from(entry, side, risk, pol, levels)
                    except KeyError: continue
                    row = pd.Series({"regime_idx": r["regime_idx"], "regime_own": r["regime_own"]})
                    C.add_trade(rows, sym=sym, D=D, N=N, row=row, setup=setup, variant=pol,
                                primary=pol == "2R", day=day, entry_ts=et, entry=entry,
                                side=side, stop=stop, target=target,
                                extra={"signal_ts": ts + pd.Timedelta(minutes=5),
                                       "sweep_level": level,
                                       "sweep_extreme": float(bar["l"] if side > 0 else bar["h"])})
    return pd.DataFrame(rows)


def main() -> int:
    t = collect(); OUT.mkdir(parents=True, exist_ok=True)
    t.to_parquet(OUT / "liquidity_sweep_trades.parquet", index=False)
    result = {"run_utc": datetime.now(timezone.utc).isoformat(), "amendment": True,
              "definition": {"sweep_em": SWEEP_EM, "reclaim_em": RECLAIM_EM, "wick_stop_em": WICK_STOP_EM},
              "cells": {}}
    for (sym, setup, variant), g in t.groupby(["sym", "setup", "variant"]):
        key=f"{sym}|{setup}|{variant}"; result["cells"][key]={}
        for name, sub in (("full",g),("train",g[pd.to_datetime(g.D)<C.HOLDOUT]),
                          ("holdout",g[pd.to_datetime(g.D)>=C.HOLDOUT])):
            result["cells"][key][name]=C.metrics(sub,2.0) if len(sub) else {"n":0}
    (OUT / "liquidity_sweep_results.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    for k,v in result["cells"].items():
        if not k.endswith("|2R"): continue
        f=v["full"]; h=v["holdout"]
        print(f"{k:45s} full n={f['n']} mean={f.get('mean_net_bps',np.nan):+.2f} t={f.get('t',np.nan):+.2f} "
              f"hold n={h['n']} mean={h.get('mean_net_bps',np.nan):+.2f}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
