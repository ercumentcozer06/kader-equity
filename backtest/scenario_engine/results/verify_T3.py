"""
verify_T3.py — ADVERSARIAL independent recomputation of study T3 (flip reclaim reversal).

Recomputes FROM PANEL + 1-MIN BARS directly, with OWN 15-min aggregation
(floor-to-15min grid, label = bar start; independent of se_panel.bars15 logic,
then cross-checked against bars15 as a consistency audit):

  1) breakdown count + reclaim count per symbol (+ failed, late-after-cutoff)
  2) reclaim trade net EV bps + hit rate per symbol (pooled regimes)
  3) unconditional control overall mean gross bps per symbol (all days, bars start<=14:45)
  4) failed-reclaim EV for QQQ (breakdown bar close -> c1)
  plus: time-of-day-matched baseline (audit e), per-day panel-join PIT checks (audit a).

Locked definitions taken literally from se_config docstring:
  BREAKDOWN = first 15-min bar close < flip[D] in session N
  RECLAIM   = first SUBSEQUENT 15-min bar close > flip[D]; only counts if that
              bar's START <= 14:45 ET (entries only at bars starting <=14:45)
  TRADE     = LONG at reclaim bar close -> exit at session close c1
  CONTROL   = ALL panel days, ALL 15-min bar closes with bar start <= 14:45,
              LONG bar close -> c1
  strict inequalities; flip NaN days skipped from events (control keeps all days).

Run:  cd kader-equity && <venv python> backtest/scenario_engine/results/verify_T3.py
Out:  backtest/scenario_engine/results/verify_T3.json
"""
from __future__ import annotations

import json
import sys
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent          # .../scenario_engine/results
SE_DIR = HERE.parent                            # .../scenario_engine
sys.path.insert(0, str(SE_DIR))
import se_config as SE  # noqa: E402
import se_panel  # noqa: E402  (allowed helpers: rth_minutes; bars15 only for cross-check)

CUTOFF = dtime(14, 45)
COST_BPS = SE.COST_RT * 1e4
Z = 1.959963984540054


def wilson(k, n):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + Z * Z / n
    ctr = (p + Z * Z / (2 * n)) / den
    half = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return (round(ctr - half, 4), round(ctr + half, 4))


def tstat(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return None
    sd = np.std(x, ddof=1)
    return round(float(np.mean(x) / (sd / np.sqrt(n))), 2) if sd > 0 else None


def my_bars15(mins: pd.DataFrame) -> pd.DataFrame:
    """OWN 15-min close aggregation: floor 1-min ET timestamps to the :00/:15/:30/:45
    grid (RTH starts 09:30 -> grid-aligned). Bar labeled by START. Close = last 1-min close."""
    df = mins.copy()
    df["bar_start"] = df.index.floor("15min")
    g = df.groupby(["date", "bar_start"], sort=True)
    out = g.agg(c=("c", "last")).reset_index()
    return out


def run_sym(sym: str) -> dict:
    panel = pd.read_parquet(SE.panel_path(sym))
    mins = se_panel.rth_minutes(sym)
    b15_mine = my_bars15(mins)
    b15_study = se_panel.bars15(mins)  # cross-check only

    # ---- cross-check my aggregation vs se_panel.bars15 (closes must be identical) ----
    sb = b15_study.reset_index()
    sb_key = list(zip(sb["date"], [ts.time() for ts in sb["index" if "index" in sb else sb.columns[0]]]))
    # robust: rebuild with named index
    sb = b15_study.copy()
    sb["bar_start_t"] = [ts.time() for ts in sb.index]
    mine_map = {(d, ts.time()): c for d, ts, c in
                zip(b15_mine["date"], b15_mine["bar_start"], b15_mine["c"])}
    n_bar_mismatch = 0
    for d, t, c in zip(sb["date"], sb["bar_start_t"], sb["c"]):
        mc = mine_map.get((d, t))
        if mc is None or abs(mc - c) > 1e-9:
            n_bar_mismatch += 1
    bars_crosscheck = {"study_bars": int(len(sb)), "my_bars": int(len(b15_mine)),
                       "close_mismatches": int(n_bar_mismatch)}

    by_day = {d: g.sort_values("bar_start") for d, g in b15_mine.groupby("date")}
    sessions = sorted(by_day.keys())

    # ---- daily last 1-min close (independent c1 check) ----
    day_close = mins.groupby("date")["c"].last()

    # ---- audit (a): panel join PIT — N must be the immediate NEXT session after D ----
    join_bad = 0
    c1_bad = 0
    for D, row in panel.iterrows():
        N = pd.Timestamp(row["N"])
        if not (N > D):
            join_bad += 1
            continue
        later = [s for s in sessions if s > D]
        if not later or later[0] != N:
            join_bad += 1
        if N in day_close.index and abs(float(row["c1"]) - float(day_close.loc[N])) > 1e-9:
            c1_bad += 1

    # ---- events ----
    n_bd = n_rc = n_late = n_flip_nan = 0
    rc_gross, rc_hours = [], []
    fail_gross = []
    cutoff_violations = 0
    reclaim_not_after_bd = 0
    for D, row in panel.iterrows():
        flip = row["flip"]
        if not np.isfinite(flip):
            n_flip_nan += 1
            continue
        N = pd.Timestamp(row["N"])
        bars = by_day.get(N)
        if bars is None or len(bars) == 0:
            continue
        closes = bars["c"].to_numpy()
        starts = [ts.time() for ts in bars["bar_start"]]
        below = np.where(closes < flip)[0]
        if len(below) == 0:
            continue
        bi = int(below[0])
        n_bd += 1
        c1 = float(row["c1"])
        above_after = [j for j in range(bi + 1, len(closes)) if closes[j] > flip]
        rc = None
        if above_after:
            j = above_after[0]
            if starts[j] <= CUTOFF:
                rc = j
            else:
                n_late += 1
        if rc is not None:
            if rc <= bi:
                reclaim_not_after_bd += 1
            if starts[rc] > CUTOFF:
                cutoff_violations += 1
            n_rc += 1
            rc_gross.append((c1 / float(closes[rc]) - 1.0) * 1e4)
            rc_hours.append(starts[rc].hour)
        else:
            fail_gross.append((c1 / float(closes[bi]) - 1.0) * 1e4)

    rc_gross = np.array(rc_gross)
    fail_gross = np.array(fail_gross)
    net = rc_gross - COST_BPS
    k_hit = int((rc_gross > 0).sum())

    # ---- control: ALL panel days (incl. flip-NaN), all bars start <= 14:45 ----
    ctrl_g, ctrl_h = [], []
    for D, row in panel.iterrows():
        N = pd.Timestamp(row["N"])
        bars = by_day.get(N)
        if bars is None:
            continue
        c1 = float(row["c1"])
        for ts, c in zip(bars["bar_start"], bars["c"]):
            if ts.time() <= CUTOFF:
                ctrl_g.append((c1 / float(c) - 1.0) * 1e4)
                ctrl_h.append(ts.hour)
    ctrl_g = np.array(ctrl_g)
    ctrl_h = np.array(ctrl_h)

    # ---- time-of-day matched baseline (audit e): weight per-hour UNROUNDED control means
    #      by reclaim-entry hour distribution ----
    matched = None
    if len(rc_hours):
        hw = pd.Series(rc_hours).value_counts(normalize=True)
        ms = ws = 0.0
        for h, w in hw.items():
            m = ctrl_h == h
            if m.sum():
                ms += w * float(ctrl_g[m].mean())
                ws += w
        matched = round(ms / ws, 2) if ws > 0 else None

    return {
        "n_days": int(len(panel)),
        "n_flip_nan": n_flip_nan,
        "n_breakdowns": n_bd,
        "n_reclaims": n_rc,
        "n_failed": n_bd - n_rc,
        "n_late_reclaim_after_cutoff": n_late,
        "reclaim": {
            "n": int(len(rc_gross)),
            "mean_gross_bps": round(float(rc_gross.mean()), 2) if len(rc_gross) else None,
            "mean_net_bps": round(float(net.mean()), 2) if len(net) else None,
            "t_gross": tstat(rc_gross),
            "hit_gross": round(k_hit / len(rc_gross), 4) if len(rc_gross) else None,
            "hit_wilson95": list(wilson(k_hit, len(rc_gross))),
        },
        "control_overall": {
            "n": int(len(ctrl_g)),
            "mean_gross_bps": round(float(ctrl_g.mean()), 2) if len(ctrl_g) else None,
        },
        "matched_baseline_gross_bps": matched,
        "reclaim_minus_matched": (round(float(rc_gross.mean()) - matched, 2)
                                  if (matched is not None and len(rc_gross)) else None),
        "failed": {
            "n": int(len(fail_gross)),
            "mean_gross_bps": round(float(fail_gross.mean()), 2) if len(fail_gross) else None,
            "t_gross": tstat(fail_gross),
        },
        "audits": {
            "bars15_crosscheck": bars_crosscheck,
            "panel_join_N_not_next_session": join_bad,
            "panel_c1_vs_last_1min_close_mismatch": c1_bad,
            "reclaim_entries_after_cutoff": cutoff_violations,
            "reclaim_not_strictly_after_breakdown": reclaim_not_after_bd,
        },
    }


def main():
    out = {}
    for sym in SE.SYMS:
        out[sym] = run_sym(sym)

    # readable table
    for sym in SE.SYMS:
        r = out[sym]
        a = r["audits"]
        print(f"\n========== VERIFY T3 — {sym} ==========")
        print(f"days={r['n_days']}  flipNaN={r['n_flip_nan']}  breakdown={r['n_breakdowns']}  "
              f"reclaim={r['n_reclaims']}  failed={r['n_failed']}  late>14:45={r['n_late_reclaim_after_cutoff']}")
        rc = r["reclaim"]
        print(f"RECLAIM: n={rc['n']}  gross={rc['mean_gross_bps']}bps  net={rc['mean_net_bps']}bps  "
              f"t={rc['t_gross']}  hit={rc['hit_gross']}  CI{rc['hit_wilson95']}")
        co = r["control_overall"]
        print(f"CONTROL: n={co['n']}  mean_gross={co['mean_gross_bps']}bps  "
              f"matched={r['matched_baseline_gross_bps']}bps  reclaim-matched={r['reclaim_minus_matched']}bps")
        fl = r["failed"]
        print(f"FAILED : n={fl['n']}  gross={fl['mean_gross_bps']}bps  t={fl['t_gross']}")
        print(f"AUDITS : bars15 closes mismatched={a['bars15_crosscheck']['close_mismatches']} "
              f"(study {a['bars15_crosscheck']['study_bars']} vs mine {a['bars15_crosscheck']['my_bars']} bars) | "
              f"N-not-next-session={a['panel_join_N_not_next_session']} | "
              f"c1-vs-last-1min-mismatch={a['panel_c1_vs_last_1min_close_mismatch']} | "
              f"entry-after-cutoff={a['reclaim_entries_after_cutoff']} | "
              f"reclaim<=breakdown-bar={a['reclaim_not_strictly_after_breakdown']}")

    path = HERE / "verify_T3.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nyazildi: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
