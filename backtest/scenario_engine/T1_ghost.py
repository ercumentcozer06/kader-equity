"""
T1 — GHOST GAP-FADE study (spec LOCKED in se_config.py docstring; no new thresholds).

EVENT (PRIMARY): |o1 − ghost| >= 0.25×em1 ; ALT filter: |o1 − ghost| >= 0.0015×spot_D (side-by-side).
Direction: o1 > ghost -> short toward ghost ("down"); o1 < ghost -> long toward ghost ("up").
Metrics per (sym × filter × direction ∪ pooled):
  1) P(touch ghost in session N)        (1-min low<=ghost down / high>=ghost up), n + Wilson 95.
  2) time-to-touch                      (minutes from 09:30, BAR-END convention; shares <=30 / <=120 / rest).
  3) MAE before touch (em1 units)       (adverse excursion from o1 up to AND INCLUDING touch bar; non-touch:
                                         full-session adverse excursion). median + p90.
  4) TRADE EV o1 -> ghost (TP) else c1  (no-stop) + STOP VARIANT (hard stop at 1.0×em1 adverse, 1-min bars,
                                         same-minute TP&stop -> STOP, fill at stop level). gross+net bps,
                                         hit rates, t-stats.
  5) PLACEBO bucket table               (DIST_BUCKETS midpoints; unconditional touch P over ALL panel days
                                         of a target d_mid×em1 below/above open vs event-conditional rate).
  6) NEVER-FILL next-session continuation  sign(o1−ghost) × (c1p−o1p)/o1p from the panel row indexed N.

Run:  cd kader-equity && <venv python> backtest/scenario_engine/T1_ghost.py
Out:  backtest/scenario_engine/results/T1_{spy,qqq}.json + stdout tables.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import se_config as SE  # noqa: E402
import se_panel  # noqa: E402

Z95 = 1.96
INSUFF = "YETERSIZ-N"


# ----------------------------------------------------------------------------- stats helpers
def wilson(k: int, n: int):
    """Wilson 95% CI for a proportion. Returns (rate, lo, hi) or (None, None, None)."""
    if n == 0:
        return None, None, None
    p = k / n
    z2 = Z95 * Z95
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return p, center - half, center + half


def prop_cell(k: int, n: int) -> dict:
    rate, lo, hi = wilson(k, n)
    cell = {"n": int(n), "k": int(k),
            "rate": None if rate is None else round(rate, 4),
            "wilson95": None if rate is None else [round(lo, 4), round(hi, 4)]}
    if n < 10:
        cell["flag"] = INSUFF
    return cell


def tstat(x: np.ndarray):
    n = len(x)
    if n < 2:
        return None
    s = float(np.std(x, ddof=1))
    if s == 0:
        return None
    return float(np.mean(x) / (s / math.sqrt(n)))


def ev_cell(gross: np.ndarray) -> dict:
    """EV summary: gross + net (= gross − COST_RT per round trip), bps, t-stats, hit rates."""
    n = len(gross)
    if n == 0:
        return {"n": 0, "flag": INSUFF}
    net = gross - SE.COST_RT
    tg, tn = tstat(gross), tstat(net)
    cell = {
        "n": int(n),
        "mean_gross_bps": round(float(np.mean(gross)) * 1e4, 2),
        "mean_net_bps": round(float(np.mean(net)) * 1e4, 2),
        "std_bps": round(float(np.std(gross, ddof=1)) * 1e4, 2) if n >= 2 else None,
        "t_gross": None if tg is None else round(tg, 2),
        "t_net": None if tn is None else round(tn, 2),
        "hit_gross": prop_cell(int((gross > 0).sum()), n),
        "hit_net": prop_cell(int((net > 0).sum()), n),
    }
    if n < 10:
        cell["flag"] = INSUFF
    return cell


def q(x, p):
    return None if len(x) == 0 else round(float(np.percentile(np.asarray(x, float), p)), 3)


# ----------------------------------------------------------------------------- per-day computation
def day_records(sym: str):
    """One record per panel row: event flags, touch/timing/MAE/EV primitives. PIT: row-D levels,
    session-N 1-min bars only."""
    panel = pd.read_parquet(SE.panel_path(sym))
    mins = se_panel.rth_minutes(sym)
    groups = {d: g for d, g in mins.groupby("date")}
    recs, missing_bars = [], 0
    for D, r in panel.iterrows():
        N = r["N"]
        g = groups.get(N)
        if g is None or len(g) == 0:
            missing_bars += 1
            continue
        o1, c1, ghost, em1, spot = (float(r["o1"]), float(r["c1"]), float(r["ghost"]),
                                    float(r["em1"]), float(r["spot_D"]))
        h = g["h"].to_numpy(float)
        l = g["l"].to_numpy(float)
        dist = o1 - ghost
        absd = abs(dist)
        d_em = absd / em1
        direction = "down" if dist > 0 else ("up" if dist < 0 else None)

        rec = {"D": D, "N": N, "o1": o1, "c1": c1, "ghost": ghost, "em1": em1,
               "dist": dist, "d_em": d_em,
               "event_primary": bool(d_em >= SE.EM1_MIN_DIST),
               "event_alt": bool(absd >= SE.ALT_DIST_PCT * spot),
               "direction": direction,
               "day_low": float(l.min()), "day_high": float(h.max())}

        if direction is not None:
            short = direction == "down"
            tp_mask = (l <= ghost) if short else (h >= ghost)
            touch = bool(tp_mask.any())
            rec["touch"] = touch
            if touch:
                ti = int(np.argmax(tp_mask))
                t_bar = g.index[ti]
                sess_open = t_bar.normalize() + pd.Timedelta(hours=9, minutes=30)
                # BAR-END convention: touch certain by end of the touching 1-min bar
                rec["ttt_min"] = float((t_bar - sess_open).total_seconds() / 60.0) + 1.0
                adverse = (h[: ti + 1].max() - o1) if short else (o1 - l[: ti + 1].min())
            else:
                rec["ttt_min"] = None
                adverse = (h.max() - o1) if short else (o1 - l.min())
            rec["mae_em1"] = max(0.0, float(adverse)) / em1

            # --- trade EV, no-stop: enter o1 toward ghost; TP=ghost if touched else exit c1
            exit_ns = ghost if touch else c1
            rec["gross_nostop"] = (o1 - exit_ns) / o1 if short else (exit_ns - o1) / o1

            # --- stop variant: hard stop at 1.0×em1 adverse from o1, 1-min bars, stop-first tie
            stop_lvl = o1 + SE.T1_STOP_EM1 * em1 if short else o1 - SE.T1_STOP_EM1 * em1
            stop_mask = (h >= stop_lvl) if short else (l <= stop_lvl)
            si = int(np.argmax(stop_mask)) if stop_mask.any() else None
            pi = int(np.argmax(tp_mask)) if touch else None
            if si is not None and (pi is None or si <= pi):     # same minute -> STOP (conservative)
                exit_s, kind = stop_lvl, "stop"
            elif pi is not None:
                exit_s, kind = ghost, "tp"
            else:
                exit_s, kind = c1, "eod"
            rec["gross_stop"] = (o1 - exit_s) / o1 if short else (exit_s - o1) / o1
            rec["stop_kind"] = kind
        recs.append(rec)
    return panel, recs, missing_bars


# ----------------------------------------------------------------------------- aggregation
def build_cell(events: list, panel: pd.DataFrame) -> dict:
    """All T1 metrics for one (filter × direction) subset of event records."""
    n = len(events)
    out = {"n_events": n}
    if n < 10:
        out["flag"] = INSUFF
    touched = [e for e in events if e["touch"]]
    out["p_touch"] = prop_cell(len(touched), n)

    ttt = np.array([e["ttt_min"] for e in touched], float)
    n_t = len(ttt)
    out["time_to_touch"] = {
        "n_touch": n_t,
        "median_min": q(ttt, 50),
        "share_le30": prop_cell(int((ttt <= SE.QUICK_FILL_MIN).sum()), n_t),
        "share_le120": prop_cell(int((ttt <= 120).sum()), n_t),
        "share_gt120": prop_cell(int((ttt > 120).sum()), n_t),
    }
    if n_t < 10:
        out["time_to_touch"]["flag"] = INSUFF

    mae_t = [e["mae_em1"] for e in touched]
    mae_nt = [e["mae_em1"] for e in events if not e["touch"]]
    out["mae_em1"] = {
        "touched": {"n": len(mae_t), "median": q(mae_t, 50), "p90": q(mae_t, 90),
                    **({"flag": INSUFF} if len(mae_t) < 10 else {})},
        "non_touch_full_session": {"n": len(mae_nt), "median": q(mae_nt, 50), "p90": q(mae_nt, 90),
                                   **({"flag": INSUFF} if len(mae_nt) < 10 else {})},
    }

    out["ev_nostop"] = ev_cell(np.array([e["gross_nostop"] for e in events], float))
    out["ev_stop"] = ev_cell(np.array([e["gross_stop"] for e in events], float))
    out["ev_stop"]["exits"] = {k: sum(1 for e in events if e["stop_kind"] == k)
                               for k in ("tp", "stop", "eod")}

    # never-fill continuation: next-session (panel row indexed N) open->close, signed by gap side
    conts, skipped = [], 0
    for e in (x for x in events if not x["touch"]):
        if e["N"] in panel.index:
            nr = panel.loc[e["N"]]
            conts.append(math.copysign(1.0, e["dist"]) * (float(nr["c1"]) - float(nr["o1"])) / float(nr["o1"]))
        else:
            skipped += 1
    conts = np.array(conts, float)
    nf = ev_cell(conts) if len(conts) else {"n": 0, "flag": INSUFF}
    nf["hit"] = prop_cell(int((conts > 0).sum()), len(conts))
    nf["n_never_fill"] = len(conts) + skipped
    nf["skipped_no_next_panel_row"] = skipped
    out["never_fill_continuation"] = nf
    return out


def placebo_table(recs: list, events_by_filter: dict) -> dict:
    """Bucket table: conditional event touch rate vs unconditional placebo at d_mid×em1 from open."""
    n_all = len(recs)
    table = {}
    for (lo, hi) in SE.DIST_BUCKETS:
        d_mid = 2.5 if hi >= 99.0 else (lo + hi) / 2.0
        key = f"{lo}-{'2+' if hi >= 99.0 else hi}"
        k_dn = sum(1 for r in recs if r["day_low"] <= r["o1"] - d_mid * r["em1"])
        k_up = sum(1 for r in recs if r["day_high"] >= r["o1"] + d_mid * r["em1"])
        bucket = {"d_mid": d_mid,
                  "placebo": {"down": prop_cell(k_dn, n_all), "up": prop_cell(k_up, n_all)}}
        for filt, evs in events_by_filter.items():
            cond = {}
            for dirn in ("down", "up"):
                sub = [e for e in evs if e["direction"] == dirn and lo <= e["d_em"] < hi]
                cond[dirn] = prop_cell(sum(1 for e in sub if e["touch"]), len(sub))
            bucket[f"conditional_{filt}"] = cond
        table[key] = bucket
    return table


# ----------------------------------------------------------------------------- main per symbol
def run_sym(sym: str) -> dict:
    panel, recs, missing_bars = day_records(sym)
    ev_p = [r for r in recs if r["event_primary"] and r["direction"]]
    ev_a = [r for r in recs if r["event_alt"] and r["direction"]]
    alt_below_bucket = sum(1 for e in ev_a if e["d_em"] < SE.DIST_BUCKETS[0][0])

    cells = {}
    for filt, evs in (("primary", ev_p), ("alt", ev_a)):
        cells[filt] = {dirn: build_cell([e for e in evs if e["direction"] == dirn], panel)
                       for dirn in ("down", "up")}
        cells[filt]["pooled"] = build_cell(evs, panel)
    cells["placebo_buckets"] = placebo_table(recs, {"primary": ev_p, "alt": ev_a})

    caveats = [
        "PIT: row-D (D-EOD) ghost/em1/spot vs session-N 1-min bars only; entry at o1 = 09:30 open "
        "(15-min 14:45 entry-cutoff trivially satisfied; exits anytime).",
        "time-to-touch BAR-END convention: minutes from 09:30 to END of first touching 1-min bar "
        "(touch in 09:30 bar = 1 min); quick-fill = <=30 min (se_config.QUICK_FILL_MIN); offsets from "
        "timestamps (robust to missing minutes).",
        "MAE-before-touch INCLUDES the touching bar's adverse extreme (intra-bar ordering unknown -> "
        "conservative, MAE biased high); non-touch days = full-session adverse excursion from o1.",
        "Stop variant: hard stop at o1 adverse-side 1.0×em1 (T1_STOP_EM1), checked on 1-min bars from the "
        "09:30 entry bar onward; stop and TP in the SAME 1-min bar -> STOP counted first (conservative); "
        "stop fill price = stop level exactly (no slippage beyond COST_RT).",
        "Net = gross − COST_RT (2.0 bps) per round trip; hit rates reported on BOTH gross>0 and net>0.",
        f"ALT filter (0.15%×spot_D): {alt_below_bucket} event(s) with d_em < 0.25 fall below the lowest "
        "configured DIST_BUCKET and are EXCLUDED from the bucket table (still in alt direction/pooled cells); "
        "no new bucket invented (locked spec).",
        "Placebo = unconditional touch P over ALL panel days of a target at d_mid×em1 below/above o1 "
        "(d_mid = bucket midpoint, 2.5 for 2+), same-day em1, session extremes from 1-min bars; ghost is a "
        "real magnet only if conditional > placebo consistently per bucket+direction.",
        "Never-fill continuation needs the panel row indexed N (its o1/c1 = the FOLLOWING session); events "
        "whose N is not a panel index (e.g. the last panel day) are skipped (count reported per cell).",
        "Bucket assignment lo <= d_em < hi; d_em == 0.25 exactly -> first bucket (matches PRIMARY threshold).",
        "Sample = 237 sessions, single year 2025-06..2026-06 (one regime era); descriptive scenario scoring, "
        "no multiplicity correction — all cells reported (se_config multi-test note).",
        f"Sessions skipped for missing minute bars: {missing_bars}.",
    ]
    meta = {
        "sym": sym,
        "n_days": len(recs),
        "n_events_primary": len(ev_p),
        "n_events_alt": len(ev_a),
        "n_events_primary_down": sum(1 for e in ev_p if e["direction"] == "down"),
        "n_events_primary_up": sum(1 for e in ev_p if e["direction"] == "up"),
        "n_events_alt_down": sum(1 for e in ev_a if e["direction"] == "down"),
        "n_events_alt_up": sum(1 for e in ev_a if e["direction"] == "up"),
        "em1_pct_of_spot_median": round(float((panel["em1"] / panel["spot_D"]).median()) * 100, 3),
        "cost_rt_bps": SE.COST_RT * 1e4,
        "panel_window": [str(panel.index.min().date()), str(panel.index.max().date())],
    }
    return {"meta": meta, "cells": cells, "caveats": caveats}


# ----------------------------------------------------------------------------- stdout report
def fmt_prop(c):
    if c is None or c.get("rate") is None:
        return "  -  (n=%d)" % c.get("n", 0) if c else "  -"
    s = f"{100*c['rate']:5.1f}% [{100*c['wilson95'][0]:.1f},{100*c['wilson95'][1]:.1f}] n={c['n']}"
    return s + (" " + INSUFF if c.get("flag") else "")


def fmt_ev(c):
    if c.get("n", 0) == 0:
        return "  -   n=0 " + INSUFF
    s = (f"gross {c['mean_gross_bps']:+7.1f} / net {c['mean_net_bps']:+7.1f} bps  "
         f"t={c['t_gross'] if c['t_gross'] is not None else float('nan'):+5.2f}/"
         f"{c['t_net'] if c['t_net'] is not None else float('nan'):+5.2f}  "
         f"hit(g) {fmt_prop(c['hit_gross'])}")
    return s + (" " + INSUFF if c.get("flag") else "")


def print_report(res: dict):
    m = res["meta"]
    print(f"\n{'='*100}\nT1 GHOST GAP-FADE — {m['sym']}  |  days={m['n_days']}  "
          f"events: primary={m['n_events_primary']} (dn {m['n_events_primary_down']}/up {m['n_events_primary_up']})"
          f"  alt={m['n_events_alt']} (dn {m['n_events_alt_down']}/up {m['n_events_alt_up']})"
          f"  |  em1 median {m['em1_pct_of_spot_median']}% of spot\n{'='*100}")
    for filt in ("primary", "alt"):
        print(f"\n--- filter = {filt.upper()} "
              f"({'|o1-ghost| >= 0.25*em1' if filt == 'primary' else '|o1-ghost| >= 0.15%*spot_D'}) ---")
        for dirn in ("down", "up", "pooled"):
            c = res["cells"][filt][dirn]
            t = c["time_to_touch"]
            mae = c["mae_em1"]
            nf = c["never_fill_continuation"]
            print(f"\n  [{dirn.upper():6}] n_events={c['n_events']}"
                  + (f"  {INSUFF}" if c.get("flag") else ""))
            print(f"    P(touch ghost)     : {fmt_prop(c['p_touch'])}")
            print(f"    time-to-touch      : median {t['median_min']} min | <=30m {fmt_prop(t['share_le30'])}"
                  f" | <=120m {fmt_prop(t['share_le120'])} | >120m {fmt_prop(t['share_gt120'])}")
            print(f"    MAE before touch   : touched med {mae['touched']['median']} p90 {mae['touched']['p90']} em1"
                  f" (n={mae['touched']['n']}) | non-touch full-sess med {mae['non_touch_full_session']['median']}"
                  f" p90 {mae['non_touch_full_session']['p90']} (n={mae['non_touch_full_session']['n']})")
            print(f"    EV no-stop         : {fmt_ev(c['ev_nostop'])}")
            print(f"    EV 1.0xem1 stop    : {fmt_ev(c['ev_stop'])}  exits={c['ev_stop'].get('exits')}")
            print(f"    never-fill cont.   : n={nf['n']} (skip {nf['skipped_no_next_panel_row']})"
                  f"  mean {nf.get('mean_gross_bps')}/{nf.get('mean_net_bps')} bps (g/n)"
                  f"  hit {fmt_prop(nf['hit'])}" + (f"  {INSUFF}" if nf.get("flag") else ""))
    print(f"\n--- PLACEBO bucket table (conditional touch vs unconditional same-distance target) ---")
    print(f"  {'bucket':10} {'d_mid':5} {'dir':5} | {'placebo (all days)':38} | {'cond PRIMARY':38} | cond ALT")
    for key, b in res["cells"]["placebo_buckets"].items():
        for dirn in ("down", "up"):
            print(f"  {key:10} {b['d_mid']:5} {dirn:5} | {fmt_prop(b['placebo'][dirn]):38} | "
                  f"{fmt_prop(b['conditional_primary'][dirn]):38} | {fmt_prop(b['conditional_alt'][dirn])}")


def count_cells(res: dict) -> int:
    n = 0
    for filt in ("primary", "alt"):
        for dirn in ("down", "up", "pooled"):
            n += 6  # p_touch, time_to_touch, mae, ev_nostop, ev_stop, never_fill
    n += len(res["cells"]["placebo_buckets"]) * 2 * 3  # per bucket × dir × (placebo, cond_primary, cond_alt)
    return n


def main():
    SE.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    total_cells = 0
    for sym in SE.SYMS:
        res = run_sym(sym)
        out = SE.RESULTS_DIR / f"T1_{sym.lower()}.json"
        out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
        print_report(res)
        total_cells += count_cells(res)
        print(f"\n  JSON -> {out}")
    print(f"\nTOTAL REPORTED CELLS (both symbols): {total_cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
