"""
verify_T1.py — ADVERSARIAL independent recomputation of T1 (ghost gap-fade) key numbers.

Fresh code path: everything recomputed from panel_{sym}.parquet daily columns (o1/h1/l1/c1),
NOT from the study's per-minute event loop. Minute bars used only to CROSS-CHECK that the
panel session extremes equal the minute-derived extremes (so the panel path is valid), and
to validate the PIT join (N = first session after D; levels = livematch[D]).

Recomputed independently:
  1) PRIMARY event count per symbol (+ down/up split)
  2) pooled P(touch ghost) + per direction (Wilson 95)
  3) no-stop trade EV net bps pooled (enter o1, TP ghost if touched else exit c1)
  4) placebo touch rate, bucket 0.5-1.0 em1, DOWN direction (target = o1 - 0.75*em1, ALL days)
  5) never-fill next-session continuation mean bps, SPY pooled PRIMARY

Compare vs results/T1_{spy,qqq}.json. Tolerances: proportions +-1.5pp, EV +-2bps or +-10% rel,
counts exact +-1.

Run: cd kader-equity && <venv python> backtest/scenario_engine/results/verify_T1.py
Out: results/verify_T1_results.json + stdout table.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # .../scenario_engine/results
SCEN = HERE.parent                              # .../scenario_engine
sys.path.insert(0, str(SCEN))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import se_config as SE  # noqa: E402
import se_panel  # noqa: E402  (helpers allowed; used only for bars cross-check)

Z = 1.96


def wilson(k: int, n: int):
    if n == 0:
        return None, None, None
    p = k / n
    z2 = Z * Z
    c = (p + z2 / (2 * n)) / (1 + z2 / n)
    h = Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return p, c - h, c + h


def tstat(x: np.ndarray):
    n = len(x)
    if n < 2:
        return None
    s = float(np.std(x, ddof=1))
    return None if s == 0 else float(np.mean(x) / (s / math.sqrt(n)))


def pp(rate, lo, hi, n):
    return f"{100*rate:5.1f}% [{100*lo:.1f},{100*hi:.1f}] n={n}"


# ------------------------------------------------------------------ recompute per symbol
def recompute(sym: str) -> dict:
    panel = pd.read_parquet(SE.panel_path(sym))

    # ---- cross-check 1: panel o1/h1/l1/c1 == minute-derived session OHLC for session N
    mins = se_panel.rth_minutes(sym)
    daily = mins.groupby("date").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    chk = panel.join(daily, on="N")
    bars_ok = (np.allclose(chk["o1"], chk["o"]) and np.allclose(chk["h1"], chk["h"])
               and np.allclose(chk["l1"], chk["l"]) and np.allclose(chk["c1"], chk["c"]))

    # ---- cross-check 2 (PIT): N strictly after D and N = FIRST bar-session after D;
    #      panel levels equal livematch level_series at D (not at N)
    sessions = np.array(sorted(daily.index))
    first_after = {D: sessions[sessions > D][0] for D in panel.index}
    n_join_ok = all(panel.loc[D, "N"] == first_after[D] for D in panel.index)
    lv = SE.level_series(sym, "livematch")
    lv_d = lv.loc[panel.index]
    pit_levels_ok = (np.allclose(panel["ghost"], lv_d["ghost"]) and np.allclose(panel["em1"], lv_d["em1"])
                     and np.allclose(panel["spot_D"], lv_d["spot"]))

    o1 = panel["o1"].to_numpy(float)
    h1 = panel["h1"].to_numpy(float)
    l1 = panel["l1"].to_numpy(float)
    c1 = panel["c1"].to_numpy(float)
    ghost = panel["ghost"].to_numpy(float)
    em1 = panel["em1"].to_numpy(float)

    dist = o1 - ghost
    d_em = np.abs(dist) / em1
    down = dist > 0
    up = dist < 0

    # 1) PRIMARY events: |o1-ghost| >= 0.25*em1 (zero-gap days have no direction)
    ev = (d_em >= SE.EM1_MIN_DIST) & (dist != 0)
    ev_dn, ev_up = ev & down, ev & up

    # 2) touch ghost during session N: down -> session low <= ghost ; up -> session high >= ghost
    touch = np.where(down, l1 <= ghost, np.where(up, h1 >= ghost, False)).astype(bool)

    def prop(mask_num, mask_den):
        k, n = int((mask_num & mask_den).sum()), int(mask_den.sum())
        r, lo, hi = wilson(k, n)
        return {"k": k, "n": n, "rate": r, "lo": lo, "hi": hi}

    p_pool = prop(touch, ev)
    p_dn = prop(touch, ev_dn)
    p_up = prop(touch, ev_up)

    # 3) no-stop EV pooled: short (down) o1 -> ghost|c1 ; long (up) o1 -> ghost|c1
    exit_px = np.where(touch, ghost, c1)
    gross = np.where(down, (o1 - exit_px) / o1, (exit_px - o1) / o1)[ev]
    net = gross - SE.COST_RT
    ev_nostop = {"n": int(ev.sum()),
                 "mean_gross_bps": float(np.mean(gross)) * 1e4,
                 "mean_net_bps": float(np.mean(net)) * 1e4,
                 "t_net": tstat(net)}

    # 4) placebo, bucket 0.5-1.0, DOWN: unconditional P(session low <= o1 - 0.75*em1) over ALL days
    k_pl = int((l1 <= o1 - 0.75 * em1).sum())
    r, lo, hi = wilson(k_pl, len(panel))
    placebo_dn_05_1 = {"k": k_pl, "n": len(panel), "rate": r, "lo": lo, "hi": hi}
    # context: conditional touch rate of PRIMARY down events in d_em bucket [0.5, 1.0)
    in_b = ev_dn & (d_em >= 0.5) & (d_em < 1.0)
    cond_dn_05_1 = prop(touch, in_b)

    # 5) never-fill continuation (pooled PRIMARY): events not touched -> next-session o->c,
    #    signed by gap side, read from the panel row indexed N (its o1/c1 = following session)
    conts, skipped = [], 0
    for D in panel.index[ev & ~touch]:
        N = panel.loc[D, "N"]
        if N in panel.index:
            nr = panel.loc[N]
            conts.append(math.copysign(1.0, panel.loc[D, "o1"] - panel.loc[D, "ghost"])
                         * (float(nr["c1"]) - float(nr["o1"])) / float(nr["o1"]))
        else:
            skipped += 1
    conts = np.asarray(conts, float)
    hit_k = int((conts > 0).sum())
    hr, hlo, hhi = wilson(hit_k, len(conts))
    never_fill = {"n": int(len(conts)), "skipped": skipped,
                  "mean_gross_bps": float(np.mean(conts)) * 1e4,
                  "mean_net_bps": float(np.mean(conts - SE.COST_RT)) * 1e4,
                  "hit_rate": hr, "hit_lo": hlo, "hit_hi": hhi}

    return {"n_days": int(len(panel)), "bars_ok": bool(bars_ok), "n_join_ok": bool(n_join_ok),
            "pit_levels_ok": bool(pit_levels_ok),
            "events_primary": int(ev.sum()), "events_dn": int(ev_dn.sum()), "events_up": int(ev_up.sum()),
            "p_touch_pooled": p_pool, "p_touch_down": p_dn, "p_touch_up": p_up,
            "ev_nostop_pooled": ev_nostop,
            "placebo_dn_0.5-1.0": placebo_dn_05_1, "cond_dn_0.5-1.0": cond_dn_05_1,
            "never_fill_pooled": never_fill}


# ------------------------------------------------------------------ compare vs study JSON
def close_prop(a, b):  # proportions +-1.5pp
    return abs(a - b) <= 0.015


def close_ev(a, b):    # EV +-2 bps or +-10% relative
    return abs(a - b) <= 2.0 or (abs(b) > 1e-9 and abs(a - b) / abs(b) <= 0.10)


def close_cnt(a, b):   # counts exact +-1
    return abs(a - b) <= 1


def main():
    out = {"recompute": {}, "comparison": []}
    for sym in ("SPY", "QQQ"):
        r = recompute(sym)
        out["recompute"][sym] = r
        study = json.loads((HERE / f"T1_{sym.lower()}.json").read_text(encoding="utf-8"))
        m, prim = study["meta"], study["cells"]["primary"]

        rows = [
            (f"{sym} events PRIMARY", r["events_primary"], m["n_events_primary"], close_cnt),
            (f"{sym} events dn", r["events_dn"], m["n_events_primary_down"], close_cnt),
            (f"{sym} events up", r["events_up"], m["n_events_primary_up"], close_cnt),
            (f"{sym} P(touch) pooled", r["p_touch_pooled"]["rate"], prim["pooled"]["p_touch"]["rate"], close_prop),
            (f"{sym} P(touch) down", r["p_touch_down"]["rate"], prim["down"]["p_touch"]["rate"], close_prop),
            (f"{sym} P(touch) up", r["p_touch_up"]["rate"], prim["up"]["p_touch"]["rate"], close_prop),
            (f"{sym} EV no-stop net bps", r["ev_nostop_pooled"]["mean_net_bps"],
             prim["pooled"]["ev_nostop"]["mean_net_bps"], close_ev),
            (f"{sym} placebo dn 0.5-1.0", r["placebo_dn_0.5-1.0"]["rate"],
             study["cells"]["placebo_buckets"]["0.5-1.0"]["placebo"]["down"]["rate"], close_prop),
        ]
        if sym == "SPY":
            nf = prim["pooled"]["never_fill_continuation"]
            rows += [
                ("SPY never-fill n", r["never_fill_pooled"]["n"], nf["n"], close_cnt),
                ("SPY never-fill gross bps", r["never_fill_pooled"]["mean_gross_bps"], nf["mean_gross_bps"], close_ev),
                ("SPY never-fill net bps", r["never_fill_pooled"]["mean_net_bps"], nf["mean_net_bps"], close_ev),
            ]
        for name, mine, theirs, fn in rows:
            out["comparison"].append({"name": name, "verify": mine, "study": theirs,
                                      "match": bool(fn(mine, theirs))})

    # audit (d): every DIST_BUCKET cell present in both study JSONs
    bucket_keys = [f"{lo}-{'2+' if hi >= 99.0 else hi}" for lo, hi in SE.DIST_BUCKETS]
    audit_d = {}
    for sym in ("SPY", "QQQ"):
        study = json.loads((HERE / f"T1_{sym.lower()}.json").read_text(encoding="utf-8"))
        pb = study["cells"]["placebo_buckets"]
        ok = all(k in pb and all(
            d in pb[k][grp] for grp in ("placebo", "conditional_primary", "conditional_alt")
            for d in ("down", "up")) for k in bucket_keys)
        audit_d[sym] = {"all_bucket_cells_present": ok, "keys": list(pb.keys())}
    out["audit_buckets"] = audit_d

    # ------------------------------------------------------------- stdout
    print("=" * 96)
    print("VERIFY T1 — independent recomputation (panel path) vs study JSON")
    print("=" * 96)
    for sym in ("SPY", "QQQ"):
        r = out["recompute"][sym]
        print(f"\n[{sym}] days={r['n_days']}  bars_xcheck={'OK' if r['bars_ok'] else 'FAIL'}  "
              f"N-join={'OK' if r['n_join_ok'] else 'FAIL'}  PIT-levels(D)={'OK' if r['pit_levels_ok'] else 'FAIL'}")
        p = r["p_touch_pooled"]
        print(f"  events primary {r['events_primary']} (dn {r['events_dn']} / up {r['events_up']})"
              f"  | P(touch) pooled {pp(p['rate'], p['lo'], p['hi'], p['n'])}")
        d, u = r["p_touch_down"], r["p_touch_up"]
        print(f"  P(touch) down {pp(d['rate'], d['lo'], d['hi'], d['n'])} | up {pp(u['rate'], u['lo'], u['hi'], u['n'])}")
        e = r["ev_nostop_pooled"]
        print(f"  EV no-stop pooled gross {e['mean_gross_bps']:+.2f} / net {e['mean_net_bps']:+.2f} bps  "
              f"t_net {e['t_net']:+.2f}  n={e['n']}")
        pl, cd = r["placebo_dn_0.5-1.0"], r["cond_dn_0.5-1.0"]
        print(f"  placebo dn 0.5-1.0 {pp(pl['rate'], pl['lo'], pl['hi'], pl['n'])}"
              f"  | cond dn 0.5-1.0 {pp(cd['rate'], cd['lo'], cd['hi'], cd['n'])}"
              + ("  YETERSIZ-N" if cd["n"] < 10 else ""))
        nf = r["never_fill_pooled"]
        print(f"  never-fill cont. n={nf['n']} (skip {nf['skipped']})  gross {nf['mean_gross_bps']:+.2f} / "
              f"net {nf['mean_net_bps']:+.2f} bps  hit {pp(nf['hit_rate'], nf['hit_lo'], nf['hit_hi'], nf['n'])}")

    print("\n" + "-" * 96)
    print(f"{'metric':34} {'verify':>12} {'study':>12}  match")
    print("-" * 96)
    n_bad = 0
    for c in out["comparison"]:
        v = c["verify"] if isinstance(c["verify"], int) else round(float(c["verify"]), 4)
        s = c["study"] if isinstance(c["study"], int) else round(float(c["study"]), 4)
        flag = "OK" if c["match"] else "<<< MISMATCH"
        n_bad += 0 if c["match"] else 1
        print(f"{c['name']:34} {v:>12} {s:>12}  {flag}")
    for sym, a in audit_d.items():
        print(f"audit(d) {sym} all bucket cells present: {a['all_bucket_cells_present']}")
    print(f"\nMISMATCHES: {n_bad} / {len(out['comparison'])}")

    (HERE / "verify_T1_results.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"JSON -> {HERE / 'verify_T1_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
