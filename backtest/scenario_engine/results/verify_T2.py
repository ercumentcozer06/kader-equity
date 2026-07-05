"""
verify_T2.py — ADVERSARIAL VERIFIER for the T2 wall-study (T2_walls.py).
Independent re-implementation from PANEL + 1-min BARS (no import of T2_walls logic).

Recomputes:
  1) P(CW touch) and P(break|CW touch) per symbol per regime_own (+1/-1/all)
  2) S2 CW-BREAK-MOM Variant A net EV bps + n, QQQ, regime_own = -1 and +1
  3) S4 PW-BREAK-MOM Variant B net EV bps + n, SPY, regime_own = -1
  4) S1 skip count per symbol (all)
Plus free extras computed in the same pass (S2A/S4A/S4B 'all', S2B QQQ own splits,
P(VU|break) all) for additional cross-checks.

Definitions implemented fresh from the LOCKED spec (se_config.py docstring), using the
study's documented conservative readings where the spec is ambiguous:
  - touch(up): any 1-min high >= level; touch(down): any 1-min low <= level (whole session)
  - BREAK-CONFIRM: first 09:30-anchored 15-min bar CLOSE strictly through the wall whose
    bar START <= 14:45 ET; if the only close-through starts later -> neither break nor reject
    (study caveat 1; entries impossible there anyway)
  - entry at the confirm/touch bar close; TP/SL evaluated on bars strictly AFTER the entry bar
  - Variant A: TP = 1-min intrabar touch of VU/VD inside a 15-min bar; soft-stop = 15-min bar
    close back through the wall; same-bar TP+stop -> STOP (conservative tie, se_config)
  - Variant B: VU/VD 1-min touch disables the stop and holds to session close, UNLESS the same
    15-min bar also stop-closes -> STOP (study's conservative extension)
  - no TP/stop -> exit at session close (c1)
  - net = gross - 2.0 bps; t = mean / (std(ddof=1)/sqrt(n)) with the gross SD (constant shift)
  - S1 skip: CW-touch bar starts <= 14:45 AND closes < CW, but neither ghost nor mid_up sits
    at least TP_MIN_GAP below entry (NaN candidate -> condition False)

Run:  cd kader-equity && <venv-python> backtest/scenario_engine/results/verify_T2.py
Out:  backtest/scenario_engine/results/verify_T2.json + stdout table
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # .../scenario_engine/results
SEDIR = HERE.parent                              # .../scenario_engine
sys.path.insert(0, str(SEDIR))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import se_config as SE  # noqa: E402
import se_panel  # noqa: E402

Z = 1.959963984540054
CUT = pd.Timestamp(SE.ENTRY_CUTOFF).time()
COST = SE.COST_RT * 1e4
GAP = SE.TP_MIN_GAP
B15 = pd.Timedelta(minutes=15)


def wilson(k: int, n: int):
    if n == 0:
        return None, None, None
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fv(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def bar_start_of(ts: pd.Timestamp) -> pd.Timestamp:
    m = ts.hour * 60 + ts.minute
    bs = 570 + ((m - 570) // 15) * 15
    return ts.replace(hour=int(bs // 60), minute=int(bs % 60), second=0,
                      microsecond=0, nanosecond=0)


def sim_A(bd, md, entry_start, entry, side, tp, stop_fn, c1):
    """Variant A. Returns (gross_bps, reason). Tie (TP-touch + stop-close same bar) -> STOP."""
    for s, bar in bd[bd.index > entry_start].iterrows():
        sl = md[(md.index >= s) & (md.index < s + B15)]
        close = float(bar["c"])
        stop = stop_fn(close)
        tp_hit = False
        if tp is not None and len(sl):
            tp_hit = bool((sl["h"] >= tp).any()) if side > 0 else bool((sl["l"] <= tp).any())
        if stop:  # stop-first: covers both pure stop and the same-bar tie
            return side * (close / entry - 1.0) * 1e4, ("stop_tie" if tp_hit else "stop")
        if tp_hit:
            return side * (tp / entry - 1.0) * 1e4, "tp"
    return side * (c1 / entry - 1.0) * 1e4, "eod"


def sim_B(bd, md, entry_start, entry, side, casc, stop_fn, c1):
    """Variant B (cascade-hold). Touch of casc disables stop unless same bar stop-closes."""
    disabled = False
    for s, bar in bd[bd.index > entry_start].iterrows():
        sl = md[(md.index >= s) & (md.index < s + B15)]
        close = float(bar["c"])
        touched = False
        if casc is not None and len(sl):
            touched = bool((sl["h"] >= casc).any()) if side > 0 else bool((sl["l"] <= casc).any())
        stop = (not disabled) and stop_fn(close)
        if touched and stop:
            return side * (close / entry - 1.0) * 1e4, "stop_tie"
        if touched:
            disabled = True
            continue
        if stop:
            return side * (close / entry - 1.0) * 1e4, "stop"
    return side * (c1 / entry - 1.0) * 1e4, ("eod_cascade" if disabled else "eod")


def ev_stats(gross: list):
    n = len(gross)
    if n == 0:
        return {"n": 0, "mean_net_bps": None, "t_net": None}
    g = np.asarray(gross, float)
    mean_net = float(g.mean() - COST)
    sd = float(g.std(ddof=1)) if n >= 2 else None
    t = (mean_net / (sd / math.sqrt(n))) if (sd is not None and sd > 0) else None
    return {"n": n, "mean_net_bps": mean_net, "t_net": (None if t is None else float(t))}


def run_sym(sym: str) -> dict:
    panel = pd.read_parquet(SE.panel_path(sym))
    mins = se_panel.rth_minutes(sym)
    b15 = se_panel.bars15(mins)
    dm = {d: g for d, g in mins.groupby("date")}
    db = {d: g for d, g in b15.groupby("date")}
    sessions = sorted(dm.keys())

    audit = {"rows": len(panel), "join_mismatch": 0, "N_not_next_session": 0,
             "cw_late_only": 0, "tie_stops_A": 0, "tie_stops_B": 0}
    days = []
    s2a, s2b, s4a, s4b = [], [], [], []
    s1 = {"skips": 0, "trades": 0}
    s1_skips_le = 0  # study's <= boundary variant (sanity)

    for D, r in panel.iterrows():
        N = pd.Timestamp(r["N"])
        md, bd = dm.get(N), db.get(N)
        if md is None or bd is None or len(bd) == 0:
            continue
        # ---- audit (a): PIT join -- N must be the first session strictly after D,
        #      and o1/h1/l1/c1 must equal session-N minute aggregates
        nxt = [s for s in sessions if s > D]
        if not nxt or nxt[0] != N:
            audit["N_not_next_session"] += 1
        if (abs(float(md["o"].iloc[0]) - float(r["o1"])) > 1e-6
                or abs(float(md["c"].iloc[-1]) - float(r["c1"])) > 1e-6
                or abs(float(md["h"].max()) - float(r["h1"])) > 1e-6
                or abs(float(md["l"].min()) - float(r["l1"])) > 1e-6):
            audit["join_mismatch"] += 1

        cw, pw = fv(r["call_wall"]), fv(r["put_wall"])
        vu, vd = fv(r["vu"]), fv(r["vd"])
        ghost, mid_up = fv(r["ghost"]), fv(r["mid_up"])
        c1 = float(r["c1"])
        reg = int(r["regime_own"])
        ev = {"reg": reg, "cw_touch": None, "cw_break": None, "vu_reach": None}

        if cw is not None:
            th = md.index[md["h"] >= cw]
            ev["cw_touch"] = bool(len(th))
            ct = bd.index[bd["c"] > cw]
            brk = bool(len(ct) and ct[0].time() <= CUT)
            if len(ct) and ct[0].time() > CUT:
                audit["cw_late_only"] += 1
            ev["cw_break"] = brk
            # S1 skip / trade decision
            if len(th):
                tb = bar_start_of(th[0])
                if tb.time() <= CUT:
                    tbar = bd.loc[tb]
                    entry = float(tbar["c"])
                    if entry < cw:
                        tp = None
                        if ghost is not None and ghost < entry * (1 - GAP):
                            tp = ghost
                        elif mid_up is not None and mid_up < entry * (1 - GAP):
                            tp = mid_up
                        if tp is None:
                            s1["skips"] += 1
                        else:
                            s1["trades"] += 1
                        tp_le = None  # study's <= reading
                        if ghost is not None and ghost <= entry * (1 - GAP):
                            tp_le = ghost
                        elif mid_up is not None and mid_up <= entry * (1 - GAP):
                            tp_le = mid_up
                        if tp_le is None:
                            s1_skips_le += 1
            if brk:
                cb = ct[0]
                entry = float(bd.loc[cb, "c"])
                if vu is not None:
                    after = md[md.index >= cb + B15]
                    ev["vu_reach"] = bool((after["h"] >= vu).any())
                gA, rA = sim_A(bd, md, cb, entry, +1, vu, lambda c, w=cw: c < w, c1)
                gB, rB = sim_B(bd, md, cb, entry, +1, vu, lambda c, w=cw: c < w, c1)
                s2a.append({"reg": reg, "g": gA})
                s2b.append({"reg": reg, "g": gB})
                audit["tie_stops_A"] += int(rA == "stop_tie")
                audit["tie_stops_B"] += int(rB == "stop_tie")

        if pw is not None:
            pct_ = bd.index[bd["c"] < pw]
            pbrk = bool(len(pct_) and pct_[0].time() <= CUT)
            if pbrk:
                pb = pct_[0]
                entry = float(bd.loc[pb, "c"])
                gA, rA = sim_A(bd, md, pb, entry, -1, vd, lambda c, w=pw: c > w, c1)
                gB, rB = sim_B(bd, md, pb, entry, -1, vd, lambda c, w=pw: c > w, c1)
                s4a.append({"reg": reg, "g": gA})
                s4b.append({"reg": reg, "g": gB})
                audit["tie_stops_A"] += int(rA == "stop_tie")
                audit["tie_stops_B"] += int(rB == "stop_tie")
        days.append(ev)

    def probs(rg):
        sub = days if rg == "all" else [d for d in days if d["reg"] == int(rg)]
        den = [d for d in sub if d["cw_touch"] is not None]
        k_t = sum(d["cw_touch"] for d in den)
        p_t, lo_t, hi_t = wilson(k_t, len(den))
        tch = [d for d in den if d["cw_touch"]]
        k_b = sum(d["cw_break"] for d in tch)
        p_b, lo_b, hi_b = wilson(k_b, len(tch))
        return {"cw_touch": {"k": k_t, "n": len(den), "p": p_t, "lo": lo_t, "hi": hi_t},
                "break_given_touch": {"k": k_b, "n": len(tch), "p": p_b, "lo": lo_b, "hi": hi_b}}

    def cell(lst, rg):
        g = [x["g"] for x in lst if rg == "all" or x["reg"] == int(rg)]
        return ev_stats(g)

    vu_den = [d for d in days if d["cw_break"] and d["vu_reach"] is not None]
    vu_k = sum(d["vu_reach"] for d in vu_den)
    return {
        "audit": audit,
        "probs_own": {rg: probs(rg) for rg in ("+1", "-1", "all")},
        "p_vu_given_break_all": {"k": vu_k, "n": len(vu_den),
                                 "p": (vu_k / len(vu_den) if vu_den else None)},
        "S1": {**s1, "skips_le_boundary": s1_skips_le},
        "S2A": {rg: cell(s2a, rg) for rg in ("+1", "-1", "all")},
        "S2B": {rg: cell(s2b, rg) for rg in ("+1", "-1", "all")},
        "S4A": {rg: cell(s4a, rg) for rg in ("+1", "-1", "all")},
        "S4B": {rg: cell(s4b, rg) for rg in ("+1", "-1", "all")},
    }


# ---------------------------------------------------------------------------- comparison
def close_pp(a, b, tol=0.015):
    return a is not None and b is not None and abs(a - b) <= tol


def close_ev(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= 2.0 or (abs(b) > 1e-12 and abs(a - b) / abs(b) <= 0.10)


def close_n(a, b):
    return abs(int(a) - int(b)) <= 1


def main():
    out = {"comparisons": [], "verify": {}}
    study = {s: json.loads((HERE / f"T2_{s}.json").read_text(encoding="utf-8"))
             for s in ("spy", "qqq")}
    comps = out["comparisons"]

    for sym in ("SPY", "QQQ"):
        v = run_sym(sym)
        out["verify"][sym] = v
        sj = study[sym.lower()]
        po = sj["prob_table_own"]
        so = sj["setups_own"]

        # 1) P(CW touch), P(break|touch) per regime_own
        for rg in ("+1", "-1", "all"):
            mine = v["probs_own"][rg]
            st_t = po[rg]["p_cw_touch"]
            st_b = po[rg]["p_break_given_cw_touch"]
            comps.append({"name": f"{sym} P(CW touch) own={rg}",
                          "study": f"p={st_t['p']:.4f} k={st_t['k']} n={st_t['n']}",
                          "mine": f"p={mine['cw_touch']['p']:.4f} k={mine['cw_touch']['k']} n={mine['cw_touch']['n']}",
                          "match": bool(close_pp(mine["cw_touch"]["p"], st_t["p"])
                                        and close_n(mine["cw_touch"]["n"], st_t["n"])
                                        and close_n(mine["cw_touch"]["k"], st_t["k"]))})
            comps.append({"name": f"{sym} P(break|CW touch) own={rg}",
                          "study": f"p={st_b['p']:.4f} k={st_b['k']} n={st_b['n']}",
                          "mine": f"p={mine['break_given_touch']['p']:.4f} k={mine['break_given_touch']['k']} n={mine['break_given_touch']['n']}",
                          "match": bool(close_pp(mine["break_given_touch"]["p"], st_b["p"])
                                        and close_n(mine["break_given_touch"]["n"], st_b["n"])
                                        and close_n(mine["break_given_touch"]["k"], st_b["k"]))})

        # 4) S1 skip count (all)
        st_sk = so["S1_CW_REJECT_FADE"]["all"]["skips"]
        comps.append({"name": f"{sym} S1 skips (all)",
                      "study": str(st_sk), "mine": str(v["S1"]["skips"]),
                      "match": bool(close_n(v["S1"]["skips"], st_sk))})
        # free extras: S2A/S4A/S4B all
        for st_name, key in (("S2A_CW_BREAK_VU_TP", "S2A"), ("S4A_PW_BREAK_VD_TP", "S4A"),
                             ("S4B_PW_BREAK_CASCADE", "S4B")):
            sc = so[st_name]["all"]
            mc = v[key]["all"]
            comps.append({"name": f"{sym} {key} all net bps (extra)",
                          "study": f"{sc['mean_net_bps']:.2f} n={sc['n']}",
                          "mine": f"{mc['mean_net_bps']:.2f} n={mc['n']}",
                          "match": bool(close_ev(mc["mean_net_bps"], sc["mean_net_bps"])
                                        and close_n(mc["n"], sc["n"]))})

    # 2) S2A QQQ own -1 / +1
    for rg in ("-1", "+1"):
        sc = study["qqq"]["setups_own"]["S2A_CW_BREAK_VU_TP"][rg]
        mc = out["verify"]["QQQ"]["S2A"][rg]
        comps.append({"name": f"QQQ S2A net bps own={rg}",
                      "study": f"{sc['mean_net_bps']:.2f} t={sc['t_net']:.2f} n={sc['n']}",
                      "mine": f"{mc['mean_net_bps']:.2f} t={mc['t_net']:.2f} n={mc['n']}",
                      "match": bool(close_ev(mc["mean_net_bps"], sc["mean_net_bps"])
                                    and close_n(mc["n"], sc["n"]))})
        # free extra: S2B QQQ own splits (study's claimed best cell)
        sc = study["qqq"]["setups_own"]["S2B_CW_BREAK_CASCADE"][rg]
        mc = out["verify"]["QQQ"]["S2B"][rg]
        comps.append({"name": f"QQQ S2B net bps own={rg} (extra)",
                      "study": f"{sc['mean_net_bps']:.2f} n={sc['n']}",
                      "mine": f"{mc['mean_net_bps']:.2f} n={mc['n']}",
                      "match": bool(close_ev(mc["mean_net_bps"], sc["mean_net_bps"])
                                    and close_n(mc["n"], sc["n"]))})

    # 3) S4B SPY own -1
    sc = study["spy"]["setups_own"]["S4B_PW_BREAK_CASCADE"]["-1"]
    mc = out["verify"]["SPY"]["S4B"]["-1"]
    comps.append({"name": "SPY S4B net bps own=-1",
                  "study": f"{sc['mean_net_bps']:.2f} t={sc['t_net']:.2f} n={sc['n']}",
                  "mine": f"{mc['mean_net_bps']:.2f} t={mc['t_net']:.2f} n={mc['n']}",
                  "match": bool(close_ev(mc["mean_net_bps"], sc["mean_net_bps"])
                                and close_n(mc["n"], sc["n"]))})

    # P(VU|break) all, extra
    for sym in ("SPY", "QQQ"):
        sc = study[sym.lower()]["prob_table_own"]["all"]["p_vu_given_cw_break"]
        mc = out["verify"][sym]["p_vu_given_break_all"]
        comps.append({"name": f"{sym} P(VU|CW break) all (extra)",
                      "study": f"p={sc['p']:.4f} k={sc['k']} n={sc['n']}",
                      "mine": f"p={mc['p']:.4f} k={mc['k']} n={mc['n']}",
                      "match": bool(close_pp(mc["p"], sc["p"]) and close_n(mc["n"], sc["n"]))})

    (HERE / "verify_T2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"{'comparison':42s} {'study':>30s} {'verify':>30s}  match")
    print("-" * 112)
    for c in comps:
        print(f"{c['name']:42s} {c['study']:>30s} {c['mine']:>30s}  {'OK' if c['match'] else '** MISMATCH **'}")
    n_bad = sum(not c["match"] for c in comps)
    for sym in ("SPY", "QQQ"):
        a = out["verify"][sym]["audit"]
        s1 = out["verify"][sym]["S1"]
        print(f"\n{sym} audit: rows={a['rows']} join_mismatch={a['join_mismatch']} "
              f"N_not_next={a['N_not_next_session']} cw_late_only={a['cw_late_only']} "
              f"tieA={a['tie_stops_A']} tieB={a['tie_stops_B']} | "
              f"S1 skips(strict<)={s1['skips']} skips(<=)={s1['skips_le_boundary']} trades={s1['trades']}")
    print(f"\nMISMATCHES: {n_bad}/{len(comps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
