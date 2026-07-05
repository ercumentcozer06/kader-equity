"""
T3 — GEX-FLIP RECLAIM REVERSAL study (se_config docstring = LOCKED spec).

Indicator claim: "a reclaim of the GEX Flip after a breakdown is one of the
highest-probability reversals".

EVENT per session N (levels row D):
  BREAKDOWN = first 15-min bar close < flip[D]
  RECLAIM   = first SUBSEQUENT 15-min bar close > flip[D], bar start <= 14:45 ET
TRADE: LONG at reclaim bar close -> exit at session close c1.
CONTROL: unconditional baseline — all panel days, all 15-min bars start <= 14:45,
  LONG bar-close -> session close; overall + by start hour; time-of-day-matched
  baseline weighted by the reclaim entries' hour distribution.
FAILED: breakdown days with NO reclaim -> EV breakdown bar close -> c1 (context).
NEXT-DAY: (c1p-o1p)/o1p from the panel row whose INDEX equals N.
REGIME CONTEXT: (h1-l1)/o1 and |c1/c0-1| by regime_own over all panel days.

Run:  cd kader-equity && <venv python> backtest/scenario_engine/T3_flip.py
Out:  backtest/scenario_engine/results/T3_{spy,qqq}.json
"""
from __future__ import annotations

import json
import sys
from datetime import time as dtime
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

COST_BPS = SE.COST_RT * 1e4                       # 2.0 bps round-trip
CUTOFF = dtime(*[int(x) for x in SE.ENTRY_CUTOFF.split(":")])  # 14:45 ET
Z95 = 1.959963984540054

CELLS_TESTED = 0  # global counter of reported stat cells


def wilson(k: int, n: int):
    if n == 0:
        return None, None
    p = k / n
    den = 1 + Z95 * Z95 / n
    ctr = (p + Z95 * Z95 / (2 * n)) / den
    half = Z95 * np.sqrt(p * (1 - p) / n + Z95 * Z95 / (4 * n * n)) / den
    return round(ctr - half, 4), round(ctr + half, 4)


def ev_stats(gross_bps: np.ndarray) -> dict:
    """Full EV cell: n, mean gross/net bps, t-stats, hit (gross>0 and net>0) + Wilson CI."""
    global CELLS_TESTED
    CELLS_TESTED += 1
    g = np.asarray(gross_bps, float)
    g = g[np.isfinite(g)]
    n = int(len(g))
    out = {"n": n}
    if n == 0:
        out.update({"mean_gross_bps": None, "mean_net_bps": None, "t_gross": None,
                    "t_net": None, "hit_gross": None, "hit_gross_wilson95": [None, None],
                    "hit_net": None, "hit_net_wilson95": [None, None], "flag": "YETERSIZ-N"})
        return out
    net = g - COST_BPS
    sd = float(np.std(g, ddof=1)) if n >= 2 else np.nan
    t_g = float(np.mean(g) / (sd / np.sqrt(n))) if n >= 2 and sd > 0 else None
    t_n = float(np.mean(net) / (sd / np.sqrt(n))) if n >= 2 and sd > 0 else None
    kg, kn = int((g > 0).sum()), int((net > 0).sum())
    out.update({
        "mean_gross_bps": round(float(np.mean(g)), 2),
        "mean_net_bps": round(float(np.mean(net)), 2),
        "median_gross_bps": round(float(np.median(g)), 2),
        "std_bps": round(sd, 2) if np.isfinite(sd) else None,
        "t_gross": round(t_g, 2) if t_g is not None else None,
        "t_net": round(t_n, 2) if t_n is not None else None,
        "hit_gross": round(kg / n, 4), "hit_gross_wilson95": list(wilson(kg, n)),
        "hit_net": round(kn / n, 4), "hit_net_wilson95": list(wilson(kn, n)),
    })
    if n < 10:
        out["flag"] = "YETERSIZ-N"
    return out


def split_cells(df: pd.DataFrame, col: str, valcol: str = "gross_bps") -> dict:
    cells = {}
    for v in sorted(df[col].dropna().unique()):
        sub = df[df[col] == v]
        cells[f"{int(v):+d}"] = ev_stats(sub[valcol].values)
    return cells


def build_events(sym: str):
    """Scan each session N's 15-min bars vs flip[D]; return events df + per-day records."""
    panel = pd.read_parquet(SE.panel_path(sym))
    mins = se_panel.rth_minutes(sym)
    b15 = se_panel.bars15(mins)
    bars_by_day = {d: g.sort_index() for d, g in b15.groupby("date")}

    ev_rows, n_flip_nan, n_no_bars = [], 0, 0
    for D, row in panel.iterrows():
        flip = row["flip"]
        if not np.isfinite(flip):
            n_flip_nan += 1
            continue
        N = pd.Timestamp(row["N"])
        bars = bars_by_day.get(N)
        if bars is None or len(bars) == 0:
            n_no_bars += 1
            continue
        closes = bars["c"].values
        starts = [ts.time() for ts in bars.index]
        bd_idx = next((i for i, c in enumerate(closes) if c < flip), None)
        if bd_idx is None:
            continue  # no breakdown this session
        opened_below = bool(row["o1"] < flip)
        # first SUBSEQUENT close > flip
        rc_idx_any = next((j for j in range(bd_idx + 1, len(closes)) if closes[j] > flip), None)
        rc_idx = rc_idx_any if (rc_idx_any is not None and starts[rc_idx_any] <= CUTOFF) else None
        rec = {
            "D": D, "N": N, "regime_own": int(row["regime_own"]),
            "regime_idx": int(row["regime_idx"]), "flip": flip, "c1": float(row["c1"]),
            "opened_below": opened_below,
            "bd_close": float(closes[bd_idx]), "bd_start": str(starts[bd_idx]),
            "reclaim": rc_idx is not None,
            "late_reclaim_after_cutoff": bool(rc_idx_any is not None and rc_idx is None),
        }
        if rc_idx is not None:
            entry = float(closes[rc_idx])
            rec.update({
                "entry": entry, "entry_hour": bars.index[rc_idx].hour,
                "entry_start": str(starts[rc_idx]),
                "gross_bps": (rec["c1"] / entry - 1.0) * 1e4,
            })
        else:
            rec["failed_gross_bps"] = (rec["c1"] / rec["bd_close"] - 1.0) * 1e4
        ev_rows.append(rec)
    ev = pd.DataFrame(ev_rows)
    return panel, bars_by_day, ev, n_flip_nan, n_no_bars


def control_baseline(panel: pd.DataFrame, bars_by_day: dict) -> tuple[dict, pd.DataFrame]:
    """Unconditional LONG bar-close -> session close over ALL panel days, bars start <= 14:45."""
    recs = []
    for D, row in panel.iterrows():
        N = pd.Timestamp(row["N"])
        bars = bars_by_day.get(N)
        if bars is None or len(bars) == 0:
            continue
        c1 = float(row["c1"])
        for ts, c in zip(bars.index, bars["c"].values):
            if ts.time() <= CUTOFF:
                recs.append({"hour": ts.hour, "gross_bps": (c1 / c - 1.0) * 1e4})
    cb = pd.DataFrame(recs)
    overall = ev_stats(cb["gross_bps"].values)
    by_hour = {}
    for h in sorted(cb["hour"].unique()):
        by_hour[f"{h:02d}"] = ev_stats(cb.loc[cb["hour"] == h, "gross_bps"].values)
    return {"overall": overall, "by_hour": by_hour}, cb


def run_sym(sym: str) -> dict:
    panel, bars_by_day, ev, n_flip_nan, n_no_bars = build_events(sym)
    rec = ev[ev["reclaim"]] if len(ev) else ev
    fail = ev[~ev["reclaim"]] if len(ev) else ev

    # --- reclaim trade cells ---
    reclaim = {"all": ev_stats(rec["gross_bps"].values if len(rec) else np.array([]))}
    reclaim["by_regime_own"] = split_cells(rec, "regime_own") if len(rec) else {}
    reclaim["by_regime_idx"] = split_cells(rec, "regime_idx") if len(rec) else {}

    # --- next-day follow-through: panel row whose INDEX equals N ---
    nd_rows, nd_drop = [], 0
    for _, e in rec.iterrows():
        N = pd.Timestamp(e["N"])
        if N in panel.index:
            pr = panel.loc[N]
            nd_rows.append({"regime_own": e["regime_own"],
                            "gross_bps": (float(pr["c1"]) / float(pr["o1"]) - 1.0) * 1e4})
        else:
            nd_drop += 1
    nd = pd.DataFrame(nd_rows)
    next_day = {"all": ev_stats(nd["gross_bps"].values if len(nd) else np.array([])),
                "by_regime_own": split_cells(nd, "regime_own") if len(nd) else {},
                "n_dropped_no_next_panel_row": nd_drop}

    # --- control ---
    control, cb = control_baseline(panel, bars_by_day)
    hour_dist = {}
    if len(rec):
        vc = rec["entry_hour"].value_counts(normalize=True).sort_index()
        hour_dist = {f"{int(h):02d}": round(float(w), 4) for h, w in vc.items()}
    matched = None
    if hour_dist:
        ms, ws = 0.0, 0.0
        for h, w in hour_dist.items():
            bh = control["by_hour"].get(h)
            if bh and bh["mean_gross_bps"] is not None:
                ms += w * bh["mean_gross_bps"]
                ws += w
        matched = round(ms / ws, 2) if ws > 0 else None
    control["reclaim_entry_hour_distribution"] = hour_dist
    control["matched_baseline_mean_gross_bps"] = matched
    control["matched_baseline_mean_net_bps"] = round(matched - COST_BPS, 2) if matched is not None else None
    ra = reclaim["all"]["mean_gross_bps"]
    control["reclaim_minus_matched_gross_bps"] = (
        round(ra - matched, 2) if (ra is not None and matched is not None) else None)

    # --- failed reclaim (context, not a trade) ---
    failed = {"all": ev_stats(fail["failed_gross_bps"].values if len(fail) else np.array([])),
              "by_regime_own": split_cells(fail, "regime_own", "failed_gross_bps") if len(fail) else {},
              "n_late_reclaim_after_cutoff": int(fail["late_reclaim_after_cutoff"].sum()) if len(fail) else 0}

    # --- regime context (all panel days) ---
    rng = (panel["h1"] - panel["l1"]) / panel["o1"]
    ccr = (panel["c1"] / panel["c0"] - 1.0).abs()
    regime_context = {}
    for v in (1, -1):
        m = panel["regime_own"] == v
        regime_context[f"{v:+d}"] = {
            "n": int(m.sum()),
            "intraday_range_pct_mean": round(float(rng[m].mean()) * 100, 3),
            "intraday_range_pct_median": round(float(rng[m].median()) * 100, 3),
            "abs_cc_ret_pct_mean": round(float(ccr[m].mean()) * 100, 3),
            "abs_cc_ret_pct_median": round(float(ccr[m].median()) * 100, 3),
        }

    n_bd = int(len(ev))
    n_rc = int(len(rec))
    meta = {
        "sym": sym, "n_days": int(len(panel)), "n_days_flip_nan_skipped": n_flip_nan,
        "n_days_no_bars": n_no_bars, "n_breakdowns": n_bd, "n_reclaims": n_rc,
        "n_failed": int(len(fail)),
        "share_breakdowns_opened_below_flip": round(float(ev["opened_below"].mean()), 4) if n_bd else None,
        "share_reclaims_opened_below_flip": round(float(rec["opened_below"].mean()), 4) if n_rc else None,
        "entry_cutoff_et": SE.ENTRY_CUTOFF, "cost_rt_bps": COST_BPS,
        "control_n_bars": control["overall"]["n"],
    }
    return {"meta": meta, "reclaim": reclaim, "next_day": next_day, "control": control,
            "failed": failed, "regime_context": regime_context,
            "_events": ev, "_panel": panel}


def fmt_cell(name: str, c: dict) -> str:
    if c["n"] == 0:
        return f"  {name:<22} n=0  YETERSIZ-N"
    w = c["hit_gross_wilson95"]
    fl = f"  [{c.get('flag','')}]" if c.get("flag") else ""
    return (f"  {name:<22} n={c['n']:<4} gross={c['mean_gross_bps']:>8.2f}bps "
            f"net={c['mean_net_bps']:>8.2f}bps t={c['t_gross'] if c['t_gross'] is not None else float('nan'):>5.2f} "
            f"hit={c['hit_gross']:.2%} CI[{w[0]:.2f},{w[1]:.2f}]{fl}")


def main():
    out_dir = SE.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    results, events_by_sym = {}, {}
    for sym in SE.SYMS:
        results[sym] = run_sym(sym)
        events_by_sym[sym] = results[sym].pop("_events")
        results[sym].pop("_panel")

    # --- pooled both-symbol cells (identical block embedded in both JSONs) ---
    pooled_rec = pd.concat([events_by_sym[s][events_by_sym[s]["reclaim"]] for s in SE.SYMS],
                           ignore_index=True)
    pooled = {"all": ev_stats(pooled_rec["gross_bps"].values),
              "by_regime_own": split_cells(pooled_rec, "regime_own"),
              "note": "SPY+QQQ pooled; days overlap across symbols -> NOT independent samples"}

    caveats_common = [
        "Tek donem (2025-06-13..2026-06-09, 237 gun) — tek-rejim ornek, cok-dongulu degil.",
        "RECLAIM tanimi kilitli-muhafazakar: ilk breakdown-sonrasi 15-dk close>flip BAR BASLANGICI <=14:45 ET olmali; "
        "ilk close>flip 14:45'ten SONRAki barda gerceklesen gunler FAILED sayildi (n_late_reclaim_after_cutoff).",
        "Acilis zaten flip altinda olan seanslar dogal olarak dahil (ozel-durum yok); pay meta'da: "
        "share_breakdowns_opened_below_flip / share_reclaims_opened_below_flip.",
        "FAILED hucresi islem degil baglam olcumu: breakdown bar-kapanisindan c1'e LONG-yonlu getiri, "
        "bar saati kisiti YOK (son barda breakdown ~0 getiri verir); net sutunu sadece simetri icin.",
        "close==flip esitligi kesin esitsizlikle ele alindi (ne breakdown ne reclaim).",
        "Kontrol bazi TUM panel gunleri (olay gunleri dahil) — kosulsuz, literal okuma.",
        "pooled_syms hucresi SPY+QQQ ayni gunleri icerir — bagimsiz orneklem DEGIL, t-stat iyimser.",
        "regime_idx==0 olan 1'er gun by_regime_idx'te ayri hucre olarak raporlanir (YETERSIZ-N).",
        "Cikis c1 = panel gunluk kapanis (= son 1-dk close); MOC slippage modellenmedi, maliyet sabit 2bps RT.",
    ]
    for sym in SE.SYMS:
        r = results[sym]
        r["reclaim"]["pooled_syms"] = pooled
        cav = list(caveats_common)
        cav.insert(0, f"{sym}: flip NaN {r['meta']['n_days_flip_nan_skipped']} gun atlandi; "
                      f"bar'i olmayan gun {r['meta']['n_days_no_bars']}.")
        r["caveats"] = cav
        path = out_dir / f"T3_{sym.lower()}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2, default=str)
        print(f"yazildi: {path}")

    # ---------- stdout table ----------
    for sym in SE.SYMS:
        r = results[sym]
        m = r["meta"]
        print(f"\n================ T3 FLIP-RECLAIM — {sym} ================")
        print(f"gun={m['n_days']}  breakdown={m['n_breakdowns']}  reclaim={m['n_reclaims']}  "
              f"failed={m['n_failed']}  (flip-NaN atlanan={m['n_days_flip_nan_skipped']})")
        print(f"acilis<flip payi: breakdown'larda {m['share_breakdowns_opened_below_flip']:.0%}, "
              f"reclaim'lerde {m['share_reclaims_opened_below_flip']:.0%}")
        print("RECLAIM LONG -> seans kapanisi:")
        print(fmt_cell("all", r["reclaim"]["all"]))
        for k, c in r["reclaim"]["by_regime_own"].items():
            print(fmt_cell(f"regime_own {k}", c))
        for k, c in r["reclaim"]["by_regime_idx"].items():
            print(fmt_cell(f"regime_idx {k}", c))
        print("NEXT-DAY (o->c, panel[N]):")
        print(fmt_cell("all", r["next_day"]["all"]))
        for k, c in r["next_day"]["by_regime_own"].items():
            print(fmt_cell(f"regime_own {k}", c))
        co = r["control"]
        print("KONTROL (kosulsuz bar-close->close, start<=14:45):")
        print(fmt_cell("overall", co["overall"]))
        for h, c in co["by_hour"].items():
            print(fmt_cell(f"hour {h}:xx", c))
        print(f"  reclaim saat-dagilimi: {co['reclaim_entry_hour_distribution']}")
        print(f"  saat-eslesik baz gross={co['matched_baseline_mean_gross_bps']}bps  "
              f"reclaim-eksi-baz={co['reclaim_minus_matched_gross_bps']}bps")
        print("FAILED (breakdown, reclaim yok; bd-close -> c1, baglam):")
        print(fmt_cell("all", r["failed"]["all"]))
        for k, c in r["failed"]["by_regime_own"].items():
            print(fmt_cell(f"regime_own {k}", c))
        print(f"  gec-reclaim(>14:45) failed icinde: {r['failed']['n_late_reclaim_after_cutoff']}")
        print("REJIM BAGLAMI (tum gunler):")
        for k, c in r["regime_context"].items():
            print(f"  regime_own {k}: n={c['n']}  range mean/med={c['intraday_range_pct_mean']}/"
                  f"{c['intraday_range_pct_median']}%  |cc-ret| mean/med={c['abs_cc_ret_pct_mean']}/"
                  f"{c['abs_cc_ret_pct_median']}%")
    print("\nPOOLED (SPY+QQQ, ayni gunler -> bagimsiz degil):")
    print(fmt_cell("all", pooled["all"]))
    for k, c in pooled["by_regime_own"].items():
        print(fmt_cell(f"regime_own {k}", c))
    print(f"\nTOPLAM RAPORLANAN ISTATISTIK HUCRESI (ev_stats): {CELLS_TESTED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
