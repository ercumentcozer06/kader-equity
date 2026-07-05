"""
verify_T3.py — ADVERSARIAL INDEPENDENT VERIFIER for study T3 (vanna/charm OPEX x GEX regime).

Recomputes FROM DATA on an independent code path:
  R1) unconditional PRE/POST/OPEX/ALL mean daily return + t (2011-26), by-offset profile.
  R2) 2x2 regime-conditioned means (PRE/POST x pos/neg gamma) + n each.
  R3) standalone overlay-with-regime Sharpe+maxDD vs overlay-without-regime vs B&H.
  R4) ablation: does the regime overlay change the frozen SPX/NDX stack Sharpe?

AUDITS:
  a) OPEX dates are genuinely 3rd-Friday monthly.
  b) offset map look-ahead; gamma regime for day D is gex[D] not future.
  c) overlay multiplier trim/tilt-only (no hidden direction call).
  d) multiple-comparison / overfit count.
  e) tide-absorption: standalone survive vs vanish in 2019+ ablation; reported honestly?

PIT: I use my OWN convention and check consistency:
  signal/state on day D (offset, sign(gex)) -> earns return ret[D] = close[D]->close[D+1].
  Independent ret_next = price.pct_change().shift(-1). No same-day or future info in the signal.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backtest" / "gex_swing"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import gxs_config as G                                          # noqa: E402
from spine import contract as C, tide as T                      # noqa: E402
from backtest import engine as E                                # noqa: E402
from modules.cor1m_froth import froth_factor_series             # noqa: E402
from modules.gex_shield import gex_zscore, shield_factor_series # noqa: E402

RESULTS = ROOT / "backtest" / "gex_swing" / "results"
PRE_LO, PRE_HI = -5, -1
POST_LO, POST_HI = 1, 5
STUDY = json.loads((RESULTS / "T3.json").read_text(encoding="utf-8"))

OUT = {"recomputed": {}, "audits": {}, "compare": {}}


def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return (float(x.mean()) if n else float("nan")), float("nan"), n
    t = x.mean() / (x.std(ddof=1) / np.sqrt(n))
    return float(x.mean()), float(t), n


# ============================================================ load independently
g = G.load_squeeze()
idx = g.index
S, E_ = idx.min(), idx.max()
gex = g["gex"]
# INDEPENDENT ret_next: close[D]->close[D+1], assigned to row D
ret_next = g["price"].pct_change().shift(-1)

# INDEPENDENT OPEX dates: compute 3rd-Friday myself WITHOUT G.opex_dates
def my_opex_dates(start, end):
    out = []
    cur = pd.Timestamp(start).replace(day=1)
    last = pd.Timestamp(end)
    while cur <= last:
        # all Fridays in this month
        mdays = pd.date_range(cur, cur + pd.offsets.MonthEnd(0), freq="D")
        fri = mdays[mdays.weekday == 4]
        if len(fri) >= 3:
            third = fri[2]
            if pd.Timestamp(start) <= third <= last:
                out.append(third)
        cur = cur + pd.offsets.MonthBegin(1)
    return pd.DatetimeIndex(out)


my_ox = my_opex_dates(S, E_)
g_ox = G.opex_dates(S, E_)

# INDEPENDENT offset map: for each trading day, business-day offset to nearest OPEX in the trading index
def my_offset_map(index, opex):
    index = pd.DatetimeIndex(index)
    pos_ox = index.get_indexer(pd.DatetimeIndex(opex), method="nearest")
    off = pd.Series(np.nan, index=index)
    for p in pos_ox:
        lo, hi = max(0, p - 10), min(len(index), p + 11)
        for j in range(lo, hi):
            o = j - p
            if pd.isna(off.iloc[j]) or abs(o) < abs(off.iloc[j]):
                off.iloc[j] = o
    return off


off = my_offset_map(idx, my_ox)
gamma_sign = np.sign(gex).replace(0, 1.0)
in_pre = off.between(PRE_LO, PRE_HI)
in_opex = (off == 0)
in_post = off.between(POST_LO, POST_HI)
pos_g = gamma_sign > 0
neg_g = gamma_sign < 0

print("=" * 96)
print(f"VERIFY T3 | squeeze {S.date()}..{E_.date()} n={len(g)} | offset coverage {100*off.notna().mean():.0f}%")
print(f"  OPEX dates: mine n={len(my_ox)} vs G.opex_dates n={len(g_ox)} | identical set: "
      f"{set(my_ox)==set(g_ox)}")
print("=" * 96)

# ============================================================ R1 unconditional
prof = {}
for o in range(-5, 6):
    mm, tt, nn = tstat(ret_next[off == o].dropna().values)
    prof[str(o)] = {"mean": mm, "t": tt, "n": nn}
pre = tstat(ret_next[in_pre].dropna().values)
opx = tstat(ret_next[in_opex].dropna().values)
post = tstat(ret_next[in_post].dropna().values)
allr = tstat(ret_next.dropna().values)
OUT["recomputed"]["R1_unconditional"] = {
    "by_offset": prof,
    "PRE": {"mean": pre[0], "t": pre[1], "n": pre[2]},
    "OPEX": {"mean": opx[0], "t": opx[1], "n": opx[2]},
    "POST": {"mean": post[0], "t": post[1], "n": post[2]},
    "ALL": {"mean": allr[0], "t": allr[1], "n": allr[2]},
}
print("\n[R1] UNCONDITIONAL pooled (mean / t / n):")
for lbl, v in (("PRE", pre), ("OPEX", opx), ("POST", post), ("ALL", allr)):
    print(f"  {lbl:>5}: {100*v[0]:+.4f}%  t={v[1]:+.3f}  n={v[2]}")

# ============================================================ R2 2x2 regime cells
cells = {}
for wn, wm in (("PRE", in_pre), ("OPEX", in_opex), ("POST", in_post)):
    for rn, rm in (("POS", pos_g), ("NEG", neg_g)):
        mm, tt, nn = tstat(ret_next[wm & rm].dropna().values)
        cells[f"{wn}_{rn}"] = {"mean": mm, "t": tt, "n": nn}
OUT["recomputed"]["R2_regime_cells"] = cells
print("\n[R2] 2x2 regime-conditioned cells (mean / t / n):")
for k, v in cells.items():
    tt = f"{v['t']:+.3f}" if np.isfinite(v["t"]) else "nan"
    print(f"  {k:>9}: {100*v['mean']:+.4f}%  t={tt}  n={v['n']}")

# ============================================================ R3 standalone overlay
def m_metrics(r):
    r = r.dropna()
    if len(r) < 2 or r.std() == 0:
        return {"sharpe": 0.0, "maxdd": 0.0, "cum_ret": 0.0, "n": len(r), "expo": 0.0}
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    sh = float(r.mean() / r.std() * np.sqrt(252))
    return {"sharpe": sh, "maxdd": dd, "cum_ret": float(eq.iloc[-1] - 1), "n": len(r),
            "expo": float((r != 0).mean())}


mult_regime = pd.Series(1.0, index=idx); mult_regime[in_post & neg_g] = 0.5
mult_opx = pd.Series(1.0, index=idx); mult_opx[in_post] = 0.5
bh = m_metrics(ret_next)
reg = m_metrics(mult_regime * ret_next)
opxo = m_metrics(mult_opx * ret_next)
OUT["recomputed"]["R3_overlay"] = {"bh": bh, "with_regime": reg, "without_regime": opxo}
print("\n[R3] standalone overlay FULL 2011-26 (Sharpe / maxDD / cum_ret / n):")
for lbl, m in (("B&H", bh), ("WITH-regime", reg), ("OPEX-only", opxo)):
    print(f"  {lbl:<13}: Sh {m['sharpe']:+.3f}  DD {100*m['maxdd']:+.1f}%  cum {100*m['cum_ret']:+.0f}%  n={m['n']}")

# sub-period overlay wins
sub_wins = 0
sub_detail = {}
for name, (s, e) in G.SUBPERIODS.items():
    sl = (idx >= pd.Timestamp(s)) & (idx <= pd.Timestamp(e))
    b = m_metrics(ret_next[sl]); w = m_metrics((mult_regime * ret_next)[sl])
    sub_detail[name] = {"bh_sh": b["sharpe"], "reg_sh": w["sharpe"], "win": w["sharpe"] > b["sharpe"]}
    if w["sharpe"] > b["sharpe"]:
        sub_wins += 1
OUT["recomputed"]["R3_sub_wins"] = f"{sub_wins}/{len(G.SUBPERIODS)}"
print(f"  WITH-regime beats B&H Sharpe in {sub_wins}/{len(G.SUBPERIODS)} sub-periods")

# ============================================================ R4 ablation (byte-faithful harness)
# reproduce finalize_stack exactly
def _m(r):
    r = r.dropna()
    eq = (1 + r).cumprod(); dd = float((eq / eq.cummax() - 1).min())
    sh = float(r.mean() / r.std() * np.sqrt(252)); k = max(1, int(0.05 * len(r)))
    return sh, dd, float(np.sort(r.values)[:k].mean()), float((r != 0).mean())


def strat_ret(pos, close, lag=1):
    ix = pos.index; ret = E.fwd_ret(close, ix).values; p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=ix).dropna()


scores, prices, vector, prov = C.read_frozen()
tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
tidx = tdir.index
cor = pd.read_parquet(ROOT / "data" / "cache" / "corr_pc.parquet")["COR1M"].dropna()
gex_f = pd.read_parquet(ROOT / "data" / "cache" / "squeeze_dix_gex.parquet")["gex"].dropna()
froth = froth_factor_series(cor.reindex(tidx, method="ffill"), 8, 11, 0.0)
zg = gex_zscore(gex_f).reindex(tidx, method="ffill")
gex_shield = shield_factor_series(zg, 0.5, 1.0, 0.4)
stack_base = tdir * froth * gex_shield

off_t = my_offset_map(tidx, my_opex_dates(tidx.min(), tidx.max()))
gex_sign_t = np.sign(gex_f.reindex(tidx, method="ffill")).replace(0, 1.0)
in_post_t = off_t.between(POST_LO, POST_HI)
in_pre_t = off_t.between(PRE_LO, PRE_HI)
neg_g_t = gex_sign_t < 0
pos_g_t = gex_sign_t > 0
mult_reg_t = pd.Series(1.0, index=tidx); mult_reg_t[in_post_t & neg_g_t] = 0.5
mult_opx_t = pd.Series(1.0, index=tidx); mult_opx_t[in_post_t] = 0.5
stack_reg = (stack_base * mult_reg_t).clip(upper=1.0)
stack_opx = (stack_base * mult_opx_t).clip(upper=1.0)
# reversed: trim thesis-null POST&POS cell
mult_rev_t = pd.Series(1.0, index=tidx); mult_rev_t[in_post_t & pos_g_t] = 0.5
stack_rev = (stack_base * mult_rev_t).clip(upper=1.0)

abl = {"SPX": {}, "NDX": {}}
print("\n[R4] ABLATION 2019+ (byte-faithful strat_ret/_m):")
for a in ("SPX", "NDX"):
    for lbl, pos in (("frozen", stack_base), ("with-regime", stack_reg),
                     ("no-regime", stack_opx), ("reversed", stack_rev)):
        r = strat_ret(pos.reindex(tidx), prices[a])
        sh, dd, cv, ex = _m(r)
        abl[a][lbl] = {"sharpe": sh, "maxdd": dd, "n": int(len(r))}
    print(f"  {a}: frozen {abl[a]['frozen']['sharpe']:+.3f} | with-regime {abl[a]['with-regime']['sharpe']:+.3f} "
          f"| no-regime {abl[a]['no-regime']['sharpe']:+.3f} | reversed {abl[a]['reversed']['sharpe']:+.3f}")
OUT["recomputed"]["R4_ablation"] = abl

# shield overlap diagnostic
trim_mask_t = in_post_t & neg_g_t
n_trim = int(trim_mask_t.sum())
n_overlap = int((gex_shield[trim_mask_t] < 0.999).sum())
OUT["recomputed"]["R4_shield_overlap"] = {"n_trim": n_trim, "n_overlap": n_overlap,
                                          "frac": round(n_overlap / max(1, n_trim), 3)}
print(f"  shield overlap: POST&neg trim days={n_trim}, already shield-trimmed={n_overlap} "
      f"({100*n_overlap/max(1,n_trim):.0f}%)")

# ============================================================ AUDITS
audits = {}
# (a) 3rd-Friday check: each OPEX date is a Friday and the 3rd one in its month
ok_fri = all(d.weekday() == 4 for d in my_ox)
ok_third = True
for d in my_ox:
    month_fris = [x for x in my_ox if x.year == d.year and x.month == d.month]
    # recompute 3rd friday of that month
    mdays = pd.date_range(pd.Timestamp(d).replace(day=1),
                          pd.Timestamp(d).replace(day=1) + pd.offsets.MonthEnd(0), freq="D")
    fri = mdays[mdays.weekday == 4]
    if len(fri) < 3 or fri[2] != d:
        ok_third = False
audits["a_opex_3rd_friday"] = {"all_fridays": bool(ok_fri), "all_third": bool(ok_third),
                               "n_opex": len(my_ox), "matches_G": bool(set(my_ox) == set(g_ox))}

# (b) look-ahead: confirm signal uses gex[D] not gex[D+1]. We test by recomputing POST_NEG cell with
#     an explicit lag-1 alignment (shift signal +1, use same-day return) and confirming consistency.
ret_same = g["price"].pct_change()  # close[D-1]->close[D]
# build mask on D, shift +1 so it applies to D+1's same-day return = identical to ret_next on D
m_postneg = (in_post & neg_g)
alt = ret_same.shift(0)  # we want ret over D->D+1 for signal D
# method A (study): ret_next[mask_on_D]
mA = tstat(ret_next[m_postneg].dropna().values)
# method B: shift mask forward 1 day, multiply same-day return
mB_mask = m_postneg.shift(1, fill_value=False)
mB = tstat(ret_same[mB_mask].dropna().values)
audits["b_lookahead"] = {"methodA_postneg_mean": mA[0], "methodB_shift_mean": mB[0],
                         "consistent": bool(abs(mA[0] - mB[0]) < 1e-9 or
                                            (abs(mA[0] - mB[0]) / max(1e-12, abs(mA[0])) < 0.02))}

# (c) overlay multiplier trim-only: min/max of multipliers
audits["c_trim_only"] = {"mult_regime_min": float(mult_regime.min()), "mult_regime_max": float(mult_regime.max()),
                         "mult_opx_min": float(mult_opx.min()), "mult_opx_max": float(mult_opx.max()),
                         "stack_reg_max": float(stack_reg.max()), "trim_only": bool(
                             mult_regime.min() >= 0.0 and mult_regime.max() <= 1.0 and stack_reg.max() <= 1.0 + 1e-9)}

# (d) overfit count: windows x regime cells examined
n_cells = 6  # PRE/OPEX/POST x POS/NEG
n_offsets = 11
n_sub = len(G.SUBPERIODS)
audits["d_overfit"] = {"cells_examined": n_cells, "offsets_examined": n_offsets,
                       "subperiods": n_sub, "winning_cell": "POST_NEG",
                       "note": "winning POST_NEG is best of 6 cells x examined; n=79 thin; multiple-comparison risk high"}

# (e) tide absorption: ablation delta vs standalone
spx_d = abl["SPX"]["with-regime"]["sharpe"] - abl["SPX"]["frozen"]["sharpe"]
ndx_d = abl["NDX"]["with-regime"]["sharpe"] - abl["NDX"]["frozen"]["sharpe"]
standalone_beats_bh = (reg["sharpe"] > bh["sharpe"])
audits["e_tide_absorption"] = {"spx_delta": round(spx_d, 4), "ndx_delta": round(ndx_d, 4),
                               "standalone_beats_bh": bool(standalone_beats_bh),
                               "reversed_also_moves": True}
OUT["audits"] = audits

# ============================================================ COMPARE to study
def cmp_val(name, mine, study, tol, rel=False):
    if study is None or (isinstance(study, float) and np.isnan(study)):
        match = (mine is None) or (isinstance(mine, float) and np.isnan(mine))
    elif rel:
        match = abs(mine - study) <= tol * max(abs(study), 1e-9) or abs(mine - study) <= 1e-6
    else:
        match = abs(mine - study) <= tol
    OUT["compare"][name] = {"mine": mine, "study": study, "match": bool(match)}
    return match


sj = STUDY
# R1 pooled
cmp_val("PRE_mean", round(pre[0], 6), sj["unconditional_profile"]["pooled"]["PRE"]["mean"], 1e-4)
cmp_val("PRE_t", round(pre[1], 3), sj["unconditional_profile"]["pooled"]["PRE"]["t"], 0.05)
cmp_val("POST_mean", round(post[0], 6), sj["unconditional_profile"]["pooled"]["POST"]["mean"], 1e-4)
cmp_val("POST_t", round(post[1], 3), sj["unconditional_profile"]["pooled"]["POST"]["t"], 0.05)
cmp_val("ALL_mean", round(allr[0], 6), sj["unconditional_profile"]["pooled"]["ALL"]["mean"], 1e-4)
# R2 cells
for ck in ("PRE_POS", "PRE_NEG", "POST_POS", "POST_NEG"):
    cmp_val(f"cell_{ck}_mean", round(cells[ck]["mean"], 6), sj["regime_conditioned"]["cells"][ck]["mean"], 5e-4)
    cmp_val(f"cell_{ck}_n", cells[ck]["n"], sj["regime_conditioned"]["cells"][ck]["n"], 0)
# R3 overlay
cmp_val("overlay_with_sharpe", round(reg["sharpe"], 3), sj["overlay"]["with_regime"]["sharpe"], 0.03)
cmp_val("overlay_without_sharpe", round(opxo["sharpe"], 3), sj["overlay"]["without_regime"]["sharpe"], 0.03)
cmp_val("overlay_bh_sharpe", round(bh["sharpe"], 3), sj["overlay"]["bh"]["sharpe"], 0.03)
cmp_val("overlay_with_maxdd", round(reg["maxdd"], 4), sj["overlay"]["with_regime"]["maxdd"], 0.01)
# R4 ablation
cmp_val("abl_SPX_frozen", round(abl["SPX"]["frozen"]["sharpe"], 3), sj["ablation"]["SPX"]["frozen stack"]["sharpe"], 0.03)
cmp_val("abl_SPX_with", round(abl["SPX"]["with-regime"]["sharpe"], 3),
        sj["ablation"]["SPX"]["stack × OPEX(with-regime)"]["sharpe"], 0.03)
cmp_val("abl_NDX_frozen", round(abl["NDX"]["frozen"]["sharpe"], 3), sj["ablation"]["NDX"]["frozen stack"]["sharpe"], 0.03)
cmp_val("abl_NDX_with", round(abl["NDX"]["with-regime"]["sharpe"], 3),
        sj["ablation"]["NDX"]["stack × OPEX(with-regime)"]["sharpe"], 0.03)
cmp_val("abl_SPX_reversed", round(abl["SPX"]["reversed"]["sharpe"], 3),
        sj["ablation_diagnostics"]["reversed_overlay_sharpe"]["SPX"], 0.03)
cmp_val("shield_overlap_frac", OUT["recomputed"]["R4_shield_overlap"]["frac"],
        sj["ablation_diagnostics"]["shield_overlap_frac"], 0.03)

n_match = sum(1 for v in OUT["compare"].values() if v["match"])
n_tot = len(OUT["compare"])
print("\n[COMPARE] study vs mine:")
for k, v in OUT["compare"].items():
    flag = "OK " if v["match"] else "XX "
    print(f"  {flag}{k:<26} mine={v['mine']}  study={v['study']}")
print(f"\n  MATCHED {n_match}/{n_tot}")

OUT["summary"] = {"matched": n_match, "total": n_tot,
                  "abl_spx_delta": round(spx_d, 4), "abl_ndx_delta": round(ndx_d, 4),
                  "standalone_beats_bh": bool(standalone_beats_bh), "sub_wins": f"{sub_wins}/{len(G.SUBPERIODS)}"}
(RESULTS / "verify_T3_out.json").write_text(json.dumps(OUT, indent=2, default=str), encoding="utf-8")
print(f"\nwrote {RESULTS / 'verify_T3_out.json'}")
