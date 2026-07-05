"""
T3 — VANNA/CHARM OPEX CYCLE × GEX REGIME.

Emir most-hopeful test. PRIOR WARNING: plain seasonality was tide-ABSORBED before; the NEW angle is
REGIME-CONDITIONING (split the OPEX window by gamma regime = sign(gex)). Be adversarial with the result.

Pipeline:
  1) UNCONDITIONAL daily-return profile by OPEX offset (-5..+5): mean + t + n, pooled PRE/OPEX/POST.
  2) REGIME-CONDITIONED 2x2 (window × sign(gex)): PRE/POST mean+t+n in POS vs NEG gamma, + by sub-period.
  3) TRADEABLE standalone overlay (trim/tilt-only): boost-to-1.0 PRE∩pos, trim-to-0.5 POST∩neg, else 1.0.
     vs an UNCONDITIONED OPEX-only overlay (same windows, no regime split). full + sub-period vs B&H.
  4) ABLATION over the tide (2019+): tide×froth×shield × OPEX-overlay vs frozen stack; with-regime vs
     without-regime. If the tide already captures it → verdict TIDE_ABSORBED with the numbers.

PIT: signal[D] (offset + sign(gex) at EOD D) earns return D->D+1. STANDALONE uses
     ret = price.pct_change().shift(-1) aligned to D (explicit, look-ahead-free). ABLATION reuses
     finalize_stack.strat_ret(pos, close, lag=1) byte-faithfully. The window/regime state on day D is
     known at EOD D (no future info). gex regime is lagged via the same +1 exec lag as the position.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backtest" / "gex_swing"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import gxs_config as G                                          # noqa: E402
from spine import contract as C, tide as T                      # noqa: E402
from modules.cor1m_froth import froth_factor_series             # noqa: E402
from modules.gex_shield import gex_zscore, shield_factor_series # noqa: E402
import screen.finalize_stack as FS                              # noqa: E402

RESULTS = ROOT / "backtest" / "gex_swing" / "results"
PRE_LO, PRE_HI = -G.OPEX_PRE, -1     # [-5..-1]
POST_LO, POST_HI = 1, G.OPEX_POST    # [1..5]


# ----------------------------------------------------------------------------- helpers
def tstat(x: np.ndarray):
    """Mean daily return (%), t-stat of mean vs 0, n. x = daily simple returns."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return (float(x.mean()) if n else float("nan")), float("nan"), n
    t = x.mean() / (x.std(ddof=1) / np.sqrt(n))
    return float(x.mean()), float(t), n


def fmt_mean_t(m, t, n):
    return f"{100*m:+.3f}% t={t:+.2f} n={n}"


# ----------------------------------------------------------------------------- load standalone data
g = G.load_squeeze()                              # price(=SPX), dix, gex ; 2011-05..2026-06
idx = g.index
S, E = idx.min(), idx.max()
off = G.opex_offset_map(idx, S, E)                # business-day offset to nearest monthly OPEX
gex = g["gex"]
ret_next = g["price"].pct_change().shift(-1)      # ret[D] = close[D]->close[D+1] ; signal on D earns this
gamma_sign = np.sign(gex).replace(0, 1.0)         # +1 = long gamma (above flip) ; -1 = short gamma (below)

# window membership (evaluated on day D)
in_pre = off.between(PRE_LO, PRE_HI)
in_opex = (off == 0)
in_post = off.between(POST_LO, POST_HI)

print("=" * 100)
print("T3 — VANNA/CHARM OPEX × GEX REGIME")
print(f"squeeze {S.date()}..{E.date()} n={len(g)} | offset coverage %{100*off.notna().mean():.0f}")
print(f"  windows: PRE=[{PRE_LO}..{PRE_HI}]  OPEX=0  POST=[{POST_LO}..{POST_HI}]")
print("=" * 100)

# ============================================================ 1) UNCONDITIONAL offset profile
print("\n[1] UNCONDITIONAL daily-return profile by OPEX offset (signal[D] -> ret D->D+1)")
print(f"  {'offset':>7}{'mean_ret':>12}{'t':>8}{'n':>6}")
prof = {}
for o in range(-G.OPEX_PRE, G.OPEX_POST + 1):
    m = (off == o)
    x = ret_next[m].dropna().values
    mm, tt, nn = tstat(x)
    prof[str(o)] = {"mean": round(mm, 6), "t": round(tt, 3) if np.isfinite(tt) else None, "n": nn}
    print(f"  {o:>+7}{100*mm:>+11.3f}%{tt:>+8.2f}{nn:>6}")

pre_m, pre_t, pre_n = tstat(ret_next[in_pre].dropna().values)
opx_m, opx_t, opx_n = tstat(ret_next[in_opex].dropna().values)
post_m, post_t, post_n = tstat(ret_next[in_post].dropna().values)
all_m, all_t, all_n = tstat(ret_next.dropna().values)
print(f"\n  POOLED  PRE [{PRE_LO}..{PRE_HI}]: {fmt_mean_t(pre_m,pre_t,pre_n)}")
print(f"          OPEX day 0   : {fmt_mean_t(opx_m,opx_t,opx_n)}")
print(f"          POST[{POST_LO}..{POST_HI}] : {fmt_mean_t(post_m,post_t,post_n)}")
print(f"          ALL days     : {fmt_mean_t(all_m,all_t,all_n)}")
thesis_pre = "PRE drift POSITIVE" if pre_m > 0 else "PRE drift NOT positive"
thesis_post = "POST weakness (negative)" if post_m < 0 else "POST NOT negative"
print(f"  THESIS CHECK: vanna→{thesis_pre} ({pre_m>0}); charm→{thesis_post} ({post_m<0})")
print(f"  (note: ALL-day mean is +{100*all_m:.3f}%; PRE/POST must beat/lag THIS baseline to mean anything)")

unconditional = {
    "by_offset": prof,
    "pooled": {
        "PRE":  {"mean": round(pre_m, 6), "t": round(pre_t, 3), "n": pre_n},
        "OPEX": {"mean": round(opx_m, 6), "t": round(opx_t, 3) if np.isfinite(opx_t) else None, "n": opx_n},
        "POST": {"mean": round(post_m, 6), "t": round(post_t, 3), "n": post_n},
        "ALL":  {"mean": round(all_m, 6), "t": round(all_t, 3), "n": all_n},
    },
    "thesis": {"pre_positive": bool(pre_m > 0), "post_negative": bool(post_m < 0),
               "pre_beats_allday": bool(pre_m > all_m), "post_lags_allday": bool(post_m < all_m)},
}

# ============================================================ 2) REGIME-CONDITIONED 2x2
print("\n[2] REGIME-CONDITIONED (window × sign(gex)) — the key test")
pos_g = gamma_sign > 0       # long gamma / above flip
neg_g = gamma_sign < 0       # short gamma / below flip
print(f"  regime mix: POS-gamma %{100*pos_g.mean():.0f} | NEG-gamma %{100*neg_g.mean():.0f}")


def cell(win_mask, regime_mask):
    x = ret_next[win_mask & regime_mask].dropna().values
    return tstat(x)


cells = {}
print(f"  {'window':>8}{'regime':>10}{'mean_ret':>12}{'t':>8}{'n':>7}")
for wn, wm in (("PRE", in_pre), ("OPEX", in_opex), ("POST", in_post)):
    for rn, rm in (("POS", pos_g), ("NEG", neg_g)):
        mm, tt, nn = cell(wm, rm)
        cells[f"{wn}_{rn}"] = {"mean": round(mm, 6), "t": round(tt, 3) if np.isfinite(tt) else None, "n": nn}
        ts = f"{tt:+.2f}" if np.isfinite(tt) else "  nan"
        print(f"  {wn:>8}{rn:>10}{100*mm:>+11.3f}%{ts:>8}{nn:>7}")

# hypotheses
h1 = cells["PRE_POS"]["mean"] > cells["PRE_NEG"]["mean"]   # PRE drift stronger in POS gamma
h2 = cells["POST_NEG"]["mean"] < cells["POST_POS"]["mean"] # POST weakness sharper in NEG gamma
print(f"\n  H1 PRE drift stronger in POS gamma  : {h1}  (PRE_POS {100*cells['PRE_POS']['mean']:+.3f}% vs "
      f"PRE_NEG {100*cells['PRE_NEG']['mean']:+.3f}%)")
print(f"  H2 POST weakness sharper in NEG gamma: {h2}  (POST_NEG {100*cells['POST_NEG']['mean']:+.3f}% vs "
      f"POST_POS {100*cells['POST_POS']['mean']:+.3f}%)")

# by sub-period stability (PRE_POS and POST_NEG — the two cells the overlay trades on)
print("\n  sub-period stability of the two TRADED cells (PRE∩POS boost, POST∩NEG trim):")
sub_cells = {}
print(f"  {'subperiod':>16}{'PRE_POS':>20}{'POST_NEG':>20}")
for name, (s, e) in G.SUBPERIODS.items():
    sl = (idx >= pd.Timestamp(s)) & (idx <= pd.Timestamp(e))
    pp = ret_next[in_pre & pos_g & pd.Series(sl, index=idx)].dropna().values
    pn = ret_next[in_post & neg_g & pd.Series(sl, index=idx)].dropna().values
    a, at, an = tstat(pp); b, bt, bn = tstat(pn)
    sub_cells[name] = {"PRE_POS": {"mean": round(a, 6), "t": round(at, 3) if np.isfinite(at) else None, "n": an},
                       "POST_NEG": {"mean": round(b, 6), "t": round(bt, 3) if np.isfinite(bt) else None, "n": bn}}
    pps = f"{100*a:+.3f}% n={an}" if an else "n=0"
    pns = f"{100*b:+.3f}% n={bn}" if bn else "n=0"
    print(f"  {name:>16}{pps:>20}{pns:>20}")

regime_conditioned = {"cells": cells, "hypotheses": {"H1_pre_pos_stronger": bool(h1),
                      "H2_post_neg_sharper": bool(h2)}, "subperiods": sub_cells,
                      "regime_mix": {"pos_frac": round(float(pos_g.mean()), 3), "neg_frac": round(float(neg_g.mean()), 3)}}

# ============================================================ 3) TRADEABLE standalone overlay
print("\n[3] TRADEABLE standalone overlay (trim/tilt-only, long-bias book)")
print("  IMPL: base book = always-long (1.0). WITH-REGIME mult = 1.0 normal; POST∩NEG-gamma → 0.5 (trim).")
print("        (PRE∩POS 'boost-to-1.0' is a no-op on a 1.0 book — stated honestly; the only active move is the trim.)")
print("        WITHOUT-REGIME (OPEX-only) mult = 1.0 normal; POST (any regime) → 0.5 (trim).")

# PIT: state on day D (off, gamma_sign) -> multiplier on D -> earns ret D->D+1.
# This is look-ahead-free: off[D] and gex[D] are EOD-D known; ret_next[D] is D->D+1.
mult_regime = pd.Series(1.0, index=idx)
mult_regime[in_post & neg_g] = 0.5

mult_opexonly = pd.Series(1.0, index=idx)
mult_opexonly[in_post] = 0.5

# Optional sharper variant that also USES the PRE∩POS tailwind vs a <1 baseline is NOT defined (would add a
# free parameter / a non-1.0 base). We keep the base book at 1.0 so 'boost-to-1.0' is genuinely a no-op and
# the ONLY thing tested is whether trimming the POST window (regime-split vs not) helps. Honest + minimal.

bh_ret = ret_next.copy()                          # always-long book daily returns


def overlay_metrics(mult, sl_mask=None):
    r = (mult * ret_next)
    if sl_mask is not None:
        r = r[sl_mask]
    return G.metrics(r.dropna())


def bh_metrics(sl_mask=None):
    r = bh_ret.copy()
    if sl_mask is not None:
        r = r[sl_mask]
    return G.metrics(r.dropna())


full_bh = bh_metrics()
full_reg = overlay_metrics(mult_regime)
full_opx = overlay_metrics(mult_opexonly)
print(f"\n  FULL 2011-26:")
print(f"    {'variant':<22}{'Sharpe':>8}{'maxDD':>9}{'cum_ret':>10}{'expo':>7}{'n':>7}")
for lbl, mm in (("B&H (always-long)", full_bh), ("WITH-regime overlay", full_reg), ("OPEX-only overlay", full_opx)):
    print(f"    {lbl:<22}{mm['sharpe']:>+8.2f}{100*mm['maxdd']:>+8.0f}%{100*mm['cum_ret']:>+9.0f}%"
          f"{100*mm['expo']:>+6.0f}%{mm['n']:>7}")

# sub-period overlay
print(f"\n  SUB-PERIOD (Sharpe | maxDD | cum_ret):")
print(f"    {'subperiod':>16}{'B&H':>26}{'WITH-regime':>26}{'OPEX-only':>26}")
overlay_sub = {}
for name, (s, e) in G.SUBPERIODS.items():
    sl = (idx >= pd.Timestamp(s)) & (idx <= pd.Timestamp(e))
    sl = pd.Series(sl, index=idx)
    b = bh_metrics(sl); w = overlay_metrics(mult_regime, sl); o = overlay_metrics(mult_opexonly, sl)
    overlay_sub[name] = {"bh": b, "with_regime": w, "opex_only": o}

    def c(m):
        return f"{m['sharpe']:+.2f}|{100*m['maxdd']:+.0f}%|{100*m['cum_ret']:+.0f}%"
    print(f"    {name:>16}{c(b):>26}{c(w):>26}{c(o):>26}")

overlay = {
    "impl": "base=always-long 1.0; with_regime: POST&neg-gamma->0.5; opex_only: POST(any)->0.5; PRE-boost is no-op on 1.0 book",
    "full": {"bh": full_bh, "with_regime": full_reg, "opex_only": full_opx},
    "subperiods": overlay_sub,
}

# ============================================================ 4) ABLATION over the tide (2019+)
print("\n[4] ABLATION over the tide (2019+) — does the OPEX×regime overlay improve the frozen stack?")
scores, prices, vector, prov = C.read_frozen()
tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
tidx = tdir.index
cor = pd.read_parquet(ROOT / "data" / "cache" / "corr_pc.parquet")["COR1M"].dropna()
gex_f = pd.read_parquet(ROOT / "data" / "cache" / "squeeze_dix_gex.parquet")["gex"].dropna()
froth = froth_factor_series(cor.reindex(tidx, method="ffill"), 8, 11, 0.0)
zg = gex_zscore(gex_f).reindex(tidx, method="ffill")
gex_shield = shield_factor_series(zg, 0.5, 1.0, 0.4)
stack_base = tdir * froth * gex_shield            # frozen stack position (0..1)

# OPEX overlay multiplier on the 2019+ index (same windows; regime from frozen-era gex sign, ffill+lag via strat_ret)
off_t = G.opex_offset_map(tidx, tidx.min(), tidx.max())
gex_sign_t = np.sign(gex_f.reindex(tidx, method="ffill")).replace(0, 1.0)
in_pre_t = off_t.between(PRE_LO, PRE_HI)
in_post_t = off_t.between(POST_LO, POST_HI)
pos_g_t = gex_sign_t > 0
neg_g_t = gex_sign_t < 0

mult_reg_t = pd.Series(1.0, index=tidx)
mult_reg_t[in_post_t & neg_g_t] = 0.5
mult_opx_t = pd.Series(1.0, index=tidx)
mult_opx_t[in_post_t] = 0.5

stack_with_regime = (stack_base * mult_reg_t).clip(upper=1.0)
stack_opex_only = (stack_base * mult_opx_t).clip(upper=1.0)

ablation = {"SPX": {}, "NDX": {}}
print(f"  {'asset':<6}{'variant':<26}{'Sharpe':>8}{'maxDD':>8}{'CVaR':>8}{'expo':>7}{'P>frozen':>10}{'DSR':>7}")
for a in ("SPX", "NDX"):
    base_r = FS.strat_ret(stack_base.reindex(tidx), prices[a])      # frozen stack returns (byte-faithful)
    variants = {
        "frozen stack":            stack_base,
        "stack × OPEX(with-regime)": stack_with_regime,
        "stack × OPEX(no-regime)":   stack_opex_only,
    }
    for lbl, pos in variants.items():
        r = FS.strat_ret(pos.reindex(tidx), prices[a])
        sh, dd, cv, ex = FS._m(r)
        from screen._util import paired_win_prob
        wp = paired_win_prob(base_r, r)
        ds = FS.dsr(r)
        ablation[a][lbl] = {"sharpe": round(sh, 3), "maxdd": round(dd, 4), "cvar5": round(cv, 5),
                            "expo": round(ex, 3), "p_gt_frozen": (round(wp, 3) if wp is not None else None),
                            "dsr": (round(ds, 4) if ds is not None else None), "n": int(len(r))}
        wps = f"{wp:.0%}" if wp is not None else "-"
        dss = f"{ds:.3f}" if ds is not None else "-"
        print(f"  {a:<6}{lbl:<26}{sh:>+8.2f}{100*dd:>+7.0f}%{100*cv:>+7.2f}%{100*ex:>+6.0f}%{wps:>10}{dss:>7}")

# ADVERSARIAL DIAGNOSTICS: is the tiny ablation gain the OPEX×regime STRUCTURE, or just incidental
# de-risking that overlaps the GEX-shield already in the stack?
trim_mask_t = in_post_t & neg_g_t
n_trim = int(trim_mask_t.sum())
n_shield_overlap = int((gex_shield[trim_mask_t] < 0.999).sum())
mult_rev_t = pd.Series(1.0, index=tidx)
mult_rev_t[in_post_t & pos_g_t] = 0.5                       # REVERSE: trim the thesis-NULL cell (POST&POS)
stack_rev = (stack_base * mult_rev_t).clip(upper=1.0)
rev_sharpe = {}
for a in ("SPX", "NDX"):
    r = FS.strat_ret(stack_rev.reindex(tidx), prices[a])
    rev_sharpe[a] = round(FS._m(r)[0], 3)
print(f"\n  ADVERSARIAL: POST&neg trimmed days = {n_trim}; of which GEX-shield ALREADY <1.0 = {n_shield_overlap} "
      f"(%{100*n_shield_overlap/max(1,n_trim):.0f} overlap)")
print(f"  REVERSED overlay (trim POST&POS = thesis-NULL cell) Sharpe: SPX {rev_sharpe['SPX']:+.2f} "
      f"NDX {rev_sharpe['NDX']:+.2f} (frozen SPX {ablation['SPX']['frozen stack']['sharpe']:+.2f} "
      f"NDX {ablation['NDX']['frozen stack']['sharpe']:+.2f})")
diagnostics = {"post_neg_trim_days_2019plus": n_trim, "shield_already_trimming": n_shield_overlap,
               "shield_overlap_frac": round(n_shield_overlap / max(1, n_trim), 3),
               "reversed_overlay_sharpe": rev_sharpe,
               "note": "72%+ of trimmed days are already shield-trimmed; reversed overlay still moves Sharpe "
                       "-> the small gain is incidental de-risking inside a single regime, not the OPEX/charm structure"}

# verdict logic
sp_base = ablation["SPX"]["frozen stack"]["sharpe"]
sp_reg = ablation["SPX"]["stack × OPEX(with-regime)"]["sharpe"]
nx_base = ablation["NDX"]["frozen stack"]["sharpe"]
nx_reg = ablation["NDX"]["stack × OPEX(with-regime)"]["sharpe"]
spx_dd_base = ablation["SPX"]["frozen stack"]["maxdd"]
spx_dd_reg = ablation["SPX"]["stack × OPEX(with-regime)"]["maxdd"]
nx_dd_base = ablation["NDX"]["frozen stack"]["maxdd"]
nx_dd_reg = ablation["NDX"]["stack × OPEX(with-regime)"]["maxdd"]

# standalone regime edge present?
standalone_helps = (full_reg["sharpe"] > full_bh["sharpe"]) and (full_reg["maxdd"] > full_bh["maxdd"])
# regime conditioning adds over plain OPEX standalone?
regime_over_plain = (full_reg["sharpe"] >= full_opx["sharpe"])
# ablation: improves both assets' Sharpe meaningfully?
abl_sharpe_help = (sp_reg > sp_base + 0.03) and (nx_reg > nx_base + 0.03)
abl_dd_help = (spx_dd_reg > spx_dd_base) and (nx_dd_reg > nx_dd_base)  # less negative = better
abl_helps = abl_sharpe_help or abl_dd_help

print("\n" + "=" * 100)
print("VERDICT LOGIC:")
print(f"  standalone regime-overlay beats B&H (Sharpe & maxDD): {standalone_helps}")
print(f"  regime-conditioning >= plain-OPEX standalone Sharpe  : {regime_over_plain}")
print(f"  ablation SPX {sp_base:+.2f}->{sp_reg:+.2f} | NDX {nx_base:+.2f}->{nx_reg:+.2f}")
print(f"  ablation maxDD SPX {100*spx_dd_base:+.0f}%->{100*spx_dd_reg:+.0f}% | NDX {100*nx_dd_base:+.0f}%->{100*nx_dd_reg:+.0f}%")
print(f"  ablation improves stack (Sharpe>+0.03 both OR maxDD better both): {abl_helps}")

# multi-subperiod robustness of standalone regime overlay (does WITH-regime beat B&H Sharpe in most subs?)
sub_wins = sum(1 for n, d in overlay_sub.items()
               if d["with_regime"]["sharpe"] > d["bh"]["sharpe"])
n_subs = len(overlay_sub)
print(f"  standalone WITH-regime beats B&H Sharpe in {sub_wins}/{n_subs} sub-periods")

# ADVERSARIAL verdict: the tiny +0.03/+0.04 ablation bump is NOT the OPEX×regime structure —
# (a) standalone overlay LOSES to B&H (0/7 subs), (b) the charm thesis is reversed (POST is the best
# window; POST&neg is the single BEST cell, so the overlay trims the best cell), (c) 72% of the trimmed
# days are already shield-trimmed, and (d) the reversed (thesis-null) overlay also moves Sharpe.
# => the stack's existing GEX-shield already absorbs these stress days; the calendar mask adds noise.
incidental = (diagnostics["shield_overlap_frac"] >= 0.5)
if abl_helps and standalone_helps and sub_wins >= n_subs - 2 and not incidental:
    verdict = "PAYS"
elif abl_helps and incidental and not standalone_helps:
    verdict = "TIDE_ABSORBED"   # ablation bump exists but is shield-overlap / single-regime, not the structure
elif standalone_helps and not abl_helps:
    verdict = "TIDE_ABSORBED"
elif (full_reg["sharpe"] > full_bh["sharpe"]) or (abl_helps and not incidental):
    verdict = "MARGINAL"
else:
    verdict = "TIDE_ABSORBED" if abl_helps else "DEAD"
print(f"\n  >>> VERDICT: {verdict}")
print("=" * 100)

caveats = [
    "PRE-window 'boost-to-1.0' is a no-op on a 1.0-capped long-bias book; the only active move is the "
    "POST-window trim-to-0.5. Trim-only / rebound-safe by construction (never short, never >1.0).",
    "Standalone is multi-regime 2011-26 but the OPEX×regime overlay reduces EXPOSURE (trims POST&neg days); "
    "a Sharpe gain from de-risking is variance-reduction, not predictive alpha.",
    "Ablation window is 2019+ (m9-era single regime); frozen stack already includes a GEX-shield that fires "
    "on the SAME deep-negative-gamma days the POST&neg trim targets — heavy overlap, so the OPEX overlay has "
    "little orthogonal room left inside the stack.",
    "OPEX dates derived from calendar 3rd-Friday (np.busday); no holiday-shift handling — a few OPEX days may "
    "be off by the index 'nearest' snap. Offset coverage ~95%.",
    f"DSR uses N=60 (optimistic lower bound per finalize_stack note); honest N≈150-200 lowers it. n reported per cell.",
]

out = {
    "meta": {"test": "T3_opex_vanna_charm_x_gex_regime", "date": "2026-06-12",
             "baseline_stack": {"SPX": 1.64, "NDX": 1.77, "source": "finalize_stack @2019+"},
             "standalone_span": f"{S.date()}..{E.date()}", "n_standalone": int(len(g)),
             "windows": {"PRE": [PRE_LO, PRE_HI], "OPEX": 0, "POST": [POST_LO, POST_HI]},
             "pit": "signal[D] earns D->D+1; standalone ret=pct_change().shift(-1); ablation=strat_ret lag=1"},
    "unconditional_profile": unconditional,
    "regime_conditioned": regime_conditioned,
    "overlay": {"with_regime": overlay["full"]["with_regime"],
                "without_regime": overlay["full"]["opex_only"],
                "bh": overlay["full"]["bh"],
                "impl": overlay["impl"],
                "subperiods": overlay["subperiods"]},
    "ablation": ablation,
    "ablation_diagnostics": diagnostics,
    "verdict": verdict,
    "verdict_logic": {"standalone_helps": bool(standalone_helps), "regime_over_plain": bool(regime_over_plain),
                      "ablation_helps_raw": bool(abl_helps), "ablation_gain_is_incidental": bool(incidental),
                      "sub_wins": f"{sub_wins}/{n_subs}",
                      "charm_thesis_holds": bool(unconditional["thesis"]["post_negative"]),
                      "H2_post_neg_sharper": bool(regime_conditioned["hypotheses"]["H2_post_neg_sharper"])},
    "caveats": caveats,
}
RESULTS.mkdir(parents=True, exist_ok=True)
(RESULTS / "T3.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
print(f"\nwrote {RESULTS / 'T3.json'}")
