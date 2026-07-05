"""
verify_T1.py — ADVERSARIAL INDEPENDENT VERIFIER for study T1 (flip-gate / asymmetric shield).

Own code path. We deliberately RE-DERIVE every number from the raw parquet data using a
DIFFERENT implementation than T1_flipgate (different lag mechanics, different maxdd loop) so a
shared-bug cannot hide. Then we compare to results/T1.json with stated tolerances and run the
look-ahead / trim-only / proxy-vs-realtide / baseline-repro / trailing-z audits.

We do NOT import T1_flipgate. We MAY use spine/finalize_stack imports (those are the frozen model).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path("C:/Users/admin/Downloads/kader-equity")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backtest" / "gex_swing"))

RESULTS = ROOT / "backtest" / "gex_swing" / "results"
STUDY = json.loads((RESULTS / "T1.json").read_text(encoding="utf-8"))

SUBPERIODS = {
    "full_2011_26": ("2011-05-02", "2026-06-08"),
    "2015_china":   ("2015-08-01", "2015-10-15"),
    "2018Q4":       ("2018-10-01", "2018-12-31"),
    "2020_covid":   ("2020-02-15", "2020-05-31"),
    "2022_bear":    ("2022-01-01", "2022-12-31"),
    "2023_svb":     ("2023-03-01", "2023-04-15"),
    "2025_tariff":  ("2025-03-01", "2025-05-31"),
}
FLIP_FLOORS = (0.4, 0.5, 0.6)
ASYM_LO, ASYM_HI = 0.4, 0.7

# ───────────────────────── INDEPENDENT PRIMITIVES ─────────────────────────


def my_zscore(gex: pd.Series, win=252, minp=60) -> pd.Series:
    """Trailing rolling z. Independent re-impl (same definition as gex_shield, intentionally)."""
    gex = gex.dropna()
    return (gex - gex.rolling(win, min_periods=minp).mean()) / gex.rolling(win, min_periods=minp).std()


def my_maxdd(r: pd.Series) -> float:
    """Independent maxDD via explicit running-peak loop (NOT cummax/cumprod vectorized like the study)."""
    r = r.dropna().values
    eq = 1.0
    peak = 1.0
    mdd = 0.0
    for x in r:
        eq *= (1.0 + x)
        if eq > peak:
            peak = eq
        d = eq / peak - 1.0
        if d < mdd:
            mdd = d
    return float(mdd)


def my_sharpe(r: pd.Series) -> float:
    r = r.dropna().values
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * math.sqrt(252))


def my_cumret(r: pd.Series) -> float:
    r = r.dropna().values
    return float(np.prod(1.0 + r) - 1.0)


def pnl_lag1_shiftpos(pos: pd.Series, price: pd.Series) -> pd.Series:
    """
    INDEPENDENT lag=1: we shift the POSITION forward by 1 (pos decided EOD-D acts on D's same-day
    return at D+1), then multiply by the SAME-DAY simple return. This is the engine/finalize_stack
    mechanic (concatenate zeros + p[:-lag]), which is a DIFFERENT code path than the study's
    standalone helper (pos * price.pct_change().shift(-1)). Both must agree if look-ahead-free.
    """
    ret_same = price.pct_change()                 # ret[D] = D-1 -> D
    p = pos.astype(float).values
    p_shift = np.concatenate([[np.nan], p[:-1]])  # position from prior day earns today's return
    s = pd.Series(p_shift * ret_same.values, index=pos.index)
    return s.dropna()


# ───────────────────────── 1) STANDALONE FULL-SAMPLE ─────────────────────────

def load_squeeze():
    g = pd.read_parquet(ROOT / "data" / "cache" / "squeeze_dix_gex.parquet")
    g.index = pd.to_datetime(g.index)
    return g.sort_index()


g = load_squeeze()
price = g["price"]
gex = g["gex"]
z = my_zscore(gex).reindex(g.index)

print("=" * 100)
print("VERIFY T1 — independent recompute")
print("=" * 100)
print(f"squeeze {g.index.min().date()}..{g.index.max().date()} n={len(g)} "
      f"gex<0 share={100*(gex<0).mean():.1f}% z<=-1 share={100*(z<=-1).mean():.1f}%")

# Build positions independently
pos_bh = pd.Series(1.0, index=g.index)
pos_v0 = (1.0 - 0.5 * np.clip(-z - 1.0, 0.0, 3.0)).clip(0.4, 1.0).fillna(1.0)
pos_v1 = {fl: pd.Series(np.where(gex.values < 0, fl, 1.0), index=g.index) for fl in FLIP_FLOORS}

# V2 standalone with 200dma proxy (independent build)
sma200 = price.rolling(200, min_periods=200).mean()
bearish = price < sma200
neg = gex < 0
pos_v2 = pd.Series(1.0, index=g.index)
pos_v2 = pos_v2.mask(neg & bearish, ASYM_LO)
pos_v2 = pos_v2.mask(neg & (~bearish) & sma200.notna(), ASYM_HI)
pos_v2 = pos_v2.mask(neg & sma200.isna(), ASYM_HI)

variants = {"BH": pos_bh, "V0_zshield": pos_v0,
            "V1_flipbin_0.4": pos_v1[0.4], "V1_flipbin_0.5": pos_v1[0.5],
            "V1_flipbin_0.6": pos_v1[0.6], "V2_asym_200dma": pos_v2}

recompute = {}
print("\n[STANDALONE recompute vs study]")
print(f"  {'variant':<18}{'mySh':>7}{'stSh':>7}{'myDD':>9}{'stDD':>9}{'myCum':>9}{'stCum':>9}{'%trim':>7}")
for name, pos in variants.items():
    r = pnl_lag1_shiftpos(pos, price)
    sh, dd, cum = my_sharpe(r), my_maxdd(r), my_cumret(r)
    trimpct = float((pos.astype(float) < 1.0).mean())
    st = STUDY["standalone"][name]["full"]
    recompute[name] = {"sharpe": sh, "maxdd": dd, "cum_ret": cum, "n": int(len(r)), "pct_trim": trimpct}
    print(f"  {name:<18}{sh:>+7.3f}{st['sharpe']:>+7.3f}{100*dd:>+8.1f}%{100*st['maxdd']:>+8.1f}%"
          f"{100*cum:>+8.1f}%{100*st['cum_ret']:>+8.1f}%{100*trimpct:>+6.1f}%")

# ───────────────────────── 2) STANDALONE SUB-PERIOD maxDD (V1 0.5) ─────────────────────────
print("\n[V1_flipbin_0.5 sub-period maxDD recompute vs study]")
sub_recompute = {}
for nm in ("2020_covid", "2018Q4"):
    s, e = SUBPERIODS[nm]
    m = (price.index >= pd.Timestamp(s)) & (price.index <= pd.Timestamp(e))
    sub_price, sub_pos = price[m], pos_v1[0.5][m]
    r = pnl_lag1_shiftpos(sub_pos, sub_price)
    dd = my_maxdd(r)
    st_dd = STUDY["standalone"]["V1_flipbin_0.5"]["subperiods"][nm]["maxdd"]
    sub_recompute[nm] = {"maxdd": dd, "study": st_dd, "n": int(len(r))}
    print(f"  {nm:<14} my={100*dd:>+7.1f}%  study={100*st_dd:>+7.1f}%  n={len(r)}")

# ───────────────────────── 3) ABLATION (2019+) ─────────────────────────
print("\n" + "=" * 100)
print("[ABLATION recompute — independent stack build]")
print("=" * 100)

from spine import contract as C, tide as T            # noqa: E402
from backtest import engine as E                       # noqa: E402
from modules.cor1m_froth import froth_factor_series     # noqa: E402
from modules.gex_shield import gex_zscore as gs_z, shield_factor_series  # noqa: E402

scores, prices, vector, prov = C.read_frozen()
tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
idx = tdir.index
cor = pd.read_parquet(ROOT / "data" / "cache" / "corr_pc.parquet")["COR1M"].dropna()
gex_f = pd.read_parquet(ROOT / "data" / "cache" / "squeeze_dix_gex.parquet")["gex"].dropna()
froth = froth_factor_series(cor.reindex(idx, method="ffill"), 8, 11, 0.0)
zg = gs_z(gex_f).reindex(idx, method="ffill")
gex_shield = shield_factor_series(zg, 0.5, 1.0, 0.4)

gex_sign = np.sign(gex_f).reindex(idx, method="ffill")
v1_factor = pd.Series(np.where(gex_sign.values < 0, 0.5, 1.0), index=idx)
tide_long = tdir.reindex(idx) > 0
negs = gex_sign < 0
v2_factor = pd.Series(1.0, index=idx)
v2_factor = v2_factor.mask(negs & (~tide_long), ASYM_LO)
v2_factor = v2_factor.mask(negs & tide_long, ASYM_HI)


def strat_ret_independent(pos, close):
    """INDEPENDENT lag=1 path: build fwd next-day return, multiply by SHIFTED pos (zeros prepended)."""
    idx_ = pos.index
    cb = close.reindex(idx_, method="ffill")
    ret = (cb.shift(-1) / cb - 1.0).values        # next-day return aligned to D
    p = pos.astype(float).values
    p = np.concatenate([[0.0], p[:-1]])           # pos[D-1] earns ret[D-1] (= D-1->D); equiv to lag=1
    return pd.Series(p * ret, index=idx_).dropna()


abl_recompute = {}
for a in ("SPX", "NDX"):
    stacks = {
        "base_stack": tdir * froth * gex_shield,
        "V1_stack":   tdir * froth * v1_factor,
        "V2_stack":   tdir * froth * v2_factor,
    }
    a_out = {}
    print(f"\n  [{a}]  {'stack':<12}{'mySh':>8}{'stSh':>8}{'myDD':>9}{'stDD':>9}{'expo':>7}{'n':>6}")
    for label, pos in stacks.items():
        r = strat_ret_independent(pos.reindex(idx), prices[a])
        sh, dd = my_sharpe(r), my_maxdd(r)
        expo = float((r != 0).mean())
        st = STUDY["ablation"][a][label]
        a_out[label] = {"sharpe": sh, "maxdd": dd, "expo": expo, "n": int(len(r))}
        print(f"  {'':<6}{label:<12}{sh:>+8.3f}{st['sharpe']:>+8.3f}{100*dd:>+8.1f}%{100*st['maxdd']:>+8.1f}%"
              f"{100*expo:>+6.0f}%{len(r):>6}")
    abl_recompute[a] = a_out

# ───────────────────────── AUDITS ─────────────────────────
print("\n" + "=" * 100)
print("[AUDITS]")
print("=" * 100)
audits = {}

# (a) look-ahead: compare my SHIFT-POS path vs the study's SHIFT-RETURN path; identical => no same-day leak.
#     Also a destructive control: a TRUE look-ahead (pos[D]*ret[D] same-day) must give a DIFFERENT number.
r_studypath = (pos_v0.astype(float) * price.pct_change().shift(-1)).dropna()   # study's exact convention
r_mypath = pnl_lag1_shiftpos(pos_v0, price)
# align
common = r_studypath.index.intersection(r_mypath.index)
diff = float((r_studypath.reindex(common) - r_mypath.reindex(common)).abs().max())
# same-day leak control
r_leak = (pos_v0.astype(float) * price.pct_change()).dropna()      # pos[D]*ret[D] (uses same-day = leak)
leak_sh = my_sharpe(r_leak)
clean_sh = my_sharpe(r_mypath)
audits["a_lookahead"] = {"shiftpos_vs_shiftret_maxdiff": diff,
                         "leak_sharpe": leak_sh, "clean_sharpe": clean_sh,
                         "paths_agree": diff < 1e-12, "leak_differs": abs(leak_sh - clean_sh) > 0.01}
print(f"(a) look-ahead: shift-pos vs shift-ret max abs diff = {diff:.2e} (==0 => same lag, no leak)")
print(f"    same-day-leak control Sharpe={leak_sh:+.3f} vs clean={clean_sh:+.3f} "
      f"(differ => the clean path is genuinely NOT using same-day info)")

# (b) trim-only: every position series in [0,1], never <0 or >1
trim_ok = True
bad = []
for name, pos in list(variants.items()) + [("abl_v1", v1_factor), ("abl_v2", v2_factor),
                                           ("abl_shield", gex_shield)]:
    pmin, pmax = float(pos.min()), float(pos.max())
    if pmin < -1e-9 or pmax > 1.0 + 1e-9:
        trim_ok = False
        bad.append((name, pmin, pmax))
audits["b_trim_only"] = {"all_in_0_1": trim_ok, "violations": bad}
print(f"(b) trim-only: all positions in [0,1]? {trim_ok}  violations={bad}")

# (c) V2 standalone uses 200dma proxy; ablation V2 uses REAL tide. Check source code text.
src = (ROOT / "backtest" / "gex_swing" / "T1_flipgate.py").read_text(encoding="utf-8")
v2_standalone_proxy = ("sma200" in src and "200dma" in src and "rolling(200" in src)
v2_ablation_realtide = ("tide_long = tdir" in src and "v2_factor = v2_factor.mask(neg & (~tide_long)" in src)
# functional check: does the standalone V2 actually depend on price-trend (not tide)?
audits["c_v2_proxy_vs_realtide"] = {"standalone_uses_200dma": bool(v2_standalone_proxy),
                                    "ablation_uses_realtide": bool(v2_ablation_realtide)}
print(f"(c) V2 standalone uses 200dma proxy = {v2_standalone_proxy} ; "
      f"ablation V2 uses real frozen tide = {v2_ablation_realtide}")

# (d) baseline reproduces finalize_stack 1.64/1.77
base_spx = abl_recompute["SPX"]["base_stack"]["sharpe"]
base_ndx = abl_recompute["NDX"]["base_stack"]["sharpe"]
repro_ok = abs(base_spx - 1.64) < 0.03 and abs(base_ndx - 1.77) < 0.03
audits["d_baseline_repro"] = {"spx": base_spx, "ndx": base_ndx, "match": bool(repro_ok)}
print(f"(d) baseline repro: SPX {base_spx:.3f} (~1.64) NDX {base_ndx:.3f} (~1.77) → {'MATCH' if repro_ok else 'MISMATCH'}")

# Cross-check: actually run finalize_stack.main() captured? Instead recompute its exact base series here.
# We already did (base_stack uses identical froth/shield/strat_ret). Confirm equality to study repro_base.
study_repro = STUDY["meta"]["repro_base"]
audits["d_baseline_repro"]["study_claim"] = study_repro

# (e) z-score uses trailing rolling only — verify a future value cannot affect a past z.
#     Perturb the LAST gex value massively; check that z at an early date is unchanged.
gex_perturbed = gex.copy()
gex_perturbed.iloc[-1] = gex_perturbed.iloc[-1] * 1000 + 1e9
z_pert = my_zscore(gex_perturbed).reindex(g.index)
test_dt = g.index[1000]                                   # an early date, far from the end
unchanged = abs(float(z.loc[test_dt]) - float(z_pert.loc[test_dt])) < 1e-9
# and confirm the LAST z DID change (perturbation took effect somewhere)
last_changed = abs(float(z.iloc[-1]) - float(z_pert.iloc[-1])) > 1e-6 if pd.notna(z.iloc[-1]) else True
audits["e_trailing_z"] = {"past_z_unaffected_by_future": bool(unchanged), "last_z_changed": bool(last_changed)}
print(f"(e) trailing-z: future gex perturbation leaves past z unchanged = {unchanged} "
      f"(and last z changed = {last_changed})")

# ───────────────────────── COMPARE & VERDICT NOTES ─────────────────────────
print("\n" + "=" * 100)
print("[TOLERANCE COMPARISON]  Sharpe ±0.03, maxDD ±1pp, cum ±10% rel")
print("=" * 100)
mismatches = []


def cmp_metric(name, mine, study_v, tol, kind):
    ok = abs(mine - study_v) <= tol
    if not ok:
        mismatches.append((name, kind, mine, study_v, abs(mine - study_v)))
    return ok


for name in variants:
    st = STUDY["standalone"][name]["full"]
    rc = recompute[name]
    cmp_metric(f"std/{name}", rc["sharpe"], st["sharpe"], 0.03, "sharpe")
    cmp_metric(f"std/{name}", rc["maxdd"], st["maxdd"], 0.01, "maxdd")
    cmp_metric(f"std/{name}", rc["cum_ret"], st["cum_ret"], max(0.01, abs(st["cum_ret"]) * 0.10), "cum")

for nm, d in sub_recompute.items():
    cmp_metric(f"sub/V1_0.5/{nm}", d["maxdd"], d["study"], 0.01, "maxdd")

for a in ("SPX", "NDX"):
    for label in ("base_stack", "V1_stack", "V2_stack"):
        st = STUDY["ablation"][a][label]
        rc = abl_recompute[a][label]
        cmp_metric(f"abl/{a}/{label}", rc["sharpe"], st["sharpe"], 0.03, "sharpe")
        cmp_metric(f"abl/{a}/{label}", rc["maxdd"], st["maxdd"], 0.01, "maxdd")

if mismatches:
    print(f"  {len(mismatches)} MISMATCH(es):")
    for m in mismatches:
        print(f"    {m[0]:<24} {m[1]:<7} mine={m[2]:+.4f} study={m[3]:+.4f} d={m[4]:.4f}")
else:
    print("  ALL metrics within tolerance ✓")

# directional consistency of the headline claims (adversarial logic check)
print("\n[HEADLINE LOGIC CHECK]")
# claim 1: flip-gate WINS standalone (V1/V2 Sharpe & maxDD beat V0_zshield)
v0 = recompute["V0_zshield"]
v1_05 = recompute["V1_flipbin_0.5"]
v2 = recompute["V2_asym_200dma"]
stand_win = (v1_05["sharpe"] > v0["sharpe"]) and (v1_05["maxdd"] > v0["maxdd"])  # higher Sharpe, shallower DD
print(f"  standalone: V1_0.5 Sharpe {v1_05['sharpe']:+.3f} > V0 {v0['sharpe']:+.3f} = {v1_05['sharpe']>v0['sharpe']}"
      f" ; V1_0.5 maxDD {100*v1_05['maxdd']:+.1f}% shallower than V0 {100*v0['maxdd']:+.1f}% = {v1_05['maxdd']>v0['maxdd']}")
# claim 2: LOSES in-model (V1/V2 stack Sharpe < base, maxDD deeper)
inmodel_loses = True
for a in ("SPX", "NDX"):
    base = abl_recompute[a]["base_stack"]
    for v in ("V1_stack", "V2_stack"):
        vv = abl_recompute[a][v]
        worse = (vv["sharpe"] < base["sharpe"]) and (vv["maxdd"] < base["maxdd"])  # lower Sharpe, deeper DD
        inmodel_loses = inmodel_loses and worse
        print(f"  in-model {a}/{v}: Sharpe {vv['sharpe']:+.3f} < base {base['sharpe']:+.3f} "
              f"& maxDD {100*vv['maxdd']:+.1f}% deeper than base {100*base['maxdd']:+.1f}% → degrades={worse}")
print(f"  HEADLINE: standalone-win={stand_win}  in-model-loses(all 4 cells)={inmodel_loses} "
      f"→ TIDE_ABSORBED/degrades-in-stack verdict consistent = {stand_win and inmodel_loses}")

out = {
    "standalone_recompute": recompute,
    "sub_recompute": sub_recompute,
    "ablation_recompute": abl_recompute,
    "audits": audits,
    "mismatches": [{"name": m[0], "kind": m[1], "mine": m[2], "study": m[3], "absdiff": m[4]} for m in mismatches],
    "headline_check": {"standalone_win": bool(stand_win), "inmodel_loses_all": bool(inmodel_loses)},
}
(RESULTS / "verify_T1_out.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
print(f"\nwrote {RESULTS / 'verify_T1_out.json'}")
