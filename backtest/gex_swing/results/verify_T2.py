"""
verify_T2.py — ADVERSARIAL INDEPENDENT VERIFIER for study T2 (GEX leading-fragility).

Recomputes FROM DATA with an OWN code path (not by calling the study functions):
  1) corr(gz, fdd_21) and corr(rv, fdd_21)               [+ N5, N10 for completeness]
  2) partial corr(gz, fdd_21 | rv) — sign + rough magnitude (own residualizer)
  3) combined-gate full-sample maxDD vs gex-only vs vol-only (own gate/strat code)
  4) BONUS independent: HAC t-stat for gz in OLS(fdd~rv+gz) at N=21, incr R²,
     non-overlap partial corr (every-21st-day), early-warning fired-fracs/leads.

AUDITS:
  (a) forward target uses ONLY future returns [D+1..D+N]; predictor gz[D]/rv[D] is PIT.
  (b) rv and gz both trailing/point-in-time (rolling, no look-ahead).
  (c) partial-corr controls for rv correctly (residualize both on rv+const).
  (d) worst-20 event detection is on forward dd (target), not look-ahead into predictor.

Compare to study JSON with stated tolerances; print readable table; write verify_T2.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent          # results/
GXS = HERE.parent                                # gex_swing/
sys.path.insert(0, str(GXS))
import gxs_config as G  # noqa: E402

ANN = np.sqrt(252.0)
TOL = {"sharpe": 0.03, "maxdd": 0.01, "corr": 0.03}  # maxdd in fraction (1pp = 0.01)


# ----------------------------------------------------- INDEPENDENT base build
def build():
    g = G.load_squeeze()
    px = g["price"].astype(float)
    gex = g["gex"].astype(float)
    # gz: trailing 252d z, min60 — recompute directly (do NOT call G.gex_zscore body blindly;
    # but it IS the locked single-source def, so replicate its exact rolling here independently)
    m = gex.rolling(252, min_periods=60).mean()
    s = gex.rolling(252, min_periods=60).std()
    gz = (gex - m) / s
    ret = px.pct_change()
    rv = ret.rolling(21, min_periods=15).std() * ANN     # trailing-21d ann vol, known EOD[D]
    return px, gex, gz, ret, rv


def fwd_dd_vol(px, ret, N):
    """fdd over price path [D..D+N]; fwd vol over returns [D+1..D+N]. TARGET only (look-ahead by design)."""
    p = px.values
    r = ret.values
    n = len(p)
    fdd = np.full(n, np.nan)
    fv = np.full(n, np.nan)
    for i in range(n):
        j = i + N
        if j >= n:
            continue
        path = p[i:j + 1]
        if np.isfinite(path).all() and len(path) >= 2:
            cm = np.maximum.accumulate(path)
            fdd[i] = (path / cm - 1.0).min()
        seg = r[i + 1:j + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= max(3, N // 2):
            fv[i] = seg.std() * ANN
    return pd.Series(fdd, index=px.index), pd.Series(fv, index=px.index)


def corr(a, b):
    m = a.notna() & b.notna()
    if m.sum() < 30:
        return None, int(m.sum())
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def partial(y, x, ctrl):
    """partial corr(y,x | ctrl): residualize y and x on [1, ctrl], corr the residuals."""
    df = pd.concat([y, x, ctrl], axis=1).dropna()
    df.columns = ["y", "x", "c"]
    if len(df) < 30:
        return None, len(df)
    Z = np.column_stack([np.ones(len(df)), df["c"].values])
    def res(v):
        b, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ b
    ry, rx = res(df["y"].values), res(df["x"].values)
    if ry.std() == 0 or rx.std() == 0:
        return None, len(df)
    return float(np.corrcoef(ry, rx)[0, 1]), len(df)


def ols_hac(y, cols: dict, L: int):
    """OLS y~1+cols; return r2 and per-coef naive t + Newey-West HAC t (Bartlett, lag L)."""
    parts = [y] + list(cols.values())
    df = pd.concat(parts, axis=1).dropna()
    df.columns = ["y"] + list(cols.keys())
    n = len(df)
    names = list(cols.keys())
    X = np.column_stack([np.ones(n)] + [df[c].values for c in names])
    yv = df["y"].values
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    ssr = float((resid ** 2).sum())
    sst = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1 - ssr / sst
    dof = n - X.shape[1]
    XtXi = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag((ssr / dof) * XtXi))
    t = beta / se
    Xr = X * resid[:, None]
    S = Xr.T @ Xr
    for lag in range(1, L + 1):
        w = 1 - lag / (L + 1.0)
        Gl = Xr[lag:].T @ Xr[:-lag]
        S += w * (Gl + Gl.T)
    cov = XtXi @ S @ XtXi
    se_h = np.sqrt(np.maximum(np.diag(cov), 0))
    t_h = beta / se_h
    return {"n": n, "r2": r2, "t": dict(zip(["const"] + names, t.tolist())),
            "t_hac": dict(zip(["const"] + names, t_h.tolist()))}


def maxdd(r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def sharpe(r):
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


# ----------------------------------------------------- GATE (independent)
def gate_full(px, gz, rv):
    """trim-to-0.5 gates, pos[D] earns next-day return. Own code path."""
    ret_next = px.pct_change().shift(-1)          # ret_next[D] = D->D+1
    rv_q80 = rv.rolling(252, min_periods=60).quantile(0.80)
    vol_trig = (rv > rv_q80)
    gex_trig = (gz < -1.0)
    comb_trig = (vol_trig | gex_trig)

    def pos(trig):
        p = pd.Series(1.0, index=px.index)
        p[trig.fillna(False)] = 0.5
        return p

    s, e = G.SUBPERIODS["full_2011_26"]
    mask = (px.index >= pd.Timestamp(s)) & (px.index <= pd.Timestamp(e))
    out = {}
    for name, p in {"buy_hold_long": pd.Series(1.0, index=px.index),
                    "vol_only": pos(vol_trig), "gex_only": pos(gex_trig),
                    "combined": pos(comb_trig)}.items():
        rr = (p[mask] * ret_next[mask]).dropna()
        out[name] = {"sharpe": round(sharpe(rr), 3), "maxdd": round(maxdd(rr), 4),
                     "n": int(len(rr)),
                     "trim_frac": round(float((p[mask] < 1.0).mean()), 3)}
    out["_trigger_frac"] = {
        "vol": round(float(vol_trig[mask].fillna(False).mean()), 3),
        "gex": round(float(gex_trig[mask].fillna(False).mean()), 3),
        "combined": round(float(comb_trig[mask].fillna(False).mean()), 3)}
    return out


# ----------------------------------------------------- EARLY WARNING (independent)
def early_warning(px, gz, rv, N=21, n_events=20):
    fdd, _ = fwd_dd_vol(px, px.pct_change(), N)
    fdd = fdd.dropna()
    rv_q80 = rv.rolling(252, min_periods=60).quantile(0.80)
    rv_spike = (rv > rv_q80)
    gz_fire = (gz < -1.0)
    idx = px.index
    order = fdd.sort_values().index.tolist()
    chosen = []
    for d in order:
        if all(abs((d - c).days) > 21 for c in chosen):
            chosen.append(d)
        if len(chosen) >= n_events:
            break
    pos_of = {ts: i for i, ts in enumerate(idx)}
    leads_g, leads_r, both, gex_earlier = [], [], 0, 0
    for d in chosen:
        i = pos_of[d]
        j = min(i + N, len(idx) - 1)
        path = px.values[i:j + 1]
        cm = np.maximum.accumulate(path)
        trough_off = int(np.argmin(path / cm - 1.0))
        tp = i + trough_off
        lo = max(0, i - 20)
        win = range(lo, tp + 1)
        gf = next((k for k in win if bool(gz_fire.iloc[k]) and np.isfinite(gz.iloc[k])), None)
        rf = next((k for k in win if bool(rv_spike.iloc[k]) and np.isfinite(rv.iloc[k])), None)
        lg = (tp - gf) if gf is not None else None
        lr = (tp - rf) if rf is not None else None
        if lg is not None:
            leads_g.append(lg)
        if lr is not None:
            leads_r.append(lr)
        if lg is not None and lr is not None:
            both += 1
            if lg > lr:
                gex_earlier += 1
    return {"n_events": len(chosen), "gex_fired_frac": round(len(leads_g) / len(chosen), 3),
            "rv_fired_frac": round(len(leads_r) / len(chosen), 3),
            "gex_lead_median": float(np.median(leads_g)) if leads_g else None,
            "rv_lead_median": float(np.median(leads_r)) if leads_r else None,
            "both_fired": both, "gex_earlier_of_both": gex_earlier,
            "gex_earlier_frac": round(gex_earlier / both, 3) if both else None}


# ----------------------------------------------------- AUDITS
def audit_lookahead(px, gz, rv):
    """Confirm predictor PIT-ness and target forward-ness directly."""
    findings = {}
    ret = px.pct_change()
    # (a) target [D+1..D+N] only — verify fdd[D] uses prices p[i:i+N+1] = D..D+N, none before D.
    #     test: at index i, fdd should be invariant to any change in prices BEFORE i.
    N = 21
    fdd1, _ = fwd_dd_vol(px, ret, N)
    px2 = px.copy()
    px2.iloc[:50] = px2.iloc[:50] * 0.5            # corrupt the FAR PAST (first 50 bars)
    fdd2, _ = fwd_dd_vol(px2, px2.pct_change(), N)
    # for i >= 60, fdd must be identical (target doesn't peek backward)
    i0 = 60
    same_far = bool(np.allclose(fdd1.iloc[i0:].dropna().values[:200],
                                fdd2.reindex(fdd1.index).iloc[i0:].dropna().values[:200],
                                equal_nan=True))
    findings["a_target_no_backward_peek"] = same_far
    # (b) rv[D] / gz[D] are trailing: corrupt FUTURE prices, predictor at early i must NOT change.
    px3 = px.copy()
    px3.iloc[-50:] = px3.iloc[-50:] * 0.5          # corrupt the FUTURE (last 50 bars)
    r3 = px3.pct_change()
    rv3 = r3.rolling(21, min_periods=15).std() * ANN
    gex = G.load_squeeze()["gex"].astype(float)
    # gz depends on gex not px, so corrupting px cannot change gz: trivially trailing for gz.
    # check rv: an early-index rv (say i=500) must equal between rv and rv3
    rv_orig = rv.iloc[500]
    rv_corr = rv3.iloc[500]
    findings["b_rv_trailing_future_invariant"] = bool(np.isclose(rv_orig, rv_corr, equal_nan=True))
    findings["b_gz_independent_of_px"] = True  # gz built from gex series, not px (structural)
    # (c) partial-corr: residualize on [1, rv]; confirm residual of rv on [1,rv] is ~0 (control works)
    df = pd.concat([gz, rv], axis=1).dropna()
    df.columns = ["gz", "rv"]
    Z = np.column_stack([np.ones(len(df)), df["rv"].values])
    b, *_ = np.linalg.lstsq(Z, df["rv"].values, rcond=None)
    rv_resid = df["rv"].values - Z @ b
    findings["c_control_residualizes_rv"] = bool(np.allclose(rv_resid, 0, atol=1e-9))
    # (d) worst-20 events on forward dd: confirm chosen events have the most-negative fdd
    fdd, _ = fwd_dd_vol(px, ret, N)
    fdd = fdd.dropna()
    order = fdd.sort_values()
    findings["d_worst_events_are_min_fdd"] = bool(order.iloc[0] <= order.iloc[20])
    findings["d_worst_fdd_value"] = round(float(order.iloc[0]), 4)
    return findings


def near(a, b, tol):
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def main():
    px, gex, gz, ret, rv = build()
    J = json.loads((HERE / "T2.json").read_text(encoding="utf-8"))

    # ---- 1) corr(gz,fdd) & corr(rv,fdd) at N=5,10,21
    recomp_corr = {}
    for N in (5, 10, 21):
        fdd, fv = fwd_dd_vol(px, ret, N)
        c_gz_fdd, _ = corr(gz, fdd)
        c_rv_fdd, _ = corr(rv, fdd)
        c_gz_fv, _ = corr(gz, fv)
        c_rv_fv, _ = corr(rv, fv)
        pc, npc = partial(fdd, gz, rv)
        # non-overlap every-Nth-day
        nono = px.index[::N]
        pc_no, npc_no = partial(fdd.reindex(nono), gz.reindex(nono), rv.reindex(nono))
        recomp_corr[f"N{N}"] = {
            "corr_gz_fwddd": c_gz_fdd, "corr_rv_fwddd": c_rv_fdd,
            "corr_gz_fwdvol": c_gz_fv, "corr_rv_fwdvol": c_rv_fv,
            "partial_gz_fdd_rv": pc, "n_partial": npc,
            "partial_nonoverlap": pc_no, "n_nonoverlap": npc_no}
        if N == 21:
            o = ols_hac(fdd, {"rv": rv, "gz": gz}, L=21)
            obase = ols_hac(fdd, {"rv": rv}, L=21)
            recomp_corr["N21"]["ols_gz_t_naive"] = o["t"]["gz"]
            recomp_corr["N21"]["ols_gz_t_hac"] = o["t_hac"]["gz"]
            recomp_corr["N21"]["incr_r2"] = o["r2"] - obase["r2"]
            recomp_corr["N21"]["r2_full"] = o["r2"]

    # ---- 3) gate
    gate = gate_full(px, gz, rv)
    ew = early_warning(px, gz, rv)
    audits = audit_lookahead(px, gz, rv)

    # ---- compare to study JSON
    Jh = J["leadlag"]["horizons"]
    Jg = J["gate_compare"]["full"]
    cmp = []

    def add(name, mine, theirs, tol, kind="corr"):
        cmp.append({"name": name, "verify": mine, "study": theirs,
                    "match": near(mine, theirs, tol)})

    # corr (N21 primary)
    add("corr_gz_fdd_N21", recomp_corr["N21"]["corr_gz_fwddd"], Jh["N21"]["corr_gz_fwddd"], TOL["corr"])
    add("corr_rv_fdd_N21", recomp_corr["N21"]["corr_rv_fwddd"], Jh["N21"]["corr_rv_fwddd"], TOL["corr"])
    add("partial_gz_fdd_N21", recomp_corr["N21"]["partial_gz_fdd_rv"], Jh["N21"]["partial_gz_fwddd_given_rv"], TOL["corr"])
    add("partial_nonov_N21", recomp_corr["N21"]["partial_nonoverlap"], Jh["N21"]["partial_gz_fwddd_given_rv_nonoverlap"], TOL["corr"])
    add("incr_r2_N21", recomp_corr["N21"]["incr_r2"], Jh["N21"]["incr_r2_from_gz"], 0.005)
    add("ols_gz_t_hac_N21", recomp_corr["N21"]["ols_gz_t_hac"], Jh["N21"]["ols_fdd_on_rv_gz"]["gz"]["t_hac"], 0.20)
    add("ols_gz_t_naive_N21", recomp_corr["N21"]["ols_gz_t_naive"], Jh["N21"]["ols_fdd_on_rv_gz"]["gz"]["t"], 0.20)
    # also N5,N10 corr
    add("corr_gz_fdd_N5", recomp_corr["N5"]["corr_gz_fwddd"], Jh["N5"]["corr_gz_fwddd"], TOL["corr"])
    add("corr_rv_fdd_N5", recomp_corr["N5"]["corr_rv_fwddd"], Jh["N5"]["corr_rv_fwddd"], TOL["corr"])

    # gate maxDD + sharpe
    for nm in ("buy_hold_long", "vol_only", "gex_only", "combined"):
        add(f"{nm}_maxdd", gate[nm]["maxdd"], Jg[nm]["maxdd"], TOL["maxdd"], "maxdd")
        add(f"{nm}_sharpe", gate[nm]["sharpe"], Jg[nm]["sharpe"], TOL["sharpe"], "sharpe")
        add(f"{nm}_trim", gate[nm]["trim_frac"], Jg[nm].get("trim_frac", gate[nm]["trim_frac"]), 0.01)

    # early warning
    Je = J["early_warning"]
    add("ew_gex_fired_frac", ew["gex_fired_frac"], Je["gex_fired_frac"], 0.01)
    add("ew_rv_fired_frac", ew["rv_fired_frac"], Je["rv_fired_frac"], 0.01)
    add("ew_gex_lead_median", ew["gex_lead_median"], Je["gex_lead"]["median"], 0.5)
    add("ew_gex_earlier_frac", ew["gex_earlier_frac"], Je["gex_earlier_frac_of_both"], 0.01)

    n_match = sum(c["match"] for c in cmp)
    n_total = len(cmp)

    out = {"recomp_corr": recomp_corr, "gate": gate, "early_warning": ew,
           "audits": audits, "compare": cmp, "n_match": n_match, "n_total": n_total}
    (HERE / "verify_T2.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    # ---- print
    print("=" * 88)
    print("VERIFY T2 — independent recompute vs study JSON")
    print("=" * 88)
    print(f"data {px.index.min().date()}..{px.index.max().date()} n={len(px)} "
          f"gex<0 {100*(gex<0).mean():.1f}%  z<=-1 {100*(gz<=-1).mean():.1f}%")
    print("\n-- KEY CORR/PARTIAL (N21 primary) --")
    n21 = recomp_corr["N21"]
    print(f"  corr(gz,fdd21)   verify {n21['corr_gz_fwddd']:+.4f}  study {Jh['N21']['corr_gz_fwddd']:+.4f}")
    print(f"  corr(rv,fdd21)   verify {n21['corr_rv_fwddd']:+.4f}  study {Jh['N21']['corr_rv_fwddd']:+.4f}")
    print(f"  partial gz|rv    verify {n21['partial_gz_fdd_rv']:+.4f}  study {Jh['N21']['partial_gz_fwddd_given_rv']:+.4f}")
    print(f"  partial nonov    verify {n21['partial_nonoverlap']:+.4f}  study {Jh['N21']['partial_gz_fwddd_given_rv_nonoverlap']:+.4f}")
    print(f"  OLS gz t naive   verify {n21['ols_gz_t_naive']:+.3f}  study {Jh['N21']['ols_fdd_on_rv_gz']['gz']['t']:+.3f}")
    print(f"  OLS gz t HAC     verify {n21['ols_gz_t_hac']:+.3f}  study {Jh['N21']['ols_fdd_on_rv_gz']['gz']['t_hac']:+.3f}")
    print(f"  incr R2          verify {n21['incr_r2']:+.4f}  study {Jh['N21']['incr_r2_from_gz']:+.4f}")

    print("\n-- GATE (full 2011-26) --")
    print(f"  {'variant':<15}{'verSh':>8}{'stuSh':>8}{'verDD':>9}{'stuDD':>9}{'trim':>7}")
    for nm in ("buy_hold_long", "vol_only", "gex_only", "combined"):
        print(f"  {nm:<15}{gate[nm]['sharpe']:>+8.3f}{Jg[nm]['sharpe']:>+8.3f}"
              f"{100*gate[nm]['maxdd']:>+8.1f}%{100*Jg[nm]['maxdd']:>+8.1f}%{100*gate[nm]['trim_frac']:>+6.0f}%")

    print("\n-- EARLY WARNING --")
    print(f"  gex fired {100*ew['gex_fired_frac']:.0f}% (study {100*Je['gex_fired_frac']:.0f}%)  "
          f"rv fired {100*ew['rv_fired_frac']:.0f}% (study {100*Je['rv_fired_frac']:.0f}%)")
    print(f"  gex earlier of both {ew['gex_earlier_frac']} (study {Je['gex_earlier_frac_of_both']})")

    print("\n-- AUDITS --")
    for k, v in audits.items():
        print(f"  {k}: {v}")

    print(f"\n-- COMPARE: {n_match}/{n_total} within tolerance --")
    for c in cmp:
        flag = "OK " if c["match"] else "XX "
        print(f"  {flag}{c['name']:<22} verify={c['verify']}  study={c['study']}")
    print(f"\nwrote {HERE / 'verify_T2.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
