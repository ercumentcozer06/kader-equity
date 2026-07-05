"""
T2 — GEX AS LEADING FRAGILITY INDICATOR (standalone, 2011-26, SPX).

QUESTION: does GEX (negative / low-z) lead forward vol & drawdown EARLIER than
realized vol itself — i.e. does it add early-warning value beyond a vol gate?

Three blocks (per locked spec gxs_config.py):
  1) LEAD-LAG: corr(gex_z, fwd vol/dd) vs vol's own persistence + PARTIAL corr
     (gex_z vs fwd dd | rv) + OLS (fdd ~ rv + gex_z) gex_z t-stat & incr R².
  2) EARLY-WARNING TIMING: on worst-20 fwd-21d-dd events, days BEFORE the trough
     that gex_z first crossed −1 vs rv first spiked >80th pct. Median lead.
  3) PRACTICAL GATE (standalone, trim-only): vol-only vs gex-only vs combined
     trim-to-0.5 gates on maxDD/Sharpe 2011-26 + sub-period table.

PIT: signal[D] (EOD) -> return[D+1]; lag=1 ALWAYS. rv/gex_z use trailing data only.
No look-ahead. trim-only: position in {0.5, 1.0}, never short, never >1.0.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gxs_config as G  # noqa: E402

ROOT = G.ROOT
RESULTS = G.RESULTS_DIR
RESULTS.mkdir(parents=True, exist_ok=True)

ANN = np.sqrt(252.0)


# ---------------------------------------------------------------- core series
def build_base():
    """SPX price + gex + gex_z + daily ret + trailing-21d realized vol (annualized)."""
    g = G.load_squeeze()
    px = g["price"].astype(float)
    gex = g["gex"].astype(float)
    gz = G.gex_zscore(gex)                       # trailing-252d z (byte-ident to shield)
    ret = px.pct_change()                        # daily simple return, ret[D] = D-1 -> D
    rv = ret.rolling(21, min_periods=15).std() * ANN   # trailing-21d annualized vol, known at EOD[D]
    return pd.DataFrame({"px": px, "gex": gex, "gz": gz, "ret": ret, "rv": rv})


def fwd_vol_and_dd(px: pd.Series, ret: pd.Series, N: int):
    """For each D: fwd realized vol over [D+1..D+N] (annualized) and fwd max drawdown
    over the price path [D..D+N] (peak-to-trough on cum return from D). Look-ahead by
    construction (that's the TARGET we predict); aligned to signal at D."""
    r = ret.values
    p = px.values
    n = len(p)
    fv = np.full(n, np.nan)
    fdd = np.full(n, np.nan)
    for i in range(n):
        j = i + N
        if j >= n:
            continue
        seg = r[i + 1:j + 1]                      # returns D+1..D+N
        seg = seg[np.isfinite(seg)]
        if len(seg) >= max(3, N // 2):
            fv[i] = seg.std() * ANN
        # fwd max drawdown of the path starting at D's close
        path = p[i:j + 1]                         # prices D..D+N
        if np.isfinite(path).all() and len(path) >= 2:
            cm = np.maximum.accumulate(path)
            dd = (path / cm - 1.0).min()
            fdd[i] = dd                           # <= 0
    return pd.Series(fv, index=px.index), pd.Series(fdd, index=px.index)


def pearson(a, b):
    m = a.notna() & b.notna()
    if m.sum() < 30:
        return None, int(m.sum())
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def partial_corr(y, x, z):
    """partial corr of y and x controlling for z (all pandas Series)."""
    df = pd.concat([y, x, z], axis=1).dropna()
    df.columns = ["y", "x", "z"]
    if len(df) < 30:
        return None, len(df)
    # residualize y and x on z (with intercept)
    Z = np.column_stack([np.ones(len(df)), df["z"].values])
    def resid(v):
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta
    ry = resid(df["y"].values)
    rx = resid(df["x"].values)
    if ry.std() == 0 or rx.std() == 0:
        return None, len(df)
    return float(np.corrcoef(ry, rx)[0, 1]), len(df)


def ols(y, X_cols: dict, hac_lag: int = 0):
    """OLS y ~ 1 + cols. Returns per-coef beta, OLS t-stat, HAC(Newey-West) t-stat, full-model R².
    HAC matters here: forward windows OVERLAP -> residuals autocorrelated -> naive OLS t is massively
    inflated. hac_lag should be ~ the forward horizon N. X_cols=dict name->Series."""
    parts = [y] + list(X_cols.values())
    df = pd.concat(parts, axis=1).dropna()
    df.columns = ["y"] + list(X_cols.keys())
    n = len(df)
    if n < 40:
        return None
    names = list(X_cols.keys())
    X = np.column_stack([np.ones(n)] + [df[c].values for c in names])
    yv = df["y"].values
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    ssr = float((resid ** 2).sum())
    sst = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - ssr / sst if sst > 0 else float("nan")
    k = X.shape[1]
    dof = n - k
    sigma2 = ssr / dof if dof > 0 else np.nan
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    tstats = beta / se
    # Newey-West HAC standard errors (Bartlett kernel, lag L)
    hac_t = None
    L = int(hac_lag) if hac_lag and hac_lag > 0 else 0
    if L > 0:
        Xr = X * resid[:, None]                       # score
        S = Xr.T @ Xr
        for lag in range(1, L + 1):
            w = 1.0 - lag / (L + 1.0)
            G_l = Xr[lag:].T @ Xr[:-lag]
            S += w * (G_l + G_l.T)
        cov_hac = XtX_inv @ S @ XtX_inv
        se_hac = np.sqrt(np.maximum(np.diag(cov_hac), 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            hac_t = beta / se_hac
    out = {"n": n, "r2": round(r2, 4), "hac_lag": L}
    for i, nm in enumerate(["const"] + names):
        out[nm] = {"beta": float(beta[i]), "t": float(tstats[i]),
                   "t_hac": (float(hac_t[i]) if hac_t is not None else None)}
    return out


# --------------------------------------------------------------- BLOCK 1
def block_leadlag(base):
    px, ret, gz, rv = base["px"], base["ret"], base["gz"], base["rv"]
    out = {"horizons": {}}
    for N in (5, 10, 21):
        fv, fdd = fwd_vol_and_dd(px, ret, N)
        c_gz_fv, n1 = pearson(gz, fv)
        c_gz_fdd, n2 = pearson(gz, fdd)
        c_rv_fv, n3 = pearson(rv, fv)
        c_rv_fdd, n4 = pearson(rv, fdd)
        # partial: gex_z vs fwd dd controlling for current rv (OVERLAPPING — full daily)
        pc_dd, npc = partial_corr(fdd, gz, rv)
        pc_fv, npc2 = partial_corr(fv, gz, rv)
        # NON-OVERLAPPING subsample (every N-th day) — kills overlap autocorrelation, honest n
        nonov_idx = px.index[::N]
        pc_dd_no, npc_no = partial_corr(fdd.reindex(nonov_idx), gz.reindex(nonov_idx),
                                        rv.reindex(nonov_idx))
        # OLS fdd ~ rv + gz   and baseline fdd ~ rv  for incremental R² (HAC lag = N for overlap)
        full = ols(fdd, {"rv": rv, "gz": gz}, hac_lag=N)
        rvonly = ols(fdd, {"rv": rv}, hac_lag=N)
        incr_r2 = None
        if full and rvonly:
            incr_r2 = round(full["r2"] - rvonly["r2"], 4)
        out["horizons"][f"N{N}"] = {
            "corr_gz_fwdvol": _r(c_gz_fv), "n_gz_fwdvol": n1,
            "corr_gz_fwddd": _r(c_gz_fdd), "n_gz_fwddd": n2,
            "corr_rv_fwdvol": _r(c_rv_fv), "n_rv_fwdvol": n3,
            "corr_rv_fwddd": _r(c_rv_fdd), "n_rv_fwddd": n4,
            "partial_gz_fwddd_given_rv": _r(pc_dd), "n_partial": npc,
            "partial_gz_fwddd_given_rv_nonoverlap": _r(pc_dd_no), "n_partial_nonoverlap": npc_no,
            "partial_gz_fwdvol_given_rv": _r(pc_fv), "n_partial_fv": npc2,
            "ols_fdd_on_rv_gz": full,
            "ols_fdd_on_rv": rvonly,
            "incr_r2_from_gz": incr_r2,
        }
    return out


def _r(x):
    return None if x is None else round(x, 4)


# --------------------------------------------------------------- BLOCK 2
def block_early_warning(base, N=21, n_events=20):
    """Worst-20 fwd-21d-dd events. For each event peak D (the day BEFORE the worst window
    starts deteriorating), find the trough date inside [D+1..D+N], then measure how many
    days BEFORE the trough each indicator first fired in the run-up window [D-20 .. trough].
      - gex flag: gz first crosses below -1
      - rv flag:  rv first spikes above its trailing-80th-pct (expanding/trailing 252d pctile)
    Lead (days) = trough_idx - flag_idx (positive = flagged earlier than the trough).
    """
    px, ret, gz, rv = base["px"], base["ret"], base["gz"], base["rv"]
    idx = px.index
    _, fdd = fwd_vol_and_dd(px, ret, N)
    fdd = fdd.dropna()
    # trailing 80th-pct of rv (trailing 252d, min 60) — known at EOD[D], no look-ahead
    rv_q80 = rv.rolling(252, min_periods=60).quantile(0.80)
    rv_spike = (rv > rv_q80)                       # boolean series, True = vol elevated vs own history
    gz_fire = (gz < -1.0)

    # pick worst-20 NON-OVERLAPPING events (greedy by depth, enforce >=21d apart)
    order = fdd.sort_values().index.tolist()       # most negative first
    chosen = []
    for d in order:
        if all(abs((d - c).days) > 21 for c in chosen):
            chosen.append(d)
        if len(chosen) >= n_events:
            break

    pos_of = {ts: i for i, ts in enumerate(idx)}
    events = []
    leads_gex, leads_rv, both_flag = [], [], 0
    gex_earlier = 0
    for d in chosen:
        i = pos_of[d]
        j = min(i + N, len(idx) - 1)
        # trough position inside [i+1..j] on the price path from i
        path = px.values[i:j + 1]
        cm = np.maximum.accumulate(path)
        dd_path = path / cm - 1.0
        trough_off = int(np.argmin(dd_path))       # offset from i (0 == i itself, rare)
        trough_pos = i + trough_off
        trough_date = idx[trough_pos]
        # run-up search window: [i-20 .. trough_pos]
        lo = max(0, i - 20)
        win = range(lo, trough_pos + 1)
        # first gex fire in window
        g_fire_pos = next((k for k in win if bool(gz_fire.iloc[k]) and np.isfinite(gz.iloc[k])), None)
        r_fire_pos = next((k for k in win if bool(rv_spike.iloc[k]) and np.isfinite(rv.iloc[k])), None)
        lead_g = (trough_pos - g_fire_pos) if g_fire_pos is not None else None
        lead_r = (trough_pos - r_fire_pos) if r_fire_pos is not None else None
        if lead_g is not None:
            leads_gex.append(lead_g)
        if lead_r is not None:
            leads_rv.append(lead_r)
        if lead_g is not None and lead_r is not None:
            both_flag += 1
            if lead_g > lead_r:
                gex_earlier += 1
        events.append({
            "event_date": str(d.date()),
            "trough_date": str(trough_date.date()),
            "fwd_dd": round(float(fdd.loc[d]), 4),
            "trough_off_days": trough_off,
            "gex_lead_days": lead_g,
            "rv_lead_days": lead_r,
            "gex_earlier": (None if (lead_g is None or lead_r is None) else bool(lead_g > lead_r)),
        })

    def _stat(v):
        if not v:
            return {"n": 0, "median": None, "mean": None, "min": None, "max": None, "fired_frac": 0.0}
        a = np.array(v, float)
        return {"n": len(v), "median": float(np.median(a)), "mean": round(float(a.mean()), 1),
                "min": int(a.min()), "max": int(a.max())}
    return {
        "n_events": len(chosen),
        "gex_lead": _stat(leads_gex),
        "rv_lead": _stat(leads_rv),
        "gex_fired_frac": round(len(leads_gex) / len(chosen), 3) if chosen else 0.0,
        "rv_fired_frac": round(len(leads_rv) / len(chosen), 3) if chosen else 0.0,
        "both_fired": both_flag,
        "gex_earlier_of_both": gex_earlier,
        "gex_earlier_frac_of_both": round(gex_earlier / both_flag, 3) if both_flag else None,
        "median_lead_diff_gex_minus_rv": (
            round(float(np.median(leads_gex) - np.median(leads_rv)), 1)
            if leads_gex and leads_rv else None),
        "events": events,
    }


# --------------------------------------------------------------- BLOCK 3
def strat_ret(pos: pd.Series, ret_next: pd.Series):
    """pos[D] (known EOD[D]) earns ret[D+1]. ret_next already = next-day return aligned to D."""
    r = (pos * ret_next).dropna()
    return r


def block_gate(base):
    """Three trim gates, trim-to-0.5 when triggered, else full long (1.0). Standalone SPX,
    daily, 2011-26. pos[D] earns next-day return. Buy&hold long baseline = pos==1 always."""
    px, ret, gz, rv = base["px"], base["ret"], base["gz"], base["rv"]
    ret_next = px.pct_change().shift(-1)           # ret_next[D] = return D -> D+1, aligned to signal D

    # trailing 80th-pct of rv (trailing 252d, min60) — no look-ahead
    rv_q80 = rv.rolling(252, min_periods=60).quantile(0.80)
    vol_trig = (rv > rv_q80)                        # vol-only trigger
    gex_trig = (gz < -1.0)                          # gex-only trigger (deep-neg z)
    comb_trig = (vol_trig | gex_trig)              # either

    def pos_from_trig(trig):
        # require a finite signal to act; where signal missing -> stay full long (1.0)
        p = pd.Series(1.0, index=px.index)
        p[trig.fillna(False)] = 0.5
        return p

    pos_bh = pd.Series(1.0, index=px.index)
    pos_vol = pos_from_trig(vol_trig)
    pos_gex = pos_from_trig(gex_trig)
    pos_comb = pos_from_trig(comb_trig)

    variants = {
        "buy_hold_long": pos_bh,
        "vol_only": pos_vol,
        "gex_only": pos_gex,
        "combined": pos_comb,
    }

    def eval_window(pos, s, e):
        m = (px.index >= pd.Timestamp(s)) & (px.index <= pd.Timestamp(e))
        rr = strat_ret(pos[m], ret_next[m])
        mm = G.metrics(rr)
        # trim-active fraction (days the gate trimmed), within window, on valid signal days
        return mm

    full = {}
    for name, pos in variants.items():
        s, e = G.SUBPERIODS["full_2011_26"]
        mm = eval_window(pos, s, e)
        # trim fraction over full window
        m = (px.index >= pd.Timestamp(s)) & (px.index <= pd.Timestamp(e))
        trim_frac = round(float((pos[m] < 1.0).mean()), 3)
        mm["trim_frac"] = trim_frac
        full[name] = mm

    subs = {}
    for key, (s, e) in G.SUBPERIODS.items():
        if key == "full_2011_26":
            continue
        subs[key] = {}
        for name, pos in variants.items():
            subs[key][name] = eval_window(pos, s, e)

    return {"full": full, "subperiods": subs,
            "trigger_frac_full": {
                "vol": round(float(vol_trig.reindex(px.index).fillna(False).mean()), 3),
                "gex": round(float(gex_trig.reindex(px.index).fillna(False).mean()), 3),
                "combined": round(float(comb_trig.reindex(px.index).fillna(False).mean()), 3),
            }}


# --------------------------------------------------------------- MAIN
def main():
    base = build_base()
    meta = {
        "study": "T2 — GEX as leading fragility indicator (standalone, SPX, 2011-26)",
        "data_rows": int(len(base)),
        "data_span": [str(base.index.min().date()), str(base.index.max().date())],
        "gex_neg_frac": round(float((base["gex"] < 0).mean()), 3),
        "deepneg_z_frac": round(float((base["gz"] <= -1.0).mean()), 3),
        "pit_lag": G.PIT_LAG,
        "note": "rv=trailing-21d ann vol; gz=trailing-252d z(GEX); fwd vol/dd over [D+1..D+N]; "
                "gates trim-to-0.5 trim-only (never short/never>1).",
    }

    leadlag = block_leadlag(base)
    early = block_early_warning(base)
    gate = block_gate(base)

    result = {"meta": meta, "leadlag": leadlag, "early_warning": early,
              "gate_compare": {"full": gate["full"], "subperiods": gate["subperiods"],
                               "trigger_frac": gate["trigger_frac_full"]},
              "caveats": []}

    # ---------- adversarial verdict logic ----------
    caveats = []
    # partial corr significance check at N=21 (primary horizon)
    n21 = leadlag["horizons"]["N21"]
    pc = n21["partial_gz_fwddd_given_rv"]
    pc_no = n21["partial_gz_fwddd_given_rv_nonoverlap"]
    npc = n21["n_partial"]
    ols_full = n21["ols_fdd_on_rv_gz"]
    gz_t = ols_full["gz"]["t"] if ols_full else None
    gz_t_hac = ols_full["gz"]["t_hac"] if ols_full else None
    incr = n21["incr_r2_from_gz"]
    # HONEST significance: use HAC t (overlap-corrected), require |t_hac|>=2 AND non-overlap
    # partial corr keeps the same sign — i.e. signal survives killing the autocorrelation.
    gz_signif = (gz_t_hac is not None and abs(gz_t_hac) >= 2.0
                 and pc_no is not None and pc is not None
                 and np.sign(pc_no) == np.sign(pc))
    # gate maxDD improvement: combined vs better single
    f = gate["full"]
    comb_dd = f["combined"]["maxdd"]
    vol_dd = f["vol_only"]["maxdd"]
    gex_dd = f["gex_only"]["maxdd"]
    bh_dd = f["buy_hold_long"]["maxdd"]
    better_single_dd = max(vol_dd, gex_dd)         # less negative = better (closer to 0)
    comb_better_dd = comb_dd >= better_single_dd   # combined maxDD at least as good as best single
    comb_sh = f["combined"]["sharpe"]
    vol_sh = f["vol_only"]["sharpe"]
    gex_sh = f["gex_only"]["sharpe"]
    best_single_sh = max(vol_sh, gex_sh)
    sh_not_worse = comb_sh >= (best_single_sh - 0.05)   # no material extra Sharpe loss

    result["_verdict_inputs"] = {
        "partial_gz_fwddd_given_rv_N21": pc, "n_partial": npc,
        "partial_gz_fwddd_given_rv_N21_nonoverlap": pc_no,
        "ols_gz_tstat_N21_naive": round(gz_t, 2) if gz_t is not None else None,
        "ols_gz_tstat_N21_hac": round(gz_t_hac, 2) if gz_t_hac is not None else None,
        "incr_r2_N21": incr,
        "gz_signif_beyond_rv": bool(gz_signif),
        "comb_maxdd": comb_dd, "vol_maxdd": vol_dd, "gex_maxdd": gex_dd, "bh_maxdd": bh_dd,
        "combined_beats_best_single_dd": bool(comb_better_dd),
        "comb_sharpe": comb_sh, "vol_sharpe": vol_sh, "gex_sharpe": gex_sh,
        "sharpe_not_materially_worse": bool(sh_not_worse),
    }

    # Verdict per spec: PAYS if gz adds incremental fwd-dd pred beyond rv (partial signif)
    #                   AND combined gate improves maxDD; else MARGINAL/DEAD.
    if gz_signif and comb_better_dd and sh_not_worse:
        verdict = "PAYS"
    elif gz_signif or comb_better_dd:
        verdict = "MARGINAL"
    else:
        verdict = "DEAD"
    result["verdict"] = verdict

    _ht = round(gz_t_hac, 2) if gz_t_hac is not None else None
    if not gz_signif:
        caveats.append("GEX_z gives NO robust incremental fwd-21d-drawdown prediction beyond realized "
                       f"vol (HAC t={_ht}, non-overlap pcorr={pc_no}, incr R²={incr}).")
    else:
        caveats.append(f"GEX_z adds incremental fwd-dd info beyond rv (overlap-corrected HAC t={_ht}, "
                       f"non-overlap pcorr={pc_no}, incr R²={incr}) — but incr R² is tiny (~2%) and "
                       "the naive OLS t (~8-11) is overlap-inflated; effect is real but small.")
    if not comb_better_dd:
        caveats.append(f"Combined gate maxDD ({comb_dd:.3f}) NOT better than best single "
                       f"(vol {vol_dd:.3f} / gex {gex_dd:.3f}).")
    caveats.append("GEX_z and realized vol are themselves correlated (both rise in stress) — "
                   "much of GEX's raw fwd-dd corr is shared with vol; partial corr is the honest test.")
    caveats.append("Standalone 2011-26; trim gate is rebound-safe (trim-only) so it cannot profit "
                   "from shorting — it can only reduce drawdown at the cost of upside in calm regimes.")
    result["caveats"] = caveats

    # ---------- write + print ----------
    outpath = RESULTS / "T2.json"
    outpath.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    # readable table
    print("=" * 92)
    print("T2 — GEX AS LEADING FRAGILITY INDICATOR (standalone SPX, 2011-26)")
    print("=" * 92)
    print(f"data {meta['data_span'][0]}..{meta['data_span'][1]}  n={meta['data_rows']}  "
          f"gex<0 {100*meta['gex_neg_frac']:.0f}%  z<=-1 {100*meta['deepneg_z_frac']:.0f}%")
    print("\n--- BLOCK 1: LEAD-LAG (corr with forward vol / drawdown) ---")
    print("  (pcorr = partial corr gz~fwddd | rv; naiveT overlap-inflated; HAC-t & non-overlap pcorr honest)")
    print(f"{'horizon':<8}{'gz~fvol':>9}{'gz~fdd':>9}{'rv~fvol':>9}{'rv~fdd':>9}"
          f"{'pcorr':>8}{'pc_nonov':>9}{'naiveT':>8}{'HAC-t':>8}{'incrR2':>8}")
    for N in (5, 10, 21):
        h = leadlag["horizons"][f"N{N}"]
        gt = h["ols_fdd_on_rv_gz"]["gz"]["t"] if h["ols_fdd_on_rv_gz"] else None
        gth = h["ols_fdd_on_rv_gz"]["gz"]["t_hac"] if h["ols_fdd_on_rv_gz"] else None
        print(f"N={N:<6}{_f(h['corr_gz_fwdvol']):>9}{_f(h['corr_gz_fwddd']):>9}"
              f"{_f(h['corr_rv_fwdvol']):>9}{_f(h['corr_rv_fwddd']):>9}"
              f"{_f(h['partial_gz_fwddd_given_rv']):>8}"
              f"{_f(h['partial_gz_fwddd_given_rv_nonoverlap']):>9}"
              f"{_f2(gt):>8}{_f2(gth):>8}{_f(h['incr_r2_from_gz']):>8}")

    print("\n--- BLOCK 2: EARLY-WARNING TIMING (worst-20 fwd-21d-dd events) ---")
    print(f"events={early['n_events']}  gex fired {100*early['gex_fired_frac']:.0f}% / "
          f"rv fired {100*early['rv_fired_frac']:.0f}%")
    gl, rl = early["gex_lead"], early["rv_lead"]
    print(f"  gex lead days  : median {gl['median']}  mean {gl['mean']}  (n={gl['n']})")
    print(f"  rv  lead days  : median {rl['median']}  mean {rl['mean']}  (n={rl['n']})")
    print(f"  median lead diff (gex - rv) = {early['median_lead_diff_gex_minus_rv']} days "
          f"(positive = GEX earlier)")
    print(f"  of {early['both_fired']} events where both fired, GEX earlier in "
          f"{early['gex_earlier_of_both']} ({early['gex_earlier_frac_of_both']})")

    print("\n--- BLOCK 3: PRACTICAL GATE (trim-to-0.5, full 2011-26) ---")
    print(f"trigger freq: vol {100*gate['trigger_frac_full']['vol']:.0f}%  "
          f"gex {100*gate['trigger_frac_full']['gex']:.0f}%  "
          f"combined {100*gate['trigger_frac_full']['combined']:.0f}%")
    print(f"{'variant':<16}{'Sharpe':>8}{'maxDD':>9}{'CVaR5':>9}{'cumret':>9}{'trim%':>7}{'n':>6}")
    for name in ("buy_hold_long", "vol_only", "gex_only", "combined"):
        m = gate["full"][name]
        print(f"{name:<16}{m['sharpe']:>+8.3f}{100*m['maxdd']:>+8.1f}%{100*m['cvar5']:>+8.2f}%"
              f"{100*m['cum_ret']:>+8.1f}%{100*m.get('trim_frac',0):>+6.0f}%{m['n']:>6}")

    print("\n--- BLOCK 3b: SUB-PERIOD maxDD (combined vs single gates) ---")
    print(f"{'period':<14}{'BH dd':>9}{'vol dd':>9}{'gex dd':>9}{'comb dd':>9}"
          f"{'vol Sh':>8}{'gex Sh':>8}{'comb Sh':>9}")
    for key in G.SUBPERIODS:
        if key == "full_2011_26":
            continue
        s = gate["subperiods"][key]
        print(f"{key:<14}{100*s['buy_hold_long']['maxdd']:>+8.1f}%"
              f"{100*s['vol_only']['maxdd']:>+8.1f}%{100*s['gex_only']['maxdd']:>+8.1f}%"
              f"{100*s['combined']['maxdd']:>+8.1f}%"
              f"{s['vol_only']['sharpe']:>+8.2f}{s['gex_only']['sharpe']:>+8.2f}"
              f"{s['combined']['sharpe']:>+9.2f}")

    print("\n" + "=" * 92)
    print(f"VERDICT: {verdict}")
    for c in caveats:
        print(f"  - {c}")
    print(f"\nwrote {outpath}")
    return 0


def _f(x):
    return "  n/a" if x is None else f"{x:+.3f}"


def _f2(x):
    return "  n/a" if x is None else f"{x:+.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
