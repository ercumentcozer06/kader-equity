"""
backtest/karsan/sp_settlement_probe.py — PRE-REGISTERED SETTLEMENT-CHARM PROBE.

Equity analog of the BTC charm_f4 satellite (Deribit monthly settle 00:00->08:00 UTC
pre-settle short, +0.325%/event, DSR 0.485 = small tradeable, NOT alpha).

MECHANIC (locked):
 - SPY/QQQ ETF options = PM-settled at 16:00 ET close -> "into-settle" = RTH run into 16:00 on 3rd-Fri.
 - SPX/NDX index options (quad-witch = 3rd Fri Mar/Jun/Sep/Dec) = AM-settled to SOQ at 09:30 open
   -> "into-settle" = OVERNIGHT before the Friday open (daily OPEN vs prior CLOSE).

LOCKED spec: NO grid-search, NO threshold tuning, NO added signals. Two-sided primary always.
Directional net uses ONLY the pre-registered sign. Costs applied to NET (3 bps RT everywhere).
BH-FDR + Bonferroni + MC-null(5000 perm) + DSR(n_trials = 37 prior + N_this).

  & <venv> backtest/karsan/sp_settlement_probe.py  -> results/sp_settlement_probe.json + stdout table.

ANTI-LOOKAHEAD: entry uses only info known at entry instant; exit realized later.
 gap: entry=close[F] (known), exit=next open. No full-sample z/centering. Events = calendar-mechanical 3rd Fridays.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import k_config as K
import k_stats as KS
import k_data as KD
from k_phase1 import third_fridays   # reuse locked 3rd-Friday logic

RNG = K.boot_rng()                       # bootstrap rng (seed K.SEED)
MC_RNG = np.random.default_rng(K.SEED)   # MC-null permutation rng (seed K.SEED)

DAILY_COST = 0.0003                      # 3 bps round-trip (ES/NQ or SPY/QQQ)
INTRA_COST = 0.0003                      # 3 bps round-trip SPY/QQQ intraday
N_MC = 5000

N_THIS = 16                              # 5 daily signals x2 assets + 3 intraday x2 assets
DSR_NTRIALS = 37 + N_THIS               # honest cumulative memory (37 prior Karsan + this probe)

TRIALS = []


# ----------------------------------------------------------------------------- MC-null helpers
def mc_onesample(pool: np.ndarray, k: int, obs_mean: float, n=N_MC) -> float:
    """Permute event-day membership among candidate days: draw k days from pool, mean. Two-sided p vs 0."""
    pool = np.asarray(pool, float); pool = pool[~np.isnan(pool)]
    if len(pool) < k or k < 1:
        return float("nan")
    idx = np.argsort(MC_RNG.random((n, len(pool))), axis=1)[:, :k]
    draws = pool[idx].mean(1)
    return float((np.abs(draws) >= abs(obs_mean)).mean())


def mc_meandiff(vals: np.ndarray, mask: np.ndarray, obs: float, n=N_MC) -> float:
    """Permute the in/out label among candidate days; recompute mean(in)-mean(out). Two-sided p."""
    v = np.asarray(vals, float); m = np.asarray(mask, bool)
    ok = ~np.isnan(v); v, m = v[ok], m[ok]
    N = len(v); k = int(m.sum())
    if k < 1 or k >= N:
        return float("nan")
    perm = np.argsort(MC_RNG.random((n, N)), axis=1)[:, :k]
    sums_in = v[perm].sum(1)
    means_in = sums_in / k
    means_out = (v.sum() - sums_in) / (N - k)
    draws = means_in - means_out
    return float((np.abs(draws) >= abs(obs)).mean())


# ----------------------------------------------------------------------------- daily metric builder
def daily_metric(px: pd.DataFrame, kind: str) -> pd.Series:
    """Return per-date metric series (indexed at the ENTRY-reference date t). No look-ahead in entry decision."""
    o, c = px["o"], px["c"]
    if kind == "gap_next":   # short at close[t], cover next open -> open[t+1]/close[t]-1
        return o.shift(-1) / c - 1.0
    if kind == "gap_into":   # overnight into AM SOQ: enter close[t-1], exit open[t] -> open[t]/close[t-1]-1
        return o / c.shift(1) - 1.0
    if kind == "intraday":   # quad AM-settle day: enter open[t], exit close[t] -> close[t]/open[t]-1
        return c / o - 1.0
    if kind == "fwd3":       # enter close[t], exit close[t+3 sessions] -> close[t+3]/close[t]-1
        return c.shift(-3) / c - 1.0
    raise ValueError(kind)


# ----------------------------------------------------------------------------- per-signal processors
def process_onesample(sid, family, asset, ev_dates, metric, theory_dir, cost, note):
    """Daily one-sample event signal. Two-sided primary via block-boot; directional net via pre-reg sign only."""
    ev = metric.reindex(ev_dates).values.astype(float)
    ev = ev[~np.isnan(ev)]
    pool = metric.dropna().values.astype(float)
    n_ev = int(len(ev))
    two = KS.one_sample_block_boot(ev, RNG)
    raw_p = float(two["p"])
    mean_raw = float(ev.mean())

    sign = -1.0 if theory_dir == "short" else (1.0 if theory_dir == "long" else 1.0)
    directional = theory_dir in ("short", "long")
    net_ev = sign * ev - cost
    gross_bps = 1e4 * sign * mean_raw
    net_bps = 1e4 * float(net_ev.mean())
    sd = float(net_ev.std())
    sharpe_net = float(net_ev.mean() / sd) if sd > 0 else 0.0
    if directional:
        sign_matches = (mean_raw < 0) if theory_dir == "short" else (mean_raw > 0)
    else:
        sign_matches = False

    mcnull_p = mc_onesample(pool, n_ev, mean_raw)
    dsrv = KS.dsr(sharpe_net, n_ev, DSR_NTRIALS)

    TRIALS.append(dict(id=sid, family=family, asset=asset, n_events=n_ev,
                       gross_bps=round(gross_bps, 3), net_bps=round(net_bps, 3),
                       sharpe_net=round(sharpe_net, 4), raw_p=raw_p,
                       mcnull_p=mcnull_p, dsr=dsrv, theory_dir=theory_dir,
                       sign_matches_theory=bool(sign_matches),
                       _ev_dates=[str(pd.Timestamp(d).date()) for d in ev_dates if not pd.isna(metric.get(d, np.nan))],
                       _ev_vals=ev.tolist(), _sign=sign, _cost=cost, _directional=directional,
                       _footprint=False, note=note))


def process_meandiff(sid, family, asset, vals, mask, ev_dates, theory_dir, cost, footprint, note):
    """Intraday event-vs-baseline signal. Two-sided primary = block-boot mean-diff (in vs out)."""
    v = np.asarray(vals, float); m = np.asarray(mask, bool)
    ok = ~np.isnan(v); v_ok, m_ok = v[ok], m[ok]
    res = KS.mean_diff_boot(v, m, RNG)
    raw_p = float(res["p"]); obs = float(res["obs"])
    n_ev = int(m_ok.sum())
    ev_vals = v_ok[m_ok]

    if footprint:
        gross_bps = 1e4 * obs; net_bps = 1e4 * obs
        sharpe_net = 0.0; sign_matches = False; dsrv = None
    else:
        mean_ev = float(ev_vals.mean())
        net_ev = ev_vals - cost                    # two-sided/no-direction -> long reporting convention
        gross_bps = 1e4 * mean_ev; net_bps = 1e4 * float(net_ev.mean())
        sd = float(net_ev.std()); sharpe_net = float(net_ev.mean() / sd) if sd > 0 else 0.0
        sign_matches = False
        dsrv = KS.dsr(sharpe_net, n_ev, DSR_NTRIALS)

    mcnull_p = mc_meandiff(v, m, obs)

    TRIALS.append(dict(id=sid, family=family, asset=asset, n_events=n_ev,
                       gross_bps=round(gross_bps, 3), net_bps=round(net_bps, 3),
                       sharpe_net=round(sharpe_net, 4), raw_p=raw_p,
                       mcnull_p=mcnull_p, dsr=dsrv, theory_dir=theory_dir,
                       sign_matches_theory=bool(sign_matches),
                       _ev_dates=[str(pd.Timestamp(d).date()) for d in ev_dates],
                       _ev_vals=ev_vals.tolist(), _sign=1.0, _cost=cost, _directional=False,
                       _footprint=footprint, note=note))


# ----------------------------------------------------------------------------- daily block
def run_daily():
    yf = KD.fetch_yf()
    for asset, key in (("SPX", "SPX"), ("NDX", "NDX")):
        px = yf[asset].copy()
        px.index = pd.to_datetime(px.index)
        tf = third_fridays(px.index)
        quad = pd.DatetimeIndex([d for d in tf if d.month in K.QUAD_MONTHS])

        gap_next = daily_metric(px, "gap_next")
        gap_into = daily_metric(px, "gap_into")
        intr = daily_metric(px, "intraday")
        fwd3 = daily_metric(px, "fwd3")

        # D1 un-pinning gap (monthly) — theory SHORT
        process_onesample(f"D1.{asset}", "D1_unpin_gap_monthly", asset, tf, gap_next, "short", DAILY_COST,
                          "post-settle un-pin: short close[F]->next open; theory tailwind removed")
        # D2 un-pinning gap (quad subset) — theory SHORT
        process_onesample(f"D2.{asset}", "D2_unpin_gap_quad", asset, quad, gap_next, "short", DAILY_COST,
                          "D1 restricted to quad-witch Fridays")
        # D3 into-AM-settle gap (quad) — overnight into SOQ; BTC prior SHORT
        process_onesample(f"D3.{asset}", "D3_into_amsettle_quad", asset, quad, gap_into, "short", DAILY_COST,
                          "overnight into AM SOQ (open[F]/prevclose-1); BTC into-settle prior short")
        # D4 quad-day drift (quad) — two-sided, no pre-reg direction
        process_onesample(f"D4.{asset}", "D4_quadday_drift", asset, quad, intr, "two-sided", DAILY_COST,
                          "AM-settle hedge-unwind day close/open-1; no pre-reg sign")
        # D5 post-settle 3-day drift (monthly) — theory SHORT
        process_onesample(f"D5.{asset}", "D5_postsettle_drift_monthly", asset, tf, fwd3, "short", DAILY_COST,
                          "un-pin week weak: close[F+3]/close[F]-1; FLAG overlap w/ Karsan Phase-2 day-after-opex")


# ----------------------------------------------------------------------------- intraday block
def load_1m(fn):
    p = ROOT / "data" / "historical_bars" / f"{fn}.parquet"
    if not p.exists():
        return None
    b = pd.read_parquet(p)
    ts = pd.to_datetime(b.index.get_level_values("timestamp")).tz_convert("America/New_York")
    df = pd.DataFrame({"o": b["open"].values, "c": b["close"].values}, index=ts).sort_index()
    df = df[(df.index.time >= pd.Timestamp("09:30").time()) & (df.index.time < pd.Timestamp("16:00").time())]
    df["date"] = pd.to_datetime(df.index.date)      # tz-naive session date
    df["ret1"] = df.groupby("date")["c"].pct_change()
    return df


def intraday_frame(df):
    """Per-session metrics: close-window cumret [15:00,16:00), close-window sum|ret| (compression), morning cumret [09:30,10:30)."""
    tt = df.index.time
    cwin = (tt >= pd.Timestamp("15:00").time()) & (tt < pd.Timestamp("16:00").time())
    mwin = (tt >= pd.Timestamp("09:30").time()) & (tt < pd.Timestamp("10:30").time())

    dfc = df[cwin]
    gc = dfc.groupby("date")
    close_cumret = gc["c"].last() / gc["o"].first() - 1.0       # enter 15:00 open, exit 16:00 close
    close_sumabs = dfc.assign(a=dfc["ret1"].abs()).groupby("date")["a"].sum()

    dfm = df[mwin]
    gm = dfm.groupby("date")
    morn_cumret = gm["c"].last() / gm["o"].first() - 1.0        # enter 09:30 open, exit 10:30 close

    D = pd.DataFrame({"close_cumret": close_cumret, "close_sumabs": close_sumabs,
                      "morn_cumret": morn_cumret}).sort_index()
    return D


def run_intraday():
    for asset, fn in (("SPY", "alpaca_spy_1m"), ("QQQ", "alpaca_qqq_1m")):
        df = load_1m(fn)
        if df is None:
            TRIALS.append(dict(id=f"I1.{asset}", family="I1_pin_into_close", asset=asset, n_events=0,
                               gross_bps=0.0, net_bps=0.0, sharpe_net=0.0, raw_p=1.0, mcnull_p=float("nan"),
                               dsr=None, theory_dir="two-sided", sign_matches_theory=False,
                               _ev_dates=[], _ev_vals=[], _sign=1.0, _cost=INTRA_COST, _directional=False,
                               _footprint=False, note="DATA MISSING: 1-min parquet not found"))
            continue
        D = intraday_frame(df)
        didx = pd.DatetimeIndex(D.index)
        tf = third_fridays(didx)
        opex_mask = didx.isin(tf)
        quad = pd.DatetimeIndex([d for d in tf if d.month in K.QUAD_MONTHS])
        fri_mask = (didx.weekday == 4)

        # I1 PM-settle pin-into-close (monthly): [15:00->16:00] cumret, opex vs non-opex. two-sided; theory ~0.
        v1 = D["close_cumret"].values
        process_meandiff(f"I1.{asset}", "I1_pin_into_close", asset, v1, opex_mask, tf,
                          "two-sided", INTRA_COST, False,
                          "DIRECT BTC PM-settle analog; Karsan pin -> ~0/slight support; SINGLE-REGIME ~5.8y")

        # I2 PM-settle compression (monthly): sum|ret| [15:00->16:00], opex vs non-opex. footprint (not directional).
        v2 = D["close_sumabs"].values
        process_meandiff(f"I2.{asset}", "I2_compression", asset, v2, opex_mask, tf,
                          "footprint", INTRA_COST, True,
                          "pin=compression (lower abs); footprint not a strategy; SINGLE-REGIME")

        # I3 post-AM-settle morning (quad): [09:30->10:30] cumret, quad-Fri vs NON-quad Fridays. UNDERPOWERED report-only.
        v3 = D["morn_cumret"].values
        quad_mask = didx.isin(quad)
        fri_v = v3[fri_mask]
        fri_quadmask = quad_mask[fri_mask]
        fri_dates = didx[fri_mask]
        process_meandiff(f"I3.{asset}", "I3_post_amsettle_morning", asset, fri_v, fri_quadmask,
                          quad, "two-sided", INTRA_COST, False,
                          "quad-Fri vs non-quad-Fri morning; ~23 events UNDERPOWERED -> report only, never sell")


# ----------------------------------------------------------------------------- robustness (passers only)
def robustness(tr):
    """For a passer: regime sign stability. Daily = pre2008/2008-2015/2015+ + drop2020 + costx2/x3 + halves. Intraday=single-regime."""
    dates = pd.to_datetime(tr["_ev_dates"]); vals = np.asarray(tr["_ev_vals"], float)
    sign = tr["_sign"]; cost = tr["_cost"]
    out = {}
    if len(dates) != len(vals) or len(vals) == 0:
        return {"note": "no per-event series"}
    def netmean(mask, c):
        x = vals[mask]
        if len(x) == 0:
            return None
        return float((sign * x - c).mean())
    if tr["id"].startswith("I"):
        out["regime"] = "SINGLE-REGIME (~5.8y 1-min) — no multi-regime split possible"
    else:
        yrs = dates.year
        splits = {"pre2008": yrs < 2008, "2008_2015": (yrs >= 2008) & (yrs < 2015), "2015plus": yrs >= 2015}
        signs = {}
        for nm, mk in splits.items():
            mv = netmean(np.asarray(mk), cost)
            signs[nm] = None if mv is None else (1 if mv > 0 else -1)
            out[f"net_{nm}"] = None if mv is None else round(1e4 * mv, 2)
        pos = [s for s in signs.values() if s == 1]
        out["sign_stable_ge2_regimes"] = len(pos) >= 2
        # drop-2020
        out["net_drop2020"] = round(1e4 * netmean(np.asarray(yrs != 2020), cost), 2)
        # halves
        med = np.median(np.arange(len(vals)))
        out["net_firsthalf"] = round(1e4 * netmean(np.arange(len(vals)) <= med, cost), 2)
        out["net_secondhalf"] = round(1e4 * netmean(np.arange(len(vals)) > med, cost), 2)
    # cost sensitivity (all)
    out["net_costx2"] = round(1e4 * netmean(np.ones(len(vals), bool), cost * 2), 2)
    out["net_costx3"] = round(1e4 * netmean(np.ones(len(vals), bool), cost * 3), 2)
    return out


# ----------------------------------------------------------------------------- main
def main():
    print("=" * 100)
    print("  PRE-REGISTERED SETTLEMENT-CHARM PROBE — equity analog of BTC charm_f4 (locked spec)")
    print("=" * 100)
    print("  Building daily signals (SPX/NDX, 1990+, multi-regime)...")
    run_daily()
    print("  Building intraday signals (SPY/QQQ 1-min, ~2020-09..2026-06, SINGLE-REGIME)...")
    run_intraday()

    # ---- multiple testing over ALL trials in THIS probe ----
    raw_ps = [t["raw_p"] for t in TRIALS]
    bh_adj = KS.bh_fdr(raw_ps)
    N = len(TRIALS)
    for t, a in zip(TRIALS, bh_adj):
        t["bh_p"] = float(a)
        t["pass_bh"] = bool(a < K.FDR_ALPHA)
        t["bonf_p"] = float(min(t["raw_p"] * N, 1.0))
        t["pass_bonf"] = bool(t["bonf_p"] < 0.05)
        dsr_ok = (t["dsr"] is not None) and (t["dsr"] > 0.5)
        t["tradeable_candidate"] = bool(t["pass_bh"] and t["pass_bonf"]
                                        and t["sharpe_net"] > 0 and t["sign_matches_theory"]
                                        and t["_directional"] and dsr_ok)

    # ---- robustness for any passer (BH or Bonferroni) ----
    for t in TRIALS:
        if t["pass_bh"] or t["pass_bonf"]:
            t["robustness"] = robustness(t)

    # ---- persist ----
    clean = []
    for t in TRIALS:
        c = {k: v for k, v in t.items() if not k.startswith("_")}
        clean.append(c)
    out_json = {
        "probe": "sp_settlement_probe",
        "n_trials_this_probe": N,
        "dsr_n_trials_cumulative": DSR_NTRIALS,
        "prior_karsan_trials": 37,
        "costs_bps_rt": {"daily": 1e4 * DAILY_COST, "intraday": 1e4 * INTRA_COST},
        "mc_null_perms": N_MC,
        "seed": K.SEED,
        "trials": clean,
    }
    K.KRESULTS.mkdir(parents=True, exist_ok=True)
    (K.KRESULTS / "sp_settlement_probe.json").write_text(
        json.dumps(out_json, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

    # ---- stdout table ----
    print(f"\n  N_trials (this probe) = {N}   |   DSR n_trials (cumulative honest) = 37 + {N} = {DSR_NTRIALS}")
    print(f"  Costs: daily {1e4*DAILY_COST:.0f}bps RT, intraday {1e4*INTRA_COST:.0f}bps RT  |  MC-null perms = {N_MC}  |  seed = {K.SEED}")
    print("=" * 100)
    hdr = (f"  {'id':9}{'family':26}{'n':>4}{'gross':>8}{'net':>8}{'shN':>7}"
           f"{'rawp':>7}{'bhp':>7}{'bonfp':>7}{'mcp':>7}{'dsr':>6}  pass")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for t in TRIALS:
        dsr_s = "  -  " if t["dsr"] is None else f"{t['dsr']:.3f}"
        pm = "BH" if t["pass_bh"] else " ."
        pb = "BF" if t["pass_bonf"] else " ."
        trd = "TRADE" if t["tradeable_candidate"] else ""
        print(f"  {t['id']:9}{t['family'][:25]:26}{t['n_events']:>4}{t['gross_bps']:>8.1f}{t['net_bps']:>8.1f}"
              f"{t['sharpe_net']:>7.3f}{t['raw_p']:>7.3f}{t['bh_p']:>7.3f}{t['bonf_p']:>7.3f}"
              f"{t['mcnull_p']:>7.3f}{dsr_s:>6}  {pm}/{pb} {trd}")
    print("  " + "-" * (len(hdr) - 2))
    n_bh = sum(t["pass_bh"] for t in TRIALS)
    n_bf = sum(t["pass_bonf"] for t in TRIALS)
    n_tr = sum(t["tradeable_candidate"] for t in TRIALS)
    print(f"  passes: BH-FDR={n_bh}  Bonferroni={n_bf}  tradeable-candidates={n_tr}")
    print("  NOTE: theory_dir 'short' signals net = (-gap)-cost; 'two-sided'/'footprint' report only (no pre-reg sign).")
    print("  NOTE: I1/I2/I3 SINGLE-REGIME; I3 ~23 quad events UNDERPOWERED (report only, never sell).")
    print("  -> results/sp_settlement_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
