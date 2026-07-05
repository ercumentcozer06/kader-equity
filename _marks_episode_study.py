import sys, pandas as pd, numpy as np
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\Users\admin\Downloads\kader-equity")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)

panel = pd.read_parquet(r"C:\Users\admin\Downloads\kader-equity\data\cache\credit_froth_panel.parquet")
panel = panel.sort_index()
# Use full HY-OAS history for the z-score baseline so the expanding window is well-seeded by 2000.
hy_full = pd.read_parquet(r"C:\Users\admin\Downloads\kader-equity\data\cache\hy_oas_full.parquet")["hy_oas"].sort_index()

# HY-OAS is strongly right-skewed (spreads spike up violently, compress slowly): a Gaussian z
# never reaches -1.5/-2 on the DOWNSIDE because 2008/2020 upside outliers inflate the std.
# Marks' "extreme tight" = spread in its lowest percentile band -> use an expanding PERCENTILE RANK
# (robust to skew, no look-ahead), and keep the z as a secondary cross-check.
exp_mean = hy_full.expanding(min_periods=252).mean()
exp_std  = hy_full.expanding(min_periods=252).std()
z_full = (hy_full - exp_mean) / exp_std
z = z_full.reindex(panel.index).ffill()
panel["z"] = z

# expanding percentile rank of the level (fraction of history <= today's level). LOW = tight.
def exp_rank(s):
    out = pd.Series(index=s.index, dtype=float)
    arr = s.values
    import bisect
    sorted_hist = []
    for i, v in enumerate(arr):
        if i < 252:
            out.iloc[i] = np.nan
        else:
            # rank among history up to and including today
            le = np.sum(arr[:i+1] <= v)
            out.iloc[i] = le / (i+1)
    return out

pr_full = exp_rank(hy_full)
panel["pr"] = pr_full.reindex(panel.index).ffill()

spx = panel["spx_close"]
ndx = panel["ndx_close"]
cape = panel["cape"]
vix = panel["vix"]

# Rolling percentiles for confluence (expanding, to avoid look-ahead) of CAPE (high=frothy) and VIX (low=complacent).
cape_pct = cape.expanding(min_periods=252).apply(lambda s: (s <= s.iloc[-1]).mean(), raw=False)
vix_pct  = vix.expanding(min_periods=252).apply(lambda s: (s <= s.iloc[-1]).mean(), raw=False)
panel["cape_pct"] = cape_pct
panel["vix_pct"] = vix_pct


def episodes_from_flag(flag: pd.Series, gap_days=63):
    """Cluster True days into episodes; new episode if gap > gap_days between flagged days."""
    days = flag[flag].index
    eps = []
    if len(days) == 0:
        return eps
    cur = [days[0]]
    for d in days[1:]:
        if (d - cur[-1]).days > gap_days:
            eps.append(cur)
            cur = [d]
        else:
            cur.append(d)
    eps.append(cur)
    return eps


def trace(signal_date, price, horizon_days):
    """From signal_date, look forward horizon_days. Return max drawdown from the price level
    AT THE SIGNAL DATE (peak-to-trough relative to entry... but Marks cares about eventual loss,
    so we report two things: (a) drawdown from the running peak after signal (true peak-to-trough),
    (b) worst return vs the signal-day price). Plus lag months to the trough of (b)."""
    p = price.loc[signal_date:].iloc[: horizon_days + 1].dropna()
    if len(p) < 5:
        return None
    p0 = p.iloc[0]
    # worst close relative to entry
    worst_rel = (p / p0 - 1.0).min()
    worst_rel_date = (p / p0 - 1.0).idxmin()
    lag_m = (worst_rel_date - signal_date).days / 30.44
    # true peak-to-trough within the window (running-max drawdown)
    run_max = p.cummax()
    dd = (p / run_max - 1.0)
    max_dd = dd.min()
    dd_trough_date = dd.idxmin()
    dd_lag_m = (dd_trough_date - signal_date).days / 30.44
    fwd_end = p.iloc[-1] / p0 - 1.0
    return dict(worst_rel=worst_rel, worst_rel_lag_m=lag_m,
                max_dd=max_dd, max_dd_lag_m=dd_lag_m,
                fwd_ret_end=fwd_end, n=len(p))


def run(pr_thr, label, gap_days=90):
    flag = panel["pr"] < pr_thr
    eps = episodes_from_flag(flag, gap_days=gap_days)
    print(f"\n{'='*110}\nEXTREME-TIGHT EPISODES at HY-OAS percentile-rank < {pr_thr}  ({label}); gap-merge={gap_days}d ; n_eps={len(eps)}\n{'='*110}")
    rows = []
    for ep in eps:
        start = ep[0]
        end = ep[-1]
        prser = panel.loc[ep, "pr"]
        prmin = prser.min(); prmin_date = prser.idxmin()
        zser = panel.loc[ep, "z"]; zmin = zser.min()
        dur = (end - start).days
        c_pct = panel.loc[start, "cape_pct"]; v_pct = panel.loc[start, "vix_pct"]
        cape_v = panel.loc[start, "cape"]; vix_v = panel.loc[start, "vix"]
        hy_v = panel.loc[start, "hy_oas"]
        row = dict(ep_start=start.date(), ep_end=end.date(), dur_d=dur, n_days=len(ep),
                   hy=round(hy_v,2), pr_min=round(prmin,3), z_min=round(zmin,2),
                   cape=round(cape_v,1), cape_pct=round(c_pct,2) if pd.notna(c_pct) else np.nan,
                   vix=round(vix_v,1), vix_pct=round(v_pct,2) if pd.notna(v_pct) else np.nan)
        for H, hn in [(252,"1Y"),(504,"2Y"),(756,"3Y")]:
            for nm, ser in [("SPX",spx),("NDX",ndx)]:
                t = trace(start, ser, H)
                if t is None:
                    row[f"{nm}_{hn}_dd"] = np.nan; row[f"{nm}_{hn}_ddlag"] = np.nan
                    row[f"{nm}_{hn}_fwd"] = np.nan
                else:
                    row[f"{nm}_{hn}_dd"] = round(t["max_dd"]*100,1)
                    row[f"{nm}_{hn}_ddlag"] = round(t["max_dd_lag_m"],1)
                    row[f"{nm}_{hn}_fwd"] = round(t["fwd_ret_end"]*100,1)
        rows.append(row)
    return pd.DataFrame(rows)


cols_base = ["ep_start","ep_end","dur_d","n_days","hy","pr_min","z_min","cape","cape_pct","vix","vix_pct"]

# Lowest decile = extreme tight (Marks)
df10 = run(0.10, "lowest decile (tight)", gap_days=90)
print("\n--- EPISODE METADATA (pr<0.10) ---")
print(df10[cols_base].to_string(index=False))
print("\n--- pr<0.10: eventual peak-to-trough DRAWDOWNS within window + lag(months) + fwd-return-at-horizon ---")
print(df10[["ep_start","hy","cape_pct","vix_pct",
            "SPX_1Y_dd","SPX_2Y_dd","SPX_2Y_ddlag","SPX_2Y_fwd",
            "NDX_2Y_dd","NDX_2Y_ddlag",
            "SPX_3Y_dd","SPX_3Y_ddlag","SPX_3Y_fwd"]].to_string(index=False))

# Lowest 5% = deep extreme
df05 = run(0.05, "lowest 5% (DEEP tight)", gap_days=90)
print("\n--- EPISODE METADATA (pr<0.05) ---")
print(df05[cols_base].to_string(index=False))
print("\n--- pr<0.05: eventual drawdowns ---")
print(df05[["ep_start","hy","cape_pct","vix_pct","SPX_2Y_dd","SPX_2Y_ddlag","NDX_2Y_dd","SPX_3Y_dd","SPX_3Y_ddlag","SPX_3Y_fwd"]].to_string(index=False))

df10.to_csv(r"C:\Users\admin\Downloads\kader-equity\_marks_ep10.csv", index=False)
df05.to_csv(r"C:\Users\admin\Downloads\kader-equity\_marks_ep05.csv", index=False)
print("\nsaved.")
