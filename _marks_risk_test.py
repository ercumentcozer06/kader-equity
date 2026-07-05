import sys, pandas as pd, numpy as np
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\Users\admin\Downloads\kader-equity")

panel = pd.read_parquet(r"C:\Users\admin\Downloads\kader-equity\data\cache\credit_froth_panel.parquet").sort_index()
hy_full = pd.read_parquet(r"C:\Users\admin\Downloads\kader-equity\data\cache\hy_oas_full.parquet")["hy_oas"].sort_index()

# expanding percentile rank of HY level
arr = hy_full.values
pr = np.full(len(arr), np.nan)
for i in range(len(arr)):
    if i >= 252:
        pr[i] = np.sum(arr[:i+1] <= arr[i]) / (i+1)
pr_full = pd.Series(pr, index=hy_full.index)
panel["pr"] = pr_full.reindex(panel.index).ffill()

spx = panel["spx_close"]; ndx = panel["ndx_close"]
cape = panel["cape"]; vix = panel["vix"]
cape_pct = cape.expanding(252).apply(lambda s:(s<=s.iloc[-1]).mean(), raw=False)
vix_pct  = vix.expanding(252).apply(lambda s:(s<=s.iloc[-1]).mean(), raw=False)
panel["cape_pct"]=cape_pct; panel["vix_pct"]=vix_pct

def fwd_maxdd(price, H):
    """For each day, forward peak-to-trough max drawdown over next H trading days (running-max based)."""
    out = pd.Series(index=price.index, dtype=float)
    p = price.dropna()
    vals = p.values; idx = p.index
    for i in range(len(p)):
        w = vals[i:i+H+1]
        if len(w) < 20:
            out.loc[idx[i]] = np.nan; continue
        rm = np.maximum.accumulate(w)
        dd = (w/rm - 1.0).min()
        out.loc[idx[i]] = dd
    return out.reindex(price.index)

def fwd_ret(price, H):
    p = price.dropna(); vals=p.values; idx=p.index
    out=pd.Series(index=price.index, dtype=float)
    for i in range(len(p)):
        if i+H < len(p):
            out.loc[idx[i]] = vals[i+H]/vals[i]-1.0
    return out.reindex(price.index)

for H, hn in [(504,"2Y")]:
    dd_spx = fwd_maxdd(spx, H); ret_spx = fwd_ret(spx, H)
    dd_ndx = fwd_maxdd(ndx, H); ret_ndx = fwd_ret(ndx, H)
    panel[f"spx_dd_{hn}"]=dd_spx; panel[f"spx_ret_{hn}"]=ret_spx
    panel[f"ndx_dd_{hn}"]=dd_ndx; panel[f"ndx_ret_{hn}"]=ret_ndx

print("="*100)
print("BASE-RATE vs CONDITIONAL forward 2Y peak-to-trough max-drawdown (per-day, SPX)")
print("="*100)
valid = panel.dropna(subset=["spx_dd_2Y","pr"])
def summarize(mask, name):
    s = valid.loc[mask, "spx_dd_2Y"]*100
    r = valid.loc[mask, "spx_ret_2Y"]*100
    print(f"{name:38s} n={len(s):5d}  meanDD={s.mean():7.1f}%  medDD={s.median():7.1f}%  p10DD={s.quantile(0.10):7.1f}%  worstDD={s.min():7.1f}%  | mean2Yret={r.mean():6.1f}%  med={r.median():6.1f}%")
summarize(valid.index==valid.index, "ALL DAYS (base rate)")
summarize(valid["pr"]<0.50, "spread BELOW median (mild tight)")
summarize(valid["pr"]<0.10, "spread lowest DECILE (extreme tight)")
summarize(valid["pr"]<0.05, "spread lowest 5% (DEEP extreme)")
summarize(valid["pr"]>0.90, "spread highest decile (WIDE/panic)")
# confluence: tight AND frothy CAPE AND low VIX
conf = (valid["pr"]<0.10) & (valid["cape_pct"]>0.80) & (valid["vix_pct"]<0.40)
summarize(conf, "CONFLUENCE: tight+CAPE>80pct+VIX<40pct")
conf2 = (valid["pr"]<0.10) & (valid["cape_pct"]>0.80)
summarize(conf2, "tight + CAPE>80pct (froth, ignore VIX)")

print("\nSame for NDX:")
v2 = panel.dropna(subset=["ndx_dd_2Y","pr"])
def summ_ndx(mask,name):
    s=v2.loc[mask,"ndx_dd_2Y"]*100; r=v2.loc[mask,"ndx_ret_2Y"]*100
    print(f"{name:38s} n={len(s):5d}  meanDD={s.mean():7.1f}%  medDD={s.median():7.1f}%  worstDD={s.min():7.1f}%  | mean2Yret={r.mean():6.1f}%")
summ_ndx(v2.index==v2.index,"ALL DAYS (base rate)")
summ_ndx(v2["pr"]<0.10,"lowest decile (extreme tight)")
summ_ndx((v2["pr"]<0.10)&(v2["cape_pct"]>0.80),"CONFLUENCE tight+CAPE>80")

# FAVORABLE-ASYMMETRY: de-risk-at-extreme overlay. Baseline = always-in SPX. Overlay = go 50% (or 0%) when
# CONFLUENCE-extreme, else 100%. Compare Sharpe, maxDD, returns on the daily series 2000-2026.
print("\n"+"="*100)
print("FAVORABLE-ASYMMETRY: does DE-RISKING at confluence-extreme improve risk-adjusted profile?")
print("="*100)
ret = spx.pct_change().fillna(0.0)
# signal: in confluence-extreme today -> reduce exposure (Marks: de-risk at euphoria). Lag 1 day (act next day).
sig_conf = ((panel["pr"]<0.10) & (panel["cape_pct"]>0.80)).reindex(ret.index).fillna(False)
sig_tight = (panel["pr"]<0.10).reindex(ret.index).fillna(False)
def perf(w, name):
    pr_ = (w.shift(1).fillna(1.0) * ret)
    eq = (1+pr_).cumprod()
    ann = pr_.mean()*252; vol = pr_.std()*np.sqrt(252); sh = ann/vol if vol>0 else 0
    rm = eq.cummax(); dd = (eq/rm-1).min()
    print(f"{name:42s} CAGR~{ (eq.iloc[-1]**(252/len(pr_))-1)*100:6.1f}%  Sharpe={sh:5.2f}  maxDD={dd*100:7.1f}%  totRet={ (eq.iloc[-1]-1)*100:8.1f}%")
    return sh, dd
perf(pd.Series(1.0,index=ret.index), "BUY&HOLD SPX (always in)")
for frac in [0.5, 0.0]:
    w = pd.Series(1.0,index=ret.index); w[sig_conf]=frac
    perf(w, f"de-risk to {int(frac*100)}% on CONFLUENCE-extreme")
for frac in [0.5, 0.0]:
    w = pd.Series(1.0,index=ret.index); w[sig_tight]=frac
    perf(w, f"de-risk to {int(frac*100)}% on tight-alone (decile)")
print("\n(NDX same overlay)")
retn = ndx.pct_change().fillna(0.0)
sig_conf_n = sig_conf.reindex(retn.index).fillna(False)
def perfn(w,name):
    pr_=(w.shift(1).fillna(1.0)*retn); eq=(1+pr_).cumprod()
    ann=pr_.mean()*252; vol=pr_.std()*np.sqrt(252); sh=ann/vol if vol>0 else 0
    rm=eq.cummax(); dd=(eq/rm-1).min()
    print(f"{name:42s} Sharpe={sh:5.2f}  maxDD={dd*100:7.1f}%  totRet={(eq.iloc[-1]-1)*100:8.1f}%")
perfn(pd.Series(1.0,index=retn.index),"BUY&HOLD NDX")
w=pd.Series(1.0,index=retn.index); w[sig_conf_n]=0.5; perfn(w,"de-risk to 50% on CONFLUENCE-extreme")
w=pd.Series(1.0,index=retn.index); w[sig_conf_n]=0.0; perfn(w,"de-risk to 0% on CONFLUENCE-extreme")
