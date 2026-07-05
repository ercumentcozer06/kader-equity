import sys, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

panel = pd.read_parquet(r"C:\Users\admin\Downloads\kader-equity\data\cache\credit_froth_panel.parquet").sort_index()
hy_full = pd.read_parquet(r"C:\Users\admin\Downloads\kader-equity\data\cache\hy_oas_full.parquet")["hy_oas"].sort_index()
arr = hy_full.values
pr = np.full(len(arr), np.nan)
for i in range(len(arr)):
    if i >= 252: pr[i] = np.sum(arr[:i+1] <= arr[i]) / (i+1)
panel["pr"] = pd.Series(pr, index=hy_full.index).reindex(panel.index).ffill()
cape=panel["cape"]; vix=panel["vix"]
panel["cape_pct"]=cape.expanding(252).apply(lambda s:(s<=s.iloc[-1]).mean(),raw=False)
spx=panel["spx_close"]; ndx=panel["ndx_close"]

# When is the confluence/tight signal ON, by year? (shows it's really only 2 distinct macro events)
sig_conf = (panel["pr"]<0.10)&(panel["cape_pct"]>0.80)
sig_tight = (panel["pr"]<0.10)
print("Days ON by year -- tight-decile vs confluence:")
yr = pd.DataFrame({"tight":sig_tight.astype(int),"conf":sig_conf.astype(int)}); yr["y"]=yr.index.year
print(yr.groupby("y").sum().query("tight>0").to_string())

# DE-RISK WITH COOLDOWN: once confluence fires, stay de-risked for the next COOLDOWN trading days
# (captures the LAGGED reckoning Marks describes -- the bust comes after the signal turns off).
ret_spx = spx.pct_change().fillna(0.0); ret_ndx = ndx.pct_change().fillna(0.0)
def with_cooldown(sig, cd):
    on = sig.values.copy(); out = np.zeros(len(on), dtype=bool); timer=0
    for i in range(len(on)):
        if on[i]: timer=cd
        if timer>0: out[i]=True; timer-=1
    return pd.Series(out, index=sig.index)

def perf(w, ret, name):
    p=(w.shift(1).fillna(1.0)*ret); eq=(1+p).cumprod()
    ann=p.mean()*252; vol=p.std()*np.sqrt(252); sh=ann/vol if vol>0 else 0
    dd=(eq/eq.cummax()-1).min()
    print(f"{name:48s} Sharpe={sh:5.2f}  maxDD={dd*100:7.1f}%  totRet={(eq.iloc[-1]-1)*100:8.1f}%")

print("\nSPX -- de-risk to 0% on confluence + COOLDOWN (lagged reckoning capture):")
perf(pd.Series(1.0,index=ret_spx.index), ret_spx, "BUY&HOLD")
for cd in [126, 252, 378, 504]:
    s = with_cooldown(sig_conf, cd).reindex(ret_spx.index).fillna(False)
    w=pd.Series(1.0,index=ret_spx.index); w[s]=0.0
    perf(w, ret_spx, f"de-risk 0% conf + {cd}d cooldown (~{cd//21}mo)")
print("\nSPX -- de-risk to 0% on TIGHT-decile + cooldown:")
for cd in [252, 504]:
    s = with_cooldown(sig_tight, cd).reindex(ret_spx.index).fillna(False)
    w=pd.Series(1.0,index=ret_spx.index); w[s]=0.0
    perf(w, ret_spx, f"de-risk 0% tight + {cd}d cooldown")

print("\nNDX -- de-risk to 0% on confluence + cooldown:")
perf(pd.Series(1.0,index=ret_ndx.index), ret_ndx, "BUY&HOLD")
for cd in [252, 504]:
    s = with_cooldown(sig_conf, cd).reindex(ret_ndx.index).fillna(False)
    w=pd.Series(1.0,index=ret_ndx.index); w[s]=0.0
    perf(w, ret_ndx, f"de-risk 0% conf + {cd}d cooldown")

# How many INDEPENDENT episodes really drive the per-day DD result? Block-count.
print("\nEffective independent events: distinct confluence episodes (gap>90d):")
days = sig_conf[sig_conf].index
eps=[]; cur=[days[0]] if len(days) else []
for d in days[1:]:
    if (d-cur[-1]).days>90: eps.append((cur[0],cur[-1])); cur=[d]
    else: cur.append(d)
if cur: eps.append((cur[0],cur[-1]))
for a,b in eps: print(f"  {a.date()} -> {b.date()}")
print(f"  => {len(eps)} independent confluence episodes total (this is the REAL sample size).")
