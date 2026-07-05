import sys, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = np.random.default_rng(7)

panel = pd.read_parquet(r"C:\Users\admin\Downloads\kader-equity\data\cache\credit_froth_panel.parquet").sort_index()
hy_full = pd.read_parquet(r"C:\Users\admin\Downloads\kader-equity\data\cache\hy_oas_full.parquet")["hy_oas"].sort_index()
arr=hy_full.values; pr=np.full(len(arr),np.nan)
for i in range(len(arr)):
    if i>=252: pr[i]=np.sum(arr[:i+1]<=arr[i])/(i+1)
panel["pr"]=pd.Series(pr,index=hy_full.index).reindex(panel.index).ffill()
spx=panel["spx_close"]

# forward 2Y return per day
H=504; p=spx.dropna(); vals=p.values; idx=p.index
fr=pd.Series(index=spx.index,dtype=float)
for i in range(len(p)):
    if i+H<len(p): fr.loc[idx[i]]=vals[i+H]/vals[i]-1
panel["fr2y"]=fr
v=panel.dropna(subset=["fr2y","pr"])
tight=v["pr"]<0.10
obs = v.loc[tight,"fr2y"].mean()-v.loc[~tight,"fr2y"].mean()
print(f"Observed mean-2Y-fwd-return gap (tight-decile minus rest): {obs*100:.1f} pct points")

# CIRCULAR BLOCK BOOTSTRAP to respect ~1yr autocorrelation. Block=252 trading days.
def block_perm(series_vals, block, n):
    L=len(series_vals); out=[]
    nblocks=int(np.ceil(L/block))
    for _ in range(n):
        starts=rng.integers(0,L,size=nblocks)
        s=np.concatenate([np.take(series_vals,range(st,st+block),mode='wrap') for st in starts])[:L]
        out.append(s)
    return out

fr_vals=v["fr2y"].values; tight_vals=tight.values.astype(bool)
ntight=tight_vals.sum()
# Null: shuffle the tight-label in blocks, recompute gap
labels=tight_vals.astype(float)
perms=block_perm(labels,252,2000)
gaps=[]
for pl in perms:
    pl=pl>0.5
    if pl.sum()==0 or (~pl).sum()==0: continue
    gaps.append(fr_vals[pl].mean()-fr_vals[~pl].mean())
gaps=np.array(gaps)
pval=(np.sum(gaps<=obs)+1)/(len(gaps)+1)
print(f"Block-bootstrap (252d blocks, n={len(gaps)}): P(null gap <= observed) = {pval:.3f}")
print(f"  null gap mean={gaps.mean()*100:.1f}pp  5th pct={np.percentile(gaps,5)*100:.1f}pp  95th={np.percentile(gaps,95)*100:.1f}pp")

# Censoring check: how much forward data do recent episodes actually have?
print("\nCENSORING of recent episodes (need 504 trading days = ~2yr fwd for a clean read):")
last = spx.dropna().index.max()
for d in [pd.Timestamp("2021-06-16"),pd.Timestamp("2024-03-18"),pd.Timestamp("2024-10-18"),
          pd.Timestamp("2025-06-24"),pd.Timestamp("2025-07-03"),pd.Timestamp("2026-05-01")]:
    fwd = spx.loc[d:].dropna()
    print(f"  signal {d.date()}: {len(fwd)-1} fwd trading days available (2Y needs ~504) -> {'CENSORED' if len(fwd)-1<504 else 'complete'}")
print(f"  panel ends {last.date()}")
