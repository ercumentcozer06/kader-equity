import sys, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\Users\admin\Downloads\kader-equity")
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, norm
from screen._util import load_price_csv

CACHE = Path(r"C:\Users\admin\Downloads\kader-equity\data\cache")
BT = Path(r"C:\Users\admin\Desktop\backtesting")
vs = pd.read_parquet(CACHE/"vol_surface.parquet"); vs.index=pd.to_datetime(vs.index)
sv = pd.read_parquet(CACHE/"skew_vvix.parquet"); sv.index=pd.to_datetime(sv.index)
spx = load_price_csv(BT/"SPX_daily.csv").rename("spx"); spx.index=pd.to_datetime(spx.index)
ndx = load_price_csv(BT/"NASDAQ_daily.csv").rename("ndx"); ndx.index=pd.to_datetime(ndx.index)

def fwd(px,H): lp=np.log(px); return lp.shift(-H)-lp

# ---- Block-bootstrap p-value for IC, robust to overlapping-window autocorrelation ----
def block_boot_ic(sig, px, H, nboot=2000, block=63, seed=1):
    fr=fwd(px,H); df=pd.concat([sig.rename("s"),fr.rename("f")],axis=1,sort=True).dropna()
    s=df["s"].values; f=df["f"].values; n=len(s)
    obs,_=spearmanr(s,f)
    rng=np.random.default_rng(seed)
    # circular block bootstrap on the f series (break signal->return link) -> null dist of IC
    cnt=0; nb=n//block
    ranks_s=pd.Series(s).rank().values
    null=np.empty(nboot)
    for b in range(nboot):
        idx=[]
        while len(idx)<n:
            start=rng.integers(0,n)
            idx.extend(list(range(start,start+block)))
        idx=np.array(idx[:n])%n
        fb=f[idx]
        rho,_=spearmanr(s,fb)
        null[b]=rho
    # two-sided p
    p=(np.sum(np.abs(null)>=abs(obs))+1)/(nboot+1)
    return obs,p,n

print("="*72); print("BLOCK-BOOTSTRAP IC p-values (overlap-autocorr robust), H=63d"); print("="*72)
tests=[("ts_ratio",vs["ts_ratio"]),("VVIX",sv["VVIX"].dropna())]
pvals=[]
for snm,sig in tests:
    for pnm,px in (("SPX",spx),("NDX",ndx)):
        obs,p,n=block_boot_ic(sig,px,63)
        pvals.append((f"{snm}-{pnm}",obs,p))
        print(f"{snm:9s} {pnm}: IC={obs:+.4f} block-boot p={p:.4f} n={n}")

# BH-FDR across the 4 raw tests
import numpy as np
ps=sorted([(name,o,p) for name,o,p in pvals], key=lambda x:x[2])
m=len(ps); print("\nBH-FDR @ q=0.10 across 4 tests:")
for i,(name,o,p) in enumerate(ps,1):
    thr=0.10*i/m
    print(f"  {name}: p={p:.4f} thr={thr:.4f} {'PASS' if p<=thr else 'fail'}")

# ================= INCREMENTAL OVER TIDE (2019+) =================
print("\n"+"="*72); print("INCREMENTAL OVER TIDE (2019+): does signal improve a tide-timed sleeve?"); print("="*72)
try:
    from spine import contract as C, tide as T
    scores,prices,vector,_=C.read_frozen()
    ts_tide=T.tide_score_series(scores,vector); tdir=T.tide_dir_series(ts_tide)
    tdir.index=pd.to_datetime(tdir.index)
    print("tide_dir range",tdir.index.min(),tdir.index.max(),"vals",sorted(pd.unique(tdir.dropna())))
except Exception as e:
    print("TIDE load failed:",e); tdir=None

def daily_ret(px): return np.log(px).diff()

def sleeve_stats(pos, px, label):
    # pos: position series (0/1/scaled) aligned to dates; apply next-day return
    r=daily_ret(px)
    pos=pos.reindex(r.index).ffill().shift(1).fillna(0)
    pnl=pos*r
    s=pnl.dropna()
    sh=s.mean()/s.std()*np.sqrt(252) if s.std()>0 else np.nan
    # maxDD
    eq=s.cumsum(); dd=(eq-eq.cummax()); mdd=dd.min()
    cv=np.quantile(s,0.05); tail=s[s<=cv].mean()
    expo=(pos!=0).mean()
    return sh,mdd,tail,expo,s

if tdir is not None:
    start="2019-01-01"
    for pnm,px in (("SPX",spx),("NDX",ndx)):
        px2=px[px.index>=start]
        base_pos=(tdir>0).astype(float)  # tide-long sleeve
        sh,mdd,tail,expo,base_s=sleeve_stats(base_pos,px2,"tide")
        print(f"\n[{pnm}] TIDE-long base: Sharpe={sh:.3f} maxDD={mdd*100:.1f}% cvar5={tail*100:.2f}% expo={expo:.2f}")
        # OVERLAY A: VVIX risk-ceiling -> when VVIX in bottom decile (complacency) cut exposure to 0
        vvix=sv["VVIX"].dropna()
        vv_lo=vvix.rolling(252,min_periods=120).quantile(0.10)  # expanding-ish, use rolling to avoid look-ahead
        complacent=(vvix<=vv_lo).reindex(px2.index).ffill().fillna(False)
        pos_ceil=base_pos.reindex(px2.index).fillna(0).copy()
        pos_ceil[complacent.values]=0.0
        sh2,mdd2,tail2,expo2,_=sleeve_stats(pos_ceil,px2,"tide+vvixceil")
        print(f"[{pnm}] +VVIX low-decile RISK-CEILING (cut when complacent): Sharpe={sh2:.3f} maxDD={mdd2*100:.1f}% cvar5={tail2*100:.2f}% expo={expo2:.2f}")
        # OVERLAY B: ts_ratio backwardation add (when ts_ratio top decile -> stay long even if tide off? test as risk-on add)
        tsr=vs["ts_ratio"]
        ts_hi=tsr.rolling(252,min_periods=120).quantile(0.90)
        backw=(tsr>=ts_hi).reindex(px2.index).ffill().fillna(False)
        pos_add=base_pos.reindex(px2.index).fillna(0).copy()
        pos_add[backw.values]=1.0  # force-long on backwardation (rebound bet)
        sh3,mdd3,tail3,expo3,_=sleeve_stats(pos_add,px2,"tide+backw-add")
        print(f"[{pnm}] +ts_ratio backwardation FORCE-LONG add: Sharpe={sh3:.3f} maxDD={mdd3*100:.1f}% cvar5={tail3*100:.2f}% expo={expo3:.2f}")
