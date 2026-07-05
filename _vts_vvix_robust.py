import sys, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\Users\admin\Downloads\kader-equity")
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from screen._util import load_price_csv

CACHE = Path(r"C:\Users\admin\Downloads\kader-equity\data\cache")
BT = Path(r"C:\Users\admin\Desktop\backtesting")
vs = pd.read_parquet(CACHE/"vol_surface.parquet")
sv = pd.read_parquet(CACHE/"skew_vvix.parquet")
spx = load_price_csv(BT/"SPX_daily.csv").rename("spx"); spx.index=pd.to_datetime(spx.index)
ndx = load_price_csv(BT/"NASDAQ_daily.csv").rename("ndx"); ndx.index=pd.to_datetime(ndx.index)
vs.index=pd.to_datetime(vs.index); sv.index=pd.to_datetime(sv.index)

def fwd(px,H):
    lp=np.log(px); return (lp.shift(-H)-lp)

# ===== ERA SPLIT: rank-IC of each signal per era, both indices =====
eras = [("2008-2012","2008-01-01","2012-12-31"),
        ("2013-2017","2013-01-01","2017-12-31"),
        ("2018-2021","2018-01-01","2021-12-31"),
        ("2022-2026","2022-01-01","2026-12-31")]
def era_ic(sig, px, H):
    fr=fwd(px,H); df=pd.concat([sig.rename("s"),fr.rename("f")],axis=1,sort=True).dropna()
    res=[]
    for nm,a,b in eras:
        sub=df.loc[a:b]
        if len(sub)<150: res.append((nm,np.nan,np.nan,len(sub))); continue
        rho,p=spearmanr(sub["s"],sub["f"]); res.append((nm,rho,p,len(sub)))
    return res

print("="*72); print("ERA-SPLIT Spearman IC (multi-regime robustness), H=63d"); print("="*72)
sigs={"ts_ratio":vs["ts_ratio"],"VVIX":sv["VVIX"].dropna()}
for snm,sig in sigs.items():
    for pnm,px in (("SPX",spx),("NDX",ndx)):
        rows=era_ic(sig,px,63)
        line=" ".join(f"{nm}:{rho:+.3f}(p{p:.2f},n{n})" for nm,rho,p,n in rows)
        print(f"{snm:9s} {pnm}: {line}")

# ===== CONTROL FOR VIX LEVEL: is the edge term-structure-specific or just 'VIX high'? =====
# partial: regress fwd return on VIX level, take residual; then IC of signal vs residual
print("\n"+"="*72); print("PARTIAL OUT VIX LEVEL  (IC of signal vs VIX-residualized fwd ret), H=63d"); print("="*72)
from numpy.polynomial import polynomial as P
def partial_ic(sig, px, H, ctrl):
    fr=fwd(px,H)
    df=pd.concat([sig.rename("s"),fr.rename("f"),ctrl.rename("c")],axis=1,sort=True).dropna()
    if len(df)<300: return np.nan,np.nan,len(df)
    # residualize f on c (linear)
    c=df["c"].values; f=df["f"].values
    b=np.polyfit(c,f,1); resid=f-np.polyval(b,c)
    rho,p=spearmanr(df["s"].values, resid)
    return rho,p,len(df)
for snm,sig in sigs.items():
    for pnm,px in (("SPX",spx),("NDX",ndx)):
        rho,p,n=partial_ic(sig,px,63,vs["vix"])
        rho0,p0=spearmanr(*[x for x in [None]]) if False else (None,None)
        # raw for comparison
        fr=fwd(px,63); d=pd.concat([sig.rename("s"),fr.rename("f")],axis=1,sort=True).dropna()
        rraw,praw=spearmanr(d["s"],d["f"])
        print(f"{snm:9s} {pnm}: raw IC={rraw:+.4f}  | VIX-partialled IC={rho:+.4f} (p={p:.4f}, n={n})")

# Also: signal vs its OWN VIX-orthogonal component (residualize signal on vix), then IC
print("\n"+"="*72); print("SIGNAL ORTHOGONALIZED to VIX (residual signal) -> fwd63 IC"); print("="*72)
def orth_sig_ic(sig,px,H,ctrl):
    df=pd.concat([sig.rename("s"),ctrl.rename("c")],axis=1,sort=True).dropna()
    b=np.polyfit(df["c"],df["s"],1); sres=df["s"]-np.polyval(b,df["c"])
    sres=pd.Series(sres.values,index=df.index)
    fr=fwd(px,H); d=pd.concat([sres.rename("s"),fr.rename("f")],axis=1,sort=True).dropna()
    rho,p=spearmanr(d["s"],d["f"]); return rho,p,len(d)
for snm,sig in sigs.items():
    for pnm,px in (("SPX",spx),("NDX",ndx)):
        rho,p,n=orth_sig_ic(sig,px,63,vs["vix"])
        print(f"{snm:9s}_orthVIX {pnm}: IC={rho:+.4f} (p={p:.4f}, n={n})")
