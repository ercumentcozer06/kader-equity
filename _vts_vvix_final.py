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
vs = pd.read_parquet(CACHE/"vol_surface.parquet"); vs.index=pd.to_datetime(vs.index)
sv = pd.read_parquet(CACHE/"skew_vvix.parquet"); sv.index=pd.to_datetime(sv.index)
spx = load_price_csv(BT/"SPX_daily.csv").rename("spx"); spx.index=pd.to_datetime(spx.index)
ndx = load_price_csv(BT/"NASDAQ_daily.csv").rename("ndx"); ndx.index=pd.to_datetime(ndx.index)
def fwd(px,H): lp=np.log(px); return lp.shift(-H)-lp

# 1) Is the low-VVIX -> bad 63d tail an artifact of a FEW dated clusters? show WHEN low-VVIX-decile occurred and the events
vvix=sv["VVIX"].dropna()
vv_dec=vvix.rolling(252,min_periods=120).quantile(0.10)
lowmask=(vvix<=vv_dec)
print("Low-VVIX-decile (rolling) day count by year:")
print(lowmask[lowmask].groupby(lowmask[lowmask].index.year).count().to_dict())

# 2) VVIX redundancy with COR1M and GEX (the already-deployed signals). corr of levels + of fwd-IC overlap.
corr=pd.read_parquet(CACHE/"corr_pc.parquet"); corr.index=pd.to_datetime(corr.index)
sqz=pd.read_parquet(CACHE/"squeeze_dix_gex.parquet"); sqz.index=pd.to_datetime(sqz.index)
panel=pd.concat([vvix.rename("vvix"),vs["ts_ratio"].rename("tsr"),vs["vix"].rename("vix"),
                 corr["COR1M"].rename("cor1m"),sqz["gex"].rename("gex")],axis=1,sort=True).dropna()
print("\nLevel correlations among signals (overlap window n=%d):"%len(panel))
print(panel.corr().round(3).to_string())

# 3) HONEST sizing test: VVIX as a CONTINUOUS exposure TRIM (not binary cut), 2019+, vs tide
from spine import contract as C, tide as T
scores,prices,vector,_=C.read_frozen()
tide_s=T.tide_score_series(scores,vector); tdir=T.tide_dir_series(tide_s); tdir.index=pd.to_datetime(tdir.index)
def dret(px): return np.log(px).diff()
def stats(pos,px):
    r=dret(px); pos=pos.reindex(r.index).ffill().shift(1).fillna(0); s=(pos*r).dropna()
    sh=s.mean()/s.std()*np.sqrt(252) if s.std()>0 else np.nan
    eq=s.cumsum(); mdd=(eq-eq.cummax()).min(); cv=s[s<=np.quantile(s,0.05)].mean()
    return sh,mdd,cv,(pos!=0).mean()

print("\n"+"="*72); print("VVIX CONTINUOUS-TRIM sizing over TIDE (2019+): exposure=base*f(vvix_pctile)"); print("="*72)
start="2019-01-01"
# vvix percentile rank (expanding) -> high vvix = more exposure (rebound), low = trim
vv_rank=vvix.rolling(504,min_periods=120).apply(lambda w: (w.rank(pct=True)).iloc[-1], raw=False)
for pnm,px in (("SPX",spx),("NDX",ndx)):
    px2=px[px.index>=start]; base=(tdir>0).astype(float)
    sh,mdd,cv,ex=stats(base,px2); print(f"\n[{pnm}] base tide: Sh={sh:.3f} mdd={mdd*100:.1f}% cvar={cv*100:.2f}% expo={ex:.2f}")
    for floorv,lbl in [(0.5,"trim-to-50%"),(0.0,"trim-to-0%")]:
        scale=(floorv+(1-floorv)*vv_rank).reindex(px2.index).ffill().clip(floorv,1.0).fillna(1.0)
        pos=(base.reindex(px2.index).fillna(0)*scale)
        sh2,mdd2,cv2,ex2=stats(pos,px2)
        print(f"[{pnm}] tide*VVIX-rank {lbl}: Sh={sh2:.3f} mdd={mdd2*100:.1f}% cvar={cv2*100:.2f}% expo={ex2:.2f}")
    # inverse orientation (low vvix = MORE exposure) sanity both-ways
    scale_inv=(0.5+0.5*(1-vv_rank)).reindex(px2.index).ffill().clip(0.5,1.0).fillna(1.0)
    pos=(base.reindex(px2.index).fillna(0)*scale_inv); sh3,mdd3,cv3,ex3=stats(pos,px2)
    print(f"[{pnm}] tide*INV-VVIX (low=more): Sh={sh3:.3f} mdd={mdd3*100:.1f}% cvar={cv3*100:.2f}% expo={ex3:.2f}")
