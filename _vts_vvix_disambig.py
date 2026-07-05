import sys, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\Users\admin\Downloads\kader-equity")
import numpy as np, pandas as pd
from pathlib import Path
from screen._util import load_price_csv
CACHE = Path(r"C:\Users\admin\Downloads\kader-equity\data\cache")
BT = Path(r"C:\Users\admin\Desktop\backtesting")
vs = pd.read_parquet(CACHE/"vol_surface.parquet"); vs.index=pd.to_datetime(vs.index)
sv = pd.read_parquet(CACHE/"skew_vvix.parquet"); sv.index=pd.to_datetime(sv.index)
spx = load_price_csv(BT/"SPX_daily.csv").rename("spx"); spx.index=pd.to_datetime(spx.index)
ndx = load_price_csv(BT/"NASDAQ_daily.csv").rename("ndx"); ndx.index=pd.to_datetime(ndx.index)
from spine import contract as C, tide as T
scores,prices,vector,_=C.read_frozen()
tide_s=T.tide_score_series(scores,vector); tdir=T.tide_dir_series(tide_s); tdir.index=pd.to_datetime(tdir.index)
def dret(px): return np.log(px).diff()
def stats(pos,px):
    r=dret(px); pos=pos.reindex(r.index).ffill().shift(1).fillna(0); s=(pos*r).dropna()
    sh=s.mean()/s.std()*np.sqrt(252) if s.std()>0 else np.nan
    eq=s.cumsum(); mdd=(eq-eq.cummax()).min(); cv=s[s<=np.quantile(s,0.05)].mean()
    return sh,mdd,cv

start="2019-01-01"
vvix=sv["VVIX"].dropna(); vix=vs["vix"]
def rrank(x,w=504): return x.rolling(w,min_periods=120).apply(lambda s:s.rank(pct=True).iloc[-1],raw=False)
vv_rank=rrank(vvix); vix_rank=rrank(vix)
# realized-vol vol-target benchmark (pure price, no options data): inverse 21d realized vol scaling
def rv_scale(px):
    rv=dret(px).rolling(21).std()*np.sqrt(252)
    target=0.16
    return (target/rv).clip(0.3,1.0)

print("="*72)
print("DISAMBIGUATION (2019+): is the 'trim-on-high-VVIX' gain NOVEL or just VIX/realized-vol targeting?")
print("="*72)
for pnm,px in (("SPX",spx),("NDX",ndx)):
    px2=px[px.index>=start]; base=(tdir>0).astype(float)
    b=stats(base,px2)
    # VVIX-based (low=more exposure): scale=0.5+0.5*(1-vv_rank)
    sc_vv=(0.5+0.5*(1-vv_rank)).reindex(px2.index).ffill().clip(0.5,1.0).fillna(1.0)
    v=stats(base.reindex(px2.index).fillna(0)*sc_vv,px2)
    # VIX-based same construction (low VIX=more) -> is VIX alone enough?
    sc_vix=(0.5+0.5*(1-vix_rank)).reindex(px2.index).ffill().clip(0.5,1.0).fillna(1.0)
    vx=stats(base.reindex(px2.index).fillna(0)*sc_vix,px2)
    # realized-vol target (NO options data at all)
    sc_rv=rv_scale(px2).reindex(px2.index).ffill().clip(0.3,1.0).fillna(1.0)
    rv=stats(base.reindex(px2.index).fillna(0)*sc_rv,px2)
    # VVIX ORTHOGONAL to VIX, used as the trim -> does the options-specific part add over plain VIX-target?
    panel=pd.concat([vvix.rename("vv"),vix.rename("vx")],axis=1,sort=True).dropna()
    bb=np.polyfit(panel["vx"],panel["vv"],1); vv_orth=pd.Series(panel["vv"].values-np.polyval(bb,panel["vx"].values),index=panel.index)
    vvo_rank=rrank(vv_orth)
    sc_vvo=(0.7+0.3*(1-vvo_rank)).reindex(px2.index).ffill().clip(0.7,1.0).fillna(1.0)
    # stack VIX-target THEN add orth-VVIX trim
    combo=stats(base.reindex(px2.index).fillna(0)*sc_vix*sc_vvo,px2)
    print(f"\n[{pnm}] base tide:                Sh={b[0]:.3f} mdd={b[1]*100:5.1f}% cvar={b[2]*100:.2f}%")
    print(f"[{pnm}] +VVIX-rank trim(low=more):  Sh={v[0]:.3f} mdd={v[1]*100:5.1f}% cvar={v[2]*100:.2f}%")
    print(f"[{pnm}] +VIX-rank  trim(low=more):  Sh={vx[0]:.3f} mdd={vx[1]*100:5.1f}% cvar={vx[2]*100:.2f}%   <- plain VIX benchmark")
    print(f"[{pnm}] +realized-vol target(noopt):Sh={rv[0]:.3f} mdd={rv[1]*100:5.1f}% cvar={rv[2]*100:.2f}%   <- price-only benchmark")
    print(f"[{pnm}] VIX-trim + orthVVIX-trim:    Sh={combo[0]:.3f} mdd={combo[1]*100:5.1f}% cvar={combo[2]*100:.2f}%   <- does options-specific part ADD over VIX?")
