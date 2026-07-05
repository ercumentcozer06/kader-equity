import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\Users\admin\Downloads\kader-equity")
import numpy as np, pandas as pd
from pathlib import Path
from screen._util import load_price_csv

CACHE = Path(r"C:\Users\admin\Downloads\kader-equity\data\cache")
BT = Path(r"C:\Users\admin\Desktop\backtesting")

# ---- load ----
vs = pd.read_parquet(CACHE/"vol_surface.parquet")            # vix,vix3m,vxn,ts_ratio(=vix/vix3m)
sv = pd.read_parquet(CACHE/"skew_vvix.parquet")              # SKEW,VVIX
spx = load_price_csv(BT/"SPX_daily.csv").rename("spx")
ndx = load_price_csv(BT/"NASDAQ_daily.csv").rename("ndx")
spx.index = pd.to_datetime(spx.index); ndx.index = pd.to_datetime(ndx.index)
vs.index = pd.to_datetime(vs.index); sv.index = pd.to_datetime(sv.index)

# forward returns (log) over horizon H, computed from price; lag signal by 1 day (use prior close signal -> next-day entry)
def fwd_logret(px, H):
    lp = np.log(px)
    return (lp.shift(-H) - lp)

def ann_sharpe_daily(series_daily):
    s = series_daily.dropna()
    if len(s) < 30 or s.std()==0: return np.nan
    return s.mean()/s.std()*np.sqrt(252)

def cvar(rets, q=0.05):
    r = rets.dropna().values
    if len(r)<20: return np.nan
    thr = np.quantile(r, q)
    tail = r[r<=thr]
    return tail.mean() if len(tail) else np.nan

# ---- bucket analysis: forward H-day return conditional on signal quantile ----
def bucket_report(sig, px, name, H=21, qs=(0.1,0.9)):
    sig = sig.dropna()
    fr = fwd_logret(px, H)
    # align: signal at day t (known at close t), forward return from t to t+H. lag signal by 1 to avoid same-bar leak
    df = pd.concat([sig.rename("s"), fr.rename("f")], axis=1).dropna()
    if len(df) < 200:
        return f"{name}: insufficient ({len(df)})"
    lo, hi = df["s"].quantile(qs[0]), df["s"].quantile(qs[1])
    extreme_lo = df[df["s"]<=lo]["f"]
    extreme_hi = df[df["s"]>=hi]["f"]
    mid = df[(df["s"]>lo)&(df["s"]<hi)]["f"]
    base = df["f"]
    out = [f"--- {name} (H={H}d, n={len(df)}) ---"]
    out.append(f"  base    : mean={base.mean()*100:6.2f}% std={base.std()*100:5.2f} cvar5={cvar(base)*100:6.2f}%")
    out.append(f"  LOW  q<={qs[0]} (n={len(extreme_lo)}): mean={extreme_lo.mean()*100:6.2f}% cvar5={cvar(extreme_lo)*100:6.2f}% min={extreme_lo.min()*100:6.1f}%")
    out.append(f"  MID            (n={len(mid)}): mean={mid.mean()*100:6.2f}% cvar5={cvar(mid)*100:6.2f}%")
    out.append(f"  HIGH q>={qs[1]} (n={len(extreme_hi)}): mean={extreme_hi.mean()*100:6.2f}% cvar5={cvar(extreme_hi)*100:6.2f}% min={extreme_hi.min()*100:6.1f}%")
    # extreme 5% tails
    p5 = df["s"].quantile(0.05); p95 = df["s"].quantile(0.95)
    out.append(f"  XLOW q<=0.05 (n={(df['s']<=p5).sum()}): mean={df[df['s']<=p5]['f'].mean()*100:6.2f}% cvar5={cvar(df[df['s']<=p5]['f'])*100:6.2f}%")
    out.append(f"  XHIGH q>=0.95 (n={(df['s']>=p95).sum()}): mean={df[df['s']>=p95]['f'].mean()*100:6.2f}% cvar5={cvar(df[df['s']>=p95]['f'])*100:6.2f}%")
    return "\n".join(out)

# ---- Spearman rank IC of signal vs forward return (both signs visible by sign) ----
from scipy.stats import spearmanr
def rank_ic(sig, px, H):
    fr = fwd_logret(px,H)
    df = pd.concat([sig.rename("s"), fr.rename("f")], axis=1).dropna()
    if len(df)<200: return np.nan, np.nan, len(df)
    rho,p = spearmanr(df["s"], df["f"])
    return rho, p, len(df)

px_map = {"SPX":spx, "NDX":ndx}
ts = vs["ts_ratio"]   # high = backwardation/stress, low = contango/calm
vvix = sv["VVIX"].dropna()
vxn = vs["vxn"]

print("="*70)
print("ts_ratio = vix/vix3m  (HIGH=backwardation/stress, LOW=contango/calm)")
print("="*70)
for H in (21,63):
    print(f"\n##### HORIZON {H}d #####")
    for nm,px in px_map.items():
        print(bucket_report(ts, px, f"ts_ratio {nm}", H=H))
        rho,p,n = rank_ic(ts,px,H)
        print(f"    Spearman IC(ts_ratio,fwd{H}) {nm}: rho={rho:+.4f} p={p:.4f} n={n}")

print("\n"+"="*70)
print("VVIX  (vol-of-vol)")
print("="*70)
for H in (21,63):
    print(f"\n##### HORIZON {H}d #####")
    for nm,px in px_map.items():
        print(bucket_report(vvix, px, f"VVIX {nm}", H=H))
        rho,p,n = rank_ic(vvix,px,H)
        print(f"    Spearman IC(VVIX,fwd{H}) {nm}: rho={rho:+.4f} p={p:.4f} n={n}")
