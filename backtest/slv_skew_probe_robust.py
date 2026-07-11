"""
backtest/slv_skew_probe_robust — T1-FEAR/21g GECTI'sinin saglamlik kontrolu (suspect-own-test).

Iki bilinen artefakt-kaynagi test edilir:
  R1 ORTUSME: 133 gun / 48 epizot — 21g forward'lar ortusuyor → yalniz EPIZOT-ILK-GUNU (bagimsiz orneklem,
     n=48) ile yeniden: ortalama + IID-bootstrap CI + isabet orani.
  R2 KOMPOZISYON: FEAR gunleri yillara nasil dagiliyor? 2020-21 bogasina yigilmissa etki 'o donem long
     olmak'tan ibaret olabilir → yil-bazli katki + FEAR-epizotlarin yil-ici kosulsuz 21g ortalamasina
     gore FAZLASI (within-year excess).
  & <kader-macro venv python> backtest/slv_skew_probe_robust.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MINP = 252

df = pd.read_parquet(ROOT / "data" / "cache" / "rr_skew_slv_2016_2023.parquet")
df = df[df["t30_ok"]].copy()
rr = df["t30_rr_skew"].astype(float)
spot = df["spot"].astype(float)
vals = rr.values
pct = pd.Series(np.nan, index=rr.index)
for i in range(MINP, len(vals)):
    pct.iloc[i] = (vals[:i] < vals[i]).mean() * 100
fwd21 = spot.shift(-21) / spot - 1

m = (pct >= 95).fillna(False)
first = m & ~m.shift(1, fill_value=False)          # epizot ilk gunu
x = fwd21[first].dropna()
uncond = float(np.nanmean(fwd21.values))
rng = np.random.default_rng(9)
boots = [np.mean(rng.choice(x.values, len(x))) for _ in range(4000)]
lo, hi = np.percentile(boots, [0.42, 99.58])       # Bonferroni-esdeger %99.17 CI
print(f"R1 EPIZOT-ILK-GUN (bagimsiz, n={len(x)}): ort {1e4*x.mean():+.1f}bp vs kosulsuz {1e4*uncond:+.1f}bp")
print(f"   CI [{1e4*(lo-uncond):+.1f}, {1e4*(hi-uncond):+.1f}]bp (kosulsuza-fark) → "
      f"{'GECTI' if (lo-uncond) > 0 or (hi-uncond) < 0 else 'sifir-kapsar'}; "
      f"isabet(>kosulsuz) %{100*(x > uncond).mean():.0f}, medyan {1e4*x.median():+.1f}bp")

print("\nR2 KOMPOZISYON (FEAR-gunlerinin yillari + within-year excess):")
yr = pd.DatetimeIndex(df.index).year
for y in sorted(set(yr)):
    ym = (yr == y)
    fear_y = m.values & ym
    if fear_y.sum() == 0:
        print(f"   {y}: 0 gun")
        continue
    ex = np.nanmean(fwd21.values[fear_y]) - np.nanmean(fwd21.values[ym])
    print(f"   {y}: {int(fear_y.sum()):>3} gun | FEAR-ort {1e4*np.nanmean(fwd21.values[fear_y]):+.1f}bp | "
          f"yil-kosulsuz {1e4*np.nanmean(fwd21.values[ym]):+.1f}bp | excess {1e4*ex:+.1f}bp")
ex_all = []
for y in sorted(set(yr)):
    ym = (yr == y); fear_y = m.values & ym
    if fear_y.sum() > 0 and np.isfinite(fwd21.values[fear_y]).any():
        ex_all.append(np.nanmean(fwd21.values[fear_y]) - np.nanmean(fwd21.values[ym]))
print(f"   within-year excess ORT (yil-esit-agirlik): {1e4*np.mean(ex_all):+.1f}bp "
      f"({sum(1 for e in ex_all if e > 0)}/{len(ex_all)} yil pozitif)")
