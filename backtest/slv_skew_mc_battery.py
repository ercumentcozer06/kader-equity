"""
backtest/slv_skew_mc_battery — SLV FEAR-fade adayinin MONTE-CARLO bataryasi (2026-07-11).

Emir sorusu: forward-OOS beklemeden MC ile karar verilebilir mi? EMSAL: VIXEQ-VIX (FWER-fail ama
MC-robust → deploy). MC in-sample sansi olcer, rejim-kalicilik OLCMEZ — ama luck-hipotezini
oldurebilirse aday kitap-ablation'a terfi eder (Sharpe-gate yolu), olduremezse forward-OOS kalir.

ON-KAYIT (kod oncesi):
  Aday: FEAR = expanding-PIT rr30-persentil >= 95, epizot-ilk-gunu; etki = 21g forward excess.
  M1 PLACEBO-MC (2000 cekim): 48 rastgele gun (ayni gecerli-evren) → P(placebo epizot-ort-excess >=
     gozlenen +183bp). Coklu-test duzeltmesi: aday 6 testten secildi → esik p < 0.05/6 = 0.0083.
  M2 YIL-JACKKNIFE: her yili at → excess hala pozitif mi (6/6 sart).
  M3 TRADE-FORMU: FEAR-ilk-gun → 21 isgunu long SLV, cost 10bp round-trip → sleeve Sharpe/maxDD/expo
     (mekanik uygulanabilirlik; Sharpe-hedef degil, sigorta-sinifi sekil-kontrolu).
  GECER = M1 p<0.0083 VE M2 6/6 → kitap-ablation lab'ina terfi (deploy DEGIL — silver-model degisikligi
  Sharpe-gate ister). GECEMEZ = aday dipnot kalir, forward-OOS bekler.
  & <kader-macro venv python> backtest/slv_skew_mc_battery.py
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

MINP, H, N_MC = 252, 21, 2000
COST = 0.001

df = pd.read_parquet(ROOT / "data" / "cache" / "rr_skew_slv_2016_2023.parquet")
df = df[df["t30_ok"]].copy()
rr = df["t30_rr_skew"].astype(float)
spot = df["spot"].astype(float)
vals = rr.values
pct = pd.Series(np.nan, index=rr.index)
for i in range(MINP, len(vals)):
    pct.iloc[i] = (vals[:i] < vals[i]).mean() * 100
fwd = spot.shift(-H) / spot - 1
m = (pct >= 95).fillna(False)
first = (m & ~m.shift(1, fill_value=False))
valid = pct.notna() & fwd.notna()
ep_idx = df.index[first & valid]
unc = float(np.nanmean(fwd[valid].values))
obs_ex = float(np.nanmean(fwd.reindex(ep_idx).values)) - unc
n_ep = len(ep_idx)
print(f"gozlenen: n_ep={n_ep}, epizot-ort-excess {1e4*obs_ex:+.1f}bp (kosulsuz {1e4*unc:+.1f}bp tabani)")

# M1 placebo-MC
rng = np.random.default_rng(17)
pool = np.where(valid.values)[0]
fv = fwd.values
hits = 0
pl = []
for _ in range(N_MC):
    pick = rng.choice(pool, n_ep, replace=False)
    ex = np.nanmean(fv[pick]) - unc
    pl.append(ex)
    hits += ex >= obs_ex
p1 = hits / N_MC
print(f"\nM1 PLACEBO-MC ({N_MC}): P(rastgele-48-gun excess >= gozlenen) = {p1:.4f} "
      f"(esik 0.0083; placebo ort {1e4*np.mean(pl):+.1f}bp, p95 {1e4*np.percentile(pl, 95):+.1f}bp) → "
      f"{'GECTI' if p1 < 0.05/6 else 'GECEMEDI'}")

# M2 yil-jackknife
years = pd.DatetimeIndex(df.index).year
ep_years = pd.DatetimeIndex(ep_idx).year
print("\nM2 YIL-JACKKNIFE (yili at → kalan epizotlarin excess'i):")
jk_ok = 0
jk_n = 0
for y in sorted(set(ep_years)):
    keep = ep_idx[ep_years != y]
    if not len(keep):
        continue
    keep_valid_mask = valid.values & (years != y)
    unc_y = float(np.nanmean(fv[keep_valid_mask]))
    ex = float(np.nanmean(fwd.reindex(keep).values)) - unc_y
    jk_n += 1
    jk_ok += ex > 0
    print(f"   -{y}: excess {1e4*ex:+.1f}bp (n_ep {len(keep)}) {'+' if ex > 0 else 'NEG'}")
print(f"   → {jk_ok}/{jk_n} pozitif")

# M3 trade-formu (mekanik long-21g, cost'lu)
pos = np.zeros(len(df))
loc = {d: i for i, d in enumerate(df.index)}
for d in ep_idx:
    i = loc[d]
    pos[i + 1:i + 1 + H] = 1.0                      # sinyal gun-sonu → ertesi gun pozisyon (PIT)
ret1 = (spot.shift(-1) / spot - 1).values
sret = pos * np.nan_to_num(ret1)
turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
sret = sret - turn * COST
fin = np.isfinite(sret)
mu, sd = np.nanmean(sret[fin]), np.nanstd(sret[fin])
sh = mu / sd * np.sqrt(252) if sd > 0 else float("nan")
eq = np.cumprod(1 + np.nan_to_num(sret))
mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
print(f"\nM3 TRADE-FORMU (long-21g, 10bp cost): Sharpe {sh:+.2f}  maxDD {100*mdd:+.1f}%  "
      f"expo %{100*pos.mean():.0f}  toplam {100*(eq[-1]-1):+.0f}%")

verdict = (p1 < 0.05/6) and (jk_ok == jk_n)
print("\n" + "=" * 96)
print(f"  VERDICT: {'MC-ROBUST → kitap-ablation lab adayi (deploy DEGIL: Sharpe-gate ister)' if verdict else 'MC-kapiyi GECEMEDI → dipnot + forward-OOS'}")
