# -*- coding: utf-8 -*-
"""
LAB (b) — DISPERSION OVERLAY KALIBRASYONU  (2026-08-06, Emir emri). SALT-BACKTEST, CANLI YOLA DOKUNMAZ.

Sorun tespiti (08-06 olcum): canli factor 0.029; son 90 gun ortalama factor 0.081; son 1 yilda
zamanin %27'si "tam kapali" (<0.05). froth_pct = 756-gunluk TRAILING persentil -> girdi yapisal
trend yapiyorsa persentil 1.0'a DOYAR ve orada kalir; sinyal "asiri froth" demeye devam eder ama
ayrim gucu biter. Bu lab dort soruyu ayri ayri olcer:

  Q1 DOYGUNLUK : froth dagilimi zamanla kayiyor mu? ust-decile'da gecen zaman artiyor mu?
  Q2 ONGORU    : froth ileri EQUITY getirisini ayiriyor mu? (overlay'in VAROLUS sebebi)
  Q3 KAPI      : gercek tide stratejisi uzerinde (lo,hi,floor) taramasi + split-half + bootstrap
  Q4 ALTERNATIF: daha uzun pencere / expanding persentil / floor>0 doygunlugu cozuyor mu?

Zemin = FROZEN tide spine (kader-macro YOK, network YOK) — reproduce_baseline ile ayni sozlesme.
Tum carpanlar t-1 (PIT). Deploy-kapisi: Sharpe KORUNUR/ARTAR **ve** maxDD kotulesmez **ve**
split-half ROBUST. Aksi halde canli (0.70,0.95,0.0) DEGISMEZ ve bulgu "test edildi, eklenmedi".
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

from spine import contract as C, tide as T
from modules.dispersion_ensemble import _pit_series, froth_pct_series, ensemble_factor

SQ = np.sqrt(252); RNG = np.random.default_rng(20260806)
LIVE = (0.70, 0.95, 0.0)

# ── veri: frozen fiyat/spine + frozen dispersion girdileri ─────────────────────────────────
scores, prices, vector, prov = C.read_frozen()
tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
disp = pd.read_parquet(ROOT / "data" / "cache" / "dispersion.parquet")
cor = pd.read_parquet(ROOT / "data" / "cache" / "corr_pc.parquet")["COR1M"].dropna()

def froth(win=756, mp=252, mode="rolling"):
    if mode == "rolling":
        return froth_pct_series(cor, disp["spread"].dropna(), disp["dspx"].dropna(), win, mp)
    # expanding persentil: pencere sabit degil, tum gecmis (doygunluk testi)
    def ex(s):
        return s.expanding(min_periods=mp).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    d = pd.concat([(1-ex(cor)).rename("cor"), ex(disp["spread"].dropna()).rename("spr"),
                   ex(disp["dspx"].dropna()).rename("dsp")], axis=1)
    return d.mean(axis=1, skipna=False).rename("froth_pct")

F = {"rolling756": froth(756), "rolling1260": froth(1260), "expanding": froth(mode="exp")}
hdr = lambda s: print("\n" + "=" * 92 + f"\n{s}\n" + "=" * 92)

# ═══════════════ Q1 — DOYGUNLUK ═══════════════
hdr("Q1 — DOYGUNLUK: froth dagilimi zamanla kayiyor mu?")
for nm, f in F.items():
    f = f.dropna()
    yr = f.groupby(f.index.year).agg(ort="mean", ust_decile=lambda x: (x >= 0.90).mean())
    print(f"\n  [{nm}]  n={len(f)}  {f.index[0].date()}->{f.index[-1].date()}")
    print("   yil :", "  ".join(f"{y}:{r.ort:.2f}/{100*r.ust_decile:.0f}%" for y, r in yr.iterrows()))
print("\n  (yil: ORT_froth / zamanin yuzde kaci >=0.90). Sag tarafa dogru tirmanis = DOYGUNLUK.")

# ═══════════════ Q2 — ONGORU GUCU ═══════════════
hdr("Q2 — froth ileri EQUITY getirisini ayiriyor mu? (overlay'in varolus sebebi)")
for asset in ["SPX", "NDX"]:
    px = prices[asset].dropna()
    for H, lab in ((21, "1a"), (63, "3a")):
        fwd = px.shift(-H) / px - 1
        d = pd.DataFrame({"f": F["rolling756"], "r": fwd}).dropna()
        q = pd.qcut(d.f, 5, labels=["Q1 dusuk", "Q2", "Q3", "Q4", "Q5 froth"], duplicates="drop")
        g = d.groupby(q, observed=True)["r"].agg(n="size", ort=lambda x: 100 * x.mean())
        print(f"  {asset} ileri-{lab}: " + " | ".join(f"{k}:{v.ort:+.1f}%" for k, v in g.iterrows()))
print("\n  Overlay hakli ise Q5(froth) ileri getirisi Q1'den BELIRGIN DUSUK olmali.")

# ═══════════════ Q3 — GERCEK STRATEJI UZERINDE KAPI ═══════════════
hdr("Q3 — tide stratejisi x dispersion overlay: (lo,hi,floor) taramasi [t-1, split-half, bootstrap]")

def run(asset, f, lo, hi, fl):
    px = prices[asset].dropna()
    ret = px.pct_change()
    base = (tdir.reindex(ret.index).ffill().shift(1) * ret).dropna()
    if lo is None:
        m = pd.Series(1.0, base.index)
    else:
        m = f.reindex(base.index).ffill().shift(1).map(
            lambda x: 1.0 if pd.isna(x) else ensemble_factor(float(x), lo, hi, fl))
    net = (base * m).dropna()
    ann = net.mean() * 252; vol = net.std() * SQ
    eq = (1 + net).cumprod(); dd = (eq / eq.cummax() - 1).min()
    return dict(sharpe=ann / vol if vol else np.nan, getiri=100 * ann, maxDD=100 * dd,
                calmar=ann / abs(dd) if dd else np.nan, expo=m.reindex(net.index).mean()), net

def sh(z):
    v = z.std() * SQ; return (z.mean() * 252) / v if v else np.nan

GRID = [("OVERLAY YOK", None, None, None), ("CANLI 0.70/0.95/0.0", *LIVE),
        ("0.70/1.00/0.0", 0.70, 1.00, 0.0), ("0.80/1.00/0.0", 0.80, 1.00, 0.0),
        ("0.70/0.95/floor0.25", 0.70, 0.95, 0.25), ("0.70/0.95/floor0.50", 0.70, 0.95, 0.50),
        ("0.85/1.00/floor0.25", 0.85, 1.00, 0.25)]
for asset in ["SPX", "NDX"]:
    print(f"\n  ### {asset} ###")
    rows, nets = [], {}
    for nm, lo, hi, fl in GRID:
        s, n = run(asset, F["rolling756"], lo, hi, fl); nets[nm] = n
        rows.append(dict(sistem=nm, **{k: round(v, 3) for k, v in s.items()}))
    df = pd.DataFrame(rows).set_index("sistem")
    df["dSh_vs_CANLI"] = (df.sharpe - df.loc["CANLI 0.70/0.95/0.0", "sharpe"]).round(3)
    print(df.to_string())
    base = nets["CANLI 0.70/0.95/0.0"]; half = base.index[len(base) // 2]
    print(f"  split-half ({half.date()}) + bootstrap dSharpe vs CANLI:")
    for nm, n in nets.items():
        if nm.startswith("CANLI"): continue
        d1, d2 = sh(n[:half]) - sh(base[:half]), sh(n[half:]) - sh(base[half:])
        N = len(base); nb = int(np.ceil(N / 21))
        ix = RNG.integers(0, max(1, N - 21), size=(400, nb))
        bt = np.array([sh(pd.Series(np.concatenate([n.values[i:i+21] for i in r])[:N]))
                     - sh(pd.Series(np.concatenate([base.values[i:i+21] for i in r])[:N])) for r in ix])
        print(f"    {nm:22s} H1 {d1:+.3f} | H2 {d2:+.3f} | boot [{np.percentile(bt,5):+.3f},"
              f"{np.percentile(bt,95):+.3f}] {'ROBUST' if d1*d2>0 else 'sign-flip'}")

# ═══════════════ Q4 — ALTERNATIF PENCERELER ═══════════════
hdr("Q4 — pencere/persentil alternatifleri (canli esiklerle, doygunluk cozulur mu?)")
for asset in ["SPX", "NDX"]:
    rows = []
    for nm, f in F.items():
        s, _ = run(asset, f, *LIVE)
        rows.append(dict(froth_kaynagi=nm, **{k: round(v, 3) for k, v in s.items()}))
    s0, _ = run(asset, None, None, None, None)
    rows.append(dict(froth_kaynagi="(overlay yok)", **{k: round(v, 3) for k, v in s0.items()}))
    print(f"\n  ### {asset} ###"); print(pd.DataFrame(rows).set_index("froth_kaynagi").to_string())

print("\n" + "=" * 92)
print("KARAR NOTU: deploy ancak (Sharpe >= canli) VE (maxDD kotulesmez) VE (split-half ROBUST)")
print("saglanirsa. Aksi halde canli 0.70/0.95/0.0 DEGISMEZ — 'test edildi, eklenmedi'.")
