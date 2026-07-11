"""
screen/candidate_dix_directional — DIX'in YÖN edge'i (kompozit-kırılganlıktan AYRI test).

candidate_gex_dix DIX'i yalnız 'düşük DIX = kırılgan' TRİM yönünde denedi (strict FDR FAIL).
Ama SqueezeMetrics'in DIX iddiası YÖN: yüksek DIX = dark-pool birikimi = ileri-bullish. Bu hiç
test edilmedi. Burada PIT-dürüst (lag=1, trailing-z) iki soru:

  A) STANDALONE: z(DIX) tek başına yön içeriği taşıyor mu? (bucket-ABSOLUTE fwd-getiri + Sharpe,
     tam tarih 2011-2026, parquet'in kendi price kolonu). 'rank≠direction' kuralı → önce bucket.
  B) INCREMENTAL: 2019+ tide omurgası (long/flat) üstüne DIX-tilt Sharpe EKLİYOR mu? Ablation +
     strict paired-win-prob {SPX,NDX} + BH-FDR (cor1m/gex adaylarıyla AYNI bar).

Lookahead YOK: z rolling-trailing, fwd getiri hedef, pozisyon lag=1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from spine import contract as C, tide as T              # noqa: E402
from backtest import engine as E                         # noqa: E402
from screen._util import paired_win_prob, fdr_bh         # noqa: E402


def z(s, win=252):
    return (s - s.rolling(win, min_periods=60).mean()) / s.rolling(win, min_periods=60).std()


def _sh(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")


def _dd(r):
    eq = (1 + r.dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min())


def strat_ret(pos, close, lag=1):
    idx = pos.index
    ret = E.fwd_ret(close, idx).values
    p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def part_A_standalone(sg):
    print("=" * 92)
    print("  A) STANDALONE DIX YÖN — tam tarih (parquet price = SPX-proxy), PIT lag=1")
    print("=" * 92)
    px = sg["price"].dropna()
    zd = z(sg["dix"]).reindex(px.index).dropna()
    px = px.reindex(zd.index)

    # --- bucket-ABSOLUTE study: z(DIX) quintili → fwd-21g ortalama getiri + isabet ---
    fwd21 = (px.shift(-21) / px - 1.0)
    df = pd.DataFrame({"zd": zd, "fwd21": fwd21}).dropna()
    df["q"] = pd.qcut(df["zd"], 5, labels=["Q1 düşük-DIX", "Q2", "Q3", "Q4", "Q5 yüksek-DIX"])
    print("  z(DIX) quintili → fwd-21g ABSOLUTE getiri (yüksek-DIX bullish mi?):")
    print(f"    {'bucket':<16}{'n':>6}{'ort fwd21':>12}{'isabet+':>10}{'medyan':>10}")
    g = df.groupby("q", observed=True)
    for q, sub in g:
        print(f"    {str(q):<16}{len(sub):>6}{100*sub['fwd21'].mean():>+11.2f}%"
              f"{100*(sub['fwd21'] > 0).mean():>9.0f}%{100*sub['fwd21'].median():>+9.2f}%")
    mono = g["fwd21"].mean()
    print(f"  → monotoni (Q5−Q1): {100*(mono.iloc[-1]-mono.iloc[0]):+.2f}pp  "
          f"(pozitif = yüksek-DIX daha yüksek fwd-getiri = yön içeriği)")

    # --- standalone directional Sharpe: birkaç eşleme, lag=1 ---
    fwd1 = E.fwd_ret(px, px.index)
    def sh_pos(pos):
        p = pos.astype(float).values
        p = np.concatenate([np.zeros(1), p[:-1]])
        return _sh(pd.Series(p * fwd1.values, index=px.index).dropna())
    bh = _sh(fwd1)
    variants = {
        "B&H (referans)":            pd.Series(1.0, index=px.index),
        "long/flat zDIX>0":          (zd > 0).astype(float),
        "long/flat zDIX>0.5":        (zd > 0.5).astype(float),
        "long/short sign(zDIX)":     np.sign(zd).replace(0, np.nan).ffill(),
        "tilt clip(zDIX,-1,1)+long": (0.5 + 0.5 * zd.clip(-1, 1)),   # 0..1 long-only tilt
    }
    print(f"\n  standalone directional Sharpe (fwd-1g, lag=1)   [B&H={bh:+.3f}]:")
    for lbl, pos in variants.items():
        print(f"    {lbl:<28}{sh_pos(pos):>+8.3f}")


def part_B_incremental(sg):
    print("\n" + "=" * 92)
    print("  B) INCREMENTAL — 2019+ tide (long/flat) üstüne DIX-tilt; strict paired-win + BH-FDR")
    print("=" * 92)
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    zd = z(sg["dix"]).reindex(idx, method="ffill")

    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    # DIX-tilt formları (long-only, rebound-safe mantığı korunur):
    #  T1: reentry — tide=0 (flat) iken yüksek DIX birikimi long'a döndürür (kaçan long'u yakala)
    #  T2: conviction-add — tide=1 iken yüksek DIX pozisyonu büyütür (>1 kaldıraç yok, ×1.0-1.5 tilt)
    #  T3: low-DIX trim — tide=1 iken düşük DIX (zayıf dark-buying) long'u kısar
    forms = {
        "T1 reentry zDIX>1":   lambda: np.maximum(tdir, (zd > 1.0).astype(float)),
        "T2 add ×(1+.3·zd+)":  lambda: tdir * (1.0 + 0.3 * zd.clip(0, 3)),
        "T3 trim low-DIX":     lambda: tdir * (1.0 - 0.3 * np.clip(-zd - 1.0, 0, 3)).clip(0.4, 1.0),
    }
    for a in ("SPX", "NDX"):
        b = bases[a]
        print(f"\n  [{a}]  base Sharpe {_sh(b):+.3f}  maxDD {100*_dd(b):+.0f}%")
        print(f"    {'form':<22}{'Sharpe':>8}{'ΔSh':>7}{'maxDD':>8}{'P(v>b)':>8}")
        for lbl, fn in forms.items():
            v = strat_ret(pd.Series(fn(), index=idx).reindex(idx), prices[a])
            wp = paired_win_prob(b, v)
            print(f"    {lbl:<22}{_sh(v):>+8.3f}{_sh(v)-_sh(b):>+7.2f}{100*_dd(v):>+7.0f}%"
                  f"{(f'{wp:.0%}' if wp is not None else 'n/a'):>8}")

    # strict FDR — her form için {SPX,NDX} ikisi de geçmeli
    print("\n  " + "-" * 88)
    for lbl, fn in forms.items():
        wps = {a: paired_win_prob(bases[a], strat_ret(pd.Series(fn(), index=idx).reindex(idx), prices[a]))
               for a in ("SPX", "NDX")}
        passed = fdr_bh({a: 1.0 - w for a, w in wps.items() if w is not None}, alpha=0.05)
        both = all(passed.get(a, False) for a in ("SPX", "NDX"))
        print(f"  {lbl:<22} P(v>b) {{{', '.join(f'{a}:{wps[a]:.0%}' for a in wps)}}}  "
              f"→ {'PASS (LIVE adayı)' if both else 'FAIL (strict bar altı)'}")
    print("  " + "-" * 88)


def main():
    sg = pd.read_parquet(ROOT / "data" / "cache" / "squeeze_dix_gex.parquet")
    print(f"  DIX/GEX: {len(sg)} gün {sg.index.min().date()}..{sg.index.max().date()}\n")
    part_A_standalone(sg)
    part_B_incremental(sg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
