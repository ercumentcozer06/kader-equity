"""Adversarial denetim dogrulamasi (scratch, on-kayit-DISI): candidate_net_supply bulgularini
bagimsiz yeniden hesapla. Iddialar:
  A) sinyal ~ takvim-zamani trendi (Spearman ~ -0.726) -> dairesel-perm p'si sahte-trend riski
  B) 2005+ 252g 5-kova nfc-seviye: Q1 (dusuk-arz) EN KOTU kova (~-1.1%)
  C) 1984+ (geri-alim doneminin tamami) Spearman ~0 (p 0.78-0.85)
  D) tam-orneklem Q1 +9.8% mansetinin kaynagi: 1984-oncesi vs sonrasi ayristirma
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "data" / "cache"

N_PERM, SEED, MIN_SHIFT_Q = 2000, 77, 8


def fwd_at(close, dates, h):
    pos = close.index.searchsorted(dates, side="right") - 1
    out = np.full(len(dates), np.nan)
    ok = (pos >= 0) & (pos + h < len(close))
    cv = close.values
    out[ok] = cv[pos[ok] + h] / cv[pos[ok]] - 1.0
    return pd.Series(out, index=dates)


def spearman_perm(sig, fwd):
    df = pd.concat([sig.rename("s"), fwd.rename("f")], axis=1).dropna()
    n = len(df)
    if n < 40:
        return float("nan"), float("nan"), n
    rho = df["s"].corr(df["f"], method="spearman")
    rng = np.random.default_rng(SEED)
    sv, fv = df["s"].values, df["f"].values
    fr = pd.Series(fv).rank().values
    perm = np.empty(N_PERM)
    for i in range(N_PERM):
        k = int(rng.integers(MIN_SHIFT_Q, n - MIN_SHIFT_Q))
        sr = pd.Series(np.roll(sv, k)).rank().values
        perm[i] = np.corrcoef(sr, fr)[0, 1]
    p = float((np.abs(perm) >= abs(rho)).mean())
    return float(rho), p, n


def main():
    sup = pd.read_parquet(CACHE / "net_equity_supply.parquet").dropna(subset=["ratio4q_nfc_pct"])
    spx = pd.read_csv(CACHE / "spx_gspc_long.csv", parse_dates=["Date"]).set_index("Date")["Close"].dropna().sort_index()
    pit = pd.DatetimeIndex(sup["pit_date"])
    sig = pd.Series(sup["ratio4q_nfc_pct"].values, index=pit)
    f126 = fwd_at(spx, pit, 126)
    f252 = fwd_at(spx, pit, 252)

    # A) sinyal vs takvim-zamani
    tnum = pd.Series(np.arange(len(sig), dtype=float), index=pit)
    rho_t = sig.corr(tnum, method="spearman")
    print(f"A) Spearman(ratio4q_nfc, takvim-zamani) = {rho_t:+.3f}")

    # B) 2005+ 252g 5-kova
    m = sig.index.year >= 2005
    df = pd.concat([sig[m].rename("s"), f252[m].rename("f")], axis=1).dropna()
    q = pd.qcut(df["s"], 5, labels=False, duplicates="drop")
    b = [100 * df["f"][q == i].mean() for i in range(5)]
    print(f"B) 2005+ 252g kova (Q1 dusuk-arz -> Q5): " + "  ".join(f"{x:+.1f}" for x in b) + f"  (n={len(df)})")

    # C) 1984+ Spearman
    m84 = sig.index.year >= 1984
    for h, f in ((126, f126), (252, f252)):
        rho, p, n = spearman_perm(sig[m84], f[m84])
        print(f"C) 1984+ {h}g Spearman {rho:+.3f}  perm-p {p:.3f}  n={n}")

    # D) tam-orneklem Q1 mansetinin donem ayristirmasi
    dfall = pd.concat([sig.rename("s"), f252.rename("f")], axis=1).dropna()
    qall = pd.qcut(dfall["s"], 5, labels=False, duplicates="drop")
    q1 = dfall[qall == 0]
    pre, post = q1[q1.index.year < 1984], q1[q1.index.year >= 1984]
    print(f"D) tam-orneklem Q1: ort {100*q1['f'].mean():+.1f}% (n={len(q1)})"
          f" | 1984-oncesi {100*pre['f'].mean():+.1f}% (n={len(pre)})"
          f" | 1984+ {100*post['f'].mean():+.1f}% (n={len(post)})")
    # Q1 uyeligi hangi donemde?
    print(f"   Q1 uyelik dagilimi: 1984-oncesi {len(pre)}, 1984+ {len(post)}"
          f" | tum-orneklem 1984-oncesi pay {100*(dfall.index.year < 1984).mean():.0f}%")
    # donem ortalamalari (kova-bagimsiz taban)
    pre_a, post_a = dfall[dfall.index.year < 1984], dfall[dfall.index.year >= 1984]
    print(f"   taban fwd-252: 1984-oncesi {100*pre_a['f'].mean():+.1f}% vs 1984+ {100*post_a['f'].mean():+.1f}%")

    # D2) tum kovalarin donem-bilesimi + era-ici Spearman
    for i in range(5):
        sub = dfall[qall == i]
        pre, post = sub[sub.index.year < 1984], sub[sub.index.year >= 1984]
        pm = 100 * pre["f"].mean() if len(pre) else float("nan")
        qm = 100 * post["f"].mean() if len(post) else float("nan")
        print(f"   Q{i+1}: n={len(sub)} ort {100*sub['f'].mean():+.1f}% | pre84 n={len(pre)} {pm:+.1f}% | 84+ n={len(post)} {qm:+.1f}%")
    for lab, mask in (("pre84", dfall.index.year < 1984), ("84+", dfall.index.year >= 1984)):
        sub = dfall[mask]
        print(f"   era-ici Spearman [{lab}]: {sub['s'].corr(sub['f'], method='spearman'):+.3f} (n={len(sub)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
