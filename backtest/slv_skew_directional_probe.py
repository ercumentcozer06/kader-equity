"""
backtest/slv_skew_directional_probe — SLV RR30 persentili YON tasiyor mu? (on-kayitli probe, 2026-07-11)

Soru (Emir): cockpit'e bagladigimiz SLV skew-persentili directional edge mi, betimsel mi?
PRIOR: DUSUK — options-lead ailesi 7 kill (skew-state BTC/SPX/ETH, skew-EVENT, DIX, VRP, oil-RR x2,
flip-directional). SLV RR hic test edilmedi (arsiv bugune kadar yoktu) → ampirik cevap hakki.

ON-KAYIT (kod oncesi sabit):
  Veri: rr_skew_slv_2016_2023.parquet (t30_ok temiz; spot kolonu = SLV, ayni kaynak → self-contained).
  Sinyal: rr30'un EXPANDING persentili (min 252 obs; PIT — full-sample persentil YASAK).
  Testler (2 kuyruk × 3 horizon = 6 birincil; Bonferroni α=0.05/6):
    T1 FEAR-kuyruk  : pct ≥ 95 gunleri → forward 1/5/21g spot getirisi vs kosulsuz (blok-bootstrap CI)
    T2 MANIA-kuyruk : pct ≤ 5 gunleri → ayni
  + T3 quintile-monotonluk (Spearman; betimsel — rank'ten sign-flip TURETILMEZ, feedback_rank_vs_absolute)
  KARAR: hicbir kuyruk-horizon Bonferroni gecemezse = BETIMSEL kalir (cockpit satiri zaten oyle etiketli).
  GUC-NOTU zorunlu: kuyruk-n kucukse (≤30 bagimsiz epizot) 'guc-yetersiz' damgasi (skew-EVENT dersi).
  & <kader-macro venv python> backtest/slv_skew_directional_probe.py
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

HORIZONS = (1, 5, 21)
MINP = 252
N_BOOT, BLOCK = 2000, 21
ALPHA = 0.05 / 6


def _boot_mean_ci(x, rng):
    """Ortusen-epizot etkisine kaba blok-bootstrap: gun-indeksleri blokla cekilir."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return np.nan, np.nan
    means = []
    nblk = max(1, int(np.ceil(n / min(BLOCK, n))))
    for _ in range(N_BOOT):
        st = rng.integers(0, max(n - min(BLOCK, n) + 1, 1), nblk)
        idx = (st[:, None] + np.arange(min(BLOCK, n))[None, :]).ravel()[:n]
        means.append(np.nanmean(x[idx]))
    lo, hi = np.percentile(means, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(lo), float(hi)


def main() -> int:
    df = pd.read_parquet(ROOT / "data" / "cache" / "rr_skew_slv_2016_2023.parquet")
    df = df[df["t30_ok"]].copy()
    rr = df["t30_rr_skew"].astype(float)
    spot = df["spot"].astype(float)
    # PIT expanding persentil: o gune KADARKI dagilimda bugunun sirasi
    pct = pd.Series(np.nan, index=rr.index)
    vals = rr.values
    for i in range(MINP, len(vals)):
        pct.iloc[i] = (vals[:i] < vals[i]).mean() * 100
    fwd = {h: spot.shift(-h) / spot - 1 for h in HORIZONS}

    print("=" * 100)
    print("  SLV RR30 YON-PROBE — on-kayitli (expanding-PIT persentil; Bonferroni α=%.4f)" % ALPHA)
    print("=" * 100)
    uncond = {h: np.nanmean(fwd[h].values) for h in HORIZONS}
    print(f"  temiz gun {len(df)}, persentil-tanimli {pct.notna().sum()}; kosulsuz ort "
          + "  ".join(f"{h}g {1e4*uncond[h]:+.1f}bp" for h in HORIZONS))

    rng = np.random.default_rng(3)
    any_pass = False
    for tag, mask in (("T1 FEAR (pct>=95)", pct >= 95), ("T2 MANIA (pct<=5)", pct <= 5)):
        m = mask.fillna(False).values
        n_days = int(m.sum())
        # bagimsiz-epizot sayisi: ardisik gun-kumeleri
        eps = int(np.sum(np.diff(np.concatenate([[0], m.astype(int)])) == 1))
        pw = "  [GUC-YETERSIZ ≤30 epizot]" if eps <= 30 else ""
        print(f"\n  {tag}: {n_days} gun / {eps} epizot{pw}")
        for h in HORIZONS:
            x = fwd[h].values[m]
            mu = np.nanmean(x) if np.isfinite(x).any() else np.nan
            lo, hi = _boot_mean_ci(x - uncond[h], rng)     # kosulsuza gore FARK'in CI'si
            verdict = "GECTI" if (np.isfinite(lo) and (lo > 0 or hi < 0)) else "sifir-kapsar"
            any_pass |= (verdict == "GECTI")
            print(f"    {h:>2}g: ort {1e4*mu:+.1f}bp (kosulsuz {1e4*uncond[h]:+.1f}) | "
                  f"Δ-CI [{1e4*lo:+.1f}, {1e4*hi:+.1f}]bp → {verdict}")

    # T3 quintile-monotonluk (betimsel)
    q = pd.qcut(pct.dropna(), 5, labels=False)
    print("\n  T3 quintile ort. 21g-forward (betimsel; sign-flip turetilmez):")
    for k in range(5):
        idx = q[q == k].index
        print(f"    Q{k+1}: {1e4*np.nanmean(fwd[21].reindex(idx).values):+.1f}bp (n={len(idx)})")
    from scipy.stats import spearmanr
    sp = spearmanr(pct.dropna(), fwd[21].reindex(pct.dropna().index), nan_policy="omit")
    print(f"    Spearman(pct, fwd21) = {sp.statistic:+.3f} (p={sp.pvalue:.3f})")

    print("\n" + "=" * 100)
    print(f"  VERDICT: {'EN AZ BIR KUYRUK-HORIZON GECTI — derin teste aday' if any_pass else 'HICBIRI GECMEDI → BETIMSEL kalir (cockpit etiketi dogru)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
