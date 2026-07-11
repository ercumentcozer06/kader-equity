"""
validation/blocklen_p99 — B1 (block-length sweep) + B2 (p99/p99.5 empirik VaR).

B1: sequence-riski #3'te yalnız 21g blokla ölçüldü (sub-aylık, +0.4pp). Blok-uzunluğunu 5/21/63/126g
tarayınca çok-aylık rejim-kümelenmesi maxDD dağılımını genişletiyor mu? (uzun blok = daha çok otokorelasyon
korunur → sıra-etkisi büyürse maxDD sol-kuyruğu açılır). #3 ↔ #6 (rejim-geçiş) köprüsü.

B2: her şey %5'te kesiliyordu. p95/p99/p99.5 empirik VaR + CVaR — parametrik varsayım YOK, mevcut örnek.
EVT/GPD DEĞİL (o örnek-ince, kırılgan; ayrı iş). Stack vs B&H bağlam.

READ-ONLY. stack = tide × cor1m_froth × gex_shield (canlı reçete), lag=1 PIT.
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

from validation.mc_implied_distribution import stack_returns          # noqa: E402
from spine import contract as C                                        # noqa: E402
from backtest import engine as E                                       # noqa: E402

ANN = np.sqrt(252)


def _sh(r):
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    return float(r.mean() / r.std() * ANN) if (len(r) > 20 and r.std() > 0) else np.nan


def _maxdd(r):
    eq = np.cumprod(1 + np.asarray(r, float)); return float((eq / np.maximum.accumulate(eq) - 1).min())


def block_paths(x, n_boot, block, seed=7):
    rng = np.random.default_rng(seed); n = len(x)
    nblk = int(np.ceil(n / block)); pool = np.arange(0, n - block + 1)
    for _ in range(n_boot):
        st = rng.choice(pool, size=nblk, replace=True)
        yield np.concatenate([x[s:s + block] for s in st])[:n]


def bh_returns(asset):
    scores, prices, vector, prov = C.read_frozen()
    r = E.fwd_ret(prices[asset], prices[asset].index)
    return r.dropna()


def main():
    print("=" * 90)
    print("  B1 BLOCK-LENGTH SWEEP + B2 EMPİRİK p99 VaR — canlı stack, 2019+, lag=1 PIT, READ-ONLY")
    print("=" * 90)
    for a in ("SPX", "NDX"):
        r = stack_returns(a); x = r.values
        print(f"\n████ [{a}]  gözlenen Sharpe {_sh(x):+.3f}  maxDD {100*_maxdd(x):+.1f}%  n={len(x)}")

        # --- B1: block-length sweep ---
        print("  B1) BLOCK-LENGTH SWEEP (5k yol/blok) — uzun blok maxDD kuyruğunu açıyor mu?")
        print(f"     {'blok':>6}{'Sh p5':>9}{'Sh p50':>9}{'Sh p95':>9}{'maxDD p50':>11}{'maxDD p95(kötü)':>16}{'maxDD p99':>11}")
        for blk in (5, 21, 63, 126):
            S = []; D = []
            for p in block_paths(x, 5000, blk):
                S.append(_sh(p)); D.append(_maxdd(p))
            S = np.array(S); D = np.array(D)
            print(f"     {blk:>4}g{np.percentile(S,5):>+9.2f}{np.percentile(S,50):>+9.2f}{np.percentile(S,95):>+9.2f}"
                  f"{100*np.percentile(D,50):>+10.1f}%{100*np.percentile(D,5):>+15.1f}%{100*np.percentile(D,1):>+10.1f}%")
        print("     (maxDD p95/p99 = en KÖTÜ %5/%1 yol; blok büyüdükçe kötüleşiyorsa rejim-kümelenmesi gerçek)")

        # --- B2: empirical tail VaR ---
        bh = bh_returns(a).reindex(r.index).dropna().values
        print("  B2) EMPİRİK VaR/CVaR (günlük getiri, parametrik-varsayım YOK):")
        print(f"     {'kuantil':>10}{'VaR stack':>12}{'CVaR stack':>12}{'VaR B&H':>11}{'CVaR B&H':>11}")
        for q, lbl in ((0.05, "p95"), (0.01, "p99"), (0.005, "p99.5")):
            xs = np.sort(x); bs = np.sort(bh)
            ks = max(1, int(q * len(xs))); kb = max(1, int(q * len(bs)))
            var_s, cvar_s = xs[ks - 1], xs[:ks].mean()
            var_b, cvar_b = bs[kb - 1], bs[:kb].mean()
            print(f"     {lbl:>10}{100*var_s:>+11.2f}%{100*cvar_s:>+11.2f}%{100*var_b:>+10.2f}%{100*cvar_b:>+10.2f}%")
        worst = np.sort(x)[:3]
        print(f"     en kötü 3 gün: {', '.join(f'{100*w:+.2f}%' for w in worst)}  | "
              f"en kötü B&H 3: {', '.join(f'{100*w:+.2f}%' for w in np.sort(bh)[:3])}")
    print("\n" + "-" * 90)
    print("  READ-ONLY — pozisyon/mimari DEĞİŞMEDİ.")
    print("-" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
