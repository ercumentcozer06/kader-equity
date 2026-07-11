"""
validation/mc_implied_distribution — CANLI stack'in "implied distribution"ı (Lucid_Watcher tweet uyarlaması).

Tweet'in TRANSFER EDİLEBİLİR yarısı (decumulation/withdrawal kısmı bize UYMAZ — çekim/ruin/terminal-wealth yok):
Monte-Carlo path dağılımı + CVaR + drawdown persistence + tail-risk. Nokta-tahmin (Sharpe 1.64/1.77) yerine
canlı stack getirilerini BLOK-BOOTSTRAP ile binlerce yolda yeniden örnekleyip Sharpe/maxDD/CVaR'ın DAĞILIMINI
çıkarır → "1.64 kırılgan tek-yol mu, yoksa yeniden-örneklemde sağlam mı?" sorusuna güven-bandı.

Stack = tide_dir × cor1m_froth × gex_shield (canlı reçete, config defaults). PIT: lag=1, trailing-z (mc_boot
gerçek getirileri yeniden örnekler → look-ahead eklemez). Blok = otokorelasyonu korur (i.i.d. bootstrap Sharpe'ı
şişirir). READ-ONLY doğrulama — modele DOKUNMAZ.
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

from spine import contract as C, tide as T                       # noqa: E402
from backtest import engine as E                                  # noqa: E402
from modules.cor1m_froth import froth_factor_series, fetch_cor1m_live  # noqa: E402
from modules.gex_shield import gex_zscore, shield_factor_series   # noqa: E402


def _sh(r):
    r = np.asarray(r); r = r[~np.isnan(r)]
    return float(r.mean() / r.std() * np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")


def _dd(r):
    eq = np.cumprod(1 + np.asarray(r)); peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min())


def _cvar(r, q=0.05):
    r = np.sort(np.asarray(r)); k = max(1, int(q * len(r)))
    return float(r[:k].mean())


def stack_returns(asset):
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    # cor1m_froth faktör serisi (config defaults lo8/hi11/floor0)
    cor = fetch_cor1m_live()
    cf = froth_factor_series(cor.reindex(idx, method="ffill"), 8.0, 11.0, 0.0)
    # gex_shield faktör serisi (squeeze gex → z252 → shield k.5/thr1/fl.4)
    sg = pd.read_parquet(ROOT / "data" / "cache" / "squeeze_dix_gex.parquet")
    zg = gex_zscore(sg["gex"], 252).reindex(idx, method="ffill")
    gf = shield_factor_series(zg, 0.5, 1.0, 0.4)
    pos = (tdir * cf * gf).reindex(idx)
    ret = E.fwd_ret(prices[asset], idx).values
    p = pos.astype(float).values
    p = np.concatenate([np.zeros(1), p[:-1]])                     # lag=1 (kanıtlanmış look-ahead-free hiza)
    r = pd.Series(p * ret, index=idx).dropna()
    return r


def block_bootstrap(r, n_boot=10000, block=21, seed=7):
    """Blok-bootstrap → Sharpe/maxDD/CVaR implied dağılımı (otokorelasyon korunur)."""
    rng = np.random.default_rng(seed)
    x = r.values; n = len(x); nblk = int(np.ceil(n / block))
    starts_pool = np.arange(0, n - block + 1)
    S = np.empty(n_boot); D = np.empty(n_boot); V = np.empty(n_boot)
    for i in range(n_boot):
        st = rng.choice(starts_pool, size=nblk, replace=True)
        path = np.concatenate([x[s:s + block] for s in st])[:n]
        S[i] = _sh(path); D[i] = _dd(path); V[i] = _cvar(path)
    return S, D, V


def main():
    print("=" * 78)
    print("  CANLI STACK IMPLIED DISTRIBUTION — block-bootstrap MC (tweet uyarlaması)")
    print("  stack = tide × cor1m_froth × gex_shield | 2019+ frozen | lag=1 PIT | 10k×21g blok")
    print("=" * 78)
    for a in ("SPX", "NDX"):
        r = stack_returns(a)
        pt_s, pt_d, pt_v = _sh(r), _dd(r), _cvar(r)
        S, D, V = block_bootstrap(r)
        def pct(x, p): return float(np.percentile(x, p))
        print(f"\n  [{a}]  gözlenen nokta: Sharpe {pt_s:+.3f}  maxDD {100*pt_d:+.1f}%  CVaR5 {100*pt_v:+.2f}%")
        print(f"    {'metrik':<10}{'p5':>9}{'p50':>9}{'p95':>9}   dağılım-notu")
        print(f"    {'Sharpe':<10}{pct(S,5):>+9.2f}{pct(S,50):>+9.2f}{pct(S,95):>+9.2f}   "
              f"P(Sh>0)={100*np.mean(S>0):.1f}%  P(Sh>1)={100*np.mean(S>1):.1f}%")
        print(f"    {'maxDD %':<10}{100*pct(D,5):>+9.1f}{100*pct(D,50):>+9.1f}{100*pct(D,95):>+9.1f}   "
              f"P(DD<-25%)={100*np.mean(D<-0.25):.1f}%")
        print(f"    {'CVaR5 %':<10}{100*pct(V,5):>+9.2f}{100*pct(V,50):>+9.2f}{100*pct(V,95):>+9.2f}")
    print("\n  " + "-" * 74)
    print("  READ-ONLY doğrulama — pozisyona/mimariye DOKUNULMADI. Güven-bandı ölçümü.")
    print("  " + "-" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
