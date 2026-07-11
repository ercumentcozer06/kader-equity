"""
validation/lucid_full_eval — Lucid_Watcher SWR/decumulation tweet'inin HER kavramını kader-equity
delta-one yön-modeline çevirip GERÇEK sayıyla ölçer. Decumulation dili → trading-model analoğu:

  withdrawal/ruin/terminal-wealth  → strateji equity-eğrisi dağılımı, katastrofik-DD olasılığı
  sequence-of-returns risk         → sıra-bağımlılık: blok(sıra-korur) vs iid-shuffle(sıra-bozar) DD farkı
  vol/inflation regime embedding   → vol-rejim STRATİFİYE bootstrap (yüksek/düşük RV içinde)
  regime-transition stress         → tide flip pencerelerinde koşullu performans
  correlation-structure stress     → SPX-NDX ortak-DD, kitap CVaR
  CVaR/expected-shortfall          → var (ES≡CVaR)
  drawdown persistence             → time-underwater dağılımı, recovery süresi
  convexity                        → skew, up/down-capture, kuadratik-beta (idx^2 katsayısı)
  path dependency / tail-risk      → blok-bootstrap + kuyruk-oranları

READ-ONLY. lag=1 PIT. Stack = tide × cor1m_froth × gex_shield (canlı reçete).
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

from spine import contract as C, tide as T                            # noqa: E402
from backtest import engine as E                                       # noqa: E402
from modules.cor1m_froth import froth_factor_series, fetch_cor1m_live  # noqa: E402
from modules.gex_shield import gex_zscore, shield_factor_series        # noqa: E402

ANN = np.sqrt(252)


def _sh(r):
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    return float(r.mean() / r.std() * ANN) if (len(r) > 20 and r.std() > 0) else np.nan


def _dd_curve(r):
    eq = np.cumprod(1 + np.asarray(r, float)); peak = np.maximum.accumulate(eq)
    return eq, eq / peak - 1.0


def _maxdd(r):
    _, dd = _dd_curve(r); return float(dd.min())


def _cvar(r, q=0.05):
    r = np.sort(np.asarray(r, float)); k = max(1, int(q * len(r))); return float(r[:k].mean())


def build():
    """Canlı stack + endeks getiri serileri (SPX, NDX), 2019+ frozen."""
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    cor = fetch_cor1m_live()
    cf = froth_factor_series(cor.reindex(idx, method="ffill"), 8.0, 11.0, 0.0)
    sg = pd.read_parquet(ROOT / "data" / "cache" / "squeeze_dix_gex.parquet")
    zg = gex_zscore(sg["gex"], 252).reindex(idx, method="ffill")
    gf = shield_factor_series(zg, 0.5, 1.0, 0.4)
    pos = (tdir * cf * gf).reindex(idx)
    out = {}
    for a in ("SPX", "NDX"):
        idxret = E.fwd_ret(prices[a], idx)
        p = pos.astype(float).values
        p = np.concatenate([np.zeros(1), p[:-1]])
        stack = pd.Series(p * idxret.values, index=idx).dropna()
        out[a] = {"stack": stack, "idx": idxret.reindex(stack.index).fillna(0.0),
                  "pos": pd.Series(p, index=idx).reindex(stack.index), "price": prices[a]}
    return out, tdir


def block_paths(r, n_boot=10000, block=21, seed=7):
    rng = np.random.default_rng(seed); x = r.values; n = len(x)
    nblk = int(np.ceil(n / block)); pool = np.arange(0, n - block + 1)
    for _ in range(n_boot):
        st = rng.choice(pool, size=nblk, replace=True)
        yield np.concatenate([x[s:s + block] for s in st])[:n]


def iid_paths(r, n_boot=10000, seed=7):
    rng = np.random.default_rng(seed); x = r.values; n = len(x)
    for _ in range(n_boot):
        yield rng.choice(x, size=n, replace=True)


def time_underwater(r):
    """En uzun ve ortalama 'su altında' (yeni tepe görmeden geçen) gün sayısı + % zaman DD'de."""
    eq, dd = _dd_curve(r)
    under = dd < -1e-9
    runs = []; c = 0
    for u in under:
        if u:
            c += 1
        elif c:
            runs.append(c); c = 0
    if c:
        runs.append(c)
    return {"pct_time_underwater": float(under.mean()),
            "max_tuw_days": int(max(runs)) if runs else 0,
            "avg_tuw_days": float(np.mean(runs)) if runs else 0.0,
            "n_episodes": len(runs)}


def convexity(stack, idx):
    """Kuadratik-beta (idx^2 katsayısı>0 = konveks) + up/down capture + skew + tail-oranı."""
    s = stack.values; x = idx.values
    A = np.column_stack([np.ones_like(x), x, x * x])
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    up = x > 0; dn = x < 0
    up_cap = s[up].mean() / x[up].mean() if up.any() and x[up].mean() != 0 else np.nan
    dn_cap = s[dn].mean() / x[dn].mean() if dn.any() and x[dn].mean() != 0 else np.nan
    sk = float(pd.Series(s).skew())
    p95, p5 = np.percentile(s, 95), np.percentile(s, 5)
    return {"quad_beta_idx2": float(coef[2]), "lin_beta_idx": float(coef[1]),
            "up_capture": float(up_cap), "down_capture": float(dn_cap),
            "capture_convexity": float(up_cap - dn_cap), "skew": sk,
            "tail_ratio_p95_p5": float(p95 / abs(p5)) if p5 != 0 else np.nan}


def regime_transition(stack, tdir, win=10):
    """Tide flip (rejim geçişi) ±win pencerede vs stabil rejimde Sharpe."""
    tdir = tdir.reindex(stack.index, method="ffill")
    flips = tdir.diff().abs() > 0
    near = pd.Series(False, index=stack.index)
    fl_idx = np.where(flips.values)[0]
    for k in fl_idx:
        near.iloc[max(0, k - win):min(len(near), k + win + 1)] = True
    return {"n_flips": int(flips.sum()),
            "sharpe_transition": _sh(stack[near.values]),
            "sharpe_stable": _sh(stack[~near.values]),
            "pct_days_transition": float(near.mean())}


def vol_regime(stack, price, idx):
    """Yüksek/düşük realized-vol rejiminde STRATİFİYE performans (vol-rejim gömme)."""
    rv = price.reindex(stack.index, method="ffill").pct_change().rolling(21).std() * ANN
    med = rv.median()
    hi = (rv > med).reindex(stack.index).fillna(False).values
    return {"rv_median_ann": float(med),
            "sharpe_hi_vol": _sh(stack[hi]), "sharpe_lo_vol": _sh(stack[~hi]),
            "cvar_hi_vol": _cvar(stack[hi]), "cvar_lo_vol": _cvar(stack[~hi])}


def main():
    data, tdir = build()
    print("=" * 92)
    print("  LUCID TWEET → kader-equity TAM DEĞERLENDİRME (canlı stack, 2019+, lag=1 PIT, READ-ONLY)")
    print("=" * 92)
    for a in ("SPX", "NDX"):
        d = data[a]; stack = d["stack"]; idx = d["idx"]
        print(f"\n{'█'*4} [{a}]  gözlenen: Sharpe {_sh(stack):+.3f}  maxDD {100*_maxdd(stack):+.1f}%  "
              f"CVaR5 {100*_cvar(stack):+.2f}%  n={len(stack)}")

        # --- MC implied distribution (blok) : Sharpe / maxDD / terminal-wealth / ruin ---
        bl = list(block_paths(stack))
        S = np.array([_sh(p) for p in bl]); DD = np.array([_maxdd(p) for p in bl])
        TW = np.array([float(np.prod(1 + p)) for p in bl])
        print("  1) IMPLIED DIST (blok-bootstrap 10k):")
        print(f"     Sharpe p5/50/95 {np.percentile(S,5):+.2f}/{np.percentile(S,50):+.2f}/{np.percentile(S,95):+.2f}"
              f" | P(Sh>1)={100*np.mean(S>1):.0f}%")
        print(f"     Terminal-wealth ×  p5/50/95 {np.percentile(TW,5):.2f}/{np.percentile(TW,50):.2f}/{np.percentile(TW,95):.2f}"
              f" | P(final<1 = para kaybı)={100*np.mean(TW<1):.1f}%")
        print(f"  2) RISK-OF-RUIN (katastrofik-DD analoğu): "
              f"P(DD<-25%)={100*np.mean(DD<-0.25):.1f}%  P(DD<-35%)={100*np.mean(DD<-0.35):.1f}%  "
              f"P(DD<-50%)={100*np.mean(DD<-0.50):.1f}%")

        # --- sequence-of-returns: blok(sıra-korur) vs iid(sıra-bozar) maxDD ---
        DDi = np.array([_maxdd(p) for p in iid_paths(stack)])
        print(f"  3) SEQUENCE RISK (sıra-bağımlılık): maxDD medyan blok {100*np.median(DD):+.1f}% "
              f"vs iid-shuffle {100*np.median(DDi):+.1f}%  → sıra-etkisi {100*(np.median(DD)-np.median(DDi)):+.1f}pp"
              f" | terminal-wealth sıradan BAĞIMSIZ (çarpım komütatif)")

        # --- drawdown persistence ---
        tuw = time_underwater(stack.values)
        print(f"  4) DRAWDOWN PERSISTENCE: %zaman-DD {100*tuw['pct_time_underwater']:.0f}  "
              f"max-time-underwater {tuw['max_tuw_days']}g  ort {tuw['avg_tuw_days']:.0f}g  ({tuw['n_episodes']} epizod)")

        # --- convexity ---
        cx = convexity(stack, idx)
        print(f"  5) CONVEXITY: kuadratik-β(idx²) {cx['quad_beta_idx2']:+.2f} ({'KONVEKS' if cx['quad_beta_idx2']>0 else 'konkav'}) "
              f"| up-cap {cx['up_capture']:.2f} down-cap {cx['down_capture']:.2f} → convexity {cx['capture_convexity']:+.2f} "
              f"| skew {cx['skew']:+.2f} | tail-oranı {cx['tail_ratio_p95_p5']:.2f}")

        # --- regime-transition stress ---
        rt = regime_transition(stack, tdir)
        print(f"  6) REGIME-TRANSITION STRESS: {rt['n_flips']} flip | Sharpe geçiş {rt['sharpe_transition']:+.2f} "
              f"vs stabil {rt['sharpe_stable']:+.2f} (geçiş-günü %{100*rt['pct_days_transition']:.0f})")

        # --- vol-regime embedding ---
        vr = vol_regime(stack, d["price"], idx)
        print(f"  7) VOL-REGIME EMBED (RV-med {vr['rv_median_ann']:.0%}): Sharpe hi-vol {vr['sharpe_hi_vol']:+.2f} / "
              f"lo-vol {vr['sharpe_lo_vol']:+.2f} | CVaR hi {100*vr['cvar_hi_vol']:+.2f}% / lo {100*vr['cvar_lo_vol']:+.2f}%")

    # --- correlation-structure stress (iki varlık birlikte) ---
    ssp, snd = data["SPX"]["stack"].align(data["NDX"]["stack"], join="inner")
    book = 0.5 * ssp + 0.5 * snd
    _, ddsp = _dd_curve(ssp.values); _, ddnd = _dd_curve(snd.values)
    print(f"\n{'█'*4} [KİTAP: 50/50 SPX+NDX]  CORRELATION-STRUCTURE STRESS")
    print(f"  8) stack-korelasyon SPX~NDX {ssp.corr(snd):+.2f} | ortak-DD (ikisi de <-10%) günleri "
          f"%{100*np.mean((ddsp<-0.10)&(ddnd<-0.10)):.0f} | kitap Sharpe {_sh(book):+.2f} "
          f"maxDD {100*_maxdd(book):+.1f}% CVaR {100*_cvar(book):+.2f}% (çeşitlenme faydası)")
    print("\n" + "-" * 92)
    print("  READ-ONLY — pozisyon/mimari DEĞİŞMEDİ. Tüm sayılar canlı reçetenin ölçümü.")
    print("-" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
