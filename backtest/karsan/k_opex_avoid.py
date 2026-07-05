"""
backtest/karsan/k_opex_avoid — "OpEx / quad-witch gününde flat kal" izole testi (Emir 2026-06-12).
Empirik: bu günlerin ortalama getirisi + flat-kalma overlay'lerinin Sharpe/maxDD/PnL etkisi.
Standalone SPX+NDX 1990-2026 (max tarih, çok-rejim) + model ablation 2019+. t+1 gerekmez (gün-içi flat = o günü atla).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import k_config as K
import k_stats as KS
from k_phase1 import third_fridays, bd_offset_to_events
RNG = K.boot_rng()


def met(r):
    r = pd.Series(r).dropna()
    eq = (1 + r).cumprod(); dd = float((eq / eq.cummax() - 1).min())
    return (round(float(r.mean() / r.std() * np.sqrt(252)), 3), round(100 * dd, 1), round(100 * float(eq.iloc[-1] - 1), 0))


def main():
    yf = {a: pd.read_parquet(K.KDATA / f"yf_{a}.parquet") for a in ("SPX", "NDX")}
    print("=" * 90); print("  OpEx / QUAD-WITCH 'FLAT KAL' İZOLE TESTİ (1990-2026, long-only kitap)"); print("=" * 90)

    for a in ("SPX", "NDX"):
        px = yf[a]["c"]; ret = px.pct_change().dropna()
        idx = ret.index
        tf = third_fridays(idx)
        quad = pd.DatetimeIndex([d for d in tf if d.month in K.QUAD_MONTHS])
        off_m = bd_offset_to_events(idx, tf)
        off_q = bd_offset_to_events(idx, quad)
        is_opex_day = idx.isin(tf)
        is_quad_day = idx.isin(quad)
        quad_week = off_q.between(-4, 0)
        post_quad = off_q.between(1, 1)        # quad ertesi gün
        opex_week = off_m.between(-4, 0)

        print(f"\n  [{a}] — günlerin ORTALAMA getirisi (empirik):")
        for nm, m in (("quad-witch günü", is_quad_day), ("monthly OpEx günü", is_opex_day),
                      ("quad ertesi gün", post_quad.values), ("quad haftası", quad_week.values),
                      ("OpEx haftası", opex_week.values), ("diğer tüm günler", ~is_opex_day)):
            r = ret[m] if isinstance(m, np.ndarray) else ret[m]
            t = r.mean() / (r.std() / np.sqrt(len(r))) if len(r) > 2 and r.std() > 0 else 0
            print(f"    {nm:20} {1e4*r.mean():+7.1f} bps/gün  (t{t:+.2f}, n={len(r)})")

        print(f"\n  [{a}] — 'o günlerde FLAT kal' overlay (vs hep-long): Sharpe | maxDD | cumPnL")
        base = ret.copy()
        bs, bd, bc = met(base)
        print(f"    {'hep-long (B&H)':28} {bs:+.3f} | {bd:+6.1f}% | {bc:+.0f}%")
        for nm, mask in (("flat quad-witch günü", is_quad_day), ("flat monthly OpEx günü", is_opex_day),
                         ("flat quad ertesi gün", post_quad.values), ("flat quad haftası", quad_week.values),
                         ("flat OpEx haftası", opex_week.values)):
            ov = ret.copy(); ov[mask] = 0.0
            s, d, c = met(ov)
            print(f"    {nm:28} {s:+.3f} | {d:+6.1f}% | {c:+.0f}%   (Δsharpe {s-bs:+.3f})")

    # ---- model ablation 2019+ ----
    print("\n" + "=" * 90); print("  MODEL ABLATION 2019+ : stack'e 'flat quad-witch' eklersek"); print("=" * 90)
    from spine import contract as C, tide as T
    from backtest import engine as E
    from modules.cor1m_froth import froth_factor_series
    from modules.gex_shield import gex_zscore, shield_factor_series
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector)); idx = tdir.index
    cor = pd.read_parquet(ROOT/"data/cache/corr_pc.parquet")["COR1M"].dropna()
    gex = pd.read_parquet(ROOT/"data/cache/squeeze_dix_gex.parquet")["gex"].dropna()
    froth = froth_factor_series(cor.reindex(idx, method="ffill"), 8, 11, 0.0)
    shield = shield_factor_series(gex_zscore(gex).reindex(idx, method="ffill"), 0.5, 1.0, 0.4)
    tf = third_fridays(idx); quad = pd.DatetimeIndex([d for d in tf if d.month in K.QUAD_MONTHS])
    is_quad = pd.Series(idx.isin(quad), index=idx)
    is_opex = pd.Series(idx.isin(tf), index=idx)

    def strat(pos, close, lag=1):
        r = E.fwd_ret(close, pos.index).values; p = pos.astype(float).values
        p = np.concatenate([np.zeros(lag), p[:-lag]]); return pd.Series(p * r, index=pos.index).dropna()
    base = tdir * froth * shield
    for a in ("SPX", "NDX"):
        bs, bd, bc = met(strat(base, prices[a]))
        print(f"\n  [{a}] stack (base)                {bs:+.3f} | {bd:+6.1f}% | {bc:+.0f}%")
        for nm, mask in (("flat quad-witch günü", is_quad), ("flat monthly OpEx günü", is_opex)):
            ov = base.copy(); ov[mask] = 0.0
            s, d, c = met(strat(ov, prices[a]))
            print(f"  [{a}] +{nm:24} {s:+.3f} | {d:+6.1f}% | {c:+.0f}%   (Δ {s-bs:+.3f})")
    print("\n  NOT: 'flat o gün' = o günün getirisini atla (gün-içi düz). Vanna/charm GERÇEK greek-akışları")
    print("       per-strike OI ister → bedava tarih YOK → FORWARD-only (gamma_engine canlı biriktiriyor).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
