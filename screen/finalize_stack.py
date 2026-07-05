"""
screen/finalize_stack — kümülatif edge: tide → ×COR1M-froth → ×GEX-shield. Honest Sharpe/maxDD/CVaR +
bootstrap-CI + P(stack>tide) + DSR (deflated, N-trial seçim-iyimserliği). @2019+ frozen pencere.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from spine import contract as C, tide as T              # noqa: E402
from backtest import engine as E                         # noqa: E402
from screen._util import bootstrap_ci, paired_win_prob   # noqa: E402
from modules.cor1m_froth import froth_factor_series      # noqa: E402
from modules.gex_shield import gex_zscore, shield_factor_series  # noqa: E402

# DSR seçim-iyimserliği trial sayısı. UYARI (GÖREV 5 denetimi): 60 = SADECE COR1M+GEX seçim-gridleri
# (~24 form). Seansın TÜM reddedilen aday-evreni (vol-surface/SKEW/VVIX/COR-term/RV/COT-5yol/breadth/
# re-entry/seasonality/flows ×2-yön ×tail) ~150-300 form → 60 ALT-SINIR, DSR'yi az düzeltir = İYİMSER.
# GERÇEK getiri serisiyle DSR(N): SPX 60→0.985 / 150→0.967 / 200→0.959 / 300→0.947; NDX 60→0.994 /
# 150→0.987 / 200→0.983. DÜRÜST OKUMA: N≈150-200 → DSR ~SPX 0.96 / NDX 0.98 (hâlâ >0.95 ama 0.985/0.994 DEĞİL).
N_TRIALS = 60
GAMMA = 0.5772156649
N_TRIALS_HONEST = (150, 200, 300)      # GÖREV 5: tam seçim-evreni duyarlılığı (main'de yazdırılır)


def _m(r):
    r = r.dropna()
    eq = (1+r).cumprod(); dd = float((eq/eq.cummax()-1).min())
    sh = float(r.mean()/r.std()*np.sqrt(252)); k = max(1, int(0.05*len(r)))
    return sh, dd, float(np.sort(r.values)[:k].mean()), float((r != 0).mean())


def strat_ret(pos, close, lag=1):
    idx = pos.index; ret = E.fwd_ret(close, idx).values; p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def dsr(r, n_trials=N_TRIALS):
    r = r.dropna().values
    if len(r) < 100 or r.std() == 0:
        return None
    sr = r.mean()/r.std(); n = len(r)
    sk = float(stats.skew(r)); ku = float(stats.kurtosis(r, fisher=False))
    # SR0 = beklenen-max Sharpe (günlük) N denemede; var_sr ~ (1/n) yaklaşık
    var_sr = (1.0/n)
    z1 = stats.norm.ppf(1-1.0/n_trials); z2 = stats.norm.ppf(1-1.0/(n_trials*math.e))
    sr0 = math.sqrt(var_sr) * ((1-GAMMA)*z1 + GAMMA*z2)
    denom = math.sqrt(max(1e-9, 1 - sk*sr + ((ku-1)/4)*sr*sr))
    return float(stats.norm.cdf((sr - sr0)*math.sqrt(n-1)/denom))


def main():
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    cor = pd.read_parquet(ROOT/"data"/"cache"/"corr_pc.parquet")["COR1M"].dropna()
    gex = pd.read_parquet(ROOT/"data"/"cache"/"squeeze_dix_gex.parquet")["gex"].dropna()
    froth = froth_factor_series(cor.reindex(idx, method="ffill"), 8, 11, 0.0)
    zg = gex_zscore(gex).reindex(idx, method="ffill")          # modules/gex_shield = tek kaynak (byte-aynı)
    gex_shield = shield_factor_series(zg, 0.5, 1.0, 0.4)

    stacks = {
        "tide (base)":              tdir,
        "tide × COR1M-froth":       tdir * froth,
        "tide × froth × GEX-shield": tdir * froth * gex_shield,
    }
    print(f"  KÜMÜLATİF STACK — frozen 2019-01..2026-05 (@m9-çağı; honest forward ~1.0-1.3)")
    print("=" * 96)
    for a in ("SPX", "NDX"):
        base_r = strat_ret(tdir, prices[a])
        print(f"\n  [{a}]")
        print(f"    {'katman':<26}{'Sharpe':>8}{'maxDD':>8}{'CVaR':>8}{'expo':>7}{'boot p5':>9}{'P>tide':>8}{'DSR':>7}")
        full_r = None
        for label, pos in stacks.items():
            r = strat_ret(pos.reindex(idx), prices[a])
            sh, dd, cv, ex = _m(r)
            ci = bootstrap_ci(r); wp = paired_win_prob(base_r, r)
            ds = dsr(r)
            print(f"    {label:<26}{sh:>+8.2f}{100*dd:>+7.0f}%{100*cv:>+7.2f}%{100*ex:>+6.0f}%"
                  f"{(ci['p5'] if ci['p5'] is not None else float('nan')):>+9.2f}"
                  f"{(f'{wp:.0%}' if wp is not None else '-'):>8}{(f'{ds:.3f}' if ds is not None else '-'):>7}")
            full_r = r
        # GÖREV 5: DÜRÜST DSR — tam seçim-evreni (N=60 alt-sınır iyimser)
        if full_r is not None:
            sens = "  ".join(f"N{N}:{dsr(full_r, N):.3f}" for N in (N_TRIALS,) + N_TRIALS_HONEST)
            print(f"    └ DSR(stack) seçim-evreni duyarlılığı: {sens}  (dürüst N≈150-200)")
    print("\n" + "=" * 96)
    print("  OKU: COR1M-froth katmanı Sharpe+maxDD iyileştirir (kanıtlı alfa); GEX-shield maxDD/CVaR ek-kalkan.")
    print("  CAVEAT: 2019+ m9-çağı tek-rejim; honest forward ~1.0-1.3. DSR N=60 İYİMSER (sadece COR1M+GEX grid);")
    print("          dürüst N≈150-200 (tüm reddedilen adaylar) → DSR ~SPX 0.96 / NDX 0.98, hâlâ >0.95.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
