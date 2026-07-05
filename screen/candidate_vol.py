"""
screen/candidate_vol — wave-1 ADAY #1: VIX term-structure (VIX/VIX3M backwardation).

AŞAMA 1 (standalone, full-history 2007+): -z(ts_ratio) sinyali → SPX/NQ forward-getiri IC (spearman,
1d+21d) + hit-rate + standalone long/flat Sharpe. 8 vol-episode kapsar.
AŞAMA 2 (incremental, 2019+ tide ömrü): base = tide_dir; variant = backwardation'da tide-long'u KES
(vol-gate). Sharpe + maxDD + CVaR base-vs-variant + paired block-bootstrap P(var>base) + episode
round-trip (giriş+çıkış). PRE-REGISTER: primer = incremental Sharpe; sekonder (shield) = maxDD/CVaR
iyileşme @ Sharpe≥0. Asimetrik-rebound (crisis_emergency_gate dersi) episode round-trip'te görünür.

Araştırma aracı (kader-macro dl + walkforward import'u OK — runtime değil).
  & <kader-macro venv python> screen\candidate_vol.py
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
from screen._util import load_price_csv, paired_win_prob  # noqa: E402  (vendored — kader-macro import YOK)

DESK = Path(r"C:\Users\admin\Desktop\backtesting")
PRICES = {"SPX": "SPX_daily.csv", "NDX": "NASDAQ_daily.csv"}
EPISODES_TIDE = {"2020-COVID": ("2020-02-01", "2020-06-30"), "2022-bear": ("2022-01-01", "2022-12-31"),
                 "2023-SVB": ("2023-03-01", "2023-05-31"), "2024-Aug": ("2024-07-15", "2024-09-15")}
EPISODES_STANDALONE = {"2010-flash": ("2010-04-01", "2010-07-31"), "2011-USdg": ("2011-07-01", "2011-10-31"),
                       "2015-Aug": ("2015-08-01", "2015-10-31"), "2018-Q4": ("2018-10-01", "2018-12-31")}


def z(s: pd.Series, win: int = 252) -> pd.Series:
    return (s - s.rolling(win, min_periods=60).mean()) / s.rolling(win, min_periods=60).std()


def strat_ret(pos: pd.Series, close: pd.Series, lag: int = 1) -> pd.Series:
    idx = pos.index
    ret = E.fwd_ret(close, idx).values
    p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def _sh(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")


def _maxdd(r: pd.Series) -> float:
    eq = (1 + r.dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min())


def _cvar(r: pd.Series, q: float = 0.05) -> float:
    r = r.dropna().values
    k = max(1, int(q * len(r)))
    return float(np.sort(r)[:k].mean())


def stage1(ts: pd.Series) -> None:
    print("\n" + "=" * 92)
    print("  AŞAMA 1 — STANDALONE (full-history 2007+):  sinyal = -z(VIX/VIX3M)  (backwardation → bearish)")
    print("=" * 92)
    print(f"  {'asset':<7}{'IC_1d':>8}{'IC_21d':>8}{'hit_1d':>8}{'standalone Sh':>15}{'  episode Sh (giriş+çıkış)':<40}")
    sig_full = -z(ts)
    for a, fn in PRICES.items():
        close = load_price_csv(DESK / fn)
        idx = sig_full.dropna().index.intersection(close.index)
        sig = sig_full.reindex(idx)
        f1 = E.fwd_ret(close, idx)
        f21 = (close.reindex(idx, method="ffill").shift(-21) / close.reindex(idx, method="ffill") - 1)
        ic1 = sig.corr(f1, method="spearman")
        ic21 = sig.corr(f21, method="spearman")
        hit = float((np.sign(sig) == np.sign(f1)).mean())
        pos = (sig > 0).astype(float)                    # long when contango (calm)
        sh = _sh(strat_ret(pos, close))
        eps = []
        for name, (s, e) in EPISODES_STANDALONE.items():
            rr = strat_ret(pos, close)
            rr = rr[(rr.index >= s) & (rr.index <= e)]
            eps.append(f"{name}:{_sh(rr):+.1f}")
        print(f"  {a:<7}{ic1:>+8.3f}{ic21:>+8.3f}{hit:>+8.2f}{sh:>+15.2f}   {'  '.join(eps)}")


def stage2(ts: pd.Series) -> None:
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    ts_t = ts.reindex(idx, method="ffill")
    zt = z(ts).reindex(idx, method="ffill")

    # vol-gate varyantları (backwardation'da tide-long'u kes/kıs)
    variants = {
        "gate ts>1.0": (tdir * (ts_t <= 1.0).astype(float)),
        "gate ts>1.05": (tdir * (ts_t <= 1.05).astype(float)),
        "gate z>1.5": (tdir * (zt <= 1.5).astype(float)),
        "trim z>1 (soft)": (tdir * (1.0 - 0.5 * np.clip(zt - 1.0, 0, 2) / 2.0).clip(0, 1)),
    }

    print("\n" + "=" * 92)
    print("  AŞAMA 2 — INCREMENTAL over TIDE (2019+):  base = tide_dir,  variant = backwardation'da long-kes")
    print("  PRE-REG: primer = ΔSharpe(OOS) ; sekonder (shield) = maxDD/CVaR iyileşme @ Sharpe≥0")
    print("=" * 92)
    for a in ("SPX", "NDX"):
        close = prices[a]
        base = strat_ret(tdir, close)
        b_sh, b_dd, b_cv = _sh(base), _maxdd(base), _cvar(base)
        print(f"\n  [{a}]  base tide: Sharpe {b_sh:+.3f}  maxDD {100*b_dd:+.0f}%  CVaR {100*b_cv:+.2f}%")
        print(f"    {'variant':<18}{'Sharpe':>8}{'ΔSh':>7}{'maxDD':>8}{'ΔmaxDD':>8}{'CVaR':>8}{'P(var>base)':>12}   {'COVID rt':>9}")
        for label, vpos in variants.items():
            var = strat_ret(vpos.reindex(idx), close)
            v_sh, v_dd, v_cv = _sh(var), _maxdd(var), _cvar(var)
            wp = paired_win_prob(base, var)
            # COVID round-trip (giriş+çıkış): episode Sharpe of the variant vs base
            cv_b = base[(base.index >= "2020-02-01") & (base.index <= "2020-06-30")]
            cv_v = var[(var.index >= "2020-02-01") & (var.index <= "2020-06-30")]
            rt = f"{_sh(cv_v):+.1f}v{_sh(cv_b):+.1f}"
            print(f"    {label:<18}{v_sh:>+8.3f}{v_sh-b_sh:>+7.2f}{100*v_dd:>+7.0f}%{100*(v_dd-b_dd):>+7.0f}%"
                  f"{100*v_cv:>+7.2f}%{(f'{wp:.0%}' if wp is not None else 'n/a'):>12}   {rt:>9}")


def main() -> int:
    vs = pd.read_parquet(ROOT / "data" / "cache" / "vol_surface.parquet")
    ts = vs["ts_ratio"].dropna()
    print(f"  vol_surface: {len(ts)} gün {ts.index.min().date()}..{ts.index.max().date()}")
    stage1(ts)
    stage2(ts)
    print("\n  OKU: ΔSharpe>0 + P(var>base)>~60% = incremental ödüyor. ΔSharpe~0 ama maxDD/CVaR iyi +")
    print("       COVID-rt variant>base = drawdown-shield (sekonder PASS). İkisi de yoksa = AT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
