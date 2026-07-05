"""
engine — kader-equity GÜNLÜK backtest çekirdeği (kader-btc haftalık değil; tide günlük + 1g lag).

Math sweep4000 ile birebir: sinyal[t] → t+1 close getirisi (look-ahead-free), pozisyon = LONG/FLAT.
İstatistik: Sharpe (ann √252, ddof=0) + total + maxDD + CVaR(5%) + exposure. CVaR downstream
drawdown-shield katmanlarının ön-kayıtlı sekonder ölçütü için (tail loss).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = np.sqrt(252)


def fwd_ret(close: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """signal[t] earns t→t+1 (look-ahead-free). Son gün NaN (gelecek yok)."""
    cb = close.reindex(idx, method="ffill")
    return (cb.shift(-1) / cb - 1.0)


def _one(x: np.ndarray, mask: np.ndarray, cvar_q: float = 0.05) -> dict | None:
    fin = np.isfinite(x) & mask
    n = int(fin.sum())
    if n < 30:
        return None
    xx = x[fin]
    mu, sd = xx.mean(), xx.std()
    eq = np.cumprod(1.0 + xx)
    dd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    k = max(1, int(cvar_q * n))
    cvar = float(np.sort(xx)[:k].mean())                       # mean of worst q% days (expected shortfall)
    return {"sharpe": float(mu / sd * ANN) if sd > 0 else float("nan"),
            "total": float(eq[-1] - 1.0), "maxdd": dd, "cvar": cvar, "n": n}


def stats_from_pos(idx: pd.DatetimeIndex, pos: np.ndarray, ret: np.ndarray, *,
                   start=None, end=None, slippage: float = 0.0) -> dict:
    """pos (0/1, lag UYGULANMIŞ) + ret (next-day) → strat/bh/expo. start/end dilimler."""
    mask = np.ones(len(idx), bool)
    if start is not None:
        mask &= idx >= pd.Timestamp(start)
    if end is not None:
        mask &= idx <= pd.Timestamp(end)
    sret = pos * ret
    if slippage:
        turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
        sret = sret - turn * slippage
    return {"strat": _one(sret, mask), "bh": _one(ret, mask),
            "expo": float(pos[mask].mean()) if mask.any() else float("nan")}


def backtest_dir(tide_dir: pd.Series, close: pd.Series, *, lag: int = 1,
                 start=None, end=None, slippage: float = 0.0) -> dict:
    """Günlük LONG/FLAT yön serisini PnL'e çevir. lag = exec gecikmesi (default +1g)."""
    idx = tide_dir.index
    ret = fwd_ret(close, idx).values
    pos = tide_dir.astype(float).values
    if lag:
        pos = np.concatenate([np.zeros(lag), pos[:-lag]])
    return stats_from_pos(idx, pos, ret, start=start, end=end, slippage=slippage)


def backtest_overlay(tide_dir: pd.Series, overlay_pos: pd.Series, close: pd.Series, *,
                     lag: int = 1, start=None, end=None) -> dict:
    """Overlay'li pozisyon: overlay_pos = nihai 0..1 pozisyon (tide × trim/gate), zaten hizalı.
    Incremental screen burayı kullanır (base = backtest_dir, variant = backtest_overlay)."""
    idx = tide_dir.index
    ret = fwd_ret(close, idx).values
    pos = overlay_pos.reindex(idx).astype(float).values
    if lag:
        pos = np.concatenate([np.zeros(lag), pos[:-lag]])
    return stats_from_pos(idx, pos, ret, start=start, end=end)
