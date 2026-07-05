"""
screen/_util — vendored research utilities (kader-equity self-contained; kader-macro backtest paketini
import ETMEZ → 'backtest' isim-çakışması yok).

  • load_price_csv  : yfinance çok-başlıklı CSV (Price/Ticker/Date + veri) → günlük Close Series
  • paired_win_prob : stationary block-bootstrap P(variant Sharpe > base Sharpe) (kader-macro walkforward'dan)
  • bootstrap_ci    : tek-seri Sharpe bootstrap CI
  • fdr_bh          : Benjamini-Hochberg (çoklu-test düzeltmesi, {SPX,NDX} ailesi)
Math kader-macro/backtest/walkforward.py ile BİREBİR (B=5000, L=63, SEED=77).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

B_BOOT = 5000
L_BLOCK = 63
SEED = 77


def load_price_csv(path) -> pd.Series:
    """yfinance CSV (satır0 Price,Close,..; satır1 Ticker,..; satır2 Date,,; veri satır3+) → Close Series."""
    raw = pd.read_csv(path, skiprows=3, header=None)
    dates = pd.to_datetime(raw.iloc[:, 0])
    close = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
    s = pd.Series(close.values, index=dates).dropna()
    return s[~s.index.duplicated(keep="first")].sort_index()


def _block_starts(n: int, rng) -> np.ndarray:
    nb = max(1, n // L_BLOCK)
    return rng.integers(0, max(1, n - L_BLOCK), size=(B_BOOT, nb))


def _boot_sharpe(s: np.ndarray, starts: np.ndarray) -> np.ndarray:
    offs = np.arange(L_BLOCK)
    idx = (starts[:, :, None] + offs[None, None, :]).reshape(starts.shape[0], -1)
    idx = np.clip(idx, 0, len(s) - 1)
    x = s[idx]
    mu, sd = x.mean(axis=1), x.std(axis=1)
    return np.where(sd > 0, mu / sd * np.sqrt(252), np.nan)


def bootstrap_ci(strat: pd.Series) -> dict:
    s = strat.dropna().values
    if len(s) < L_BLOCK * 2:
        return {"mean": None, "p5": None, "p95": None}
    rng = np.random.default_rng(SEED)
    d = _boot_sharpe(s, _block_starts(len(s), rng))
    return {"mean": round(float(np.nanmean(d)), 2), "p5": round(float(np.nanpercentile(d, 5)), 2),
            "p95": round(float(np.nanpercentile(d, 95)), 2)}


def paired_win_prob(strat_base: pd.Series, strat_var: pd.Series) -> float | None:
    """Paired block-bootstrap: P(variant Sharpe > base Sharpe). Aynı blok-çekimleri ikisine de (paired)."""
    df = pd.concat([strat_base.rename("b"), strat_var.rename("v")], axis=1).dropna()
    if len(df) < L_BLOCK * 2:
        return None
    rng = np.random.default_rng(SEED)
    starts = _block_starts(len(df), rng)
    sb = _boot_sharpe(df["b"].values, starts)
    sv = _boot_sharpe(df["v"].values, starts)
    return float(np.nanmean(sv > sb))


def fdr_bh(pvals: dict, alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg: hangi per-asset tek-yönlü testler FDR alpha'da geçer. pval = 1 - P(var>base)."""
    items = [(k, v) for k, v in pvals.items() if v is not None]
    m = len(items)
    if m == 0:
        return {}
    items.sort(key=lambda kv: kv[1])
    thresh_rank = 0
    for i, (k, p) in enumerate(items, start=1):
        if p <= (i / m) * alpha:
            thresh_rank = i
    return {k: (i <= thresh_rank) for i, (k, p) in enumerate(items, start=1)}
