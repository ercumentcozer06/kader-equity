"""
backtest/karsan/k_stats.py — paylaşılan istatistik: circular block bootstrap, BH-FDR, DSR.
Tüm Faz-1 testleri bunu kullanır (determinizm: seed k_config.SEED).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from math import erf, sqrt, log, e as E_CONST
import k_config as K

GAMMA = 0.5772156649015329


def _ncdf(x): return 0.5 * (1 + erf(x / sqrt(2)))


def circ_block_indices(n, block, n_boot, rng):
    """Circular block bootstrap satır-index matrisi (n_boot × n)."""
    nblocks = int(np.ceil(n / block))
    out = np.empty((n_boot, nblocks * block), dtype=int)
    starts = rng.integers(0, n, size=(n_boot, nblocks))
    for b in range(block):
        out[:, b::block] = (starts + b) % n
    return out[:, :n]


def mean_diff_boot(values: np.ndarray, mask: np.ndarray, rng, block=None, n_boot=None):
    """Statistic = mean(values[mask]) − mean(values[~mask]); circular block bootstrap SE/t/p/CI."""
    block = block or K.BOOT_BLOCK; n_boot = n_boot or K.BOOT_N
    v = np.asarray(values, float); m = np.asarray(mask, bool)
    ok = ~np.isnan(v)
    v, m = v[ok], m[ok]
    n = len(v)
    obs = v[m].mean() - v[~m].mean()
    idx = circ_block_indices(n, block, n_boot, rng)
    vb = v[idx]; mb = m[idx]
    # her replikasyonda mean(in)-mean(out)
    sums_in = np.where(mb, vb, 0).sum(1); cnt_in = mb.sum(1)
    sums_out = np.where(~mb, vb, 0).sum(1); cnt_out = (~mb).sum(1)
    good = (cnt_in > 0) & (cnt_out > 0)
    stat = sums_in[good] / cnt_in[good] - sums_out[good] / cnt_out[good]
    se = float(stat.std())
    t = float(obs / se) if se > 0 else 0.0
    p = float(2 * (1 - _ncdf(abs(t))))
    return {"obs": float(obs), "se": se, "t": round(t, 3), "p": p,
            "ci5": float(np.percentile(stat, 2.5)), "ci95": float(np.percentile(stat, 97.5)),
            "n_in": int(m.sum()), "n_out": int((~m).sum())}


def one_sample_block_boot(x: np.ndarray, rng, block=None, n_boot=None):
    """1-D seri ortalaması H0=0 testi, circular block bootstrap (otokorelasyon-dirençli)."""
    block = block or K.BOOT_BLOCK; n_boot = n_boot or K.BOOT_N
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x); obs = float(x.mean())
    idx = circ_block_indices(n, block, n_boot, rng)
    stat = x[idx].mean(1)
    se = float(stat.std()); t = float(obs / se) if se > 0 else 0.0
    return {"obs": obs, "se": se, "t": round(t, 3), "p": float(2 * (1 - _ncdf(abs(t)))),
            "ci5": float(np.percentile(stat, 2.5)), "ci95": float(np.percentile(stat, 97.5)), "n": n}


def paired_event_boot(deltas: np.ndarray, rng, n_boot=None):
    """Olay-bazlı (FOMC vb.) ortalama testi: H0 mean=0. Basit resample (olaylar ~bağımsız)."""
    n_boot = n_boot or K.BOOT_N
    d = np.asarray(deltas, float); d = d[~np.isnan(d)]
    n = len(d); obs = float(d.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    stat = d[idx].mean(1)
    se = float(stat.std()); t = float(obs / se) if se > 0 else 0.0
    return {"obs": obs, "se": se, "t": round(t, 3), "p": float(2 * (1 - _ncdf(abs(t)))),
            "ci5": float(np.percentile(stat, 2.5)), "ci95": float(np.percentile(stat, 97.5)), "n": n}


def ols_coef_boot(y: np.ndarray, X: np.ndarray, names, rng, block=None, n_boot=None):
    """OLS β + circular block bootstrap (otokorelasyon-dirençli) coefficient CI/t/p."""
    block = block or K.BOOT_BLOCK; n_boot = n_boot or K.BOOT_N
    y = np.asarray(y, float); X = np.asarray(X, float)
    ok = ~(np.isnan(y) | np.isnan(X).any(1))
    y, X = y[ok], X[ok]
    n = len(y)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    r2 = 1 - (resid @ resid) / (((y - y.mean()) ** 2).sum())
    idx = circ_block_indices(n, block, n_boot, rng)
    boot = np.empty((n_boot, X.shape[1]))
    for i in range(n_boot):
        ii = idx[i]
        boot[i] = np.linalg.lstsq(X[ii], y[ii], rcond=None)[0]
    out = {"r2": round(float(r2), 4), "n": n, "coef": {}}
    for j, nm in enumerate(names):
        b = float(beta[j]); se = float(boot[:, j].std()); t = b / se if se > 0 else 0.0
        out["coef"][nm] = {"beta": b, "se": se, "t": round(t, 3), "p": float(2 * (1 - _ncdf(abs(t)))),
                           "ci5": float(np.percentile(boot[:, j], 2.5)), "ci95": float(np.percentile(boot[:, j], 97.5))}
    return out


def bh_fdr(pvals, alpha=None):
    """Benjamini-Hochberg. Döndürür: adjusted p (aynı sırada) + reject bool."""
    alpha = alpha or K.FDR_ALPHA
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); ranked = p[order]
    adj = ranked * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]          # monoton
    adj = np.clip(adj, 0, 1)
    out_adj = np.empty(m); out_adj[order] = adj
    return out_adj


def dsr(sharpe_daily, n, n_trials, skew=0.0, kurt=3.0):
    """Deflated Sharpe (Bailey-LdP). sharpe_daily = günlük SR. Döndürür DSR olasılığı."""
    if n < 30:
        return None
    var_sr = (1 - skew * sharpe_daily + (kurt - 1) / 4 * sharpe_daily ** 2) / (n - 1)
    if var_sr <= 0:
        return None
    from math import sqrt as _s
    # beklenen max SR, N denemede
    z1 = _ppf(1 - 1.0 / n_trials); z2 = _ppf(1 - 1.0 / (n_trials * E_CONST))
    sr0 = _s((1) / (n - 1)) * ((1 - GAMMA) * z1 + GAMMA * z2)
    return round(float(_ncdf((sharpe_daily - sr0) / _s(var_sr))), 4)


def _ppf(p):
    """Acklam Φ⁻¹."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = sqrt(-2 * log(p)); return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - pl:
        return -_ppf(1 - p)
    q = p - 0.5; r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def garman_klass(o, h, l, c):
    """Günlük GK varyans → günlük vol (annualize edilmemiş). Seri döndürür."""
    o, h, l, c = map(lambda x: np.asarray(x, float), (o, h, l, c))
    with np.errstate(divide="ignore", invalid="ignore"):
        gk = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    gk = np.where(gk < 0, np.nan, gk)
    return np.sqrt(gk)
