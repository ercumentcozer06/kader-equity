"""Vectorised Black-Scholes Greeks used by the option research archive.

Volatility is a decimal (0.20 = 20%).  Vega/vanna/vomma/ultima derivatives are
per 1.00 volatility change.  ``charm``, ``color`` and ``veta`` are calendar-time
derivatives: one year passing while expiry stays fixed.  Divide them by 365 for
an approximate one-calendar-day effect.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr


EPS = 1e-12


def _core(spot, strike, t, vol, rate, dividend, is_call):
    s, k, tau, sig = np.broadcast_arrays(
        np.asarray(spot, float), np.asarray(strike, float),
        np.asarray(t, float), np.asarray(vol, float))
    call = np.broadcast_to(np.asarray(is_call, bool), s.shape)
    valid = (s > 0) & (k > 0) & (tau > 0) & (sig > 0) & np.isfinite(s + k + tau + sig)
    tt = np.where(valid, tau, 1.0)
    vv = np.where(valid, sig, 1.0)
    ss = np.where(valid, s, 1.0)
    kk = np.where(valid, k, 1.0)
    root = np.sqrt(tt)
    d1 = (np.log(ss / kk) + (rate - dividend + 0.5 * vv * vv) * tt) / (vv * root)
    d2 = d1 - vv * root
    pdf = np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)
    dq, dr = np.exp(-dividend * tt), np.exp(-rate * tt)
    call_price = ss * dq * ndtr(d1) - kk * dr * ndtr(d2)
    put_price = kk * dr * ndtr(-d2) - ss * dq * ndtr(-d1)
    price = np.where(call, call_price, put_price)
    delta = np.where(call, dq * ndtr(d1), dq * (ndtr(d1) - 1.0))
    gamma = dq * pdf / (ss * vv * root)
    vega = ss * dq * pdf * root
    nan = np.full(s.shape, np.nan)
    return tuple(np.where(valid, x, nan) for x in (price, delta, gamma, vega, d1, d2))


def all_greeks(spot, strike, t, vol, is_call, rate: float = 0.04, dividend: float = 0.013):
    """Return first and higher-order Greeks as equally shaped numpy arrays."""
    price, delta, gamma, vega, d1, d2 = _core(spot, strike, t, vol, rate, dividend, is_call)
    s = np.asarray(spot, float)
    tau = np.asarray(t, float)
    sig = np.asarray(vol, float)
    root = np.sqrt(tau)

    vanna = -np.exp(-dividend * tau) * np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) * d2 / sig
    vomma = vega * d1 * d2 / sig
    speed = -gamma / s * (1.0 + d1 / (sig * root))
    zomma = gamma * (d1 * d2 - 1.0) / sig
    ultima = -(vega / (sig * sig)) * (
        d1 * d2 * (1.0 - d1 * d2) + d1 * d1 + d2 * d2)

    # Time sensitivities are evaluated numerically from analytical first Greeks.
    # This avoids fragile closed-form sign conventions and remains stable at 0DTE.
    h = np.maximum(np.minimum(1.0 / 3650.0, tau * 0.25), 1.0 / (365.0 * 24.0 * 60.0))
    tp, tm = tau + h, np.maximum(tau - h, 1.0 / (365.0 * 24.0 * 60.0))
    _, dp, gp, vp, _, _ = _core(s, strike, tp, sig, rate, dividend, is_call)
    _, dm, gm, vm, _, _ = _core(s, strike, tm, sig, rate, dividend, is_call)
    denom = tp - tm
    charm = np.full(np.broadcast(s, tau).shape, np.nan, dtype=float)
    color = np.full_like(charm, np.nan)
    veta = np.full_like(charm, np.nan)
    np.divide(-(dp - dm), denom, out=charm, where=denom > 0)
    np.divide(-(gp - gm), denom, out=color, where=denom > 0)
    np.divide(-(vp - vm), denom, out=veta, where=denom > 0)

    return {
        "bs_price": price, "bs_delta": delta, "bs_gamma": gamma,
        "bs_vega_per_1vol": vega, "vanna_per_1vol": vanna,
        "charm_per_year": charm, "vomma_per_1vol2": vomma,
        "veta_per_year_per_1vol": veta, "speed": speed,
        "color_per_year": color, "zomma_per_1vol": zomma,
        "ultima_per_1vol3": ultima,
    }
