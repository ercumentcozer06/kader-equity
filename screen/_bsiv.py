"""
screen/_bsiv — BS fiyat + bid/ask MID'den IV ters-çevirme (TEK KANONİK KAYNAK).

NEDEN: yfinance'in `impliedVolatility` alanı güvenilmez (intraday ~%3 gibi saçma değerler dönebiliyor) →
her kontratın bid/ask MID fiyatından Black-Scholes bisection ile IV hesapla. gamma_engine + surface_yf +
viz_surface AYNI bu fonksiyonu kullanır → motorun ürettiği GEX/flip/wall'lar ile görselin IV'si byte-aynı.

Sabitler R/Q gamma_engine/surface_yf/viz_surface ile aynı (R=0.04 risk-free, Q=0.013 SPY/QQQ temettü).
Geçersiz girdi (bid/ask yok ya da ≤0, T≤0) → None döner; ASLA ham yahoo IV'sine geri düşmez (sessiz kirlilik yasak).
"""
from __future__ import annotations

from math import erf, exp, log, sqrt

R, Q = 0.04, 0.013


def _nd(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2)))


def bs_price(S: float, K: float, T: float, s: float, right: str) -> float:
    """Black-Scholes (temettülü) opsiyon fiyatı. viz_surface.bs_price ile byte-aynı."""
    if T <= 0 or s <= 0:
        return max(0.0, (S - K) if right == "C" else (K - S))
    d1 = (log(S / K) + (R - Q + s * s / 2) * T) / (s * sqrt(T))
    d2 = d1 - s * sqrt(T)
    if right == "C":
        return S * exp(-Q * T) * _nd(d1) - K * exp(-R * T) * _nd(d2)
    return K * exp(-R * T) * _nd(-d2) - S * exp(-Q * T) * _nd(-d1)


def implied_vol(price: float | None, S: float, K: float, T: float, right: str) -> float | None:
    """MID fiyattan BS IV (60-iter bisection, viz_surface ile byte-aynı). Geçersiz/aralık-dışı → None."""
    if price is None or price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    lo, hi = 0.01, 3.0
    for _ in range(60):
        m = (lo + hi) / 2
        if bs_price(S, K, T, m, right) > price:
            hi = m
        else:
            lo = m
    iv = (lo + hi) / 2
    return iv if 0.02 < iv < 2.9 else None


def mid_iv_from_row(r, S: float, K: float, T: float, right: str) -> float | None:
    """yfinance opsiyon-zinciri satırından bid/ask MID → BS IV. bid/ask yok ya da ≤0 → None (ham IV'ye DÜŞMEZ)."""
    bid, ask = r.get("bid"), r.get("ask")
    if not bid or not ask or bid <= 0 or ask <= 0:
        return None
    return implied_vol((bid + ask) / 2.0, S, float(K), T, right)
