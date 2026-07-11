"""Ortak CBOE GEX kütüphanesi (TEŞHİS — workflow ajanları paylaşır). CBOE ücretsiz delayed_quotes
zincirini çeker, satırlara ayırır, parametrik flip hesaplar. Variant = farklı row_filter / sign / flip-tanımı."""
from __future__ import annotations

import re
from datetime import date, datetime
from math import exp, log, pi, sqrt

import numpy as np
import requests

R, Q = 0.04, 0.013
PUT_WEIGHT = 1.3        # dealer put-overweight (vendor-match) — TEK-KAYNAK: gamma_engine + surface_yf bunu import eder (birbirinden sapamaz). Vendor'dan ±%0.5 kaçarsa screen/calibrate_pw.py ile yeniden-fit edip burayı güncelle.
SYM = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0) Chrome/120 Safari/537.36"}


def _npdf(x):
    return exp(-x * x / 2) / sqrt(2 * pi)


def bs_gamma(S, K, T, s):
    if T <= 0 or s <= 0 or S <= 0 or K <= 0:
        return 0.0
    a = s * sqrt(T)
    d1 = (log(S / K) + (R - Q + s * s / 2) * T) / a
    return exp(-Q * T) * _npdf(d1) / (S * a)


def _is_third_friday(d):
    return d.weekday() == 4 and 15 <= d.day <= 21


def fetch(sym, timeout=60):
    # P1-A (denetim 2026-07-07): tek-deneme → geçici timeout/5xx = gamma ledger'ın günü kaçar (time-decay,
    # geri gelmez). 3-deneme üstel-backoff transient'i yutar (tek CBOE vendor, yfinance-fallback 2026-06'da kalktı).
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json"
    try:
        from modules._netutil import http_get_retry
    except ImportError:                                   # _netutil yoksa (bağımsız çağrı) → tek-deneme
        r = requests.get(url, timeout=timeout, headers=UA)
        r.raise_for_status()
        return r.json()
    r = http_get_retry(url, timeout=timeout, headers=UA)  # 3-deneme backoff; hepsi patlarsa fail-loud (raise)
    return r.json()


def load_rows(sym, band=0.15, timeout=60):
    """-> (spot, rows). rows: K, cp, oi, iv(decimal), g_cboe, dte, T, expd, is_monthly."""
    j = fetch(sym, timeout)
    d = j.get("data", j)
    _sp = d.get("current_price") or d.get("close") or d.get("prev_day_close") or d.get("last_trade_price")
    spot = float(_sp) if _sp not in (None, "") else float("nan")   # None/bos -> nan (float(None) TypeError'i ENGELLE)
    if not np.isfinite(spot) or spot <= 0:   # NaN/inf/0/None spot (kismi/bozuk CBOE quote): sessiz-yanlis JUNK snapshot ENGELLE (caller try/except exit 1'e cevirir; bool(nan)==True 'or' zincirini gecer)
        raise ValueError(f"gecersiz/finite-olmayan spot ({sym}): {_sp}")
    today = date.today()
    rows = []
    for o in d.get("options", []):
        name = o.get("option") or o.get("symbol") or ""
        m = SYM.match(name)
        if not m:
            continue
        _, ymd, cp, strike8 = m.groups()
        try:
            expd = datetime.strptime(ymd, "%y%m%d").date()
        except ValueError:
            continue
        dte = (expd - today).days
        if dte < 0:
            continue
        K = int(strike8) / 1000.0
        if abs(K / spot - 1) > band:
            continue
        oi = o.get("open_interest")
        if oi is None or not np.isfinite(oi) or oi <= 0:
            continue
        iv = o.get("iv")
        iv = float(iv) if iv not in (None, "") else None
        if iv is not None and iv > 3.0:
            iv /= 100.0
        gc = o.get("gamma")
        rows.append({"K": K, "cp": cp, "oi": float(oi), "iv": iv,
                     "g_cboe": float(gc) if gc not in (None, "") else None,
                     "dte": dte, "T": max(dte, 0.5) / 365.0, "expd": expd,
                     "is_monthly": _is_third_friday(expd)})
    return spot, rows


def naive_sign(cp):
    return 1.0 if cp == "C" else -1.0


def flip_bs(rows, spot, row_filter=None, sign=naive_sign, lo=-0.15, hi=0.15, n=121, use_cboe_gamma=False):
    """BS-gamma (CBOE iv) ile sıfır-gamma flip taraması. use_cboe_gamma=True: sabit CBOE gamma ile
    strike-kümülatif sıfır-geçiş (re-pricing yok). -> (flip, net_at_spot)."""
    use = [r for r in rows if (r["iv"] or use_cboe_gamma) and (row_filter is None or row_filter(r))]
    if not use:
        return None, 0.0

    if use_cboe_gamma:
        # sabit CBOE gamma: per-strike net GEX, fiyat-eksende kümülatif sıfır-geçiş
        prof = {}
        for r in use:
            if r["g_cboe"] is None:
                continue
            prof[r["K"]] = prof.get(r["K"], 0.0) + sign(r["cp"]) * r["g_cboe"] * r["oi"] * 100 * spot * spot * 0.01
        ks = sorted(prof)
        cum, flip = 0.0, None
        prev_k = prev_c = None
        for k in ks:
            cum += prof[k]
            if prev_c is not None and ((prev_c <= 0 <= cum) or (prev_c >= 0 >= cum)):
                flip = prev_k + (k - prev_k) * (0 - prev_c) / (cum - prev_c) if cum != prev_c else k
                break
            prev_k, prev_c = k, cum
        net_spot = sum(p for p in prof.values())
        return (round(flip, 2) if flip else None), net_spot

    def net(hs):
        return sum(sign(r["cp"]) * bs_gamma(hs, r["K"], r["T"], r["iv"]) * r["oi"] * 100 * hs * hs * 0.01
                   for r in use)
    grid = [(spot * (1 + p), net(spot * (1 + p))) for p in np.linspace(lo, hi, n)]
    flip = None
    for (s0, g0), (s1, g1) in zip(grid, grid[1:]):
        if (g0 <= 0 <= g1) or (g0 >= 0 >= g1):
            flip = s0 + (s1 - s0) * (0 - g0) / (g1 - g0) if g1 != g0 else s0
            break
    return (round(flip, 2) if flip else None), net(spot)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    spot, rows = load_rows("_SPX")
    print(f"SELF-TEST _SPX  spot={spot:.2f}  rows={len(rows)}  monthly-rows={sum(1 for r in rows if r['is_monthly'])}")
    f, ns = flip_bs(rows, spot)
    print(f"  baseline(all,naive,BS)  flip={f}  net@spot={ns/1e9:+.2f}B  regime={'POS' if spot>=f else 'NEG'}")
    fm, _ = flip_bs(rows, spot, row_filter=lambda r: r["is_monthly"])
    print(f"  monthly-only            flip={fm}  regime={'POS' if fm and spot>=fm else 'NEG'}")
    fc, nc = flip_bs(rows, spot, use_cboe_gamma=True)
    print(f"  cboe-gamma cumulative   flip={fc}  net@spot={nc/1e9:+.2f}B")
