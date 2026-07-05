"""
backtest/gex_price_test — GEX-yönü FİYATA karşı sınama (hakem = gerçekleşen fiyat, SqueezeMetrics DEĞİL).
GEX çerçevesinin temel iddiası: negatif-gamma (kısa) gün → hareket büyür (oynaklık↑); pozitif-gamma (uzun)
gün → fiyat çivilenir (oynaklık↓). Bunu elimizdeki gerçek+PIT veriyle test eder. PIT: gamma EOD(t) → fiyat t+1.

Veri: MarketData OI (243g) + mid → benim-GEX (±%5/DTE≥1, kaba dealer-sign) ; Alpaca 1-dk bar → gerçekleşen
intraday oynaklık. KARŞILAŞTIRMA: benim-yön VS squeeze-yön, ikisi de t+1 oynaklığını öngörüyor mu (hangisi iyi).
  & <venv python> backtest/gex_price_test.py
"""
from __future__ import annotations

import sys
from math import exp, log, pi, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "screen"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from _bsiv import implied_vol   # noqa: E402

R, Q = 0.04, 0.013


def _bsg(S, K, T, s):
    if T <= 0 or s <= 0 or S <= 0:
        return 0.0
    d1 = (log(S/K) + (R-Q+s*s/2)*T) / (s*sqrt(T))
    return exp(-Q*T) * (exp(-d1*d1/2)/sqrt(2*pi)) / (S*s*sqrt(T))


def my_gex_series(sym="SPY", band=0.05, dte_min=1):
    bars = pd.read_parquet(ROOT/"data"/"historical_bars"/f"alpaca_{sym.lower()}_1m.parquet")
    ts = pd.to_datetime(bars.index.get_level_values(-1))
    spot = pd.Series(bars["close"].values, index=ts).resample("1D").last().dropna(); spot.index = spot.index.date
    ch = pd.read_parquet(ROOT/"data"/"historical_chains"/f"md_{sym.lower()}.parquet")
    out = {}
    for d, g in ch.groupby("date"):
        dd = pd.Timestamp(d).date()
        if dd not in spot.index:
            continue
        S = float(spot[dd]); tot = 0.0
        for _, r in g.iterrows():
            K, oi, mid = r["strike"], r["open_interest"], r["mid"]
            if not K or not oi or not mid or mid <= 0 or abs(K/S-1) > band:
                continue
            dte = (pd.Timestamp(r["expiration"]) - pd.Timestamp(d)).days
            if dte < dte_min:
                continue
            T = max(dte, 0.5)/365.0; iv = implied_vol(mid, S, float(K), T, r["right"])
            if not iv:
                continue
            tot += (1 if r["right"] == "C" else -1) * _bsg(S, float(K), T, iv) * oi * 100 * S*S*0.01
        out[dd] = tot/1e9
    s = pd.Series(out).sort_index(); s.index = pd.to_datetime(s.index); return s


def realized_vol_series(sym="SPY"):
    bars = pd.read_parquet(ROOT/"data"/"historical_bars"/f"alpaca_{sym.lower()}_1m.parquet")
    ts = pd.to_datetime(bars.index.get_level_values(-1))
    px = pd.Series(bars["close"].values, index=ts)
    lr = np.log(px / px.shift(1))
    day = pd.Series(ts.date, index=ts.values)
    rv = lr.groupby(day.values).std() * sqrt(390)        # gün-içi 1-dk getiri std × √390 = günlük gerçekleşen oynaklık
    rv.index = pd.to_datetime(rv.index); return rv.dropna()


def _test(name, gex, rvol):
    """gamma(t) → rvol(t+1) [PIT] + eşzamanlı rvol(t). Negatif-gamma → oynaklık↑ beklenir."""
    df = pd.DataFrame({"gex": gex, "rvol": rvol}).dropna()
    df["rvol_next"] = df["rvol"].shift(-1)
    df = df.dropna()
    z = (df["gex"] - df["gex"].mean()) / df["gex"].std()
    neg, pos = df[df["gex"] < 0], df[df["gex"] >= 0]
    print(f"  [{name}]  n={len(df)}  (negatif-γ {len(neg)}g / pozitif-γ {len(pos)}g)")
    print(f"    EŞZAMANLI: neg-γ oynaklık {100*neg['rvol'].mean():.1f}% vs pos-γ {100*pos['rvol'].mean():.1f}%  "
          f"(neg>pos beklenir) | korr(gex,rvol) {df['gex'].corr(df['rvol']):+.2f}")
    print(f"    PIT t+1 : neg-γ→ertesi {100*neg['rvol_next'].mean():.1f}% vs pos-γ→ertesi {100*pos['rvol_next'].mean():.1f}%  "
          f"| korr(gex_t, rvol_t+1) {df['gex'].corr(df['rvol_next']):+.2f}")


def main():
    print("=" * 92)
    print("  GEX-YÖNÜ FİYAT TESTİ — hakem gerçekleşen oynaklık (SqueezeMetrics DEĞİL). negatif-γ→oynaklık↑ iddiası")
    print("=" * 92)
    rv = realized_vol_series("SPY")
    print(f"  SPY gerçekleşen oynaklık: {len(rv)}g, ort {100*rv.mean():.1f}% (yıllık)")
    mg = my_gex_series("SPY")
    _test("BENİM-GEX (kaba dealer-sign)", mg, rv)
    sq = pd.read_parquet(ROOT/"data"/"cache"/"squeeze_dix_gex.parquet")["gex"].dropna()
    sq.index = pd.to_datetime(sq.index)
    sq = sq[sq.index >= mg.index.min()]
    _test("SQUEEZE-GEX (profesyonel)", sq, rv)
    print("=" * 92)
    print("  OKU: korr NEGATİF + neg-γ-oynaklık > pos-γ → yön fiyatı öngörüyor (yöntem ne olursa olsun). İkisi de")
    print("  pozitif/sıfır → GEX-yönü bu pencerede oynaklığı öngörmüyor (free ya da paralı fark etmez).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
