"""
backtest/build_level_series — tarihsel EOD OI → günlük gamma-seviye serisi (gamma_engine ile BYTE-EŞ formüller).
Directional GEX kalibrasyonunun TEMEL TAŞI (spec'in 'load_data()' seviye-bacağı). Her tarih için tek-front-expiry
(MarketData free derinliği: tarih başına 1 expiry, DTE 0-25) zincirinden üretir:
  net_gex (+işaret=rejim), flip (zero-gamma scan), ghost (gamma-tepe), call_wall (ham-OI), put_wall (gamma),
  hvl (|gamma×OI| tepe), max_pain, atm_iv, em1 (1D exp-move).
Spot = Alpaca günlük kapanış. IV = mid'den BS-invert (_bsiv). Greeks gamma_engine._greeks'ten (byte-eş).

DÜRÜST SINIR: front-expiry-ONLY (canlı motor N_EXP=5) → seviyeler canlıyla term-structure açısından TUTARSIZ olabilir;
forward'da front-expiry-matched motorla yeniden-doğrula. Front-expiry = intraday-gamma'nın baskın kısmı (price-test ✓).
LOOK-AHEAD: bu seviyeler D-EOD'de hesaplanır, D+1 seansında kullanılır (setup motoru bunu uygular).
  & <venv python> backtest/build_level_series.py
→ data/cache/level_series_{spy,qqq}.parquet
"""
from __future__ import annotations

import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "screen"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from _bsiv import implied_vol                  # noqa: E402
from gamma_engine import _greeks               # noqa: E402  (byte-eş greeks — tek kaynak)

M = 100
BAND = 0.15                                     # gamma_engine ile aynı strike bandı
SCAN = np.linspace(-0.06, 0.06, 13)             # gamma_engine flip-scan ile aynı


def _daily_spot(sym):
    bars = pd.read_parquet(ROOT / "data" / "historical_bars" / f"alpaca_{sym.lower()}_1m.parquet")
    ts = pd.to_datetime(bars.index.get_level_values(-1))
    s = pd.Series(bars["close"].values, index=ts).resample("1D").last().dropna()
    s.index = s.index.date
    return s


def _levels_for_day(g, S):
    """Tek tarih (tek expiry) zinciri + spot → seviye sözlüğü. gamma_engine main() ile aynı mantık."""
    dte = (pd.Timestamp(g["expiration"].iloc[0]) - pd.Timestamp(g["date"].iloc[0])).days
    T = max(dte, 0.5) / 365.0
    rows = []
    for _, r in g.iterrows():
        K, oi, mid, right = r["strike"], r["open_interest"], r["mid"], r["right"]
        if K is None or oi is None or oi != oi or mid is None or mid <= 0:
            continue
        K = float(K)
        if abs(K / S - 1) > BAND:
            continue
        iv = implied_vol(float(mid), S, K, T, right)
        if not iv or iv <= 0:
            continue
        gg, _v, _c, _de = _greeks(S, K, T, iv, right)
        rows.append({"K": K, "oi": float(oi), "right": right, "iv": iv, "g": gg})
    if len(rows) < 4:
        return None
    sgn = lambda rt: 1.0 if rt == "C" else -1.0
    net_gex = sum(sgn(x["right"]) * x["g"] * x["oi"] * M * S * S * 0.01 for x in rows)

    def net_g_at(hs):
        tot = 0.0
        for x in rows:
            gg, *_ = _greeks(hs, x["K"], T, x["iv"], x["right"])
            tot += sgn(x["right"]) * gg * x["oi"] * M * hs * hs * 0.01
        return tot
    grid = [(round(S * (1 + p), 2), net_g_at(S * (1 + p))) for p in SCAN]
    flip = None
    for (s0, g0), (s1, g1) in zip(grid, grid[1:]):
        if (g0 <= 0 <= g1) or (g0 >= 0 >= g1):
            flip = round(s0 + (s1 - s0) * (0 - g0) / (g1 - g0), 2) if g1 != g0 else s0
            break

    by_k_call, by_k_put, call_oi, by_k_all = {}, {}, {}, {}
    for x in rows:
        gk = x["g"] * x["oi"]
        if x["right"] == "C":
            by_k_call[x["K"]] = by_k_call.get(x["K"], 0.0) + gk
            if x["K"] >= S:
                call_oi[x["K"]] = call_oi.get(x["K"], 0.0) + x["oi"]
        else:
            by_k_put[x["K"]] = by_k_put.get(x["K"], 0.0) + gk
        by_k_all[x["K"]] = by_k_all.get(x["K"], 0.0) + abs(gk)
    ghost = max((k for k in by_k_call if k >= S), key=lambda k: by_k_call[k], default=None)
    call_wall = max(call_oi, key=lambda k: call_oi[k], default=None) if call_oi else None
    put_wall = max((k for k in by_k_put if k <= S), key=lambda k: by_k_put[k], default=None)
    hvl = max(by_k_all, key=lambda k: by_k_all[k]) if by_k_all else None
    strikes = sorted({x["K"] for x in rows})
    coi = {}
    for x in rows:
        coi[(x["K"], x["right"])] = coi.get((x["K"], x["right"]), 0.0) + x["oi"]
    pain = lambda P: sum(coi.get((k, "C"), 0) * max(0, P - k) + coi.get((k, "P"), 0) * max(0, k - P) for k in strikes)
    max_pain = min(strikes, key=pain) if strikes else None
    atm = min(rows, key=lambda x: abs(x["K"] - S))
    em1 = S * atm["iv"] * sqrt(1 / 252)
    return dict(net_gex=net_gex, regime=1 if net_gex >= 0 else -1, flip=flip, ghost=ghost,
                call_wall=call_wall, put_wall=put_wall, hvl=hvl, max_pain=max_pain,
                atm_iv=atm["iv"], em1=em1, dte=int(dte), n_strikes=len(rows))


def build(sym):
    ch = pd.read_parquet(ROOT / "data" / "historical_chains" / f"md_{sym.lower()}.parquet")
    spot = _daily_spot(sym)
    out = []
    for d, g in ch.groupby("date"):
        dd = pd.Timestamp(d).date()
        if dd not in spot.index:
            continue
        S = float(spot[dd])
        lv = _levels_for_day(g, S)
        if lv is None:
            continue
        lv["date"] = pd.Timestamp(d)
        lv["spot"] = S
        out.append(lv)
    df = pd.DataFrame(out).set_index("date").sort_index()
    p = ROOT / "data" / "cache" / f"level_series_{sym.lower()}.parquet"
    df.to_parquet(p)
    return df, p


def main():
    for sym in ("SPY", "QQQ"):
        df, p = build(sym)
        flip_ok = df["flip"].notna().mean()
        reg = df["regime"].value_counts()
        # akıl-sağlığı: flip spot'a yakın mı, duvarlar spot'u sarıyor mu
        near_flip = ((df["flip"] - df["spot"]).abs() / df["spot"]).median()
        cw_above = (df["call_wall"] >= df["spot"]).mean()
        pw_below = (df["put_wall"] <= df["spot"]).mean()
        print(f"=== {sym} === {len(df)} gün → {p.name}")
        print(f"  rejim: +γ {int(reg.get(1,0))}g / -γ {int(reg.get(-1,0))}g  ({100*reg.get(-1,0)/len(df):.0f}% negatif)")
        print(f"  flip bulundu %{100*flip_ok:.0f}, |flip-spot|/spot medyan %{100*near_flip:.2f}")
        print(f"  call_wall≥spot %{100*cw_above:.0f}, put_wall≤spot %{100*pw_below:.0f}  (sağlık: ~%100 beklenir)")
        print(f"  em1 medyan %{100*(df['em1']/df['spot']).median():.2f}, DTE medyan {int(df['dte'].median())}")
        print(f"  örnek son gün: spot {df['spot'].iloc[-1]:.2f} flip {df['flip'].iloc[-1]} "
              f"cw {df['call_wall'].iloc[-1]} pw {df['put_wall'].iloc[-1]} hvl {df['hvl'].iloc[-1]} regime {df['regime'].iloc[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
