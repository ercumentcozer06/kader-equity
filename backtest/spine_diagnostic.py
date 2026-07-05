"""
backtest/spine_diagnostic — DIRECTIONAL GEX modelinin GO/NO-GO testi (spec D2-adım-1, ANA HİPOTEZ).
Modelin bütün tezi tek ölçüde: AYNI seviye-olayı iki rejimde TERS mi çalışıyor?
  pozitif-γ → duvar TUTAR / fiyat geri-döner (mean-revert) → FADE kârlı
  negatif-γ → duvar KIRILIR / fiyat devam eder (momentum)  → BREAKOUT kârlı
Birleşik ölçü: "mean-reversion getirisi" MR (duvardan kapanışa geri-dönüş). Tez: E[MR|+γ] > 0 > E[MR|−γ].
  Setup-1 (+γ) = MR'yi trade eder (fade);  Setup-2 (−γ) = −MR'yi trade eder (breakout). İkisi de pozitifse tez TUTAR.

PIT: D-EOD seviyeleri (level_series) × D+1 RTH seansı (Alpaca 1-dk, ET 09:30-16:00). Look-ahead yok.
Rejim kapısı: hem sign-only hem double-confirm (net_gex işareti AND spot-vs-flip, spec Ç1) raporlanır.
Bu bir TANISAL — grid/kalibrasyon DEĞİL (tek tol-varsayımı, fill=duvar-seviyesi). Edge VARSA grid'e değer.
  & <venv python> backtest/spine_diagnostic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

COST = 0.00015            # delta-one round-trip ~1.5 bps (muhafazakâr; spec D0: 1.0 fade + 0.5 breakout slippage)


def daily_rth(sym):
    """1-dk bar → ET RTH (09:30-16:00) günlük OHLC."""
    b = pd.read_parquet(ROOT / "data" / "historical_bars" / f"alpaca_{sym.lower()}_1m.parquet")
    ts = pd.to_datetime(b.index.get_level_values(-1)).tz_convert("America/New_York")
    df = pd.DataFrame({"o": b["open"].values, "h": b["high"].values, "l": b["low"].values,
                       "c": b["close"].values}, index=ts)
    t = df.index.time
    rth = df[(t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:00").time())]
    d = rth.index.date
    agg = rth.groupby(d).agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    agg.index = pd.to_datetime(agg.index)
    return agg


def build_panel(sym):
    """D-seviyeleri + D+1 seansı hizalı panel."""
    lv = pd.read_parquet(ROOT / "data" / "cache" / f"level_series_{sym.lower()}.parquet")
    rth = daily_rth(sym)
    sess = list(rth.index)
    rows = []
    for D in lv.index:
        if D not in rth.index:
            continue
        nxt = [s for s in sess if s > D]
        if not nxt:
            continue
        N = nxt[0]
        c0 = rth.loc[D, "c"]
        o1, h1, l1, c1 = rth.loc[N, ["o", "h", "l", "c"]]
        r = lv.loc[D].to_dict()
        r.update(dict(D=D, N=N, c0=c0, o1=o1, h1=h1, l1=l1, c1=c1,
                      gap=o1 / c0 - 1, intraday=c1 / o1 - 1))
        rows.append(r)
    return pd.DataFrame(rows)


def mean_reversion_return(p):
    """Duvar-dokunuşunda geri-dönüş getirisi (gross). >0 = duvar tuttu (mean-revert), <0 = kırıldı (momentum)."""
    mr = []
    for _, r in p.iterrows():
        vals = []
        if pd.notna(r["call_wall"]) and r["h1"] >= r["call_wall"]:      # dirence dokundu → kapanışa kadar geri düştü mü
            vals.append((r["call_wall"] - r["c1"]) / r["call_wall"])
        if pd.notna(r["put_wall"]) and r["l1"] <= r["put_wall"]:        # desteğe dokundu → kapanışa kadar geri çıktı mı
            vals.append((r["c1"] - r["put_wall"]) / r["put_wall"])
        mr.append(np.mean(vals) if vals else np.nan)
    return pd.Series(mr, index=p.index)


def _stat(x):
    x = x.dropna()
    if len(x) < 5:
        return 0.0, 0.0, 0
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if x.std() > 0 else 0.0
    return x.mean(), t, len(x)


def _boot_spread(a, b, n=2000):
    """E[a]−E[b] bootstrap; P(spread>0)."""
    a, b = a.dropna().values, b.dropna().values
    if len(a) < 5 or len(b) < 5:
        return 0.0, 0.5
    idx_a = np.random.default_rng(0).integers(0, len(a), (n, len(a)))
    idx_b = np.random.default_rng(1).integers(0, len(b), (n, len(b)))
    sp = a[idx_a].mean(1) - b[idx_b].mean(1)
    return float(a.mean() - b.mean()), float((sp > 0).mean())


def run(sym):
    p = build_panel(sym)
    p["mr"] = mean_reversion_return(p)
    p["reg_sign"] = p["regime"]
    p["reg_dc"] = np.where((p["regime"] == 1) & (p["spot"] > p["flip"]), 1,
                  np.where((p["regime"] == -1) & (p["spot"] < p["flip"]), -1, 0))
    print("=" * 96)
    print(f"  {sym} — SPINE GO/NO-GO  (n={len(p)} gün, duvar-dokunuşlu {p['mr'].notna().sum()})")
    print("=" * 96)

    # M1 — rejim-koşullu gap→intraday devam/dönüş (tez: −γ devam corr>0, +γ dönüş corr<0)
    print("  M1 — overnight-gap → intraday devam (corr): tez −γ POZİTİF (momentum), +γ NEGATİF (mean-revert)")
    for tag, col in (("sign", "reg_sign"), ("double-confirm", "reg_dc")):
        pos = p[p[col] == 1]; neg = p[p[col] == -1]
        cp = pos["gap"].corr(pos["intraday"]); cn = neg["gap"].corr(neg["intraday"])
        print(f"    [{tag:>14}]  +γ corr {cp:+.3f} (n{len(pos)})  |  −γ corr {cn:+.3f} (n{len(neg)})  "
              f"→ fark {cn - cp:+.3f} {'✓ tez yönünde' if cn > cp else '✗ ters'}")

    # M2 — ANA HİPOTEZ: duvar-dokunuş MR getirisi rejime göre işaret değiştiriyor mu
    print("\n  M2 — duvar-MR getirisi (ANA TEZ): E[MR|+γ] > 0 > E[MR|−γ] olmalı (bps, gross)")
    for tag, col in (("sign", "reg_sign"), ("double-confirm", "reg_dc")):
        pos = p[p[col] == 1]["mr"]; neg = p[p[col] == -1]["mr"]
        mp, tp, np_ = _stat(pos); mn, tn, nn = _stat(neg)
        spread, pgt = _boot_spread(pos, neg)
        print(f"    [{tag:>14}]  +γ MR {1e4*mp:+.1f}bps (t{tp:+.2f},n{np_})  |  "
              f"−γ MR {1e4*mn:+.1f}bps (t{tn:+.2f},n{nn})  |  spread {1e4*spread:+.1f}bps P(>0)={pgt:.2f}")

    # M3 — ima edilen tradeable net edge (double-confirm rejimde)
    print("\n  M3 — ima edilen NET edge (double-confirm, maliyet 1.5bps round-trip):")
    pos = p[p["reg_dc"] == 1]["mr"].dropna(); neg = p[p["reg_dc"] == -1]["mr"].dropna()
    s1 = pos.mean() - COST                 # Setup-1 fade: MR'yi al
    s2 = (-neg).mean() - COST              # Setup-2 breakout: −MR'yi al
    print(f"    Setup-1 (+γ fade)      : {1e4*s1:+.1f}bps/trade  (n{len(pos)}, isabet %{100*(pos>0).mean():.0f})")
    print(f"    Setup-2 (−γ breakout)  : {1e4*s2:+.1f}bps/trade  (n{len(neg)}, isabet %{100*(neg<0).mean():.0f})")
    verdict = "GO ✓ (ikisi de pozitif → tez tutar, grid'e değer)" if (s1 > 0 and s2 > 0) else \
              ("KISMİ (biri pozitif)" if (s1 > 0 or s2 > 0) else "NO-GO ✗ (ikisi de negatif → directional edge yok)")
    print(f"    → VERDİCT: {verdict}")
    return p


def main():
    for sym in ("SPY", "QQQ"):
        run(sym)
        print()
    print("  NOT: tanısal (tek tol, fill=duvar). GO ise tam grid (tol/N/DTE/timeframe) + train/holdout + DSR sırada.")
    print("  Look-ahead temiz: D-EOD seviye × D+1 seans. Front-expiry-only (canlı N_EXP=5 ile re-validate gerekir).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
