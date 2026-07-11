"""
backtest/alpaca_panel_extension — chain_lab panelinin 2024'e PROXY uzatmasi (Alpaca opsiyon barlari).

3 adim:
  1) PROXY level-serisi: alpaca_chain_{sym}.parquet → mid:=bar close (son islem), oi:=bar volume.
     Seviye formulleri build_level_series._levels_for_day'den IMPORT (byte-es — kopya degil).
     → data/cache/level_series_{sym}_alpaca.parquet  (2024-01-18 → 2026-06-08)
  2) DOGRULAMA (ortusme 2025-06-13→2026-06-08): proxy vs gercek (MarketData mid+OI) level-serisi —
     rejim isaret-uyumu, net_gex rank-corr, flip/duvar mesafeleri; panel testleri (M1/M2/M3) yan yana.
     Proxy ortusmede gercegi REPRODUCE EDEMIYORSA 2024 uzatmasi guvenilmez → rapor bunu soyler.
  3) UZATMA (2024-01-18→2025-06-12): ayni panel testleri proxy seviyelerle SADECE-yeni donem + birlesik.

PIT korunur: D-EOD seviye × D+1 RTH (spine_diagnostic ile ayni fonksiyonlar). PROXY sinirlari:
close=son-islem (mid degil, kanatlarda bayat olabilir), volume=OI degil (gun ici akis; duvarlar
'volume-duvari' anlamina gelir). Bu yuzden ad her yerde '_alpaca' — gercek seriyle karistirilmaz.
  & <kader-macro venv python> backtest/alpaca_panel_extension.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from build_level_series import _levels_for_day, _daily_spot          # noqa: E402
from alpaca_chain_backfill import expiry_for, START, END             # noqa: E402
from spine_diagnostic import daily_rth, mean_reversion_return, _stat, _boot_spread, COST  # noqa: E402

SPLIT = pd.Timestamp("2025-06-13")            # md level_series baslangici = ortusme siniri


# ── 1) PROXY level-serisi ──────────────────────────────────────────────────────
def build_proxy_levels(sym: str) -> pd.DataFrame:
    ch = pd.read_parquet(ROOT / "data" / "historical_chains" / f"alpaca_chain_{sym.lower()}.parquet")
    spot = _daily_spot(sym)
    cal = set(spot.index)
    # tarih→tek-expiry kurali: yalniz o gunun on-aylik expiry'sine ait satirlar (md metodolojisi)
    ch = ch[ch["volume"] > 0].copy()
    ch["mid"] = ch["close"]                   # PROXY: son islem ≈ mid
    ch["open_interest"] = ch["volume"]        # PROXY: hacim ≈ OI
    out = []
    for d, g in ch.groupby("date"):
        dd = pd.Timestamp(d).date()
        if dd not in spot.index:
            continue
        g = g[pd.to_datetime(g["expiration"]).dt.date == expiry_for(dd, cal)]
        if not len(g):
            continue
        lv = _levels_for_day(g, float(spot[dd]))
        if lv is None:
            continue
        lv["date"] = pd.Timestamp(d); lv["spot"] = float(spot[dd])
        out.append(lv)
    df = pd.DataFrame(out).set_index("date").sort_index()
    p = ROOT / "data" / "cache" / f"level_series_{sym.lower()}_alpaca.parquet"
    df.to_parquet(p)
    days = [x for x in sorted(cal) if START <= x <= END]
    print(f"  {sym} proxy-seri: {len(df)}/{len(days)} gun ({df.index.min().date()}→{df.index.max().date()}), "
          f"n_strikes medyan {int(df['n_strikes'].median())}, DTE medyan {int(df['dte'].median())}, "
          f"-γ orani %{100 * (df['regime'] == -1).mean():.0f}")
    return df


# ── panel + testler (spine_diagnostic ile ayni mantik, lv parametreli) ─────────
def build_panel_from(lv: pd.DataFrame, sym: str) -> pd.DataFrame:
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
        r = lv.loc[D].to_dict()
        r.update(dict(D=D, c0=rth.loc[D, "c"], o1=rth.loc[N, "o"], h1=rth.loc[N, "h"],
                      l1=rth.loc[N, "l"], c1=rth.loc[N, "c"]))
        r["gap"] = r["o1"] / r["c0"] - 1; r["intraday"] = r["c1"] / r["o1"] - 1
        rows.append(r)
    p = pd.DataFrame(rows)
    p["mr"] = mean_reversion_return(p)
    p["reg_dc"] = np.where((p["regime"] == 1) & (p["spot"] > p["flip"]), 1,
                  np.where((p["regime"] == -1) & (p["spot"] < p["flip"]), -1, 0))
    return p


def panel_line(p: pd.DataFrame, tag: str) -> None:
    pos_g = p[p["reg_dc"] == 1]; neg_g = p[p["reg_dc"] == -1]
    cp = pos_g["gap"].corr(pos_g["intraday"]); cn = neg_g["gap"].corr(neg_g["intraday"])
    pos = pos_g["mr"]; neg = neg_g["mr"]
    mp, tp, np_ = _stat(pos); mn, tn, nn = _stat(neg)
    _sp, pgt = _boot_spread(pos, neg)
    s1 = pos.dropna().mean() - COST if np_ else float("nan")
    s2 = (-neg.dropna()).mean() - COST if nn else float("nan")
    print(f"    {tag:<22} n={len(p):<4} M1: +γ{cp:+.2f}/−γ{cn:+.2f}  "
          f"M2: +γ{1e4 * mp:+.1f}bps(t{tp:+.1f},n{np_}) −γ{1e4 * mn:+.1f}bps(t{tn:+.1f},n{nn}) P(spread>0)={pgt:.2f}  "
          f"M3net: fade{1e4 * s1:+.1f} brk{1e4 * s2:+.1f}bps")


def main() -> int:
    print("ALPACA PANEL UZATMASI — proxy(close,volume) → 2024-01-18+\n")
    for sym in ("SPY", "QQQ"):
        print(f"=== {sym} ===")
        prox = build_proxy_levels(sym)
        md = pd.read_parquet(ROOT / "data" / "cache" / f"level_series_{sym.lower()}.parquet")

        # 2) ortusme dogrulamasi
        j = prox.join(md, how="inner", lsuffix="_px", rsuffix="_md")
        agree = (j["regime_px"] == j["regime_md"]).mean()
        rc = j["net_gex_px"].corr(j["net_gex_md"], method="spearman")
        fd = ((j["flip_px"] - j["flip_md"]).abs() / j["spot_md"]).median()
        cw = ((j["call_wall_px"] - j["call_wall_md"]).abs() / j["spot_md"]).median()
        pw = ((j["put_wall_px"] - j["put_wall_md"]).abs() / j["spot_md"]).median()
        print(f"  ORTUSME ({len(j)} gun): rejim-uyum %{100 * agree:.0f}, net_gex rank-corr {rc:+.2f}, "
              f"medyan |Δ|/spot: flip %{100 * fd:.2f} cw %{100 * cw:.2f} pw %{100 * pw:.2f}")
        print("  PANEL TESTLERI (M1 gap-corr / M2 duvar-MR / M3 net-edge, double-confirm):")
        panel_line(build_panel_from(md, sym), "ortusme-GERCEK(md)")
        panel_line(build_panel_from(prox[prox.index >= SPLIT], sym), "ortusme-PROXY")
        # 3) uzatma
        ext = prox[prox.index < SPLIT]
        panel_line(build_panel_from(ext, sym), "UZATMA-2024→25(proxy)")
        panel_line(build_panel_from(prox, sym), "BIRLESIK-2024→26(proxy)")
        print()
    print("  Okuma: once ortusme-PROXY, ortusme-GERCEK'i reproduce ediyor mu (isaretler/buyukluk). Ediyorsa")
    print("  UZATMA satiri 2024-01→2025-06 donemi icin anlamli. PROXY=close/volume — gercek mid/OI DEGIL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
