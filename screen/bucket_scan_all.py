"""
screen/bucket_scan_all — YÖN-AGNOSTİK denetim (Emir 2026-06-09): COR1M sign-hatası sistematik miydi?

Her metriği quintile'a böl, her bucket'ta ABSOLUTE forward-21g SPX+NDX getiri. Şekli GÖSTER (sign
SEÇME). Düşük→bearish (kontrarian/froth) mü, yüksek→bearish (stres) mi, monotonik mi, düz mü?
Böylece tek-yön kurup yanlış gömdüğüm metrikleri ifşa et. Tüm 'ölü' verdict'leri sıfırdan denetler.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from screen._util import load_price_csv                  # noqa: E402

DESK = Path(r"C:\Users\admin\Desktop\backtesting")
CACHE = ROOT / "data" / "cache"
H = 21


def _load_metrics() -> dict:
    m = {}
    vs = pd.read_parquet(CACHE / "vol_surface.parquet")
    m["VIX"] = vs["vix"]; m["VIX_TS(VIX/3M)"] = vs["ts_ratio"]; m["VXN"] = vs.get("vxn")
    sv = pd.read_parquet(CACHE / "skew_vvix.parquet")
    m["SKEW"] = sv["SKEW"]; m["VVIX"] = sv["VVIX"]
    sg = pd.read_parquet(CACHE / "squeeze_dix_gex.parquet")
    m["GEX"] = sg["gex"]; m["DIX"] = sg["dix"]
    cp = pd.read_parquet(CACHE / "corr_pc.parquet")
    m["COR1M"] = cp["COR1M"]; m["COR3M"] = cp["COR3M"]
    return {k: v.dropna() for k, v in m.items() if v is not None}


def _read(shape_spx):
    lo, hi = shape_spx[0], shape_spx[-1]
    mn, mx = min(shape_spx), max(shape_spx)
    if abs(lo - hi) < 0.3 and (mx - mn) < 0.6:
        return "düz (sinyal yok)"
    if lo < hi - 0.4 and shape_spx == sorted(shape_spx):
        return "DÜŞÜK→bearish (kontrarian/froth?)"
    if lo > hi + 0.4 and shape_spx == sorted(shape_spx, reverse=True):
        return "YÜKSEK→bearish (stres)"
    if shape_spx[0] < min(shape_spx[1:-1]) - 0.4:
        return "DÜŞÜK-uç bearish (froth?)"
    if shape_spx[-1] < min(shape_spx[:-1]) - 0.4:
        return "YÜKSEK-uç bearish (stres)"
    return "non-monotonik/karışık"


def main():
    spx, ndx = load_price_csv(DESK / "SPX_daily.csv"), load_price_csv(DESK / "NASDAQ_daily.csv")
    metrics = _load_metrics()
    # RV-ratio (compute)
    lr = np.log(spx / spx.shift(1))
    metrics["RV_ratio(5/20)"] = (lr.rolling(5).std() / lr.rolling(20).std()).dropna()

    print(f"  YÖN-AGNOSTİK BUCKET — quintile (Q1=düşük..Q5=yüksek) → mean forward-{H}g getiri")
    print("=" * 100)
    print(f"  {'metric':<16}{'tarih':>11}  {'SPX:  Q1':>9}{'Q2':>7}{'Q3':>7}{'Q4':>7}{'Q5':>7}   {'okuma (SPX)':<32}")
    for name, s in metrics.items():
        idx = s.index.intersection(spx.index).intersection(ndx.index)
        if len(idx) < 300:
            continue
        sv = s.reindex(idx)
        cb_s = spx.reindex(idx, method="ffill"); f_s = (cb_s.shift(-H) / cb_s - 1)
        cb_n = ndx.reindex(idx, method="ffill"); f_n = (cb_n.shift(-H) / cb_n - 1)
        try:
            q = pd.qcut(sv, 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        spx_b = [100 * f_s[q == i].mean() for i in range(5)]
        ndx_b = [100 * f_n[q == i].mean() for i in range(5)]
        read = _read(spx_b)
        yr = f"{idx.min().year}-{idx.max().year}"
        print(f"  {name:<16}{yr:>11}  {spx_b[0]:>+9.1f}{spx_b[1]:>+7.1f}{spx_b[2]:>+7.1f}{spx_b[3]:>+7.1f}{spx_b[4]:>+7.1f}   {read:<32}")
        print(f"  {'':16}{'NDX:':>11}  {ndx_b[0]:>+9.1f}{ndx_b[1]:>+7.1f}{ndx_b[2]:>+7.1f}{ndx_b[3]:>+7.1f}{ndx_b[4]:>+7.1f}")
    print("=" * 100)
    print("  Q1-Q5 mean fwd getiri; bir UÇ belirgin negatifse orada kontrarian sinyal var → incremental test et.")
    print("  (COR1M referans: Q1 düşük=bearish çıkmalı = froth. Diğerlerinde benzer kaçırma var mı?)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
