"""
screen/bucket_scan_tails — UÇ-KUYRUK denetimi. Quintile uç sinyalleri yıkıyor (COR1M<8 froth böyle kaçar).
Her metrik için alt-%5 / alt-%10 / orta / üst-%10 / üst-%5 forward-21g getiri (SPX+NDX) + %neg + n.
Hangi UÇ belirgin negatifse orada kontrarian sinyal var → incremental test adayı. Yön-agnostik.
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


def _metrics():
    m = {}
    vs = pd.read_parquet(CACHE / "vol_surface.parquet")
    m["VIX"] = vs["vix"]; m["VIX_TS"] = vs["ts_ratio"]; m["VXN"] = vs.get("vxn")
    sv = pd.read_parquet(CACHE / "skew_vvix.parquet")
    m["SKEW"] = sv["SKEW"]; m["VVIX"] = sv["VVIX"]
    sg = pd.read_parquet(CACHE / "squeeze_dix_gex.parquet")
    m["GEX"] = sg["gex"]; m["DIX"] = sg["dix"]
    cp = pd.read_parquet(CACHE / "corr_pc.parquet")
    m["COR1M"] = cp["COR1M"]; m["COR3M"] = cp["COR3M"]
    return {k: v.dropna() for k, v in m.items() if v is not None}


def main():
    spx, ndx = load_price_csv(DESK / "SPX_daily.csv"), load_price_csv(DESK / "NASDAQ_daily.csv")
    metrics = _metrics()
    lr = np.log(spx / spx.shift(1))
    metrics["RV_ratio"] = (lr.rolling(5).std() / lr.rolling(20).std()).dropna()

    print(f"  UÇ-KUYRUK — forward-{H}g getiri: alt%5 / alt%10 / orta(20-80) / üst%10 / üst%5   (SPX satır1, NDX satır2)")
    print("=" * 104)
    print(f"  {'metric':<10}{'  bot5%':>9}{'bot10%':>8}{'mid':>8}{'top10%':>8}{'top5%':>8}   {'sinyal (uç negatif = kontrarian)':<34}")
    for name, s in metrics.items():
        idx = s.index.intersection(spx.index).intersection(ndx.index)
        if len(idx) < 300:
            continue
        sv = s.reindex(idx)
        cb_s = spx.reindex(idx, method="ffill"); f_s = (cb_s.shift(-H) / cb_s - 1)
        cb_n = ndx.reindex(idx, method="ffill"); f_n = (cb_n.shift(-H) / cb_n - 1)
        p5, p10, p90, p95 = sv.quantile([.05, .10, .90, .95])
        masks = {"b5": sv <= p5, "b10": sv <= p10, "mid": (sv > p10) & (sv < p90),
                 "t10": sv >= p90, "t5": sv >= p95}
        sb = {k: 100 * f_s[mm].mean() for k, mm in masks.items()}
        nb = {k: 100 * f_n[mm].mean() for k, mm in masks.items()}
        # flag: hangi uç belirgin negatif (< -0.5%) ya da orta'dan >1.5% düşük
        flag = []
        if sb["b5"] < -0.5 or nb["b5"] < -0.5:
            flag.append(f"DÜŞÜK-uç bearish (SPX{sb['b5']:+.1f}/NDX{nb['b5']:+.1f})")
        if sb["t5"] < -0.5 or nb["t5"] < -0.5:
            flag.append(f"YÜKSEK-uç bearish (SPX{sb['t5']:+.1f}/NDX{nb['t5']:+.1f})")
        if not flag:
            if sb["b5"] < sb["mid"] - 1.5:
                flag.append("düşük-uç zayıf")
            elif sb["t5"] < sb["mid"] - 1.5:
                flag.append("yüksek-uç zayıf")
            else:
                flag.append("uç sinyal yok")
        print(f"  {name:<10}{sb['b5']:>+9.1f}{sb['b10']:>+8.1f}{sb['mid']:>+8.1f}{sb['t10']:>+8.1f}{sb['t5']:>+8.1f}   {' | '.join(flag):<34}")
        print(f"  {'(NDX)':<10}{nb['b5']:>+9.1f}{nb['b10']:>+8.1f}{nb['mid']:>+8.1f}{nb['t10']:>+8.1f}{nb['t5']:>+8.1f}")
    print("=" * 104)
    print("  UÇU belirgin negatif olan = gerçek kontrarian sinyal (COR1M alt-%5 gibi) → incremental + strict FDR test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
