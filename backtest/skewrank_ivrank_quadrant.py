"""
backtest/skewrank_ivrank_quadrant — SpotGamma (Brent) video-iddiasi T0: skew-rank x IV-rank ceyrek-duzlem.

Iddia: call-skew-rank YUKSEK (call'lar put'lara gore zengin, 1y-persentil >=0.90) + IV-rank YUKSEK
(opsiyonlar mutlak olarak pahali) = "crash-up" kosesi -> ileri getiri zayif/spazm riski.
T0 = ENDEKS-SEVIYE test (SPY 2010-2023, QQQ 2012-2023, mevcut rr_skew parquet'leri, YENI VERI YOK).
Tek-isim breadth (Mag-7) veri toplamaya deger mi -> bu testin sonucu karar verir.

Tanimlar (PIT, look-ahead yok):
  call_skew_rank = 1 - trailing-252g pct(t30_rr_skew)   (rr_skew POZITIF=put-primi; dusuk/negatif=call-zengin)
  iv_rank        = trailing-252g pct(t30_atm_iv)
  CRASH_UP  = call_skew_rank>=0.90 & iv_rank>=0.60   (video: "IV da son 1 yilin pahali dilimi")
  CALL_CHEAP= call_skew_rank>=0.90 & iv_rank<0.60    (normal boga: call'lar zengin ama IV ucuz)
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


def _pit(s: pd.Series, win: int = 252, mp: int = 126) -> pd.Series:
    return s.rolling(win, min_periods=mp).apply(lambda w: pd.Series(w).rank(pct=True).iloc[-1], raw=False)


def run(sym: str) -> None:
    df = pd.read_parquet(ROOT / "data" / "cache" / f"rr_skew_{sym}_{'2010' if sym=='spy' else '2012'}_2023.parquet")
    df = df[df["t30_ok"].astype(bool)].copy()
    df["csr"] = 1.0 - _pit(df["t30_rr_skew"])
    df["ivr"] = _pit(df["t30_atm_iv"])
    for h in (10, 21, 63):
        df[f"fwd{h}"] = df["spot"].shift(-h) / df["spot"] - 1
    d = df.dropna(subset=["csr", "ivr", "fwd21"])

    regimes = {
        "CRASH_UP (csr>=.9 & ivr>=.6)": (d["csr"] >= 0.90) & (d["ivr"] >= 0.60),
        "CALL_CHEAP (csr>=.9 & ivr<.6)": (d["csr"] >= 0.90) & (d["ivr"] < 0.60),
        "PUT_FEAR  (csr<=.1)": d["csr"] <= 0.10,
        "BAZ (hepsi)": pd.Series(True, index=d.index),
    }
    print(f"\n===== {sym.upper()}  ({d.index.min().date()} -> {d.index.max().date()}, n={len(d)}) =====")
    for name, m in regimes.items():
        sub = d[m]
        if len(sub) < 20:
            print(f"  {name:34s} n={len(sub):4d}  (az)")
            continue
        line = f"  {name:34s} n={len(sub):4d}"
        for h in (10, 21, 63):
            line += f"  {h}g {sub[f'fwd{h}'].mean()*100:+.2f}%/{(sub[f'fwd{h}']<0).mean()*100:.0f}%neg"
        print(line)
    # en-kotu-an (21g ileri maxDD) kiyasi: CRASH_UP vs baz
    arr = d["spot"].to_numpy()
    dd = np.array([np.min(arr[i:i+22] / arr[i] - 1) for i in range(len(arr) - 21)])
    ddx = pd.Series(dd, index=d.index[: len(dd)])
    cu = regimes["CRASH_UP (csr>=.9 & ivr>=.6)"].reindex(ddx.index).fillna(False)
    print(f"  21g ileri en-kotu-an: CRASH_UP ort {ddx[cu].mean()*100:.2f}% / p10 {ddx[cu].quantile(.1)*100:.2f}%"
          f"  |  BAZ ort {ddx.mean()*100:.2f}% / p10 {ddx.quantile(.1)*100:.2f}%")
    # csr tek-basina (ivr sartsiz) — ivr kosulu deger katiyor mu?
    solo = d[d["csr"] >= 0.90]
    print(f"  csr>=.9 TEK-BASINA                 n={len(solo):4d}  21g {solo['fwd21'].mean()*100:+.2f}%/{(solo['fwd21']<0).mean()*100:.0f}%neg")


if __name__ == "__main__":
    for s in ("spy", "qqq"):
        run(s)
