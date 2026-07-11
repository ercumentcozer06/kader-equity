"""
candidate_credit_extreme — Marks contrarian-EXTREME kredi ekseni (2026-07-10, ön-kayıtlı).

HİPOTEZ ([[credit_froth_macro_todo]]): m6 krediyi LİNEER okur (dar=iyi, geniş=kötü). Marks'ın iddiası
ortogonal: AŞIRI-dar spread = fiyatlanmış mükemmellik = complacency → İLERİ getiri kötü (trim sinyali);
aşırı-geniş = kapitülasyon → kontrarian iyi. Bu EXTREME-kuyruk okuması test edilmemişti.

ÖN-KAYITLI TASARIM (fit yok):
  1) BETİMSEL: HY-OAS trailing-756g percentile DECILE'ları → ileri 21/63g SPX/NDX getirisi (2000+).
     Marks doğruysa en-dar decile ileri-getiri en kötülerden olmalı; en-geniş decile en iyilerden.
  2) GATE ablation (canlı tide spine 2019+, vrp_verify Probe-D şablonu): BASE=tide-long;
     T1 trim-to-0 @OAS-pct≤0.05; T2 trim-to-0 @≤0.10; T3 trim-to-0.5 @≤0.10. TRIM-ONLY (ev stili).
     PIT: rolling 756g (min 252) percentile; FRED yayın gecikmesi için gate 1 GÜN shift'li.
  KARAR KURALI: kol ancak SPX ve NDX'in İKİSİNDE Sharpe>BASE VE maxDD ≤ BASE+2pp ise ADAY; yoksa RET.
  ÖNCEL: VRP kampanyası tide-üstüne-gate'lerin genelde zarar verdiğini gösterdi — beklenti mütevazı.
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

from screen._util import load_price_csv          # noqa: E402
from spine import contract as C, tide as T       # noqa: E402

DESK = Path(r"C:\Users\admin\Desktop\backtesting")
CACHE = ROOT / "data" / "cache"


def sh(x):
    x = x.dropna()
    return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan


def mdd(x):
    c = (1 + x.fillna(0)).cumprod()
    return (c / c.cummax() - 1).min()


def main() -> int:
    oas = pd.read_parquet(CACHE / "hy_oas_full.parquet")["hy_oas"].dropna().sort_index()
    pct = oas.rolling(756, min_periods=252).apply(lambda w: pd.Series(w).rank(pct=True).iloc[-1], raw=False)
    spx = load_price_csv(DESK / "SPX_daily.csv").sort_index()
    ndx = load_price_csv(DESK / "NASDAQ_daily.csv").sort_index()

    # ── 1) BETİMSEL: OAS-percentile decile → ileri getiri (2000+) ──
    print("=" * 100)
    print("  HY-OAS trailing-3y percentile DECILE → ileri getiri (Marks: en-dar kötü mü, en-geniş iyi mi?)")
    print("=" * 100)
    for name, px in (("SPX", spx), ("NDX", ndx)):
        df = pd.DataFrame({"pct": pct}).join(px.rename("px"), how="inner").dropna().loc["2000-01-01":]
        for h in (21, 63):
            fr = df["px"].shift(-h) / df["px"] - 1.0
            dec = (df["pct"] * 10).clip(0, 9.999).astype(int)
            g = fr.groupby(dec).mean() * 100
            n = fr.groupby(dec).size()
            row = " ".join(f"D{d}:{g.get(d, np.nan):+.1f}" for d in range(10))
            print(f"  {name} h={h:2}g  (D0=en-DAR spread/complacency … D9=en-GENİŞ/stres)")
            print(f"    {row}   [n≈{int(n.median())}/decile]")
        u21 = (df["px"].shift(-21) / df["px"] - 1.0).mean() * 100
        print(f"    koşulsuz 21g ort: {u21:+.2f}%")

    # ── 2) GATE ablation: canlı tide spine 2019+ ──
    print("\n" + "=" * 100)
    print("  GATE ABLATION (tide 2019+; gate 1g shift'li PIT; trim-only)  — karar kuralı başlıkta")
    print("=" * 100)
    scores, prices, vector, _ = C.read_frozen()
    ts = T.tide_score_series(scores, vector)
    tdir = T.tide_dir_series(ts).sort_index()
    verdict = {}
    for name, px in (("SPX", spx), ("NDX", ndx)):
        ret = px.pct_change().shift(-1)
        d = pd.DataFrame({"ret": ret}).join(tdir.rename("tdir"), how="inner")
        d = d.join(pct.rename("pct"), how="left")
        d["pct"] = d["pct"].ffill(limit=5).shift(1)          # yayın gecikmesi: 1g lag
        d = d.loc["2019-01-01":].dropna(subset=["ret", "tdir"])
        base = (d["tdir"] > 0).astype(float) * d["ret"]
        stats = {"BASE tide": (sh(base), mdd(base))}
        arms = {"T1 flat@pct<=0.05": (0.05, 0.0), "T2 flat@pct<=0.10": (0.10, 0.0),
                "T3 half@pct<=0.10": (0.10, 0.5)}
        for aname, (thr, floor) in arms.items():
            f = pd.Series(1.0, index=d.index)
            f[d["pct"] <= thr] = floor
            g = base * f
            stats[aname] = (sh(g), mdd(g))
        print(f"\n  [{name}]")
        for k, (s, m) in stats.items():
            print(f"    {k:22} Sharpe {s:+.3f}  maxDD {m*100:+6.1f}%")
        b = stats["BASE tide"]
        for aname in arms:
            s, m = stats[aname]
            ok = (s > b[0]) and (m >= b[1] - 0.02)
            verdict.setdefault(aname, []).append(ok)
        ndays = int((d["pct"] <= 0.10).sum())
        print(f"    pct<=0.10 gün sayısı (2019+): {ndays}")

    print("\n" + "=" * 100)
    print("  KARAR (SPX VE NDX ikisinde Sharpe>BASE ve maxDD ≤ BASE+2pp):")
    for aname, oks in verdict.items():
        print(f"    {aname:22} -> {'ADAY (mekanizma-testi gerek)' if all(oks) else 'RET / betimsel'}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
