"""
backtest/horizon_analysis — GÖREV 2 (ANALİZ, kod-değişikliği YOK). Tide YAVAŞ bir sinyal mi, edge'i kaç gün
taşımaktan geliyor? Engine bugün çoğu rejimde intraday (2g) bilet kuruyor — uyumsuzsa edge çöpe, friction maks.

Kilitli frozen panel + fiyatlardan (reproducible, ağsız):
  • tide_dir run-length (LONG kaç gün sürüyor) — doğal vade
  • tide_score otokorelasyon (sinyal ne kadar kalıcı)
  • LONG-girişten sonra getiri profili: kümülatif h=1/2/5/10/21/45g + gün-kova marjinal (1 / 2-5 / 6-10 / 11-20)
  → edge gün-1'de mi (intraday OK) yoksa haftalara mı yayılıyor (uzun DTE şart)
  & <venv python> backtest/horizon_analysis.py
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

from spine import contract as C, tide as T          # noqa: E402


def _runs(d: np.ndarray):
    out, i = [], 0
    while i < len(d):
        j = i
        while j < len(d) and d[j] == d[i]:
            j += 1
        out.append((int(d[i]), j - i)); i = j
    return out


def main() -> int:
    scores, prices, vector, prov = C.read_frozen()
    score = T.tide_score_series(scores, vector)
    idx = score.index
    d = (score > 0).astype(int)
    dd = d.values

    print("=" * 78)
    print(f"  GÖREV 2 — HORIZON UYUMU (frozen {idx[0].date()}..{idx[-1].date()}, {len(idx)}g)")
    print("=" * 78)

    # 1) run-length
    runs = _runs(dd)
    long_runs = np.array([n for v, n in runs if v == 1])
    flat_runs = np.array([n for v, n in runs if v == 0])
    long_share = dd.mean()
    print(f"  POZİSYON   LONG günleri %{long_share*100:.0f} | LONG-run say {len(long_runs)} / FLAT-run {len(flat_runs)}")
    if len(long_runs):
        print(f"  LONG-RUN   ort {long_runs.mean():.0f}g | medyan {np.median(long_runs):.0f}g | "
              f"p25 {np.percentile(long_runs,25):.0f} / p75 {np.percentile(long_runs,75):.0f} / max {long_runs.max():.0f}g")
    if len(flat_runs):
        print(f"  FLAT-RUN   ort {flat_runs.mean():.0f}g | medyan {np.median(flat_runs):.0f}g | max {flat_runs.max():.0f}g")

    # 2) otokorelasyon (tide_score)
    s = score.dropna()
    acf = {lag: round(float(s.autocorr(lag)), 3) for lag in (1, 5, 10, 21, 45)}
    print(f"  OTOKOR.    tide_score ACF: " + "  ".join(f"lag{l}={v}" for l, v in acf.items())
          + "   (yüksek+yavaş-düşüş → kalıcı/yavaş sinyal)")

    # 3) LONG-girişten sonra getiri profili (SPX & NDX)
    print("-" * 78)
    print("  LONG-GİRİŞ SONRASI GETİRİ (tide_dir 0→1 flip; kümülatif ortalama %, hit=poz oran)")
    entries = [k for k in range(1, len(dd)) if dd[k] == 1 and dd[k-1] == 0]
    HZ = [1, 2, 5, 10, 21, 45]
    for asset in ("SPX", "NDX"):
        if asset not in prices.columns:
            continue
        close = prices[asset].reindex(idx, method="ffill")
        cv = close.values
        print(f"  {asset:<5} ({len(entries)} giriş)   " +
              "  ".join(f"h{h}g" for h in HZ))
        means, hits = [], []
        for h in HZ:
            r = [(cv[k+h]/cv[k]-1) for k in entries if k+h < len(cv)]
            r = np.array(r)
            means.append(r.mean()*100 if len(r) else float("nan"))
            hits.append((r > 0).mean()*100 if len(r) else float("nan"))
        print(f"        kümül %  " + "  ".join(f"{m:>5.2f}" for m in means))
        print(f"        hit   %  " + "  ".join(f"{h:>5.0f}" for h in hits))
        # gün-kova marjinal ortalama GÜNLÜK getiri (front-load testi)
        buckets = {"g1": (0, 1), "g2-5": (1, 5), "g6-10": (5, 10), "g11-20": (10, 20)}
        bvals = {}
        for name, (a, b) in buckets.items():
            per = []
            for k in entries:
                for off in range(a, b):
                    if k+off+1 < len(cv):
                        per.append(cv[k+off+1]/cv[k+off]-1)
            bvals[name] = np.mean(per)*100 if per else float("nan")
        print(f"        marjinal günlük %  " + "  ".join(f"{n}={v:+.3f}" for n, v in bvals.items())
              + "   (eşitse edge yayılı → uzun-tut; gün1 baskınsa front-load)")
    print("-" * 78)

    # 4) vade karşılaştırması
    dte = {"intraday": 2, "swing": 21, "position": 45}
    lr = long_runs.mean() if len(long_runs) else float("nan")
    print(f"  VADE       motor DTE: intraday {dte['intraday']}g / swing {dte['swing']}g / position {dte['position']}g")
    print(f"  UYUM       ort LONG-run {lr:.0f}g vs intraday {dte['intraday']}g → "
          f"intraday bileti run'ın ~%{dte['intraday']/lr*100:.0f}'ini kapsıyor (kalanı re-entry+friction)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
