"""
backtest/block_robust — "non-stationary" verdiğine EMİN MİSİN? Tek 150/70 bölünmesi yerine düzgün denetim.
Emir itiraz etti (haklı: negatif-sonuç genelde test-hatası). gamma_inv (naive-+γ→momentum / −γ→revert) sinyalini:
  (1) 6 ardışık BLOK Sharpe — gerçekten son bloka mı sıkışmış, yoksa geneli pozitif mi?
  (2) ROLLING 63-gün Sharpe — zamanın % kaçında pozitif?
  (3) BOOTSTRAP full-sample Sharpe CI — sıfırı dışlıyor mu?
  (4) AYKIRI-GÜN — holdout'un +3.4'ü birkaç güne mi bağlı (top-3 katkı %)?
  (5) PIT vol-medyan (look-ahead temizliği kontrolü) — sonucu değiştiriyor mu?
PIT temiz: net_gex[D] + gap[D+1-open] → intraday[D+1]. pnl net = pos×intraday − 1.5bps.
  & <venv python> backtest/block_robust.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backtest"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from spine_diagnostic import build_panel       # noqa: E402

COST = 0.00015


def sharpe(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean() / x.std() * np.sqrt(252) if len(x) > 2 and x.std() > 0 else 0.0


def gamma_inv_pnl(sym):
    p = build_panel(sym).dropna(subset=["gap", "intraday", "flip"]).reset_index(drop=True)
    gap = np.sign(p["gap"].values)
    pos = np.where(p["regime"].values > 0, gap, -gap)       # naive-+γ momentum / −γ revert
    pnl = pos * p["intraday"].values - COST                 # NET
    return pd.Series(pnl, index=p["D"].values), p


def run(sym):
    pnl, p = gamma_inv_pnl(sym)
    n = len(pnl)
    print("=" * 96)
    print(f"  {sym} — gamma_inv NET robustluk denetimi (n={n})  full-Sharpe {sharpe(pnl):+.2f}")
    print("=" * 96)

    # (1) 6 ardışık blok
    print("  (1) 6 ardışık BLOK (her ~%d gün) net Sharpe:" % (n // 6))
    bs = np.array_split(np.arange(n), 6)
    blk = []
    for i, idx in enumerate(bs):
        s = sharpe(pnl.values[idx]); blk.append(s)
        d0, d1 = pd.Timestamp(pnl.index[idx[0]]).date(), pd.Timestamp(pnl.index[idx[-1]]).date()
        print(f"      blok{i+1} {d0}→{d1}: Sharpe {s:+.2f}  ort {1e4*pnl.values[idx].mean():+.1f}bps  ({(pnl.values[idx]>0).mean()*100:.0f}% gün+)")
    print(f"      → {sum(b>0 for b in blk)}/6 blok pozitif  (geneli-pozitif mi yoksa son-bloka-sıkışık mı?)")

    # (2) rolling 63g
    roll = pnl.rolling(63).apply(lambda w: sharpe(w.values), raw=False).dropna()
    print(f"  (2) ROLLING 63-gün Sharpe: medyan {roll.median():+.2f}, min {roll.min():+.2f}, max {roll.max():+.2f}, "
          f"%pozitif {100*(roll>0).mean():.0f}")

    # (3) bootstrap full-sample Sharpe
    rng = np.random.default_rng(7); v = pnl.values
    bss = np.array([sharpe(v[rng.integers(0, n, n)]) for _ in range(3000)])
    print(f"  (3) BOOTSTRAP full Sharpe: {np.median(bss):+.2f}  [%5 {np.percentile(bss,5):+.2f}, %95 {np.percentile(bss,95):+.2f}]  "
          f"P(>0)={100*(bss>0).mean():.0f}%")

    # (4) aykırı-gün: son 70g top-3 katkı
    ho = pnl.values[-70:]
    tot = ho.sum(); top3 = np.sort(ho)[-3:].sum()
    print(f"  (4) HOLDOUT(son70) toplam {1e4*tot:+.0f}bps; top-3 gün katkısı {1e4*top3:+.0f}bps = %{100*top3/tot if tot else 0:.0f} "
          f"({'birkaç-güne-bağlı' if tot and top3/tot > 0.6 else 'dağılmış'})")

    # (5) PIT-medyan vs full-medyan vol-bucket: yalnız (B) diagnostiğini etkiler — burada gamma_inv vol kullanmıyor → no-op teyidi
    print("  (5) gamma_inv vol-medyan KULLANMIYOR (yalnız net_gex-işareti+gap) → look-ahead yok; (B)-diagnostik ayrı.")
    return blk, roll


def main():
    for sym in ("SPY", "QQQ"):
        run(sym); print()
    print("  KARAR: ≥4/6 blok + rolling %pozitif yüksek + bootstrap-P(>0) yüksek + aykırı-değil → 'non-stationary' YANLIŞTI,")
    print("  edge geneli-geçerli (tek-rejim kaydıyla). Aksi (yalnız son blok + bootstrap-zayıf) → non-stationary DOĞRU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
