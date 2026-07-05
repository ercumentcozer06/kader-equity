"""
backtest/tracking_sanity — H9 (RAPOR-ONLY, eşik YOK). 'Delta-one ETF ≈ endeks' varsayımının sayısal sağlaması:
SPLG vs SPX(^GSPC) ve QQQM vs NDX(^NDX) günlük getiri korelasyon / beta / tracking-error (bps). 2019+ (QQQM
inception 2020-10 → mevcut tarihten). yfinance (bedava). Karar yok — sadece varsayımı görünür kılar.

  & <venv python> backtest/tracking_sanity.py
"""
from __future__ import annotations

import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _pair(etf: str, idx: str):
    import yfinance as yf
    df = yf.download([etf, idx], start="2019-01-01", progress=False, auto_adjust=True)["Close"].dropna()
    if df.empty or etf not in df or idx not in df:
        return None
    r = np.log(df / df.shift(1)).dropna()
    re_, ri = r[etf].values, r[idx].values
    corr = float(np.corrcoef(re_, ri)[0, 1])
    beta = float(np.cov(re_, ri)[0, 1] / np.var(ri))
    te_bps = float(np.std(re_ - ri) * np.sqrt(252) * 1e4)     # yıllık tracking-error, bps
    return {"n": len(r), "start": str(df.index.min().date()), "corr": corr, "beta": beta, "te_bps": te_bps}


def main():
    print("=" * 78)
    print("  H9 — DELTA-ONE ETF ≈ ENDEKS sağlaması (rapor-only, eşik yok; günlük log-getiri 2019+)")
    print("=" * 78)
    for etf, idx, lbl in [("SPLG", "^GSPC", "SPX"), ("QQQM", "^NDX", "NDX")]:
        try:
            s = _pair(etf, idx)
        except Exception as e:
            print(f"  {etf} vs {lbl}: çekilemedi ({type(e).__name__}: {str(e)[:60]})"); continue
        if not s:
            print(f"  {etf} vs {lbl}: veri yok"); continue
        print(f"  {etf:>5} vs {lbl:<4} (n={s['n']}, {s['start']}+): corr {s['corr']:.4f} | beta {s['beta']:.4f} | "
              f"tracking-error {s['te_bps']:.0f} bps/yıl")
    print("  NOT: QQQM inception 2020-10 → 2019 öncesi yok (FLAG). Yüksek-corr/beta≈1/düşük-TE → 'ETF≈endeks' OK.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
