"""
backtest/kaggle_rr_skew_history — SPY 25Δ risk-reversal (rr_skew) tarihcesi 2010-2023, Kaggle EOD setinden.

Kaynak: kagglehub dudesurfin/spy-options-eod-volatility-surface-2010-2023 (optionsdx format, gercek bid/ask
IV+delta hazir; OI YOK — bu is icin gerekmiyor). Tanim canli terminalle AYNI (vol_surface_live.py):
  rr_skew = (IV(25Δ put) − IV(25Δ call)) × 100, secim = en-yakin-delta (interpolasyon yok), DTE basina.
Tarihce icin sabit-vade: hedef DTE 30 ve 90 → o gune en yakin DTE'li expiry (tolerans ±%60). Hijyen:
0.01<IV<3, |Δ−0.25|≤0.08 sarti (yoksa NaN), DTE≥5. Pozitif RR = put primi (korku).
→ data/cache/rr_skew_spy_2010_2023.parquet  (date, spot, t30/t90: dte, atm_iv, put25/call25_iv, rr_skew)
  & <kader-macro venv python> backtest/kaggle_rr_skew_history.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = Path.home() / ".cache" / "kagglehub" / "datasets" / "dudesurfin" / \
    "spy-options-eod-volatility-surface-2010-2023" / "versions" / "2"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

COLS = ["[QUOTE_DATE]", "[UNDERLYING_LAST]", "[DTE]", "[STRIKE]",
        "[C_DELTA]", "[C_IV]", "[P_DELTA]", "[P_IV]"]
TARGETS = {"t30": 30, "t90": 90}
DELTA_TOL = 0.08


def _pick(g: pd.DataFrame, dcol: str, ivcol: str, target: float):
    """En-yakin-delta secimi (canli kodla ayni: min|Δ−hedef|); tolerans disiysa None."""
    v = g[[dcol, ivcol]].to_numpy(dtype=float)
    v = v[(v[:, 1] > 0.01) & (v[:, 1] < 3.0) & ~np.isnan(v[:, 0])]
    if not len(v):
        return None
    i = np.argmin(np.abs(v[:, 0] - target))
    return float(v[i, 1]) if abs(v[i, 0] - target) <= DELTA_TOL else None


def one_year(f: Path) -> list[dict]:
    df = pd.read_parquet(f, columns=COLS)
    df.columns = [c.strip("[]") for c in df.columns]
    for c in ("DTE", "C_DELTA", "C_IV", "P_DELTA", "P_IV", "UNDERLYING_LAST"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["DTE"] >= 5]
    rows = []
    for d, day in df.groupby("QUOTE_DATE"):
        rec = {"date": pd.Timestamp(str(d).strip()), "spot": float(day["UNDERLYING_LAST"].iloc[0])}
        dtes = day["DTE"].unique()
        for key, tgt in TARGETS.items():
            cand = dtes[np.abs(dtes - tgt) <= tgt * 0.6]
            if not len(cand):
                continue
            dte = float(cand[np.argmin(np.abs(cand - tgt))])
            g = day[day["DTE"] == dte]
            p25 = _pick(g, "P_DELTA", "P_IV", -0.25)
            c25 = _pick(g, "C_DELTA", "C_IV", +0.25)
            atm = _pick(g, "C_DELTA", "C_IV", +0.50)
            rec[f"{key}_dte"] = dte
            rec[f"{key}_atm_iv"] = round(atm * 100, 2) if atm else None
            rec[f"{key}_put25_iv"] = round(p25 * 100, 2) if p25 else None
            rec[f"{key}_call25_iv"] = round(c25 * 100, 2) if c25 else None
            rec[f"{key}_rr_skew"] = round((p25 - c25) * 100, 2) if (p25 and c25) else None
        rows.append(rec)
    return rows


def main() -> int:
    files = sorted(SRC.glob("spy_eod_*.parquet"))
    if not files:
        raise RuntimeError(f"Kaggle parquet bulunamadi: {SRC}")
    allrows = []
    for f in files:
        rows = one_year(f)
        allrows += rows
        ok = sum(1 for r in rows if r.get("t30_rr_skew") is not None)
        print(f"  {f.stem}: {len(rows)} gun, rr30 dolu {ok} (%{100*ok/max(len(rows),1):.0f})")
    out = pd.DataFrame(allrows).set_index("date").sort_index()
    p = ROOT / "data" / "cache" / "rr_skew_spy_2010_2023.parquet"
    out.to_parquet(p)
    print(f"\n→ {p}  ({len(out)} gun, {out.index.min().date()}→{out.index.max().date()})")

    rr = out["t30_rr_skew"].dropna()
    print(f"\nRR30 OZET: medyan {rr.median():+.2f} IVpt, p10 {rr.quantile(.1):+.2f}, p90 {rr.quantile(.9):+.2f}, "
          f"min {rr.min():+.2f}, max {rr.max():+.2f}")
    print("Yillik medyan RR30 / ATM30:")
    yr = out.groupby(out.index.year).agg(rr30=("t30_rr_skew", "median"), atm=("t30_atm_iv", "median"),
                                         n=("t30_rr_skew", "count"))
    for y, r in yr.iterrows():
        print(f"  {y}: RR {r['rr30']:+.2f}  ATM {r['atm']:.1f}%  (n{int(r['n'])})")
    print("\nEn yuksek 8 RR30 gunu (korku tepeleri):")
    for d, v in rr.nlargest(8).items():
        print(f"  {d.date()}  {v:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
