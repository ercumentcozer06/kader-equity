"""
backtest/qqq_rr_skew_history — QQQ 25Δ risk-reversal tarihcesi 2012-2023, optionsdx EOD setinden.

Kaynak: optionsdx.com QQQ Option Chains, EOD, 2012-2023 ($0 — siparis 19019, hesap=Emir'in maili).
Ham zip/txt'ler: data/raw_optionsdx_qqq/ (yillik zipler; icinde aylik .txt, optionsdx bracket-kolon
formati — Kaggle SPY setiyle AYNI format). Tanim/hijyen kaggle_rr_skew_history ile birebir ayni
(oradan import — kopya degil): 25Δ nearest-delta RR, 30g+90g sabit vade, IV/delta/DTE filtreleri.
→ data/cache/rr_skew_qqq_2012_2023.parquet (+ t30_ok/t90_ok kalite bayraklari, SPY'dekiyle ayni kural)
  & <kader-macro venv python> backtest/qqq_rr_skew_history.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from kaggle_rr_skew_history import COLS, TARGETS, _pick   # noqa: E402  (secim kurali tek kaynak)

RAW = ROOT / "data" / "raw_optionsdx_qqq"


def _frames():
    """raw_optionsdx_qqq/txt/ altindaki aylik .txt'ler (7z'lerden acilmis) → DataFrame'ler."""
    for f in sorted((RAW / "txt").rglob("*.txt")):
        yield f.name, pd.read_csv(f, low_memory=False)


def main() -> int:
    rows = []
    n_files = 0
    for name, df in _frames():
        df.columns = [c.strip() for c in df.columns]
        missing = [c for c in COLS if c.strip("[]") not in {x.strip("[]") for x in df.columns}]
        if missing:
            print(f"  {name}: kolon eksik {missing[:3]} — atla")
            continue
        df.columns = [c.strip("[]").strip() for c in df.columns]
        for c in ("DTE", "C_DELTA", "C_IV", "P_DELTA", "P_IV", "UNDERLYING_LAST"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[df["DTE"] >= 5]
        n_files += 1
        for d, day in df.groupby("QUOTE_DATE"):
            rec = {"date": pd.Timestamp(str(d).strip()),
                   "spot": float(day["UNDERLYING_LAST"].iloc[0])}
            dtes = day["DTE"].unique()
            for key, tgt in TARGETS.items():
                cand = dtes[abs(dtes - tgt) <= tgt * 0.6]
                if not len(cand):
                    continue
                dte = float(sorted(cand, key=lambda x: abs(x - tgt))[0])
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
    if not rows:
        raise RuntimeError(f"veri yok — {RAW} altinda gecerli zip/txt bulunamadi (once indirme tamamlanmali)")
    out = pd.DataFrame(rows).drop_duplicates(subset=["date"], keep="last").set_index("date").sort_index()
    for t in ("t30", "t90"):
        bad = (out[f"{t}_put25_iv"] < 0.5 * out[f"{t}_atm_iv"]) | \
              (out[f"{t}_call25_iv"] < 0.3 * out[f"{t}_atm_iv"])
        out[f"{t}_ok"] = out[f"{t}_rr_skew"].notna() & ~bad.fillna(True)
    p = ROOT / "data" / "cache" / "rr_skew_qqq_2012_2023.parquet"
    out.to_parquet(p)
    rr = out.loc[out["t30_ok"], "t30_rr_skew"]
    print(f"→ {p}  ({n_files} dosya, {len(out)} gun, {out.index.min().date()}→{out.index.max().date()})")
    print(f"temiz RR30: n={len(rr)}, medyan {rr.median():+.2f}, p10 {rr.quantile(.1):+.2f}, "
          f"p90 {rr.quantile(.9):+.2f}, max {rr.max():+.2f} ({rr.idxmax().date()})")
    yr = out.groupby(out.index.year)["t30_rr_skew"].median()
    print("yillik medyan RR30:", {int(y): round(float(v), 2) for y, v in yr.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
