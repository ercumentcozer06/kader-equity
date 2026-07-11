"""
backtest/optionsdx_rr_skew_history — GENEL optionsdx 25Δ risk-reversal tarihce ekstraktoru (sym-parametreli).

qqq_rr_skew_history'nin genellesmis hali: data/raw_optionsdx_<sym>/txt/ altindaki aylik optionsdx
dosyalarindan rr_skew tarihcesi cikarir. Tanim/hijyen SPY/QQQ arsivleriyle BIREBIR ayni (secim kurali
kaggle_rr_skew_history._pick'ten import — kopya yok): 25Δ nearest-delta RR (IVpt), 30g+90g sabit vade,
t30_ok/t90_ok kalite bayraklari. Ilk kullanim: SLV 2016-2023 (optionsdx $0, siparis 19044) — metal
skew persentil hammaddesi (deira brief SLV RR25 satirinin tarihsel baglami; cockpit baglantisi AYRI is).
NOT: SLV'de RR isaret-yapisi equity'den FARKLI olabilir (gumus manialarinda CALL-primi → negatif RR mesru).
→ data/cache/rr_skew_<sym>_<ilk>_<son>.parquet
  & <kader-macro venv python> backtest/optionsdx_rr_skew_history.py slv
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

from kaggle_rr_skew_history import COLS, TARGETS, _pick   # noqa: E402  (secim kurali tek-kaynak)


def main() -> int:
    sym = (sys.argv[1] if len(sys.argv) > 1 else "slv").lower()
    raw = ROOT / "data" / f"raw_optionsdx_{sym}" / "txt"
    files = sorted(raw.rglob("*.txt"))
    if not files:
        raise SystemExit(f"veri yok: {raw}")
    rows = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip().strip("[]").strip() for c in df.columns]
        need = {c.strip("[]") for c in COLS}
        if not need.issubset(set(df.columns)):
            print(f"  {f.name}: kolon eksik — atla")
            continue
        for c in ("DTE", "C_DELTA", "C_IV", "P_DELTA", "P_IV", "UNDERLYING_LAST"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[df["DTE"] >= 5]
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
    out = pd.DataFrame(rows).drop_duplicates(subset=["date"], keep="last").set_index("date").sort_index()
    for t in ("t30", "t90"):
        bad = (out[f"{t}_put25_iv"] < 0.5 * out[f"{t}_atm_iv"]) | \
              (out[f"{t}_call25_iv"] < 0.3 * out[f"{t}_atm_iv"])
        out[f"{t}_ok"] = out[f"{t}_rr_skew"].notna() & ~bad.fillna(True)
    y0, y1 = out.index.min().year, out.index.max().year
    p = ROOT / "data" / "cache" / f"rr_skew_{sym}_{y0}_{y1}.parquet"
    out.to_parquet(p)
    rr = out.loc[out["t30_ok"], "t30_rr_skew"]
    print(f"→ {p.name}  ({len(files)} dosya, {len(out)} gun, {out.index.min().date()}→{out.index.max().date()})")
    print(f"temiz RR30: n={len(rr)}, medyan {rr.median():+.2f}, p10 {rr.quantile(.1):+.2f}, "
          f"p90 {rr.quantile(.9):+.2f}, min {rr.min():+.2f} ({rr.idxmin().date()}), "
          f"max {rr.max():+.2f} ({rr.idxmax().date()})")
    yr = out.groupby(out.index.year)["t30_rr_skew"].median()
    print("yillik medyan RR30:", {int(y): round(float(v), 2) for y, v in yr.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
