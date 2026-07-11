"""
screen/gex_naive_sentinel — NAIVE-NET vs SqueezeMetrics GEX nobetcisi (betimsel forward-arsiv, 2026-07-11).

Amac: bizim zincir-okumamiz SM'in seri-uretimiyle ayni nabzi tutuyor mu? Her gun iki sayi arsivlenir:
  our_naive_bn = _SPX TUM-seri, naive isaret (C+1/P-1), CBOE gammasi, SM white-paper formulu (S^2*0.01)
  sm_gex_bn    = SqueezeMetrics'in YAYINLADIGI gunluk GEX (DIX.csv son satiri; kendi tarihiyle damgali)
~6 ayda korelasyon/isaret-uyumu olculur (mutlak olcek farki biliniyor ~10x — onemli olan ko-hareket).
BEST-EFFORT: collect_daily cekirdek kapisina girmez. Dedup as_of bazinda.
→ output/gex_naive_sm_ledger.parquet
  & <venv python> screen/gex_naive_sentinel.py
"""
from __future__ import annotations

import io
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "screen"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import _cboe_lib as L                                   # noqa: E402
from modules._netutil import http_get_retry             # noqa: E402

OUT = ROOT / "output" / "gex_naive_sm_ledger.parquet"


def main() -> int:
    spot, rows = L.load_rows("_SPX", band=3.0)
    naive = sum((1.0 if r["cp"] == "C" else -1.0) * r["g_cboe"] * r["oi"] * 100 * spot * spot * 0.01
                for r in rows if r["g_cboe"] is not None)
    sm_date, sm_gex = None, None
    try:
        r = http_get_retry("https://squeezemetrics.com/monitor/static/DIX.csv", timeout=60)
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip().lower() for c in df.columns]
        sm_date, sm_gex = str(df.iloc[-1]["date"]), float(df.iloc[-1]["gex"]) / 1e9
    except Exception as e:
        print(f"  [~] SM DIX.csv alinamadi ({type(e).__name__}) — our_naive yine arsivlenir")
    rec = {"as_of": date.today().isoformat(), "spot": round(float(spot), 2),
           "our_naive_bn": round(naive / 1e9, 3), "sm_date": sm_date,
           "sm_gex_bn": round(sm_gex, 3) if sm_gex is not None else None}
    old = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame()
    out = pd.concat([old, pd.DataFrame([rec])], ignore_index=True)
    out = out.drop_duplicates(subset=["as_of"], keep="last")
    out.to_parquet(OUT)
    print(f"  gex-naive nobetcisi: biz {rec['our_naive_bn']:+.2f}bn | SM({sm_date}) "
          f"{rec['sm_gex_bn']}bn → {OUT.name} ({len(out)} satir)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
