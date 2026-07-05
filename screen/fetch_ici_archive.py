"""screen/fetch_ici_archive — ICI yıllık combined-flows Excel arşivini STITCH. Her yıllık dosya ~2yıl
trailing weekly içerir; çok yılı birleştirip 2008+ tam weekly equity-flow rekonstrükte et. Çok URL-pattern dene."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
CACHE = ROOT / "data" / "cache"
UA = {"User-Agent": "Mozilla/5.0"}


def _parse(content):
    xl = pd.ExcelFile(io.BytesIO(content))
    sh = xl.sheet_names[0]
    # başlık satırını bul (içinde 'Date' + 'Equity')
    raw = xl.parse(sh, header=None, nrows=12)
    hrow = next((i for i in range(len(raw)) if raw.iloc[i].astype(str).str.contains("Date", case=False).any()
                 and raw.iloc[i].astype(str).str.contains("Equity", case=False).any()), 4)
    d = xl.parse(sh, header=hrow)
    dcol = [c for c in d.columns if "date" in str(c).lower()][0]
    ec = next((c for c in d.columns if str(c).strip().lower() == "equity"), None)
    if ec is None:
        ec = next((c for c in d.columns if "equity" in str(c).lower()), None)
    s = pd.Series(pd.to_numeric(d[ec], errors="coerce").values,
                  index=pd.to_datetime(d[dcol], errors="coerce")).dropna()
    return s[~s.index.duplicated(keep="last")]


def main() -> int:
    series = []
    for Y in range(2009, 2027):
        got = False
        for mm in ("01", "02", "03", "12"):
            for path in (f"{Y}-{mm}/combined_flows_data_{Y}.xls", f"{Y}-{mm}/combined_flows_data_{Y}.xlsx"):
                url = f"https://www.ici.org/system/files/{path}"
                try:
                    r = requests.get(url, headers=UA, timeout=30)
                    if r.status_code == 200 and len(r.content) > 10000:
                        s = _parse(r.content)
                        if len(s) > 10:
                            series.append(s)
                            print(f"  {Y}: OK ({path[:18]})  {len(s)} hafta {s.index.min().date()}..{s.index.max().date()}")
                            got = True
                            break
                except Exception:
                    pass
            if got:
                break
        if not got:
            print(f"  {Y}: bulunamadı")
    # datahub weekly de ekle
    try:
        dh = pd.read_parquet(CACHE / "flows2.parquet")["ici_weekly_equity"].dropna()
        series.append(dh)
    except Exception:
        pass
    if not series:
        print("  [!] hiç dosya bulunamadı.")
        return 1
    full = pd.concat(series).sort_index()
    full = full[~full.index.duplicated(keep="last")]
    full.to_frame("equity_flow").to_parquet(CACHE / "ici_full_weekly.parquet")
    print(f"\n  STITCH SONUCU: {len(full)} hafta {full.index.min().date()}..{full.index.max().date()} -> ici_full_weekly.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
