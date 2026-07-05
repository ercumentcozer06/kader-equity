"""screen/fetch_fundamental — S&P 500 EPS estimate Excel (resmi, free) + Shiller CAPE/earnings. İndir+incele+kaydet."""
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


def main() -> int:
    out = {}
    # ── 1) S&P resmi operating EPS + estimates ──
    try:
        r = requests.get("https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx",
                         headers=UA, timeout=60)
        print(f"  S&P EPS xlsx: HTTP {r.status_code}, {len(r.content)} byte")
        if r.status_code == 200 and len(r.content) > 10000:
            xl = pd.ExcelFile(io.BytesIO(r.content))
            print(f"  sheet'ler: {xl.sheet_names}")
            for sh in xl.sheet_names[:3]:
                df = xl.parse(sh, header=None, nrows=12)
                print(f"  --- sheet '{sh}' ilk satırlar ---")
                print(df.to_string(max_cols=8)[:1400])
            (CACHE / "_sp_eps.xlsx").write_bytes(r.content)
            print(f"  ham xlsx kaydedildi -> {CACHE / '_sp_eps.xlsx'}")
    except Exception as e:
        print(f"  S&P EPS: {type(e).__name__}: {str(e)[:80]}")

    # ── 2) Shiller CAPE + earnings (mirror) ──
    for url in ("https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9- ab02-/downloads/ie_data.xls",
                "http://www.econ.yale.edu/~shiller/data/ie_data.xls",
                "https://raw.githubusercontent.com/datasets/s-and-p-500-shiller/master/data/data.csv"):
        try:
            r = requests.get(url, headers=UA, timeout=40)
            if r.status_code != 200 or len(r.content) < 5000:
                print(f"  Shiller {url[-20:]}: HTTP {r.status_code}")
                continue
            if url.endswith(".csv"):
                df = pd.read_csv(io.StringIO(r.text))
                print(f"  Shiller CSV: shape {df.shape} kolonlar {list(df.columns)[:10]}")
            else:
                xl = pd.ExcelFile(io.BytesIO(r.content))
                d = xl.parse("Data", header=7) if "Data" in xl.sheet_names else xl.parse(0, header=7)
                print(f"  Shiller xls 'Data': shape {d.shape} kolonlar {list(d.columns)[:12]}")
            (CACHE / "_shiller.bin").write_bytes(r.content)
            print(f"  Shiller kaydedildi ({url[-20:]})")
            break
        except Exception as e:
            print(f"  Shiller {url[-20:]}: {type(e).__name__}: {str(e)[:50]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
