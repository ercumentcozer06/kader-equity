"""screen/fetch_ici_full — TAM ICI weekly flows geçmişi (2007+). datahub 2022+ idi; ICI Excel uzun-tarih.
İndir + incele (sheet/kolon/aralık). Sonra weekly contrarian-flow sinyalini UZUN pencerede re-test."""
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

URLS = [
    "https://www.ici.org/system/files/2022-01/combined_flows_data_2022.xls",
    "https://www.ici.org/system/files/combined_flows_data.xls",
    "https://www.ici.org/system/files/2025-01/combined_flows_data_2025.xls",
    "https://www.ici.org/system/files/2024-01/combined_flows_data_2024.xls",
]


def main() -> int:
    for url in URLS:
        try:
            r = requests.get(url, headers=UA, timeout=50)
            print(f"  {url[-40:]}: HTTP {r.status_code}, {len(r.content)} byte")
            if r.status_code != 200 or len(r.content) < 10000:
                continue
            xl = pd.ExcelFile(io.BytesIO(r.content))
            print(f"  sheet'ler: {xl.sheet_names}")
            for sh in xl.sheet_names[:6]:
                d = xl.parse(sh, header=None, nrows=8)
                print(f"  --- '{sh}' ---")
                print(d.to_string(max_cols=10)[:900])
            (CACHE / "_ici_full.xls").write_bytes(r.content)
            print(f"  kaydedildi -> {CACHE / '_ici_full.xls'}")
            return 0
        except Exception as e:
            print(f"  {url[-30:]}: {type(e).__name__}: {str(e)[:60]}")
    print("  [!] ICI Excel çekilemedi — alternatif: ICI weekly sayfasından / datahub farklı branch.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
