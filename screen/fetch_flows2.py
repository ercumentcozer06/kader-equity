"""
screen/fetch_flows2 — GERÇEK flow verisi (free, kazıldı): FINRA margin debt (1997+ aylık) + ICI investor
flow-of-funds (datahub, equity fund flows). İndir + parse + kaydet. Yapıyı bas (gerekirse adapte ederim).
"""
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
    # ── 1) FINRA margin debt ──
    for url in ("https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx",
                "https://www.finra.org/sites/default/files/margin-statistics.xlsx"):
        try:
            r = requests.get(url, headers=UA, timeout=45)
            if r.status_code != 200 or len(r.content) < 5000:
                print(f"  FINRA {url[-40:]}: HTTP {r.status_code} / boş")
                continue
            raw = pd.read_excel(io.BytesIO(r.content), header=None)
            # başlık satırını bul (içinde 'Debit' geçen)
            hrow = next((i for i in range(min(10, len(raw))) if raw.iloc[i].astype(str).str.contains("Debit", case=False).any()), 0)
            df = pd.read_excel(io.BytesIO(r.content), header=hrow)
            print(f"  FINRA OK {url[-30:]}  shape {df.shape}  kolonlar {list(df.columns)[:5]}")
            # tarih + debit kolonu
            dcol = next((c for c in df.columns if "month" in str(c).lower() or "date" in str(c).lower() or "year" in str(c).lower()), df.columns[0])
            debit = next((c for c in df.columns if "debit" in str(c).lower()), None)
            if debit is not None:
                s = pd.Series(pd.to_numeric(df[debit], errors="coerce").values,
                              index=pd.to_datetime(df[dcol], errors="coerce")).dropna()
                s = s[~s.index.duplicated(keep="last")].sort_index()
                out["margin_debt"] = s
                print(f"    margin_debt: {len(s)} ay {s.index.min().date()}..{s.index.max().date()}  son {s.iloc[-1]/1e9:.0f}B")
            break
        except Exception as e:
            print(f"  FINRA {url[-30:]}: {type(e).__name__}: {str(e)[:60]}")

    # ── 2) ICI investor flow of funds (datahub) ──
    for fn in ("weekly", "monthly"):
        for u in (f"https://raw.githubusercontent.com/datasets/investor-flow-of-funds-us/master/data/{fn}.csv",
                  f"https://raw.githubusercontent.com/datasets/investor-flow-of-funds-us/main/data/{fn}.csv"):
            try:
                r = requests.get(u, headers=UA, timeout=30)
                if r.status_code == 200 and "," in r.text[:300]:
                    df = pd.read_csv(io.StringIO(r.text))
                    print(f"  ICI {fn}: shape {df.shape}  kolonlar {list(df.columns)}")
                    print(f"    ilk {df.iloc[0].to_dict()}")
                    print(f"    son  {df.iloc[-1].to_dict()}")
                    dcol = [c for c in df.columns if "date" in c.lower()][0]
                    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
                    df = df.dropna(subset=[dcol]).set_index(dcol).sort_index()
                    # equity flow kolonu
                    eq = next((c for c in df.columns if c.lower().strip() in ("total equity", "equity", "domestic equity")), None)
                    if eq:
                        out[f"ici_{fn}_equity"] = pd.to_numeric(df[eq], errors="coerce").dropna()
                        print(f"    -> equity flow '{eq}' alındı ({fn})")
                    break
            except Exception as e:
                print(f"  ICI {fn} {u[-20:]}: {type(e).__name__}")
        else:
            continue

    if not out:
        print("  [!] hiçbir flow serisi alınamadı.")
        return 1
    CACHE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_parquet(CACHE / "flows2.parquet")
    print(f"\n  saved -> {CACHE / 'flows2.parquet'}  ({list(out)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
