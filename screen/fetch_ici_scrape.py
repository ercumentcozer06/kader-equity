"""screen/fetch_ici_scrape — ICI tam geçmiş AVI. datahub repo'sunun TÜM data dosyalarını (GitHub API) +
git-tarihindeki en uzun versiyonu + ICI yıllık Excel'leri tara. En uzun equity-flow serisini bul + kaydet."""
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


def _eqcol(df):
    for c in df.columns:
        if str(c).strip().lower() in ("total equity", "equity", "domestic equity"):
            return c
    return None


def main() -> int:
    best = None
    # ── 1) GitHub API: datahub repo data klasörü ──
    for branch in ("main", "master"):
        try:
            api = f"https://api.github.com/repos/datasets/investor-flow-of-funds-us/contents/data?ref={branch}"
            r = requests.get(api, headers={**UA, "Accept": "application/vnd.github+json"}, timeout=30)
            if r.status_code != 200:
                print(f"  GitHub API {branch}: HTTP {r.status_code}"); continue
            for f in r.json():
                if not str(f.get("name", "")).endswith((".csv", ".json")):
                    continue
                try:
                    rr = requests.get(f["download_url"], headers=UA, timeout=30)
                    df = pd.read_csv(io.StringIO(rr.text)) if f["name"].endswith(".csv") else pd.DataFrame(rr.json())
                    dcol = [c for c in df.columns if "date" in str(c).lower()]
                    ec = _eqcol(df)
                    if dcol and ec is not None:
                        dts = pd.to_datetime(df[dcol[0]], errors="coerce").dropna()
                        print(f"  {f['name']:<24} {len(df)} satır  {dts.min().date()}..{dts.max().date()}  eq-col '{ec}'")
                        if best is None or len(df) > best[0]:
                            s = pd.Series(pd.to_numeric(df[ec], errors="coerce").values,
                                          index=pd.to_datetime(df[dcol[0]], errors="coerce")).dropna()
                            best = (len(df), f["name"], s)
                    else:
                        print(f"  {f['name']:<24} {len(df)} satır  (date/equity kolonu yok: {list(df.columns)[:6]})")
                except Exception as e:
                    print(f"  {f['name']}: parse {type(e).__name__}")
            break
        except Exception as e:
            print(f"  GitHub API {branch}: {type(e).__name__}: {str(e)[:50]}")

    # ── 2) datahub git-history: eski commit'lerde daha uzun weekly olabilir ──
    try:
        commits = requests.get("https://api.github.com/repos/datasets/investor-flow-of-funds-us/commits?path=data/weekly.csv&per_page=100",
                               headers={**UA, "Accept": "application/vnd.github+json"}, timeout=30).json()
        print(f"  weekly.csv commit sayısı: {len(commits)}")
        if isinstance(commits, list) and len(commits) > 1:
            oldest = commits[-1]["sha"]
            rr = requests.get(f"https://raw.githubusercontent.com/datasets/investor-flow-of-funds-us/{oldest}/data/weekly.csv", headers=UA, timeout=30)
            if rr.status_code == 200:
                df = pd.read_csv(io.StringIO(rr.text)); ec = _eqcol(df); dcol = [c for c in df.columns if "date" in str(c).lower()]
                if ec and dcol:
                    dts = pd.to_datetime(df[dcol[0]], errors="coerce").dropna()
                    print(f"  EN ESKİ commit weekly.csv: {len(df)} satır {dts.min().date()}..{dts.max().date()}")
    except Exception as e:
        print(f"  git-history: {type(e).__name__}: {str(e)[:50]}")

    if best:
        best[2].to_frame("equity_flow").to_parquet(CACHE / "ici_best.parquet")
        print(f"\n  EN UZUN seri: {best[1]} ({best[0]} satır) -> ici_best.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
