"""
screen/fetch_skew_vvix — vol-surface kompozit bileşenleri: CBOE SKEW (kuyruk/put-talebi) + VVIX (vol-of-vol).

Kaynak: CBOE CDN günlük tarihçe CSV (kader-macro cboe_cache'inde VVIX olduğuna göre erişilebilir).
HIGH = stres yönü (yüksek SKEW = kuyruk-hedge talebi; yüksek VVIX = vol-belirsizliği). Read-only, ağ.
PIT: market-close → lag 0 (engine +1g uygular).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CACHE = ROOT / "data" / "cache"
URLS = {
    "SKEW": "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv",
    "VVIX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv",
}


def _parse(text: str, name: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    # tarih kolonu + son sayısal kolon (close) — CBOE CDN format değişebilir, defansif
    dcol = [c for c in df.columns if "date" in c.lower()][0]
    num = [c for c in df.columns if c != dcol and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8]
    vcol = num[-1]                                        # CLOSE genelde son sayısal kolon
    s = pd.Series(pd.to_numeric(df[vcol], errors="coerce").values,
                  index=pd.to_datetime(df[dcol], errors="coerce"), name=name).dropna()
    return s[~s.index.duplicated(keep="last")].sort_index()


def main() -> int:
    out = {}
    for name, url in URLS.items():
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            s = _parse(r.text, name)
            out[name] = s
            print(f"  {name:<6}: {len(s)} gün {s.index.min().date()}..{s.index.max().date()}  son={s.iloc[-1]:.1f}")
            print(f"          ilk satırlar CSV: {r.text.splitlines()[0][:80]}")
        except Exception as e:
            print(f"  {name:<6}: FETCH HATA — {type(e).__name__}: {str(e)[:90]}  (endpoint cdn.cboe.com)")
    if not out:
        print("  [!] hiçbiri çekilemedi.")
        return 1
    df = pd.DataFrame(out).dropna(how="all")
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE / "skew_vvix.parquet")
    print(f"\n  saved -> {CACHE / 'skew_vvix.parquet'}  ({df.shape[0]} gün, kolonlar {list(df.columns)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
