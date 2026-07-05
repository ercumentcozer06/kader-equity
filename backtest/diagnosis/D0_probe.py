"""
backtest/diagnosis/D0_probe — KOD-GERÇEĞİ forensics (TEŞHİS-ONLY, P&L üretmez).
mtime/hash karşılaştırması: level_series_*.parquet IV-mid-fix'ten (_bsiv canonical) ÖNCE mi SONRA mı üretildi?
git yoksa os.path.getmtime + sha256 ile hüküm verir.
  & <venv python> backtest/diagnosis/D0_probe.py
"""
from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def info(p: Path) -> dict:
    st = p.stat()
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return {
        "path": str(p),
        "exists": p.exists(),
        "mtime_epoch": st.st_mtime,
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "mtime_local": datetime.fromtimestamp(st.st_mtime).isoformat(),
        "size": st.st_size,
        "sha16": h,
    }


TARGETS = {
    "_bsiv.py": ROOT / "screen" / "_bsiv.py",
    "gamma_engine.py": ROOT / "screen" / "gamma_engine.py",
    "build_level_series.py": ROOT / "backtest" / "build_level_series.py",
    "marketdata_backfill.py": ROOT / "screen" / "marketdata_backfill.py",
    "level_series_spy.parquet": ROOT / "data" / "cache" / "level_series_spy.parquet",
    "level_series_qqq.parquet": ROOT / "data" / "cache" / "level_series_qqq.parquet",
    "md_spy.parquet": ROOT / "data" / "historical_chains" / "md_spy.parquet",
    "md_qqq.parquet": ROOT / "data" / "historical_chains" / "md_qqq.parquet",
    "spine_diagnostic.py": ROOT / "backtest" / "spine_diagnostic.py",
    "disentangle.py": ROOT / "backtest" / "disentangle.py",
    "block_robust.py": ROOT / "backtest" / "block_robust.py",
}


def main() -> int:
    print("=" * 88)
    print("D0_probe — mtime/hash forensics (repo git-değil; getmtime hükmü)")
    print("=" * 88)
    rows = []
    for name, p in TARGETS.items():
        if not p.exists():
            print(f"  [YOK] {name}: {p}")
            continue
        d = info(p)
        rows.append((name, d))
        print(f"  {name:<30} mtime {d['mtime_local']}  ({d['size']:,}B, sha16 {d['sha16']})")

    print("\n--- HÜKÜM: level_series vs _bsiv (IV-mid-fix kaynağı) ---")
    bsiv = next((d for n, d in rows if n == "_bsiv.py"), None)
    gam = next((d for n, d in rows if n == "gamma_engine.py"), None)
    bld = next((d for n, d in rows if n == "build_level_series.py"), None)
    for ls_name in ("level_series_spy.parquet", "level_series_qqq.parquet"):
        ls = next((d for n, d in rows if n == ls_name), None)
        if ls is None or bsiv is None:
            continue
        dt_bsiv = ls["mtime_epoch"] - bsiv["mtime_epoch"]
        dt_gam = (ls["mtime_epoch"] - gam["mtime_epoch"]) if gam else float("nan")
        dt_bld = (ls["mtime_epoch"] - bld["mtime_epoch"]) if bld else float("nan")
        verdict = "SONRA (canonical IV-mid kullanılmış)" if dt_bsiv > 0 else "ÖNCE (eski IV ile üretilmiş olabilir!)"
        print(f"  {ls_name}:")
        print(f"    level_series mtime − _bsiv mtime         = {dt_bsiv/3600:+.2f} saat  → {verdict}")
        print(f"    level_series mtime − gamma_engine mtime  = {dt_gam/3600:+.2f} saat")
        print(f"    level_series mtime − build_level mtime   = {dt_bld/3600:+.2f} saat")

    # veri-içerik teyidi: level_series'in atm_iv kolonu _bsiv ile yeniden üretilince byte-eşleşir mi?
    print("\n--- İÇERİK TEYİDİ: level_series.atm_iv dağılımı (mid-IV imzası) ---")
    for sym in ("spy", "qqq"):
        p = ROOT / "data" / "cache" / f"level_series_{sym}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        cols = list(df.columns)
        print(f"  {sym}: {len(df)} satır, kolonlar={cols}")
        if "atm_iv" in df.columns:
            iv = df["atm_iv"]
            print(f"    atm_iv  min {iv.min():.4f}  med {iv.median():.4f}  max {iv.max():.4f}  "
                  f"(<0.02 say {(iv < 0.02).sum()}, >2.0 say {(iv > 2.0).sum()})")
        if "date" in df.columns:
            print(f"    date span {df['date'].min()} → {df['date'].max()}")
        else:
            print(f"    index span {df.index.min()} → {df.index.max()}")
        if "dte" in df.columns:
            print(f"    dte min {int(df['dte'].min())} med {int(df['dte'].median())} max {int(df['dte'].max())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
