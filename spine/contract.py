"""
contract — spine erişim katmanı: donmuş equities-tide snapshot'ını okur + tazelik + KİLİTLER.

Runtime kader-macro'yu İMPORT ETMEZ (yalnız spine/gen_snapshot.py build-tool eder). Burası donmuş
parquet/json'u okur, KİLİTLERİ doğrular (RAW m2 + 8-modül + çapa Sharpe ≥ 1.40), tazelik kapısını
uygular. Parite (_tide_liveparity) kanıtı: smart-RRP m2 → 1.02/1.19 RED, top-4 → 1.16/1.28 RED.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FROZEN = Path(__file__).resolve().parent / "frozen"
ANCHOR_MIN_SHARPE = 1.40


def snapshot_freshness(as_of, max_staleness_days: int, today=None) -> dict:
    """Snapshot TAKVİM-YAŞI kapısı (Bible: bayat veri güncel çağrı olarak sunulamaz).
    age = bugün − snapshot son tarihi; age > eşik → STALE (yalnız as_of durumu, güncel call değil)."""
    today = today or datetime.now(timezone.utc).date()
    as_of_d = as_of.date() if hasattr(as_of, "date") else pd.Timestamp(as_of).date()
    age = (today - as_of_d).days
    thr = int(max_staleness_days)
    return {"as_of": str(as_of_d), "today": str(today), "age_days": int(age),
            "max_staleness_days": thr, "stale": age > thr}


def _assert_locks(prov: dict) -> None:
    """Donmuş snapshot gerçekten raw-m2 + 8-modül + çapa ≥ 1.40 tabanından mı."""
    locks = prov.get("locks", {}) or {}
    if not bool(locks.get("raw_m2", False)):
        raise RuntimeError("tide spine RAW net-liq m2 gerektirir (smart-RRP → 1.02/1.19 RED). provenance.locks.raw_m2 False.")
    if int(locks.get("n_modules_nonzero", 0)) < 8:
        raise RuntimeError(f"tide spine 8 modül gerektirir (top-4 → 1.16/1.28 RED). n_modules_nonzero={locks.get('n_modules_nonzero')}.")
    a = (prov.get("anchor", {}) or {}).get("SPX", {}) or {}
    if not (float(a.get("sharpe", 0.0)) >= ANCHOR_MIN_SHARPE):
        raise RuntimeError(f"FROZEN çapa SPX Sharpe {a.get('sharpe')} < {ANCHOR_MIN_SHARPE} — raw-winner tabanı değil. "
                           "spine/gen_snapshot.py ile yeniden üret (kader-macro venv).")


def read_frozen() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Döndürür: (module_scores_df, prices_df, vector_dict, provenance_dict). KİLİT doğrulanır."""
    prov_p = FROZEN / "provenance.json"
    if not prov_p.exists():
        raise FileNotFoundError(f"donmuş tide snapshot yok: {FROZEN}\n  → spine/gen_snapshot.py (kader-macro venv) ile üret.")
    prov = json.loads(prov_p.read_text(encoding="utf-8"))
    _assert_locks(prov)
    scores = pd.read_parquet(FROZEN / "module_scores.parquet")
    prices = pd.read_parquet(FROZEN / "prices.parquet")
    vector = json.loads((FROZEN / "vector.json").read_text(encoding="utf-8"))
    return scores, prices, vector, prov
