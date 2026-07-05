"""
tide — equities macro-tide spine: donmuş 8-modül sweep vektörü → TIDE_SCORE / TIDE_DIR.

TIDE_SCORE(t) = Σ_i w_i · score_i(t)          (eksik modül = 0 oy; w = donmuş kazanan vektör)
TIDE_DIR(t)   = 1 (LONG) if TIDE_SCORE > 0 else 0 (FLAT)
pozisyon(t)   = TIDE_DIR(t−1)                  (+1 gün exec lag → look-ahead-free; engine uygular)

Vektör (kanıtlı): m9 .563 / m5 .214 / m2(RAW) .118 / m0 .061 / m3 .025 / m6 .01 / m8 .006 / m4 .002.
Defansif yön + düşük-beta (~0.5) zemin; asıl alfa downstream overlay'lerden gelir (gate'te modüle edilir).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def tide_score_series(scores: pd.DataFrame, vector: dict) -> pd.Series:
    """Günlük TIDE_SCORE serisi. scores: [date × modül] kapped ±20; vector: {modül: w}. Eksik modül = 0."""
    cols = [m for m in vector if m in scores.columns]
    W = np.array([float(vector[m]) for m in cols])
    X = np.nan_to_num(scores[cols].values, nan=0.0)
    return pd.Series(X @ W, index=scores.index, name="tide_score")


def tide_dir_series(score: pd.Series) -> pd.Series:
    """LONG(1) if TIDE_SCORE > 0 else FLAT(0). (exec lag engine'de uygulanır.)"""
    return (score > 0).astype(float).rename("tide_dir")


def decide(scores: dict, vector: dict) -> dict:
    """Canlı tek-tarih tide. scores = {modül: kapped skor}; eksik/None modül = 0 oy.

    Audit 2026-06-19: eksik modül SESSİZCE 0-oy olarak düşüyordu (census/halt/flag yok) — m9 tek
    başına ağırlığın ~%56'sı. tide_score/tide_dir DEĞİŞMEZ (eksik=0 davranışı korunur), ama eksik
    modüller + kayıp-ağırlık ARTIK GÖRÜNÜR (degraded bayrağı): fail-open yerine fail-VISIBLE.
    """
    s = 0.0
    used, missing = [], []
    miss_w = 0.0
    for m, w in vector.items():
        v = scores.get(m)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            if abs(float(w)) > 1e-9:
                missing.append(m)
                miss_w += abs(float(w))
            continue
        s += float(w) * float(v)
        if abs(float(w)) > 1e-9:
            used.append(m)
    tot_w = sum(abs(float(w)) for w in vector.values()) or 1.0
    return {"tide_score": round(float(s), 4), "tide_dir": int(s > 0), "modules_used": used,
            "modules_missing": missing, "missing_weight_frac": round(miss_w / tot_w, 4),
            "degraded": bool(missing)}
