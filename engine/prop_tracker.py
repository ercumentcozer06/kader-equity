"""
engine/prop_tracker — B2: FTMO hesabı günlük takip + execution-drag ölçümü. Hesap-başına parquet satırı.
model_equity (endeks-getirisinden OTOMATİK) vs platform_equity (Emir manuel, opsiyonel) → execution-drag =
signal/expression-split felsefesinin prop'a uygulanması. FTMO_HESAP girilince aktive (run_daily'ye bağlı).

Şema: date, account, phase, model_equity, platform_equity, day_pnl, cum_pnl, dist_to_target_pp,
dist_to_daily_pp, dist_to_totaldd_pp, trading_days, position_exposure, lot. Yalnız pandas+parquet, ağ yok.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLS = ["date", "account", "phase", "model_equity", "platform_equity", "day_pnl", "cum_pnl",
        "dist_to_target_pp", "dist_to_daily_pp", "dist_to_totaldd_pp", "trading_days",
        "position_exposure", "lot", "note"]
# FTMO Swing 2-Step limitler (initial-balance oranı)
P_TARGET = {"P1": 0.10, "P2": 0.05}
DAILY_LIM, TOTAL_LIM = 0.05, 0.10


def ledger_path(account: str) -> Path:
    return ROOT / "output" / f"prop_tracker_{account}.parquet"


def append_day(account: str, date_: str, phase: str, day_index_return: float, position_exposure: float,
               lot: float = None, platform_equity: float = None, note: str = "") -> pd.DataFrame:
    """Bir günlük satır ekle. model_equity = önceki model_equity × (1 + pozisyon×endeks-getirisi). as_of dedup."""
    p = ledger_path(account); p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=COLS)
    for c in COLS:
        if c not in df.columns:
            df[c] = pd.NA
    prev = df[df["account"] == account]
    prev_eq = float(prev["model_equity"].iloc[-1]) if len(prev) and pd.notna(prev["model_equity"].iloc[-1]) else 1.0
    prev_td = int(prev["trading_days"].iloc[-1]) if len(prev) and pd.notna(prev["trading_days"].iloc[-1]) else 0
    day_pnl = float(position_exposure) * float(day_index_return)
    model_eq = prev_eq * (1 + day_pnl)
    td = prev_td + (1 if position_exposure and abs(position_exposure) > 1e-9 else 0)
    tgt = P_TARGET.get(phase, 0.10)
    rec = {"date": date_, "account": account, "phase": phase, "model_equity": round(model_eq, 6),
           "platform_equity": platform_equity, "day_pnl": round(day_pnl, 6), "cum_pnl": round(model_eq - 1, 6),
           "dist_to_target_pp": round(100 * (1 + tgt - model_eq), 2),          # +%X hedefe ne kadar kaldı
           "dist_to_daily_pp": round(100 * (model_eq * DAILY_LIM if False else DAILY_LIM), 2),  # günlük marj sabit %5
           "dist_to_totaldd_pp": round(100 * (model_eq - (1 - TOTAL_LIM)), 2), # toplam-DD'ye mesafe
           "trading_days": td, "position_exposure": round(float(position_exposure), 4),
           "lot": (round(float(lot), 4) if lot is not None else None), "note": note}
    df = df[~((df["account"] == account) & (df["date"].astype(str) == str(date_)))]
    df = pd.concat([df, pd.DataFrame([{c: rec.get(c) for c in COLS}])], ignore_index=True)
    df = df.sort_values(["account", "date"]).reset_index(drop=True)[COLS]
    df.to_parquet(p)
    return df


def drag_summary(account: str) -> dict:
    """execution-drag: model_equity vs platform_equity (Emir-girdiği günlerde). Haftalık brief satırı için."""
    p = ledger_path(account)
    if not p.exists():
        return {"account": account, "rows": 0, "drag": None}
    df = pd.read_parquet(p)
    df = df[df["account"] == account]
    pe = pd.to_numeric(df["platform_equity"], errors="coerce")
    me = pd.to_numeric(df["model_equity"], errors="coerce")
    both = pe.notna() & me.notna()
    drag = float((me[both] - pe[both]).mean()) if both.any() else None      # model − platform (poz = platform geride)
    return {"account": account, "rows": int(len(df)), "n_platform": int(both.sum()),
            "model_equity": (round(float(me.iloc[-1]), 4) if len(me) else None),
            "platform_equity": (round(float(pe.dropna().iloc[-1]), 4) if pe.notna().any() else None),
            "exec_drag": (round(drag, 5) if drag is not None else None),
            "trading_days": (int(df["trading_days"].iloc[-1]) if len(df) else 0),
            "phase": (df["phase"].iloc[-1] if len(df) else None)}
