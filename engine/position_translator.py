"""
engine/position_translator — B1: model exposure → ulaşılabilir LOT + delta-emir (MEKANİK, yeni DoF yok).
Katman 1-2 FROZEN; bu sadece çeviri katmanı.

  hedef_exposure = model_deploy(0..1) × eval_pos
  lot_hedef = floor(equity × hedef_exposure / (contract_size × spot) / lot_step) × lot_step   # AŞAĞI yuvarla
              (aşağı → gerçekleşen ≤ hedef → −%5 marjı korunur; A2: cs büyükse adım kaba, RMS sapma)
  realized_exposure = lot_hedef × contract_size × spot / equity
  delta_emir = lot_hedef − mevcut_lot   ("US100: 0.03 SAT" / "AL")
  felaket-stop seviyesi = prev_close × (1 − STOP_LOSS/hedef_exposure)   (A1 EVET ise; ops fail-safe)
  günlük-limit seviyesi = prev_close × (1 − 0.05/hedef_exposure)        (her durumda bilgi)

KONTRAT_SPEC parametrik (config_accounts.yaml accounts.eval.contract_spec); platform-teyit gelince FLAG kalkar.
"""
from __future__ import annotations

import math

STOP_LOSS = 0.045        # A1 ops fail-safe (eval'de opsiyonel, funded'da EVET)
DAILY_LIM = 0.05


def translate(model_deploy: float, spot: float, prev_close: float, equity: float, eval_pos: float,
              contract_size: float, lot_step: float, current_lot: float = 0.0,
              use_stop: bool = False, instrument: str = "US100", index_mult: float = 1.0) -> dict:
    """Model exposure'ı lot + delta-emir + stop/limit seviyelerine çevir. Tümü mekanik."""
    tgt_exp = float(model_deploy) * float(eval_pos)
    denom = contract_size * spot
    lot_target = math.floor(equity * tgt_exp / denom / lot_step) * lot_step if denom > 0 else 0.0
    lot_target = round(lot_target, 4)
    realized_exp = lot_target * contract_size * spot / equity if equity > 0 else 0.0
    delta = round(lot_target - float(current_lot), 4)
    side = "AL" if delta > 0 else ("SAT" if delta < 0 else "—")
    # exposure-adımı (granülarite uyarısı): 1 lot_step = ne kadar exposure
    step_exp = lot_step * contract_size * spot / equity if equity > 0 else 0.0
    # seviyeler (endeks-fiyat; underlying ya da index_mult ile gösterilir)
    stop_lvl = prev_close * (1 - STOP_LOSS / tgt_exp) if (use_stop and tgt_exp > 0) else None
    dlim_lvl = prev_close * (1 - DAILY_LIM / tgt_exp) if tgt_exp > 0 else None
    return {
        "instrument": instrument, "target_exposure": round(tgt_exp, 3), "realized_exposure": round(realized_exp, 3),
        "exposure_slip": round(realized_exp - tgt_exp, 3),               # aşağı-yuvarlama kaybı (≤0)
        "lot_target": lot_target, "current_lot": round(float(current_lot), 4),
        "delta_lot": abs(delta), "side": side,
        "exposure_step_pct": round(100 * step_exp, 1),                   # A2: cs kabaysa büyük (FLAG)
        "coarse_flag": bool(step_exp > 0.15),                            # %15+ adım = kaba ifade
        "stop_level": (round(stop_lvl, 2) if stop_lvl else None),
        "daily_limit_level": (round(dlim_lvl, 2) if dlim_lvl else None),
        "order_str": (f"{instrument}: {abs(delta):.2f} {side}" if delta else f"{instrument}: pozisyon sabit"),
    }


def check_policy(acc_cfg: dict) -> dict | None:
    """B4: eval_pos pre-registered'dan saparsa POLICY-İHLALİ FLAG (mid-eval boyut değişimi YASAK — Niederhoffer:
    sim-dışı discretionary override felaket). None → temiz."""
    ev = (acc_cfg.get("accounts", {}) or {}).get("eval", {}) or {}
    cur, reg = ev.get("eval_pos"), ev.get("eval_pos_registered")
    if cur is not None and reg is not None and abs(float(cur) - float(reg)) > 1e-9:
        return {"violation": True,
                "reason": f"eval_pos {cur} ≠ registered {reg} → PRE-REGISTERED POLICY İHLALİ (mid-eval değişim YASAK)"}
    return None


def from_config(model_deploy: float, spot: float, prev_close: float, acc_cfg: dict, current_lot: float = 0.0) -> dict:
    """config_accounts.yaml accounts.eval bloğundan parametreleri çekerek çevir."""
    ev = (acc_cfg.get("accounts", {}) or {}).get("eval", {}) or {}
    cs = (ev.get("contract_spec", {}) or {})
    return translate(
        model_deploy=model_deploy, spot=spot, prev_close=prev_close,
        equity=float(ev.get("size_usd", 10000)), eval_pos=float(ev.get("eval_pos", 1.2)),
        contract_size=float(cs.get("contract_size", 10)), lot_step=float(cs.get("lot_step", 0.01)),
        current_lot=current_lot, use_stop=bool(ev.get("eval_stop", False)),
        instrument=ev.get("instrument", "US100"))
