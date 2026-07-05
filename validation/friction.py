"""
validation/friction — GÖREV 1 friction analizi. Mevcut hesap boyutunda tipik defined-risk spread biletinin
komisyon + yarı-spread maliyeti, RİSKİN YÜZDESİ olarak. Küçük hesapta friction riskin büyük payını yiyebilir
→ eşik kararı Emir'de; bu modül yalnız SAYIYI üretir.

Konvansiyon: dikey spread = 2 bacak. close_round_trip=True → aç(2)+kapat(2)=4 fill; False → vade-sonu (2 fill).
Komisyon config'ten (costs.commission_per_contract); null ise parametrik 2 senaryo (düşük $0.65 / yüksek $1.50).
half_spread bacak-başı yarı bid/ask ($/kontrat). Hepsi a priori — optimize edilmedi.
"""
from __future__ import annotations

import math

LEGS_VERTICAL = 2                                    # dikey spread bacak sayısı
COMMISSION_SCENARIOS = {"düşük($0.65)": 0.65, "yüksek($1.50)": 1.50}   # a priori, Midas bilinmiyor


def _row(label, comm_per, half_spread, legs, fills, contracts, risk_usd):
    commission = comm_per * legs * contracts * fills
    slippage = half_spread * legs * contracts * fills
    friction = commission + slippage
    ratio = friction / risk_usd if risk_usd > 0 else None
    return {"senaryo": label, "komisyon_$": round(commission, 2), "slippage_$": round(slippage, 2),
            "friction_$": round(friction, 2), "risk_$": round(risk_usd, 2),
            "friction/risk_%": (None if ratio is None else round(ratio * 100, 1))}


def friction_table(cfg: dict, contracts: int = 1) -> dict:
    """cfg = config_accounts dict. Döndürür {risk_usd, fills, senaryolar:[...]}. Komisyon null → 2 senaryo."""
    acc = (cfg.get("accounts", {}) or {}).get("options", {}) or {}
    costs = (cfg.get("costs", {}) or {})
    size = float(acc.get("size_usd") or 0)
    maxpct = float(acc.get("max_risk_per_trade_pct", 2.0)) / 100.0
    risk_usd = size * maxpct
    half = float(costs.get("half_spread_per_contract", 1.5))
    fills = 4 if bool(costs.get("close_round_trip", True)) else 2
    comm_cfg = costs.get("commission_per_contract")

    rows = []
    if comm_cfg is not None:                          # Midas komisyonu girilmiş → tek senaryo
        rows.append(_row(f"Midas(${float(comm_cfg):.2f})", float(comm_cfg), half,
                         LEGS_VERTICAL, fills, contracts, risk_usd))
    else:                                             # parametrik 2 senaryo
        for lbl, c in COMMISSION_SCENARIOS.items():
            rows.append(_row(lbl, c, half, LEGS_VERTICAL, fills, contracts, risk_usd))
    return {"account_size_usd": size, "max_risk_pct": maxpct * 100, "risk_usd": round(risk_usd, 2),
            "contracts": contracts, "legs": LEGS_VERTICAL, "fills": fills,
            "half_spread_per_contract": half, "commission_source": ("config" if comm_cfg is not None else "parametrik"),
            "scenarios": rows}


def render(cfg: dict, contracts: int = 1) -> str:
    t = friction_table(cfg, contracts)
    L = []
    L.append(f"  FRICTION — hesap ${t['account_size_usd']:.0f}, risk/trade %{t['max_risk_pct']:.1f} = ${t['risk_usd']:.2f}, "
             f"{t['contracts']} kontrat × {t['legs']} bacak × {t['fills']} fill (half-spread ${t['half_spread_per_contract']:.2f}/bacak)")
    L.append(f"  {'senaryo':<16}{'komisyon$':>11}{'slippage$':>11}{'friction$':>11}{'friction/risk%':>16}")
    for s in t["scenarios"]:
        L.append(f"  {s['senaryo']:<16}{s['komisyon_$']:>11.2f}{s['slippage_$']:>11.2f}{s['friction_$']:>11.2f}{str(s['friction/risk_%']):>16}")
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import yaml
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "engine" / "config_accounts.yaml").read_text(encoding="utf-8"))
    print(render(cfg, contracts=1))
