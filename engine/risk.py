"""
engine/risk — dinamik boyutlama, hesaba ölçekli (config_accounts.yaml). İKİ HAVUZ:
  • futures_prop (directional, ES/NQ) — ŞU AN KAPALI (prop yok); açılınca drawdown-tampon + günlük-limit gate'li.
  • options (Midas) — AKTİF: risk-bütçesi = sermaye × max_risk% × (konviksiyon × fractional-Kelly).

Trade-başı $ risk = defined-risk yapının max-loss'u; #kontrat = floor(bütçe / max-loss-per-unit).
Placeholder sermaye ile çalışır; Emir gerçek değerleri girince otomatik ölçeklenir. Saf-sinyal.
"""
from __future__ import annotations
import math


def size(decision: dict, trade: dict, cfg: dict) -> dict:
    acc = (cfg.get("accounts", {}) or {})
    sz = (cfg.get("sizing", {}) or {})
    opt = (acc.get("options", {}) or {})
    prop = (acc.get("futures_prop", {}) or {})
    conv = float(decision.get("conviction", 0.0))
    kelly = float(sz.get("fractional_kelly", 0.5))
    klass = decision["vehicle"]["class"]

    if klass == "stand_aside" or trade.get("ticket") is None:
        return {"action": "STAND-ASIDE", "dollar_risk": 0.0, "note": "işlem yok → risk 0"}

    # risk bütçesi (options havuzu)
    cap = float(opt.get("size_usd") or 0)
    maxpct = float(opt.get("max_risk_per_trade_pct", 2.0)) / 100.0
    budget = cap * maxpct * min(1.0, conv * 2 * kelly)        # konviksiyon-ölçekli, tavan %max
    t = trade["ticket"]

    out = {"pool": "options(Midas)", "account_size_usd": cap,
           "conviction": round(conv, 3), "risk_budget_usd": round(budget, 2)}

    if klass == "directional_long":
        # cash-ETF: risk = (giriş − stop) × hisse; hisse = bütçe / (giriş−stop)
        entry, stop = t.get("entry"), t.get("stop")
        per = max((entry - stop), 1e-9)
        shares = math.floor(budget / per) if budget > 0 else 0
        notional = round(shares * entry, 2)
        out.update(action="BUY", instrument=t.get("instrument"), shares=shares,
                   notional_usd=notional, dollar_risk=round(shares * per, 2),
                   note=f"{shares} hisse {t.get('instrument')}; risk=(giriş−stop)×hisse")
    else:
        # defined-risk opsiyon: max-loss-per-contract bilinmiyor (canlı premium gerekir) → bütçeyle ifade et
        out.update(action="OPEN " + (t.get("structure") or klass),
                   contracts="≈ floor(risk_bütçesi / yapı-max-loss)", dollar_risk=round(budget, 2),
                   note=f"max-loss ≤ ${round(budget,2)} (defined-risk); kontrat = bütçe ÷ (yapının canlı max-loss'u). "
                        f"Midas'ta debit/genişlik girince netleşir.")

    if not prop.get("enabled"):
        out["futures_note"] = "prop hesabı kapalı → directional futures devre-dışı (cash-ETF/options kullanılıyor)"
    return out
