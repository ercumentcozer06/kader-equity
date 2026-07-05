"""
engine/trade — DECISION'ın araç-sınıfını SOMUT bilete çevirir: strike'lar wall/flip/expected-move'a çapalı,
giriş/stop/hedef. Defined-risk öncelikli (Midas). Vade → expiry DTE. Saf-sinyal; execution kullanıcıda.

NOT: QQQ ETF opsiyonu (×41 ≈ NDX). SPX bacağı 2026-06-17'den beri CBOE _SPX index-native (mult=1, SPY×10 DEĞİL). Index karşılığı = state.TIC_MULT (SPY→1, QQQ→41).
"""
from __future__ import annotations


def _round_strike(x: float) -> float:
    return round(x)                                  # SPY/QQQ ~$1 strike artışı


HORIZON_DTE = {"intraday": 2, "swing": 21, "position": 45, "—": 21}


def construct(decision: dict, state: dict, cfg: dict) -> dict:
    v = decision["vehicle"]
    klass = v["class"]
    spot = state.get("spot")
    em = state.get("exp_move_1d") or (0.01 * (spot or 100))
    cw, pw = state.get("call_wall"), state.get("put_wall")
    flip = state.get("gex_flip")
    # GÖREV 2: min-DTE tabanı — rejim YAPIYI seçer ama vade 21g'nin ALTINA inemez (a priori policy).
    min_dte = int(((cfg.get("accounts", {}) or {}).get("options", {}) or {}).get("min_dte", 21))
    dte = max(HORIZON_DTE.get(decision.get("horizon", "swing"), 21), min_dte)
    em5 = em * (5 ** 0.5)                             # ~hafta hareketi

    if klass == "stand_aside" or spot is None:
        return {"class": klass, "ticket": None, "note": v.get("expression", "işlem yok")}

    t = {"class": klass, "underlying": state.get("ticker"), "spot": round(spot, 2), "dte": dte}

    if klass == "directional_long":
        entry = round(spot, 2)
        stop = round(min(pw if pw else spot - em5, spot - em5), 2)     # alt: put-wall ya da −1hafta
        target = round(max(cw if cw else spot + 1.5 * em5, spot + 1.5 * em5), 2)
        # H5: dejenere genişlik (giriş≈stop; absürd-küçük em ya da wall==spot) → RR=NA + ifade skip; 1e9 saçma yok
        denom = entry - stop
        rr = round((target - entry) / denom, 2) if denom > 1e-4 * max(entry, 1.0) else None
        if rr is None:
            return {"class": klass, "ticket": None,
                    "note": f"giriş≈stop ({entry}≈{stop}) → dejenere genişlik, geçersiz R:R, ifade atlandı"}
        t.update(side="LONG", instrument=v["instrument"], entry=entry, stop=stop, target=target, rr=rr,
                 levels=f"giriş ~{entry}, stop {stop} (alt-kanat/put-wall), hedef {target} (call-wall/+1.5σhafta)")

    elif klass == "buy_convexity_put":
        long_k = _round_strike(spot)                                   # ATM long put
        short_k = _round_strike(min(pw if pw else spot - em5, spot - em5))  # put-wall'a sat
        t.update(structure="put-debit-spread", long_put=long_k, short_put=short_k, dte=dte,
                 max_loss="ödenen debit (tanımlı)", target=f"spot→{short_k} (put-wall)",
                 levels=f"AL put {long_k} / SAT put {short_k}, {dte}g — konveksite, downside konumlanma")

    elif klass == "call_debit":
        long_k = _round_strike(spot)
        short_k = _round_strike(max(cw if cw else spot + em5, spot + em5))
        t.update(structure="call-debit-spread", long_call=long_k, short_call=short_k, dte=dte,
                 max_loss="ödenen debit (tanımlı)", target=f"spot→{short_k} (call-wall)",
                 levels=f"AL call {long_k} / SAT call {short_k}, {dte}g — tanımlı-risk directional")

    elif klass == "sell_premium_condor":
        sc = _round_strike(cw if cw else spot + em5)                   # short call @ call-wall
        sp = _round_strike(pw if pw else spot - em5)                   # short put @ put-wall
        lc = _round_strike(sc + em5); lp = _round_strike(sp - em5)     # long kanatlar
        t.update(structure="iron-condor", short_call=sc, long_call=lc, short_put=sp, long_put=lp, dte=dte,
                 max_loss="(kanat genişliği − kredi) × 100 (tanımlı)", target="spot duvarlar arası kalır (pin)",
                 levels=f"SAT {sp}p/{sc}c, AL {lp}p/{lc}c, {dte}g — pin, theta topla")

    else:
        return {"class": klass, "ticket": None, "note": "tanımsız araç"}

    return {"class": klass, "ticket": t, "note": v.get("expression", "")}
