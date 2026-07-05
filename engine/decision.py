"""
engine/decision — kilitli modelden + canlı piyasa-durumundan KARAR: yön + konviksiyon + vade + ARAÇ.

Felsefe (kitaplar):
  • Taleb: doğru ifade rejime bağlı — kısa-gamma/kırılgan rejimde KONVEKSİTE al (trim değil, yapı).
  • Hull: long-gamma/pin → duvarlarda premium SAT; temiz-trend → directional.
  • Marks: konviksiyon eşik-altıysa STAND-ASIDE (çoğu gün işlem yok); agresiflik döngü-konumuna göre.
Hesaba uyarlanır: prop yoksa directional = cash ETF (SPY/QQQ Midas) ya da defined-risk opsiyon; futures değil.

Girdi: model (run.build_decision çıktısı) + state (engine.state.MarketState) + cfg (config_accounts.yaml).
Çıktı: {direction, conviction, horizon, vehicle{class,expression,instrument}, regime, rationale}. Saf-sinyal.
"""
from __future__ import annotations

# ── VRP eşikleri — A PRİORİ (literatür standardı, OPTİMİZE EDİLMEDİ; sweep/trial değil) ──
# SPX uzun-dönem ort VRP ~+3-4 vol-puanı (implied tipik realized'ı 3-4 aşar). Prim-sat yalnız implied
# anlamlı zengin iken (≥+2); konveksite-al implied ucuz iken (implied ≤ realized, VRP≤0) tercihli.
VRP_SELL_MIN = 2.0      # bu vol-puanının ÜSTÜNDE → prim-satmaya değer (vol pahalı)
VRP_CHEAP = 0.0         # bunun ALTINDA → implied ucuz → konveksite-al lehine


# ── rejim sınıflandırma ──
def classify_regime(state: dict) -> dict:
    """Canlı state'ten gamma/vol/froth rejimini çıkar (eksik alanlara dayanıklı)."""
    gex = state.get("net_gex_bn")
    flip = state.get("gex_flip")
    spot = state.get("spot")
    em1 = state.get("exp_move_1d")          # 1g expected move (underlying puan)
    # gamma rejimi
    short_gamma = (gex is not None and gex < 0)
    above_flip = (flip is not None and spot is not None and spot > flip)
    # flip'e σ-uzaklığı (em1 ≈ 1g σ): |spot-flip|/em1
    dist_flip_em = (abs(spot - flip) / em1) if (flip and spot and em1) else None
    # froth (doğrulanmış uç): COR1M < 8
    cor1m = state.get("cor1m")
    froth = (cor1m is not None and cor1m < 8.0)
    froth_soft = (cor1m is not None and cor1m < 11.0)
    # duvara yakınlık (pin adayı): spot bir wall'a ~0.5 expected-move içinde
    cw, pw = state.get("call_wall"), state.get("put_wall")
    near_wall = False
    if spot and em1:
        for w in (cw, pw):
            if w is not None and abs(spot - w) <= 0.5 * em1:
                near_wall = True
    # vol rejimi: ts_ratio = vix/vix3m (FAZ-1 düzeltmesi) → >1 backwardation/stres, <1 contango/sakin
    ts = state.get("ts_ratio")
    backwardation = (ts is not None and ts > 1.0)
    calm = (ts is not None and ts < 0.95) and not backwardation
    # VRP (GÖREV 3): implied(~30g) − realized(EWMA). vol-değer organı (yön DEĞİL, İFADE fiyatı).
    vrp = state.get("vrp")
    vrp_rich = (vrp is not None and float(vrp) > VRP_SELL_MIN)    # vol pahalı → prim-sat lehine
    vrp_cheap = (vrp is not None and float(vrp) < VRP_CHEAP)      # vol ucuz → konveksite-al lehine
    return {
        "short_gamma": short_gamma, "above_flip": above_flip, "dist_flip_em": dist_flip_em,
        "froth": froth, "froth_soft": froth_soft, "near_wall": near_wall,
        "backwardation": backwardation, "calm": calm, "cor1m": cor1m,
        "vrp": vrp, "vrp_rich": vrp_rich, "vrp_cheap": vrp_cheap,
        "ticker": state.get("ticker", "SPY"),
    }


# ── vade seçici ──
def pick_horizon(direction: str, reg: dict) -> str:
    if reg["froth"]:
        return "swing"                                  # froth-tepe → çok-günlük temkin/de-risk
    if reg["short_gamma"] and not reg["above_flip"]:
        return "intraday"                               # flip-altı kısa-gamma → kırılgan/amplifikasyon, gün-içi
    if not reg["short_gamma"] and reg["near_wall"]:
        return "intraday"                               # long-gamma + duvara yakın → pin, gün-içi mean-revert
    if direction == "LONG" and reg["above_flip"] and reg["calm"] and not reg["froth_soft"]:
        return "position"                               # temiz trend → pozisyon (haftalar)
    return "swing"


# ── araç seçici (rejime en uygun TEK ifade) ──
def pick_vehicle(direction: str, conviction: float, reg: dict, cfg: dict) -> dict:
    sz = (cfg.get("sizing", {}) or {})
    floor = float(sz.get("no_trade_conviction_floor", 0.15))
    opt = ((cfg.get("accounts", {}) or {}).get("options", {}) or {})
    defined_only = (opt.get("level", "defined_risk") == "defined_risk")
    prop_on = bool(((cfg.get("accounts", {}) or {}).get("futures_prop", {}) or {}).get("enabled"))
    dir_no_prop = ((cfg.get("vehicles", {}) or {}).get("directional_no_prop", "cash_etf"))

    lb = (cfg.get("live_book", {}) or {})
    etf_map = (lb.get("etf_map", {}) or {"SPY": "SPLG", "QQQ": "QQQM"})
    tic = reg.get("ticker", "SPY")

    def directional_instrument():
        if prop_on:
            return "futures(ES/NQ)"
        if lb.get("mode") == "delta_one":             # RİSK-1: canlı kitap delta-one ETF (SPX→SPLG, NDX→QQQM)
            return f"delta-one {etf_map.get(tic, 'SPLG' if tic == 'SPY' else 'QQQM')}"
        return "cash-ETF(SPY/QQQ)" if dir_no_prop == "cash_etf" else "long-call(SPY/QQQ)"

    # 0) Marks: konviksiyon eşik-altı VEYA model FLAT → stand-aside (ama froth-de-risk hariç)
    if direction == "FLAT" and not reg["froth_soft"]:
        return {"class": "stand_aside", "expression": "model FLAT, edge yok → işlem yok",
                "instrument": None}
    if conviction < floor and not reg["froth"]:
        return {"class": "stand_aside", "expression": f"konviksiyon {conviction:.2f} < {floor} → işlem yok",
                "instrument": None}

    # 1) FROTH-TEPE (COR1M<8) → KONVEKSİTE AL (Taleb): defined-risk put yapısı (asla naked-short)
    if reg["froth"]:
        vtag = (" [VRP düşük → konveksite ucuz ✓]" if reg.get("vrp_cheap")
                else f" [VRP +{reg['vrp']:.1f} → konveksite pahalı]" if reg.get("vrp") is not None else "")
        return {"class": "buy_convexity_put",
                "expression": f"froth-tepe (COR1M<8): koruyucu PUT / put-debit-spread (konveksite, defined-risk){vtag}",
                "instrument": "put-debit-spread(SPY/QQQ)"}

    # 2) KISA-GAMMA + FLIP-ALTI (kırılgan/amplifikasyon)
    if reg["short_gamma"] and not reg["above_flip"]:
        if direction == "LONG":
            # model hâlâ long ama kırılgan → defined-risk ile katıl (call-debit), ya da sıkı-stop directional
            return {"class": "call_debit",
                    "expression": "kısa-gamma/flip-altı + model LONG: CALL-debit-spread (tanımlı-risk ile katıl, "
                                  "kuyruğu satın al değil)",
                    "instrument": "call-debit-spread(SPY/QQQ)"}
        return {"class": "buy_convexity_put",
                "expression": "kısa-gamma/flip-altı: kırılgan → koruyucu put bias",
                "instrument": "put-debit-spread(SPY/QQQ)"}

    # 3) LONG-GAMMA + PIN (duvara yakın, sakin) → PREMIUM SAT (Hull) — AMA SADECE VRP ZENGİN (vol pahalı) İSE.
    #    GÖREV 3: VRP ≤ eşik (ucuz vol) iken prim satmak klasik kanama → aşağı düş (directional/stand-aside).
    if not reg["short_gamma"] and reg["near_wall"] and reg["calm"] and reg.get("vrp_rich"):
        vtag = f" [VRP +{reg['vrp']:.1f} zengin]" if reg.get("vrp") is not None else ""
        if defined_only:
            return {"class": "sell_premium_condor",
                    "expression": f"long-gamma/pin (duvara yakın, sakin, vol pahalı): iron-condor / credit-spread "
                                  f"(duvarlar arası premium sat, tanımlı-risk){vtag}",
                    "instrument": "iron-condor(SPY/QQQ)"}
        return {"class": "sell_premium_condor",
                "expression": f"long-gamma/pin: duvarlarda premium sat{vtag}",
                "instrument": "short-strangle/condor(SPY/QQQ)"}

    # 4) TEMİZ TREND (LONG, flip-üstü, sakin, froth yok) → DIRECTIONAL
    if direction == "LONG" and reg["above_flip"] and reg["calm"]:
        return {"class": "directional_long",
                "expression": "temiz trend (flip-üstü, sakin): DIRECTIONAL long",
                "instrument": directional_instrument()}

    # 5) DEFAULT: model LONG ama karışık rejim → tanımlı-risk directional (call-debit)
    if direction == "LONG":
        return {"class": "call_debit",
                "expression": "model LONG, karışık rejim: CALL-debit-spread (tanımlı-risk directional)",
                "instrument": "call-debit-spread(SPY/QQQ)"}
    return {"class": "stand_aside", "expression": "net kurulum yok → işlem yok", "instrument": None}


def decide(model: dict, state: dict, cfg: dict) -> dict:
    direction = model.get("direction", "FLAT")
    conviction = float(model.get("position_target", 0.0))      # 0..1 = model'in exposure/konviksiyonu
    reg = classify_regime(state)
    horizon = pick_horizon(direction, reg)
    vehicle = pick_vehicle(direction, conviction, reg, cfg)
    if vehicle["class"] == "stand_aside":
        horizon = "—"
    rationale = (f"model {direction} (konviksiyon {conviction:.2f}); "
                 f"rejim: {'kısa' if reg['short_gamma'] else 'long'}-gamma, "
                 f"flip-{'üstü' if reg['above_flip'] else 'altı'}, "
                 f"{'FROTH' if reg['froth'] else ('froth-soft' if reg['froth_soft'] else 'froth-yok')}, "
                 f"vol-{'backward/stres' if reg['backwardation'] else ('sakin' if reg['calm'] else 'nötr')}"
                 f"{', duvara-yakın' if reg['near_wall'] else ''} → {vehicle['expression']}")
    return {"direction": direction, "conviction": round(conviction, 3), "horizon": horizon,
            "vehicle": vehicle, "regime": reg, "rationale": rationale}
