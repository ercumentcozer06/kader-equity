"""
screen/gamma_engine — InSillico/Four-Seasons GAMMA LEVELS panelini PROFESYONEL CBOE verisinden üretir.

2026-06-17 CBOE-GEÇİŞİ: Eski yfinance SPY×10 proxy + 5-front-expiry naif hesabı, gerçek vendor
(MFGL/SpotGamma-klonu) flip'inden ~%0.5-1 sapıyordu → canlı rejim etiketi yanlış (Telegram'da sahte POS).
Artık gerçek CBOE delayed_quotes zinciri (_SPX index / QQQ ETF, gamma+IV+OI hazır) + DOĞRULANMIŞ reçete:
  FLIP = monthly-expiry (3. Cuma) + BS-repricing zero-gamma + put-overweight (call +1.0 / put -1.3).
Bu reçete 2026-06-16/17'de Emir'in InSillico indikatörüyle ±%0.4 örtüştü (SPX 7561, QQQ 749, NEG).
REJİM = spot vs flip (vendor konvansiyonu; NEG = spot flip ALTINDA = dealer short-gamma/amplifikasyon).

ÖNEMLİ — bu BETİMSEL bir rejim/seviye panelidir, TRADEABLE SİNYAL DEĞİL: flip directional değeri
2 bağımsız enstrümanla test edilip REDDEDİLDİ (FINDING 20-21 + 2026-06-17 backtest, adversarial-verify'li).
Değer = piyasa-OKUMA (rejim/risk karakteri) + GEX-shield (modules/gex_shield, SqueezeMetrics z, AYRI/backtest'li).
Snapshot şeması collect_daily.LEVEL_COLS ile uyumlu (gex_flip/regime/put_wall/call_wall/ghost/hvl/shield_z...).

KONVANSİYON: dealer call-long / put-short, put-overweight pw=1.3 (gerçek dealer put-skew kalibrasyonu;
saf değer değil, [1.2-1.35] bandında drift edebilir — flip vendor'dan ±%0.5 kaçarsa pw yeniden-fit, scope DEĞİL).
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from math import erf, exp, log, pi, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                        # modules.gex_shield (rejim z-damgası) için
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

try:                                                 # script (screen/ path'te) ya da paket olarak
    from _cboe_lib import load_rows as cboe_load_rows, flip_bs, PUT_WEIGHT   # noqa: E402
except ImportError:
    from screen._cboe_lib import load_rows as cboe_load_rows, flip_bs, PUT_WEIGHT  # noqa: E402

R, Q = 0.04, 0.013                                   # PUT_WEIGHT _cboe_lib'den (tek-kaynak; vendor-drift'te orada yeniden-fit)
BAND = 0.15                  # ±15% strike (load_rows zaten uygular; wall/max-pain için geniş)

# arg(ETF etiketi) -> (CBOE sembolü, indeks-etiketi, indeks-çarpanı). SPY->gerçek _SPX index; QQQ->QQQ ETF.
CFG = {"SPY": ("_SPX", "SPX", 1), "QQQ": ("QQQ", "NDX", 41)}


def _npdf(x): return exp(-x * x / 2) / sqrt(2 * pi)
def _ncdf(x): return 0.5 * (1 + erf(x / sqrt(2)))


def _greeks(S, K, T, s, right):
    if T <= 0 or s <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0, 0.0, 0.0
    a = s * sqrt(T)
    d1 = (log(S / K) + (R - Q + s * s / 2) * T) / a
    d2 = d1 - a
    gamma = exp(-Q * T) * _npdf(d1) / (S * a)
    vanna = -exp(-Q * T) * _npdf(d1) * d2 / s
    charm = (Q * exp(-Q * T) * (_ncdf(d1) if right == "C" else -_ncdf(-d1))
             - exp(-Q * T) * _npdf(d1) * (2 * (R - Q) * T - d2 * a) / (2 * T * a))
    delta = exp(-Q * T) * (_ncdf(d1) if right == "C" else -_ncdf(-d1))
    return gamma, vanna, charm, delta


def main() -> int:
    arg = (sys.argv[1] if len(sys.argv) > 1 else "SPY").upper()      # SPY|QQQ (collect_daily çağrısı)
    cboe_sym, idx_lbl, mult = CFG.get(arg, (arg, arg, 1))
    try:
        spot, crows = cboe_load_rows(cboe_sym, band=BAND)            # gerçek CBOE zinciri (gamma+IV+OI)
    except Exception as e:                                          # noqa: BLE001
        print(f"  [!] CBOE çekilemedi ({cboe_sym}): {type(e).__name__}: {str(e)[:120]}")
        return 1
    if not crows:
        print(f"  [!] CBOE zincir boş ({cboe_sym}).")
        return 1
    if not np.isfinite(spot) or spot <= 0:               # savunma (kök guard _cboe_lib'de): NaN/0 spot -> junk snapshot + sahte rejim YAZMA
        print(f"  [!] geçersiz spot ({cboe_sym}): {spot}")
        return 1

    # gamma_engine iç satır biçimi {K,oi,right,iv,g,v,c,de,mo,dte} — greek'ler CBOE iv'sinden BS ile
    rows = []
    for r in crows:
        iv = r.get("iv")
        if not iv or iv <= 0:
            continue
        g, v, c, de = _greeks(spot, r["K"], r["T"], iv, r["cp"])
        rows.append({"K": r["K"], "oi": r["oi"], "right": r["cp"], "iv": iv,
                     "g": g, "v": v, "c": c, "de": de, "mo": bool(r["is_monthly"]), "dte": r["dte"]})
    if not rows:
        print(f"  [!] geçerli IV'li kontrat yok ({cboe_sym}).")
        return 1

    M = 100  # contract multiplier

    # ── FLIP = DOĞRULANMIŞ REÇETE: monthly + BS-repricing + put-overweight ──────────────────────────
    flip, _net_at = flip_bs(crows, spot, row_filter=lambda r: r["is_monthly"],
                            sign=lambda cp: 1.0 if cp == "C" else -PUT_WEIGHT, use_cboe_gamma=False)

    # net-GEX (gösterilen) = AYNI reçete bazında (monthly + pw) → işareti spot-vs-flip ile tutarlı
    sgn_pw = lambda right: 1.0 if right == "C" else -PUT_WEIGHT
    mo_rows = [x for x in rows if x["mo"]] or rows
    net_gex = sum(sgn_pw(x["right"]) * x["g"] * x["oi"] * M * spot * spot * 0.01 for x in mo_rows)
    # vanna/charm/delta = bağlam (naif, tüm band)
    sgn = lambda right: 1.0 if right == "C" else -1.0
    net_vanna = sum(sgn(x["right"]) * x["v"] * x["oi"] * M * spot * 0.01 for x in rows)
    net_charm = sum(sgn(x["right"]) * x["c"] * x["oi"] * M * spot / 365.0 for x in rows)
    dealer_delta = -sum(x["de"] * x["oi"] * M for x in rows)

    # walls (gamma-notional / raw-OI tepe) + max pain — tüm band satırları üzerinden
    by_k_call, by_k_put = {}, {}
    for x in rows:
        gk = x["g"] * x["oi"]
        (by_k_call if x["right"] == "C" else by_k_put).setdefault(x["K"], 0.0)
        (by_k_call if x["right"] == "C" else by_k_put)[x["K"]] += gk
    ghost = max((k for k in by_k_call if k >= spot), key=lambda k: by_k_call[k], default=None)   # gamma-peak
    call_oi = {}
    for x in rows:
        if x["right"] == "C" and x["K"] >= spot:
            call_oi[x["K"]] = call_oi.get(x["K"], 0.0) + x["oi"]
    call_wall = max(call_oi, key=lambda k: call_oi[k], default=None)                              # raw-OI peak
    put_wall = max((k for k in by_k_put if k <= spot), key=lambda k: by_k_put[k], default=None)   # gamma
    by_k_all = {}
    for x in rows:
        by_k_all[x["K"]] = by_k_all.get(x["K"], 0.0) + abs(x["g"] * x["oi"])
    hvl = max(by_k_all, key=lambda k: by_k_all[k]) if by_k_all else None
    strikes = sorted({x["K"] for x in rows})
    coi = {}
    for x in rows:
        coi.setdefault((x["K"], x["right"]), 0.0)
        coi[(x["K"], x["right"])] += x["oi"]
    def pain(P): return sum(coi.get((k, "C"), 0)*max(0, P-k) + coi.get((k, "P"), 0)*max(0, k-P) for k in strikes)
    max_pain = min(strikes, key=pain) if strikes else None

    # market: exp move — kararlı ~30D ATM IV (CBOE iv'sinden; front 0-6 DTE eler)
    atm_cands = [r for r in crows if r.get("iv") and r["iv"] > 0 and r["dte"] >= 7]
    iv_em, iv_src = None, "30d"
    if atm_cands:
        tgt = min({r["dte"] for r in atm_cands}, key=lambda d: abs(d - 30))
        iv_em = min([r for r in atm_cands if r["dte"] == tgt], key=lambda r: abs(r["K"] - spot))["iv"]
    if not iv_em or iv_em <= 0:
        iv_em = min(rows, key=lambda x: abs(x["K"] - spot))["iv"]; iv_src = "front(fallback)"
    em1 = spot * iv_em * sqrt(1 / 252); em5 = spot * iv_em * sqrt(5 / 252)
    try:                                               # realized-vol oranı: yfinance FİYAT (opsiyon değil) — güvenilir
        import yfinance as yf
        h = yf.Ticker(arg).history(period="2mo")["Close"]
        lr = np.log(h / h.shift(1)).dropna()
        rv5, rv20 = lr[-5:].std()*sqrt(252), lr[-20:].std()*sqrt(252)
        rv_ratio = rv5/rv20 if rv20 else None
    except Exception:
        rv_ratio = None

    # ── REJİM = spot vs FLIP (vendor konvansiyonu — DÜZELTME) + FLIP-CİVARI ──────────────────────────
    if flip:
        regime = "SHORT GAMMA" if spot < flip else "LONG GAMMA"
    else:
        regime = "NEUTRAL GAMMA"     # flip ±%15 taramada yok → rejim belirsiz; net-işaret gürültüsünde LONG↔SHORT zıplama YAPMA
    dist_to_flip = (spot - flip) if flip else None
    flip_civari = bool(flip and abs(spot - flip) < em1)             # |spot-flip| < 1G exp-move → rejim kırılgan

    P = lambda *a: print("  " + " ".join(str(x) for x in a))
    print("=" * 64)
    P(f"GAMMA ENGINE — {idx_lbl} {spot*mult:.0f}  (CBOE {cboe_sym})   {datetime.now(timezone.utc):%Y-%m-%d %H:%MZ}")
    print("=" * 64)
    P(f"DEALER   net GEX   : {net_gex/1e9:+.2f} $bn/1%   → {regime}")
    P(f"         net VANNA : {net_vanna/1e6:+.1f} $m/1%vol")
    P(f"         net CHARM : {net_charm/1e6:+.1f} $m/gün   (delta-drift)")
    P(f"         DELTA BIAS: {dealer_delta/1e6:+.1f} $m delta  ({'Net Bearish' if dealer_delta<0 else 'Net Bullish'})")
    P(f"LEVELS   GEX FLIP  : {idx_lbl} {flip*mult:.0f}  (monthly+pw{PUT_WEIGHT})" if flip else "GEX FLIP: scan-dışı")
    P(f"         CALL WALL : {idx_lbl} {call_wall*mult:.0f} (raw-OI)" if call_wall else "CALL WALL: -")
    P(f"         GHOST     : {idx_lbl} {ghost*mult:.0f} (gamma-peak)" if ghost else "GHOST: -")
    P(f"         PUT WALL  : {idx_lbl} {put_wall*mult:.0f}" if put_wall else "PUT WALL: -")
    P(f"         MAX PAIN  : {idx_lbl} {max_pain*mult:.0f}" if max_pain else "MAX PAIN: -")
    P(f"MARKET   Exp Move  : 1D ±{em1*mult:.0f}  5D ±{em5*mult:.0f} ({idx_lbl} pts)")
    P(f"         RV ratio  : {rv_ratio:.2f} {'Rising' if rv_ratio and rv_ratio>1 else 'Falling'}" if rv_ratio else "RV ratio: -")
    if flip:
        side = "ÜSTÜNDE (stabil/söndürme)" if spot >= flip else "ALTINDA (kırılgan/amplifikasyon)"
        P(f"         spot vs FLIP: {side}  | Δ {dist_to_flip*mult:+.0f}")
        if flip_civari:
            P(f"  [!] FLIP CIVARI — |spot-flip| < 1G exp-move ({abs(dist_to_flip)*mult:.0f} < {em1*mult:.0f}); rejim kirilgan, gun ici donebilir (teyitsiz).")
    print("=" * 64)
    print("  NOT: BETİMSEL rejim/seviye paneli (CBOE EOD OI, vendor-match). TRADEABLE sinyal DEĞİL; yön testte REDDEDİLDİ.")

    # gamma rejim z-damgası: gex_shield (SqueezeMetrics SPX, market-wide; QQQ için de SPX-proxy) — DEĞİŞMEDİ
    shield_z = shield_short = None
    try:
        from modules.gex_shield import gex_zscore, fetch_gex_live
        _z = gex_zscore(fetch_gex_live())
        shield_z = round(float(_z.iloc[-1]), 2); shield_short = bool(shield_z < -1.0)
    except Exception:
        pass

    snap = {"as_of": date.today().isoformat(), "ts": datetime.now(timezone.utc).isoformat(), "underlying": cboe_sym,
            "index": idx_lbl, "spot": round(spot, 2), "net_gex_bn": round(net_gex/1e9, 3),
            "net_vanna_m": round(net_vanna/1e6, 2), "net_charm_m": round(net_charm/1e6, 2),
            "dealer_delta_m": round(dealer_delta/1e6, 2), "gex_flip": round(flip, 2) if flip else None,
            "call_wall": call_wall, "ghost": ghost, "put_wall": put_wall, "hvl": hvl, "max_pain": max_pain,
            "exp_move_1d": round(em1, 2), "atm_iv_30d": round(iv_em * 100, 2), "em_iv_src": iv_src,
            "rv_ratio": round(rv_ratio, 3) if rv_ratio else None, "regime": regime,
            "dist_to_flip": round(dist_to_flip, 2) if dist_to_flip is not None else None,
            "flip_proximity": flip_civari, "gamma_regime_value": round(dist_to_flip, 2) if dist_to_flip is not None else None,
            "put_weight": PUT_WEIGHT, "methodology": "cboe-monthly-bs-pw1.3",
            "shield_z": shield_z, "shield_short_gamma": shield_short, "source": "cboe"}
    outdir = ROOT / "data" / "cache" / f"gamma_{arg.lower()}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{date.today().isoformat()}.json").write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"  HVL ≈ {idx_lbl} {hvl*mult:.0f}  | shield-z {shield_z} ({'short-γ' if shield_short else 'normal'})" if hvl else "")
    print(f"  snapshot → {outdir / (date.today().isoformat()+'.json')}  (CBOE forward-collector {arg})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
