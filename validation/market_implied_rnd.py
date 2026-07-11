"""
validation/market_implied_rnd — A1: MARKET-implied risk-nötr yoğunluk (Lucid'in "implied distribution'ı izlerdim").

Bizim mc_implied_distribution = TARİHSEL-koşullu (geçmiş getiri bootstrap). Lucid'in kapanış cümlesi ise
İLERİYE-DÖNÜK: opsiyon-fiyatından çıkan piyasa-implied dağılım. Bu parça modelde YOKTU — canlı CBOE zinciri
zaten var (gamma_engine/_cboe_lib), o yüzden Breeden-Litzenberger ile RND çıkarıp modelin tarihsel dağılımıyla
karşılaştırıyoruz.

YÖNTEM (Breeden-Litzenberger 1978): risk-nötr yoğunluk f(K) = e^{rT} · ∂²C/∂K².
  1. CBOE zinciri → ~30-DTE tek vade, OTM smile (K<spot put-IV, K>spot call-IV).
  2. IV(K) yoğun-grid interpolasyon → BS call fiyatı C(K) → ikinci fark → RND.
  3. Normalize; implied moment (ort/vol/skew/kurtosis) + kuyruk-olasılıkları P(S_T < S_0·(1−x)).

DÜRÜST ÇERÇEVE: RND = RİSK-NÖTR (Q) ölçü; bizim bootstrap = FİZİKSEL (P). Ortalamalar farklı olur (Q drift=r−q,
P equity-risk-premium taşır) — bu fark ZATEN risk primi. ŞEKİL (vol/skew/kuyruk-asimetrisi) karşılaştırması
anlamlı: piyasa-implied sol-kuyruk fiziksel-realized'dan ŞİŞMANSA = crash-sigorta primi (variance/skew premium).
READ-ONLY, betimsel — pozisyona SIFIR etki.
"""
from __future__ import annotations

import sys
from math import erf, exp, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "screen"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

R, Q = 0.04, 0.013                     # risk-free / dividend (gamma_engine ile aynı)
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz   # numpy 2.0: trapz→trapezoid


def _ncdf(x): return 0.5 * (1 + erf(x / sqrt(2)))


def bs_call(S, K, T, sig):
    if sig <= 0 or T <= 0:
        return max(S * exp(-Q * T) - K * exp(-R * T), 0.0)
    d1 = (log(S / K) + (R - Q + 0.5 * sig * sig) * T) / (sig * sqrt(T))
    d2 = d1 - sig * sqrt(T)
    return S * exp(-Q * T) * _ncdf(d1) - K * exp(-R * T) * _ncdf(d2)


def moments_from_rnd(K, f):
    """Normalize RND, forward/vol/skew/kurtosis (getiri = K/S0−1 uzayında değil, S_T seviyesinde)."""
    w = _trapz(f, K)
    f = f / w
    m1 = _trapz(K * f, K)
    var = _trapz((K - m1) ** 2 * f, K)
    sd = sqrt(var)
    sk = _trapz((K - m1) ** 3 * f, K) / sd ** 3
    ku = _trapz((K - m1) ** 4 * f, K) / sd ** 4
    return f, m1, sd, sk, ku


def extract_rnd(sym="_SPX", target_dte=30, band=0.28):
    from _cboe_lib import load_rows
    spot, rows = load_rows(sym, band=band)
    # hedef vadeye en yakın expiry (yeterli strike'lı)
    by_exp = {}
    for r in rows:
        if r["iv"] and r["iv"] > 0:
            by_exp.setdefault(r["expd"], []).append(r)
    cand = [(e, rr) for e, rr in by_exp.items() if len(rr) >= 12]
    if not cand:
        raise RuntimeError(f"{sym}: yeterli strike'lı vade yok")
    expd, rr = min(cand, key=lambda er: abs(er[1][0]["dte"] - target_dte))
    T = max(rr[0]["dte"], 0.5) / 365.0
    dte = rr[0]["dte"]
    # OTM smile: K<spot put-IV, K>=spot call-IV (likit taraf)
    smile = {}
    for r in rr:
        otm = (r["cp"] == "P" and r["K"] < spot) or (r["cp"] == "C" and r["K"] >= spot)
        if otm:
            smile[r["K"]] = r["iv"]
    Ks = np.array(sorted(smile))
    ivs = np.array([smile[k] for k in Ks])
    if len(Ks) < 10:
        raise RuntimeError(f"{sym}: OTM smile ince ({len(Ks)})")
    # yoğun grid + IV interpolasyon (uçlarda düz-ekstrapolasyon)
    grid = np.linspace(Ks.min(), Ks.max(), 401)
    iv_grid = np.interp(grid, Ks, ivs)
    C = np.array([bs_call(spot, k, T, s) for k, s in zip(grid, iv_grid)])
    # Breeden-Litzenberger: f(K) = e^{rT} d²C/dK²
    dK = grid[1] - grid[0]
    d2C = np.gradient(np.gradient(C, dK), dK)
    f = np.exp(R * T) * d2C
    f = np.clip(f, 0, None)                       # negatif = interpolasyon-gürültüsü, kırp
    # DÜRÜSTLÜK-TEŞHİSİ (adversarial-denetim 2026-07-06): kırpma-öncesi kütle (1'den sapma=kuyruk-gürültüsü),
    # bandın σ-kapsamı (dar band = kuyruk-kenarı, aşağı-yanlı P). forward-recovery yalnız 1. moment'i valide eder.
    mass = float(_trapz(f, grid))
    fwd = spot * exp((R - Q) * T)
    atm_iv = float(np.interp(fwd, Ks, ivs))
    band_sigma = band / (atm_iv * sqrt(T)) if atm_iv > 0 else float("nan")
    diag = {"mass_pre_norm": mass, "atm_iv": atm_iv, "band_sigma": band_sigma}
    return spot, dte, grid, f, Ks, ivs, T, diag


def hist_horizon_dist(asset, h):
    """Modelin FİZİKSEL h-günlük ENDEKS getiri dağılımı (aynı ufuk, karşılaştırma için)."""
    from spine import contract as C
    scores, prices, vector, prov = C.read_frozen()
    px = prices[asset].dropna()
    fwd = (px.shift(-h) / px - 1.0).dropna()
    return fwd.values


def main():
    print("=" * 92)
    print("  A1 — MARKET-IMPLIED RND (Breeden-Litzenberger) vs modelin TARİHSEL dağılımı — canlı CBOE zinciri")
    print("=" * 92)
    for sym, asset in (("_SPX", "SPX"), ("QQQ", "NDX")):
        try:
            spot, dte, K, f, Ks, ivs, T, diag = extract_rnd(sym)
        except Exception as e:
            print(f"\n████ [{asset}] ({sym}) — RND çıkarılamadı: {type(e).__name__}: {str(e)[:90]}")
            continue
        f, m1, sd, sk, ku = moments_from_rnd(K, f)
        fwd_theo = spot * exp((R - Q) * T)
        # implied getiri-uzayı: r = K/spot − 1
        ret = K / spot - 1.0
        iv_ann = sd / spot / sqrt(T)               # implied vol (yıllık, RND std'den)
        # kuyruk olasılıkları (RND): P(S_T < spot·(1−x))
        def p_below(x):
            thr = spot * (1 - x); m = K <= thr
            return float(_trapz(f[m], K[m])) if m.any() else 0.0
        # tarihsel karşılaştırma (fiziksel, aynı ufuk ~dte)
        h = max(5, int(round(dte)))
        hist = hist_horizon_dist(asset, h)
        def hp_below(x): return float(np.mean(hist < -x))
        print(f"\n████ [{asset}] ({sym})  spot {spot:.1f}  vade {dte}g  forward(teo) {fwd_theo:.1f}  "
              f"RND-ort {m1:.1f} (≈forward doğrulama: {'OK' if abs(m1-fwd_theo)/spot<0.02 else 'SAPMA'})")
        rel = "GÜVENİLİR" if (abs(diag['mass_pre_norm']-1) < 0.03 and diag['band_sigma'] > 5) else "GÜRÜLTÜLÜ (kuyruk-momentleri kırılgan)"
        print(f"  TEŞHİS: kütle(norm-öncesi) {diag['mass_pre_norm']:.3f} | band ±{diag['band_sigma']:.1f}σ | "
              f"ATM-nokta-IV {diag['atm_iv']:.1%} → çıkarım {rel}")
        print(f"  IMPLIED (risk-nötr Q):  ATM-vol {diag['atm_iv']:.1%} / RND-std-vol {iv_ann:.1%}  skew {sk:+.2f}  "
              f"kurtosis {ku:.2f}  [skew/kurt kuyruk-hassas, betimsel]")
        print(f"  {dte}-günlük düşüş olasılıkları — PİYASA-implied (Q) vs MODEL-tarihsel (P, fiziksel):")
        print(f"     {'eşik':>8}{'P(düşüş) Q':>14}{'P(düşüş) P':>14}{'Q/P (crash primi)':>20}")
        for x in (0.03, 0.05, 0.10, 0.20):
            pq, pp = p_below(x), hp_below(x)
            ratio = (pq / pp) if pp > 1e-6 else float("inf")
            print(f"     {'-'+str(int(x*100))+'%':>8}{100*pq:>13.2f}%{100*pp:>13.2f}%{ratio:>18.1f}×")
        # tarihsel skew karşılaştırması
        print(f"  ŞEKİL karşılaştırması: implied-skew {sk:+.2f} vs tarihsel-skew {pd.Series(hist).skew():+.2f} "
              f"| implied-vol {iv_ann:.0%} vs tarihsel-vol(yıllık) {hist.std()*sqrt(252/h):.0%}")
    print("\n" + "-" * 92)
    print("  Q(risk-nötr, opsiyon) vs P(fiziksel, realized) — fark ZATEN risk primi. Betimsel, pozisyona SIFIR.")
    print("-" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
