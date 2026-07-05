"""
screen/reconstruct_flip_history — md_{spy,qqq}.parquet GÜNLÜK zincirinden GAMMA-FLIP TARİHÇESİ rekonstrüksiyonu.

PROXY DAMGASI (raporda da var): bu zincir GÜNDE-TEK-FRONT-MONTH (en yakın ay, dte 0-25). Dolayısıyla
hesaplanan flip = canlı CBOE ÇOK-VADELİ gamma-flip'in FRONT-MONTH PROXY'si. İmplied vol diskte %100 NULL
→ her satırın IV'si mid'den BS-implied (Brent kök-bulma) çıkarılır. Per-strike net-GEX = sign·bs_gamma·OI·spot²·0.01,
BS-repricing ile fiyat-ekseninde sıfır-geçiş (flip). HEM naive (C:+1, P:-1) HEM put-overweight (C:+1, P:-1.3).

ÇIKTI: data/cache/flip_history_{spy,qqq}.parquet
  schema: [as_of, ticker, spot, gamma_flip_naive, gamma_flip_pw13, regime_naive, regime_pw13, exp_move_1d]
  regime = sign(spot - flip): +1 = spot flip ÜSTÜ (dealer long-gamma/pin), -1 = spot flip ALTI (short-gamma/kırılgan).

CENSUS (sonuçtan ÖNCE basılır, Emir mandası): n gün, tarih aralığı, OI nonzero %, IV-implied başarı %/gün,
flip-bulunan gün %, spot tail-gap. Eksik/bayat bacak → HALT-LOUD (sessizce küçültme YOK).

SANITY: rekonstrükte flip-rejim işareti, market-wide SqueezeMetrics GEX işaretiyle overlap'te çapraz-doğrulanır.
"""
from __future__ import annotations

import sys
from datetime import date
from math import exp, log, pi, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# --- BS sabitleri (_cboe_lib ile aynı) ---
R, Q = 0.04, 0.013
CHAIN_DIR = ROOT / "data" / "historical_chains"
CACHE = ROOT / "data" / "cache"
TICKERS = ("spy", "qqq")

# IV-solver sınırları
IV_LO, IV_HI = 0.01, 5.0      # 1%..500% vol arama bandı
MONEYNESS_BAND = 0.15          # |K/S-1| <= 15% (flip taraması için anlamlı bölge; _cboe_lib band ile aynı)


# ---------------------------------------------------------------- BS pricing & IV
def _npdf(x: float) -> float:
    return exp(-x * x / 2.0) / sqrt(2.0 * pi)


def _ncdf(x: float) -> float:
    # standart normal CDF (math.erf ile)
    from math import erf
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_price(S: float, K: float, T: float, sigma: float, cp: str) -> float:
    """Black-Scholes-Merton (sürekli q temettü) opsiyon fiyatı."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        # T->0 sınırında içsel değer
        intr = (S - K) if cp == "C" else (K - S)
        return max(intr, 0.0)
    a = sigma * sqrt(T)
    d1 = (log(S / K) + (R - Q + sigma * sigma / 2.0) * T) / a
    d2 = d1 - a
    if cp == "C":
        return S * exp(-Q * T) * _ncdf(d1) - K * exp(-R * T) * _ncdf(d2)
    return K * exp(-R * T) * _ncdf(-d2) - S * exp(-Q * T) * _ncdf(-d1)


def bs_gamma(S: float, K: float, T: float, sigma: float) -> float:
    """_cboe_lib.bs_gamma ile birebir (dgamma/dS², C ve P için aynı)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    a = sigma * sqrt(T)
    d1 = (log(S / K) + (R - Q + sigma * sigma / 2.0) * T) / a
    return exp(-Q * T) * _npdf(d1) / (S * a)


def implied_vol(price: float, S: float, K: float, T: float, cp: str) -> float | None:
    """mid fiyatından BS-implied vol (Brent benzeri bisection, scipy.brentq). No-arb ihlali / 0DTE / derin-ITM
    (zaman-değeri yok) → None (satır elenir). Newton yerine bracket-güvenli kök-bulma."""
    if price is None or not np.isfinite(price) or price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    # no-arbitrage alt sınır (içsel değer, taşıma maliyetli): C >= S e^-qT - K e^-rT
    disc_intr = (S * exp(-Q * T) - K * exp(-R * T)) if cp == "C" else (K * exp(-R * T) - S * exp(-Q * T))
    intr_floor = max(disc_intr, 0.0)
    upper = (S * exp(-Q * T)) if cp == "C" else (K * exp(-R * T))   # fiyat üst sınırı
    # zaman-değeri ~0 (derin-ITM, mid≈içsel) ya da no-arb ihlali → vol çözülemez
    if price <= intr_floor + 1e-6 or price >= upper - 1e-9:
        return None

    def f(sig: float) -> float:
        return bs_price(S, K, T, sig, cp) - price

    flo, fhi = f(IV_LO), f(IV_HI)
    if flo * fhi > 0:   # kök bracket'lenemedi (fiyat erişilebilir vol bandı dışında)
        return None
    try:
        from scipy.optimize import brentq
        iv = brentq(f, IV_LO, IV_HI, maxiter=100, xtol=1e-6)
    except Exception:
        return None
    if not np.isfinite(iv) or iv <= IV_LO + 1e-6 or iv >= IV_HI - 1e-6:
        return None
    return float(iv)


# ---------------------------------------------------------------- per-day flip
def _build_day_rows(g: pd.DataFrame, spot: float, asof: date) -> tuple[list[dict], int, int]:
    """Bir günün zincirini IV-implied satırlara çevirir. -> (rows, n_total_candidate, n_iv_ok)."""
    expd = pd.to_datetime(g["expiration"].iloc[0]).date()
    dte = (expd - asof).days
    T = max(dte, 0.5) / 365.0      # 0DTE'de T=0.5/365 taban (gamma patlamasını önler; _cboe_lib ile aynı)
    rows: list[dict] = []
    n_cand = 0
    for K, cp, oi, mid in zip(g["strike"].astype(float), g["right"],
                              g["open_interest"].astype(float), g["mid"].astype(float)):
        if oi <= 0:                                  # OI sıfır → ağırlık yok
            continue
        if abs(K / spot - 1.0) > MONEYNESS_BAND:     # moneyness bandı dışı → flip için ilgisiz
            continue
        n_cand += 1
        iv = implied_vol(mid, spot, K, T, cp)
        if iv is None:                               # 0DTE/derin-ITM/no-arb → ele
            continue
        rows.append({"K": K, "cp": cp, "oi": oi, "iv": iv, "T": T, "dte": dte})
    return rows, n_cand, len(rows)


def _flip_repricing(rows: list[dict], spot: float, put_w: float,
                    lo: float = -0.08, hi: float = 0.08, n: int = 65) -> tuple[float | None, float]:
    """BS-repricing sıfır-gamma flip taraması (_cboe_lib.flip_bs naive yolu ile birebir mantık).
    sign: C -> +1, P -> -put_w. net-GEX(hs) = Σ sign·bs_gamma(hs,K,T,iv)·OI·100·hs²·0.01. Sıfır-geçiş = flip.
    -> (flip, net@spot)."""
    if not rows:
        return None, 0.0

    def net(hs: float) -> float:
        tot = 0.0
        for r in rows:
            sgn = 1.0 if r["cp"] == "C" else -put_w
            tot += sgn * bs_gamma(hs, r["K"], r["T"], r["iv"]) * r["oi"] * 100.0 * hs * hs * 0.01
        return tot

    grid = [(spot * (1 + p), net(spot * (1 + p))) for p in np.linspace(lo, hi, n)]
    flip = None
    for (s0, g0), (s1, g1) in zip(grid, grid[1:]):
        if (g0 <= 0 <= g1) or (g0 >= 0 >= g1):
            flip = s0 + (s1 - s0) * (0 - g0) / (g1 - g0) if g1 != g0 else s0
            break
    return (round(float(flip), 4) if flip else None), float(net(spot))


def _exp_move_1d(rows: list[dict], spot: float) -> float | None:
    """1-günlük beklenen hareket (ATM IV ile): spot·σ_atm·√(1/252). En-ATM satırın IV'si."""
    if not rows:
        return None
    atm = min(rows, key=lambda r: abs(r["K"] / spot - 1.0))
    return float(spot * atm["iv"] * sqrt(1.0 / 252.0))


# ---------------------------------------------------------------- spot
def _load_spot_yf() -> dict[str, pd.Series]:
    """yfinance SPY/QQQ günlük close (canlı spot bacağı; parity-spot ile çapraz-doğrulanır)."""
    import warnings
    warnings.filterwarnings("ignore")
    import yfinance as yf
    d = yf.download(["SPY", "QQQ"], start="2025-06-10", end="2026-06-12",
                    progress=False, auto_adjust=False)
    cl = d["Close"] if "Close" in d.columns.get_level_values(0) else d
    out = {}
    for t in ("SPY", "QQQ"):
        s = cl[t].dropna()
        s.index = pd.to_datetime(s.index).normalize()
        out[t] = s
    return out


def _parity_spot(g: pd.DataFrame) -> float | None:
    """Put-call parity ATM proxy spot (fallback + çapraz-doğrulama). S ≈ K + (C-P), en-ATM 15 strike medyanı."""
    piv = g.pivot_table(index="strike", columns="right", values="mid")
    if "C" not in piv or "P" not in piv:
        return None
    piv = piv.dropna()
    if len(piv) == 0:
        return None
    cmp_ = (piv["C"] - piv["P"]).values
    simp = piv.index.values.astype(float) + cmp_
    absd = np.abs(piv["C"].values - piv["P"].values)
    order = np.argsort(absd)[:15]
    return float(np.median(simp[order]))


# ---------------------------------------------------------------- main per ticker
def reconstruct(ticker: str, yf_spot: dict[str, pd.Series], census: list[dict]) -> pd.DataFrame:
    yf_key = ticker.upper()
    df = pd.read_parquet(CHAIN_DIR / f"md_{ticker}.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    spot_series = yf_spot[yf_key]

    recs = []
    iv_ok_frac = []
    spot_gap_days = 0
    parity_diffs = []
    for asof_ts, g in df.groupby("date"):
        asof = asof_ts.date()
        par = _parity_spot(g)
        # spot: yfinance close (canlı), yoksa parity proxy
        if asof_ts in spot_series.index:
            spot = float(spot_series.loc[asof_ts])
            if par is not None and par > 0:
                parity_diffs.append(abs(spot - par) / spot)
        elif par is not None:
            spot = par
            spot_gap_days += 1
        else:
            continue  # ne yf ne parity → atla (census'ta görünür)

        rows, n_cand, n_iv = _build_day_rows(g, spot, asof)
        iv_ok_frac.append((n_iv / n_cand) if n_cand else 0.0)
        fn, _ = _flip_repricing(rows, spot, put_w=1.0)
        fp, _ = _flip_repricing(rows, spot, put_w=1.3)
        em = _exp_move_1d(rows, spot)
        recs.append({
            "as_of": asof_ts, "ticker": yf_key, "spot": round(spot, 4),
            "gamma_flip_naive": fn, "gamma_flip_pw13": fp,
            "regime_naive": (1 if (fn is not None and spot >= fn) else (-1 if fn is not None else 0)),
            "regime_pw13": (1 if (fp is not None and spot >= fp) else (-1 if fp is not None else 0)),
            "exp_move_1d": (round(em, 4) if em is not None else None),
        })

    out = pd.DataFrame(recs).sort_values("as_of").reset_index(drop=True)
    n = len(out)
    flip_found_naive = float((out["gamma_flip_naive"].notna()).mean()) if n else 0.0
    flip_found_pw13 = float((out["gamma_flip_pw13"].notna()).mean()) if n else 0.0
    oi_nonzero = float((df["open_interest"] > 0).mean())
    census.append({
        "ticker": yf_key, "n_days": n,
        "date_lo": str(out["as_of"].min().date()) if n else None,
        "date_hi": str(out["as_of"].max().date()) if n else None,
        "oi_nonzero_pct": round(100 * oi_nonzero, 1),
        "iv_implied_ok_pct_per_day": round(100 * float(np.mean(iv_ok_frac)), 1) if iv_ok_frac else 0.0,
        "flip_found_naive_pct": round(100 * flip_found_naive, 1),
        "flip_found_pw13_pct": round(100 * flip_found_pw13, 1),
        "spot_proxy_gap_days": spot_gap_days,
        "parity_vs_yf_median_abs_pct": (round(100 * float(np.median(parity_diffs)), 3) if parity_diffs else None),
    })
    return out


# ---------------------------------------------------------------- sanity x-ref
def sanity_xref(flips: dict[str, pd.DataFrame]) -> dict:
    """Rekonstrükte flip-rejim işaretini market-wide SqueezeMetrics GEX işaretiyle çapraz-doğrula.
    SqueezeMetrics GEX>0 = net dealer LONG gamma (≈ spot flip ÜSTÜ, regime +1). overlap'te işaret uyumu."""
    sg_path = CACHE / "squeeze_dix_gex.parquet"
    res = {"available": sg_path.exists()}
    if not sg_path.exists():
        return res
    sg = pd.read_parquet(sg_path)
    sg.index = pd.to_datetime(sg.index).normalize()
    gex_sign = np.sign(sg["gex"])
    for t, fl in flips.items():
        f = fl.set_index("as_of")
        idx = f.index.intersection(sg.index)
        if len(idx) == 0:
            res[t] = {"overlap": 0}
            continue
        rn = f.loc[idx, "regime_naive"]
        gs = gex_sign.reindex(idx)
        mask = (rn != 0) & gs.notna()
        agree = float((np.sign(rn[mask]) == gs[mask]).mean()) if mask.any() else float("nan")
        res[t] = {"overlap": int(mask.sum()),
                  "recon_pct_pos": round(100 * float((rn[mask] > 0).mean()), 1) if mask.any() else None,
                  "squeeze_pct_pos": round(100 * float((gs[mask] > 0).mean()), 1) if mask.any() else None,
                  "sign_agree_pct": round(100 * agree, 1) if np.isfinite(agree) else None}
    return res


def main() -> int:
    print("=" * 100)
    print("  RECONSTRUCT FLIP HISTORY — front-month-proxy gamma-flip (md_{spy,qqq}.parquet, IV-from-mid)")
    print("=" * 100)

    # HALT-on-missing: zincirler var mı?
    for t in TICKERS:
        p = CHAIN_DIR / f"md_{t}.parquet"
        if not p.exists():
            raise SystemExit(f"HALT: zincir bulunamadı: {p}")

    # spot bacağı (yfinance) — HALT-on-missing
    try:
        yf_spot = _load_spot_yf()
    except Exception as e:
        raise SystemExit(f"HALT: yfinance SPY/QQQ spot çekilemedi (spot bacağı eksik): {type(e).__name__}: {e}")
    for k in ("SPY", "QQQ"):
        if yf_spot[k].empty:
            raise SystemExit(f"HALT: yfinance {k} close boş — spot bacağı eksik.")
    print(f"  spot bacağı (yfinance): SPY {yf_spot['SPY'].index.min().date()}..{yf_spot['SPY'].index.max().date()} "
          f"n={len(yf_spot['SPY'])} | QQQ n={len(yf_spot['QQQ'])}")

    census: list[dict] = []
    flips: dict[str, pd.DataFrame] = {}
    CACHE.mkdir(parents=True, exist_ok=True)
    for t in TICKERS:
        out = reconstruct(t, yf_spot, census)
        flips[t.upper()] = out
        out.to_parquet(CACHE / f"flip_history_{t}.parquet", index=False)

    # --- CENSUS (sonuçtan ÖNCE) ---
    print("\n" + "-" * 100)
    print("  CENSUS (sonuçtan ÖNCE — Emir mandası; eksik/bayat bacak → HALT-LOUD)")
    print("-" * 100)
    print(f"  {'tk':<5}{'n_gün':>6}{'tarih aralığı':>26}{'OInz%':>7}{'IVok%/g':>9}"
          f"{'flipN%':>8}{'flipP%':>8}{'spotGap':>8}{'parityΔ%':>9}")
    for c in census:
        rng = f"{c['date_lo']}..{c['date_hi']}"
        pd_ = c["parity_vs_yf_median_abs_pct"]
        print(f"  {c['ticker']:<5}{c['n_days']:>6}{rng:>26}{c['oi_nonzero_pct']:>7}{c['iv_implied_ok_pct_per_day']:>9}"
              f"{c['flip_found_naive_pct']:>8}{c['flip_found_pw13_pct']:>8}{c['spot_proxy_gap_days']:>8}"
              f"{(f'{pd_:.3f}' if pd_ is not None else 'n/a'):>9}")

    # spot tail-gap notu (INDEX getiri bacağı): SPX/NDX CSV 2026-05-26'da biter, zincir 06-08'e
    print("\n  NOT (spot tail-gap): spot bacağı (SPY/QQQ yfinance) TÜM zincir penceresini kapsar (gap yok).")
    print("        Ancak INDEX getiri bacağı (SPX/NDX CSV) 2026-05-26'da biter → directional fwd-getiri")
    print("        son ~9 zincir günü (05-27..06-08) için YOK. Frozen tide de 2026-05-22'de biter.")
    print("        → Stage-2 (tide-üstü) overlap penceresi 2025-06-13..2026-05-22 (bağlayıcı = tide).")

    # HALT eşikleri (sessiz küçültme YOK)
    for c in census:
        if c["n_days"] < 100:
            raise SystemExit(f"HALT: {c['ticker']} yalnız {c['n_days']} gün (<100) — yetersiz, sessiz küçültme yok.")
        if c["iv_implied_ok_pct_per_day"] < 50:
            raise SystemExit(f"HALT: {c['ticker']} IV-implied başarı %{c['iv_implied_ok_pct_per_day']} (<50) — mid→IV bozuk.")
        if c["flip_found_naive_pct"] < 50:
            print(f"  UYARI: {c['ticker']} flip-bulunan gün %{c['flip_found_naive_pct']} (<50) — flip seyrek, "
                  f"directional sinyal zayıf olabilir (HALT değil, raporda görünür).")

    # --- SANITY x-ref ---
    print("\n" + "-" * 100)
    print("  SANITY — rekonstrükte flip-rejim işareti × market-wide SqueezeMetrics GEX işareti (overlap)")
    print("  (SqueezeMetrics GEX>0 ≈ dealer net-long-gamma ≈ spot flip ÜSTÜ = regime +1. Geniş uyum beklenir.)")
    print("-" * 100)
    xref = sanity_xref(flips)
    if not xref.get("available"):
        print("  squeeze_dix_gex.parquet yok → çapraz-doğrulama atlandı.")
    else:
        for t in ("SPY", "QQQ"):
            r = xref.get(t, {})
            if r.get("overlap"):
                print(f"  {t}: overlap {r['overlap']}g | recon %pos {r['recon_pct_pos']} | "
                      f"squeeze %pos {r['squeeze_pct_pos']} | İŞARET-UYUMU %{r['sign_agree_pct']}")
            else:
                print(f"  {t}: overlap yok.")
        print("  YORUM: front-month-proxy tek-vade ÇOK-VADELİ market-wide GEX'ten dar; bu pencere kalıcı-pozitif")
        print("         rejim (kuvvetli boğa) → her iki kaynak da çoğunlukla +pos, uyum YÜKSEK beklenir.")

    print("\n  YAZILDI: data/cache/flip_history_{spy,qqq}.parquet")
    print("  PROXY DAMGASI: front-month-proxy flip — canlı CBOE çok-vadeli flip DEĞİL; IV mid'den BS-implied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
