"""
backtest/diagnosis/D6_power — İSTATİSTİKSEL GÜÇ & MİNİMUM VERİ (TEŞHİS-ONLY).

Soru: directional-GEX iddiasını (gamma_inv günlük Sharpe + wall-touch event-edge) DÜRÜST
biçimde "kanıtlanmış" saymak için kaç gözlem (N) gerekir? İki eşik birlikte:
  (i)  t ≥ 2   → SR ölçülen-değeri sıfırdan klasik-anlamlı (tek-test).
  (ii) DSR > 0 → Bailey & López de Prado Deflated-Sharpe; K=10 trial'lık çoklu-test
        null'unu (best-of-N seçim-yanlılığı) AŞ. DSR>0 = gözlenen SR, K-trial null'un
        beklenen-maksimumundan istatistiksel olarak büyük.

Üç temel (her biri için gereken-N TABLOSU):
  (a) KOŞULSUZ-GÜNLÜK : gamma_inv günlük net P&L (block_robust.gamma_inv_pnl, OKUNUR).
                        Gözlenen full Sharpe SPY +1.29 / QQQ +0.87 (script teyit eder).
  (b) REJİM-KOŞULLU   : aynı günlük P&L, +γ ve −γ dallarına AYRILMIŞ (branch başına SR & N).
  (c) WALL-TOUCH EVENT: spine_diagnostic.mean_reversion_return event-edge'i (duvar-dokunuş
                        başına MR getirisi). Per-event SE → t & gereken event-N.

SE FORMÜLLERİ + VARSAYIMLAR (hepsi script-içinde AÇIK, §0'da yazdırılır):
  - Günlük Sharpe t-stat (Lo 2002, IID yaklaşık):  t = SR_daily · √N,  SR_daily = SR_ann/√252.
    → t≥2 için  N = (2 / SR_daily)².   [Varsayım: IID, çarpıklık/kurtosis düzeltmesi YOK —
      muhafazakâr-değil; gerçek getiri fat-tail ise gereken-N daha BÜYÜK.]
  - DSR (Bailey-LdP 2014):  SR_0 = E[max_{K} null] = √(Var[SR]) · [(1−γ)Φ⁻¹(1−1/K) + γΦ⁻¹(1−1/(Ke))],
    Var[SR] ≈ (1/(N−1))·(1 − γ3·SR + (γ4−1)/4·SR²)  (SR günlük birimde).
    DSR = Φ( (SR_daily − SR_0) · √(N−1) / √(1 − γ3·SR_daily + (γ4−1)/4·SR_daily²) ).
    DSR>0 ⇔ DSR>0.5 ⇔ SR_daily > SR_0.  → gereken-N: SR_daily(gözlenen-sabit) > SR_0(N) kökü.
    [Varsayım: K=10 bağımsız-trial, getiri momentleri (γ3,γ4) gözlenen-örnekten; null SR=0.]
  - Event-edge SE (per-event):  SE = σ_event / √N_event,  t = mean / SE.
    → t≥2 için  N_event = (2·σ_event / mean_event)².  [Varsayım: olaylar IID; kümeleme YOK.]

PIT/READ-ONLY: yeni strateji/parametre/reweight YOK. block_robust.gamma_inv_pnl ve
spine_diagnostic.mean_reversion_return YALNIZ OKUNUR (mevcut P&L/level). Yeni P&L üretilmez.
  & C:/Users/admin/Downloads/kader-macro/.venv/Scripts/python.exe backtest/diagnosis/D6_power.py
"""
from __future__ import annotations

import sys
from math import sqrt, exp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backtest"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from block_robust import gamma_inv_pnl          # noqa: E402  (OKUNUR — günlük net P&L)
from spine_diagnostic import build_panel, mean_reversion_return  # noqa: E402

K_TRIALS = 10                # Bailey-LdP deflation: denenen-konfigürasyon sayısı (spec: K=10)
ANN = 252.0
EULER_G = 0.5772156649015329  # Euler-Mascheroni (DSR E[max] formülünde)


# ───────────────────────────────────────────────────────────────────── istatistik çekirdek
def ann_sharpe(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return 0.0
    return x.mean() / x.std(ddof=1) * sqrt(ANN)


def moments(x):
    """Günlük getiri serisinden (SR_daily, skew γ3, kurtosis γ4) — DSR Var[SR] için."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    sd = x.std(ddof=1)
    sr_d = x.mean() / sd if sd > 0 else 0.0
    m = x - x.mean()
    g3 = (m**3).mean() / (sd**3) if sd > 0 else 0.0           # skew (biased; DSR formülü böyle kullanır)
    g4 = (m**4).mean() / (sd**4) if sd > 0 else 3.0           # kurtosis (normal=3)
    return sr_d, g3, g4, n


def sr0_expected_max(var_sr, K):
    """Bailey-LdP: K bağımsız null-trial'ın beklenen-maks Sharpe'ı (günlük birimde).
    SR_0 = √Var[SR]·[(1−γ)·Z(1−1/K) + γ·Z(1−1/(K·e))],  Z=Φ⁻¹."""
    if K <= 1 or var_sr <= 0:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / K)
    z2 = norm.ppf(1.0 - 1.0 / (K * exp(1.0)))
    return sqrt(var_sr) * ((1 - EULER_G) * z1 + EULER_G * z2)


def var_sr_daily(sr_d, g3, g4, N):
    """SR_daily'nin örnekleme varyansı (Mertens/Lo; Bailey-LdP DSR'de kullanılan form)."""
    if N <= 1:
        return np.inf
    return (1.0 - g3 * sr_d + (g4 - 1.0) / 4.0 * sr_d**2) / (N - 1)


def dsr(sr_d, g3, g4, N, K):
    """Deflated Sharpe Ratio (olasılık ∈[0,1]); SR günlük birimde, sr_ref = SR_0(K-trial null)."""
    v_unit = (1.0 - g3 * sr_d + (g4 - 1.0) / 4.0 * sr_d**2)     # Var[SR]·(N−1)
    if v_unit <= 0 or N <= 1:
        return float("nan")
    var_sr = v_unit / (N - 1)
    sr_ref = sr0_expected_max(var_sr, K)
    z = (sr_d - sr_ref) * sqrt(N - 1) / sqrt(v_unit)
    return float(norm.cdf(z))


def n_for_t2(sr_d):
    """t = SR_daily·√N ≥ 2 → N = (2/SR_daily)². sr_d>0 varsayımı (edge yönü doğru)."""
    if sr_d <= 0:
        return float("inf")
    return (2.0 / sr_d) ** 2


def n_for_dsr_pos(sr_d, g3, g4, K):
    """DSR>0.5 (yani DSR>'0' = null'u aş) için minimum N. SR_daily SABİT (gözlenen),
    N büyüdükçe Var[SR]↓ → SR_0↓ → bir N* sonrası SR_daily>SR_0. Monoton; ikili-arama."""
    if sr_d <= 0:
        return float("inf")
    # üst sınır ara
    def ok(N):
        var_sr = var_sr_daily(sr_d, g3, g4, N)
        return sr_d > sr0_expected_max(var_sr, K)
    lo, hi = 2, 4
    while not ok(hi):
        hi *= 2
        if hi > 5_000_000:
            return float("inf")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


def fmt_n(n):
    if not np.isfinite(n):
        return "∞ (edge≤0 → asla)"
    return f"{int(round(n)):,}"


def trading_to_years(n_days):
    return n_days / 252.0


# ───────────────────────────────────────────────────────────────────── (a) KOŞULSUZ-GÜNLÜK
def panel_a(sym):
    pnl, _ = gamma_inv_pnl(sym)          # OKUNUR — günlük net P&L serisi
    x = pnl.values.astype(float)
    sr_ann = ann_sharpe(x)
    sr_d, g3, g4, N = moments(x)
    return dict(sym=sym, x=pnl, sr_ann=sr_ann, sr_d=sr_d, g3=g3, g4=g4, N=N)


def section_a(rows):
    print("=" * 104)
    print("  (a) KOŞULSUZ-GÜNLÜK — gamma_inv günlük net P&L (block_robust.gamma_inv_pnl, OKUNUR)")
    print("=" * 104)
    print(f"  {'sym':<5}{'gözlenen':>10}{'SR_daily':>11}{'skewγ3':>9}{'kurtγ4':>9}{'mevcutN':>9}"
          f"{'N(t≥2)':>10}{'N(DSR>0,K=10)':>15}{'N_gerekli=max':>15}")
    out = {}
    for r in rows:
        nt2 = n_for_t2(r["sr_d"])
        ndsr = n_for_dsr_pos(r["sr_d"], r["g3"], r["g4"], K_TRIALS)
        nreq = max(nt2, ndsr)
        out[r["sym"]] = dict(nt2=nt2, ndsr=ndsr, nreq=nreq, N=r["N"], sr_ann=r["sr_ann"])
        print(f"  {r['sym']:<5}{r['sr_ann']:>+9.2f} {r['sr_d']:>+10.4f}{r['g3']:>+9.2f}{r['g4']:>9.2f}"
              f"{r['N']:>9}{fmt_n(nt2):>10}{fmt_n(ndsr):>15}{fmt_n(nreq):>15}")
        # mevcut DSR (teşhis): bugünkü N ile DSR ne
        d_now = dsr(r["sr_d"], r["g3"], r["g4"], r["N"], K_TRIALS)
        t_now = r["sr_d"] * sqrt(r["N"])
        print(f"        └─ mevcut-N'de: t={t_now:+.2f}  DSR(K=10)={d_now:.3f}  "
              f"→ {'GEÇER' if (t_now >= 2 and d_now > 0.5) else 'GEÇMEZ'} "
              f"(t≥2 {'✓' if t_now>=2 else '✗'}, DSR>0 {'✓' if d_now>0.5 else '✗'})")
    return out


# ───────────────────────────────────────────────────────────────────── (b) REJİM-KOŞULLU
def section_b(rows_panel):
    print("\n" + "=" * 104)
    print("  (b) REJİM-KOŞULLU — günlük P&L +γ / −γ dallarına AYRILMIŞ (branch başına SR & gereken-N)")
    print("  (NOT: dal SR'ı, o dalın günlerinde alınan günlük P&L'in Sharpe'ı; mevcut-N = dal-günü sayısı)")
    print("=" * 104)
    out = {}
    for sym in ("SPY", "QQQ"):
        pnl, p = gamma_inv_pnl(sym)       # OKUNUR; p.regime ile dal ayır
        reg = p["regime"].values
        dpnl = pnl.values
        print(f"  ── {sym}")
        out[sym] = {}
        for tag, mask in (("+γ (regime=+1)", reg > 0), ("−γ (regime=−1)", reg < 0)):
            x = dpnl[mask]
            sr_ann = ann_sharpe(x)
            sr_d, g3, g4, N = moments(x)
            nt2 = n_for_t2(sr_d)
            ndsr = n_for_dsr_pos(sr_d, g3, g4, K_TRIALS)
            nreq = max(nt2, ndsr)
            t_now = sr_d * sqrt(N) if N > 1 else 0.0
            d_now = dsr(sr_d, g3, g4, N, K_TRIALS) if N > 2 else float("nan")
            out[sym][tag] = dict(N=N, sr_ann=sr_ann, nt2=nt2, ndsr=ndsr, nreq=nreq)
            print(f"     {tag:<16} mevcutN {N:>4}  SR {sr_ann:>+6.2f}  SR_d {sr_d:>+.4f}  "
                  f"t={t_now:>+5.2f}  DSR {d_now if d_now==d_now else float('nan'):.3f}  "
                  f"→ N(t≥2) {fmt_n(nt2):>10}  N(DSR>0) {fmt_n(ndsr):>10}  N_gerekli {fmt_n(nreq):>10}")
    return out


# ───────────────────────────────────────────────────────────────────── (c) WALL-TOUCH EVENT
def wall_touch_counts(sym):
    """spine_diagnostic mantığıyla: sembol × rejim × duvar-tipi wall-touch event sayımı.
    Bir 'event' = D+1 RTH seansında high≥call_wall (call-touch) VEYA low≤put_wall (put-touch).
    Aynı gün hem call hem put dokunabilir → ayrı sayılır. PIT: D-EOD seviye × D+1 seans."""
    p = build_panel(sym)
    p["reg"] = p["regime"]
    counts = {}                 # (regime, walltype) -> n
    ret_by = {}                 # (regime, walltype) -> list of geri-dönüş getirileri (bps)
    for _, r in p.iterrows():
        reg = int(r["regime"]) if pd.notna(r["regime"]) else 0
        if pd.notna(r["call_wall"]) and pd.notna(r["h1"]) and r["h1"] >= r["call_wall"]:
            counts[(reg, "call")] = counts.get((reg, "call"), 0) + 1
            ret_by.setdefault((reg, "call"), []).append((r["call_wall"] - r["c1"]) / r["call_wall"])
        if pd.notna(r["put_wall"]) and pd.notna(r["l1"]) and r["l1"] <= r["put_wall"]:
            counts[(reg, "put")] = counts.get((reg, "put"), 0) + 1
            ret_by.setdefault((reg, "put"), []).append((r["c1"] - r["put_wall"]) / r["put_wall"])
    return p, counts, ret_by


def section_c(panels):
    print("\n" + "=" * 104)
    print("  (c) WALL-TOUCH EVENT-BAZLI — spine_diagnostic.mean_reversion_return event-edge'i")
    print("      MR-getiri = duvardan-kapanışa geri-dönüş (>0 duvar-tuttu / <0 kırıldı). Edge = MR'nin ort.")
    print("      SE = σ_event/√N_event ; t = mean/SE ; N(t≥2)=(2σ/mean)². Olaylar IID-varsayımı (kümeleme YOK).")
    print("=" * 104)
    out = {}
    for sym in ("SPY", "QQQ"):
        p = panels[sym]
        mr = mean_reversion_return(p)
        p2 = p.copy(); p2["mr"] = mr
        print(f"  ── {sym}  (toplam {len(p)} gün, duvar-dokunuşlu gün {mr.notna().sum()})")
        out[sym] = {}
        # rejim-koşullu MR event-edge (spine_diagnostic ANA TEZ ile aynı tanım)
        for tag, mask in (("+γ", p2["regime"] == 1), ("−γ", p2["regime"] == -1)):
            x = p2.loc[mask, "mr"].dropna().values
            n = len(x)
            if n < 2:
                print(f"     {tag} MR: n={n} → ÖLÇÜLEMEDİ (yetersiz event)")
                continue
            mean = x.mean(); sd = x.std(ddof=1)
            se = sd / sqrt(n) if n > 0 else np.nan
            t = mean / se if se > 0 else 0.0
            nt2 = (2.0 * sd / abs(mean)) ** 2 if mean != 0 else float("inf")
            out[sym][tag] = dict(n=n, mean=mean, sd=sd, t=t, nt2=nt2)
            print(f"     {tag} MR-event:  n={n:>3}  ort {1e4*mean:>+7.1f}bps  σ {1e4*sd:>6.1f}bps  "
                  f"t={t:>+5.2f}  →  N(t≥2)={fmt_n(nt2):>10}  (|edge| yönü {'tez-içi' if (tag=='+γ' and mean>0) or (tag=='−γ' and mean<0) else 'tez-DIŞI'})")
        out[sym]["_mr_all"] = mr
    return out


def section_c_counts(panels):
    print("\n" + "-" * 104)
    print("  (c-ek) MEVCUT VERİDE WALL-TOUCH EVENT SAYIMI  (sembol × rejim × duvar-tipi)")
    print("-" * 104)
    print(f"  {'sym':<5}{'rejim':>7}{'duvar':>7}{'event-n':>9}{'ort-MR(bps)':>13}{'tez-yönü?':>11}")
    grand = {}
    for sym in ("SPY", "QQQ"):
        _, counts, ret_by = wall_touch_counts(sym)
        for reg in (1, -1):
            for wt in ("call", "put"):
                n = counts.get((reg, wt), 0)
                rs = ret_by.get((reg, wt), [])
                ort = 1e4 * np.mean(rs) if rs else float("nan")
                # tez: +γ duvar TUTAR (MR>0), −γ duvar KIRILIR (MR<0)
                inthesis = ("tez-içi" if ((reg == 1 and (np.mean(rs) > 0 if rs else False)) or
                                          (reg == -1 and (np.mean(rs) < 0 if rs else False))) else "tez-dışı") if rs else "—"
                rtag = "+γ" if reg == 1 else "−γ"
                print(f"  {sym:<5}{rtag:>7}{wt:>7}{n:>9}{ort:>13.1f}{inthesis:>11}")
                grand[(sym, reg, wt)] = n
        # sembol toplamı
        tot = sum(v for (s, _, _), v in grand.items() if s == sym)
        print(f"  {sym:<5}{'TOPLAM':>7}{'(her2)':>7}{tot:>9}{'':>13}{'':>11}")
    return grand


# ───────────────────────────────────────────────────────────────────── ÇEVİRİ
def translation(a_out, b_out, c_out, panels):
    print("\n" + "=" * 104)
    print("  ÇEVİRİ — 'directional iddia için minimum veri' (yıl-cinsinden) ve mevcut-N ile KIYAS")
    print("=" * 104)
    # mevcut çoklu-rejim kapsamı: veri penceresi
    sym0 = "SPY"
    pser = a_out[sym0]
    print(f"  Mevcut veri: günlük gözlem ~{int(pser['N'])} gün ≈ {trading_to_years(pser['N']):.2f} yıl, "
          f"TEK likidite-rejimi (2025-06→2026-06). Stres-pencereleri (2018Q4 / 2020 / 2022) KAPSAM-DIŞI.")
    print()

    print("  [A] KOŞULSUZ-GÜNLÜK directional iddia için minimum:")
    for sym in ("SPY", "QQQ"):
        o = a_out[sym]
        yreq = trading_to_years(o["nreq"])
        gap = o["nreq"] - o["N"]
        print(f"    {sym}: gereken-N(t≥2 & DSR>0,K=10) = {fmt_n(o['nreq'])} gün "
              f"≈ {yreq:.1f} yıl günlük çoklu-rejim seri.  "
              f"Mevcut {int(o['N'])} gün → {'YETERLİ (mevcut≥gereken)' if o['N'] >= o['nreq'] else f'EKSİK: {fmt_n(max(gap,0))} gün (~{trading_to_years(max(gap,0)):.1f} yıl) DAHA lazım'}.")
    print()

    print("  [B] REJİM-KOŞULLU (branch başına) directional iddia için minimum:")
    for sym in ("SPY", "QQQ"):
        for tag, d in b_out[sym].items():
            yreq = trading_to_years(d["nreq"])
            gap = d["nreq"] - d["N"]
            verdict = "YETERLİ" if d["N"] >= d["nreq"] else f"EKSİK ~{trading_to_years(max(gap,0)):.1f} yıl daha"
            print(f"    {sym} {tag}: gereken {fmt_n(d['nreq'])} dal-günü (≈{yreq:.1f} yıl o-rejimde) | "
                  f"mevcut {d['N']} → {verdict}")
    print()

    print("  [C] WALL-TOUCH EVENT directional iddia için minimum (intraday-event yolu):")
    for sym in ("SPY", "QQQ"):
        # event üretim hızı: dokunuşlu-gün / toplam-gün
        mr = c_out[sym]["_mr_all"]
        touched = int(mr.notna().sum()); total = len(mr)
        rate = touched / total if total else 0
        print(f"    {sym}: event-hızı ≈ {touched}/{total} gün = {rate:.2f} dokunuş-gün/işlem-günü "
              f"(yani ~1 event her {1/rate:.1f} günde).")
        for tag in ("+γ", "−γ"):
            d = c_out[sym].get(tag)
            if not d:
                continue
            nreq_ev = d["nt2"]
            # bu kadar event toplamak için kaç işlem-günü (event-hızıyla)
            days_needed = nreq_ev / rate if rate > 0 else float("inf")
            yrs = trading_to_years(days_needed)
            verdict = "YETERLİ" if d["n"] >= nreq_ev else "EKSİK"
            print(f"       {tag} event-edge {1e4*d['mean']:+.1f}bps (t={d['t']:+.2f}, mevcut {d['n']} event): "
                  f"t≥2 için {fmt_n(nreq_ev)} event gerek → ~{fmt_n(days_needed)} işlem-günü (≈{yrs:.1f} yıl) → {verdict} "
                  f"(mevcut {d['n']}/{fmt_n(nreq_ev)})")
    print()
    print("  KAPSAM-ŞARTI (sayıdan bağımsız metodolojik): tek-rejim N ne kadar büyürse-büyüsün, "
          "non-stationary/best-of-K riski stres-rejimi (2018Q4+2020+2022) GÖRMEDEN düşmez. "
          "Yani gereken-N ⇒ '≥X yıl GÜNLÜK ÇOK-REJİMLİ' VEYA event-yolunda intraday-bar ile event-yoğunluğunu artır.")


def main():
    print("# D6 — İSTATİSTİKSEL GÜÇ & MİNİMUM VERİ  (TEŞHİS-ONLY)\n")
    # §0 — formül + varsayım banner
    print("## §0 — SE FORMÜLLERİ & VARSAYIMLAR")
    print(f"  • Günlük-Sharpe t:  t = SR_daily·√N,  SR_daily = SR_ann/√252.  t≥2 ⇒ N=(2/SR_daily)².")
    print(f"  • DSR (Bailey-LdP, K={K_TRIALS} trial):  SR_0 = √Var[SR]·[(1−γ)Φ⁻¹(1−1/K)+γΦ⁻¹(1−1/(Ke))], γ=Euler.")
    print(f"      Var[SR]=(1−γ3·SR+(γ4−1)/4·SR²)/(N−1).  DSR=Φ((SR_d−SR_0)·√(N−1)/√(1−γ3·SR_d+(γ4−1)/4·SR_d²)).")
    print(f"      DSR>0 ⇔ DSR>0.5 ⇔ SR_d>SR_0(N).  N büyüyünce Var↓→SR_0↓ → ikili-aramayla N* bulunur.")
    print(f"  • Event-edge:  SE=σ_event/√N_event, t=mean/SE, N(t≥2)=(2σ/mean)².")
    print(f"  • VARSAYIMLAR: getiriler/eventler IID; null-trial sayısı K={K_TRIALS} bağımsız; momentler (γ3,γ4) "
          f"gözlenen-örnekten; edge-yönü doğru (SR_d>0). IID-ihlali (otokorelasyon/kümeleme) ⇒ gereken-N DAHA büyük.\n")

    rows = [panel_a("SPY"), panel_a("QQQ")]
    a_out = section_a(rows)
    b_out = section_b(rows)
    panels = {r["sym"]: r["x"] for r in rows}      # placeholder
    # (c) panelleri build_panel'den (wall sütunları lazım)
    panels = {"SPY": build_panel("SPY"), "QQQ": build_panel("QQQ")}
    c_out = section_c(panels)
    section_c_counts(panels)
    translation(a_out, b_out, c_out, panels)
    print("\n  SINIR: tüm gereken-N IID + K=10 + tek-rejim-momentleri varsayımına dayalı; fat-tail/otokorelasyon "
          "ve forward rejim-kayması gereken-N'i YUKARI iter (alt-sınır niteliğinde). TEŞHİS-ONLY: P&L üretilmedi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
