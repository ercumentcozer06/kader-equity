"""
backtest/diagnosis/D3_iv_quality.py — IV/GREEKS KALİTE TEŞHİSİ (DIAGNOSIS-ONLY).

NE YAPAR (yeni strateji/parametre/reweight YOK; sadece mevcut borunun girdi-kalitesini ÖLÇER):
  Girdi = data/historical_chains/md_{spy,qqq}.parquet (MarketData EOD OI; iv/delta kolonları BOŞ; tek-mid, bid YOK).
  IV = screen/_bsiv.implied_vol(mid, S, K, T, right)  (build_level_series ile BYTE-AYNI çağrı; _greeks de byte-aynı).
  Spot = build_level_series._daily_spot (Alpaca 1-dk close → günlük resample), birebir aynı kaynak.

  GÜN BAŞINA ölçülenler:
    (a) bisection-fail %  = implied_vol(...) None/≤0 dönen satır oranı (band-içi satırlarda).
    (b) penny/instabil-mid %  = mid≤0 VEYA mid<0.05 satır oranı. NOT: 'md'de bid-yok → MID tek-fiyat,
        bid/ask spread'i ÖLÇÜLEMEZ; mid≤0 hiç yok (MarketData mid'i 0.005'e tabanlıyor) → proxy = mid<0.05.
    (c) DTE≤2 günlerin toplam gamma$ payı = Σ_{DTE≤2} gamma·OI·100·S²·0.01  /  Σ_tüm gamma·OI·100·S²·0.01.
        (gamma = _greeks(S,K,T,iv,right)[0]; sadece IV-geçerli + band-içi satırlar.)

  net_gex SAWTOOTH (testere-dişi) testi:
    net_gex = frozen level_series_{sym}.parquet (SADECE OKUNUR — yeniden üretilmez).
    Günlük |Δnet_gex| serisini, o GÜNÜN expiry-DTE'sine göre iki kovaya ayır:
      expiry-haftası = DTE≤5   |   diğer = DTE>5
    Her kova için |Δ|'nın VARYANSI + ORAN (expiry/other). Oran»1 → roll-günlerinde sıçrama = sawtooth kanıtı.

ÇIKTI: numaralı markdown + ham sayılar (stdout). Sayı = bu script üretir; uydurma yok.
KOŞ:  & C:/Users/admin/Downloads/kader-macro/.venv/Scripts/python.exe backtest/diagnosis/D3_iv_quality.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "screen"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from _bsiv import implied_vol          # noqa: E402  (build_level_series ile byte-aynı IV-invert)
from gamma_engine import _greeks       # noqa: E402  (byte-aynı greeks; gamma = [0])

BAND = 0.15           # build_level_series ile aynı strike bandı (|K/S-1|>BAND elenir)
PENNY = 0.05          # penny/instabil-mid eşiği (mid<0.05); mid≤0 ayrıca raporlanır
M = 100               # contract multiplier


def _daily_spot(sym: str) -> pd.Series:
    """build_level_series._daily_spot ile BİREBİR aynı: Alpaca 1-dk close → günlük son."""
    bars = pd.read_parquet(ROOT / "data" / "historical_bars" / f"alpaca_{sym.lower()}_1m.parquet")
    ts = pd.to_datetime(bars.index.get_level_values(-1))
    s = pd.Series(bars["close"].values, index=ts).resample("1D").last().dropna()
    s.index = s.index.date
    return s


def per_day_quality(sym: str) -> pd.DataFrame:
    ch = pd.read_parquet(ROOT / "data" / "historical_chains" / f"md_{sym.lower()}.parquet")
    spot = _daily_spot(sym)
    out = []
    for d, g in ch.groupby("date"):
        dd = pd.Timestamp(d).date()
        if dd not in spot.index:
            continue
        S = float(spot[dd])
        exp = pd.Timestamp(g["expiration"].iloc[0])
        dte_day = (exp - pd.Timestamp(g["date"].iloc[0])).days
        T = max(dte_day, 0.5) / 365.0

        n_total = len(g)                       # md ham satır (call+put)
        n_band = 0                             # band-içi (IV denemesine giren) satır
        n_fail = 0                             # implied_vol None/≤0
        n_penny = 0                            # mid<PENNY (band-içi)
        n_mid_le0 = 0                          # mid≤0 (band-içi)
        gamma_total = 0.0
        gamma_dte_le2 = 0.0

        for _, r in g.iterrows():
            K = r["strike"]
            mid = r["mid"]
            right = r["right"]
            if K is None or K != K:
                continue
            K = float(K)
            if abs(K / S - 1) > BAND:
                continue
            n_band += 1
            mid_le0 = (mid is None) or (mid != mid) or (mid <= 0)
            penny = mid_le0 or (mid < PENNY)
            if mid_le0:
                n_mid_le0 += 1
            if penny:
                n_penny += 1
            iv = implied_vol(float(mid) if not mid_le0 else None, S, K, T, right)
            if not iv or iv <= 0:
                n_fail += 1
                continue
            gg, *_ = _greeks(S, K, T, iv, right)
            oi = float(r["open_interest"]) if r["open_interest"] == r["open_interest"] else 0.0
            gdollar = gg * oi * M * S * S * 0.01
            gamma_total += gdollar
            # DTE≤2: tek-expiry veri → günün DTE'si ≤2 ise o günün TÜM gamma$'ı DTE≤2 kovasına girer
            if dte_day <= 2:
                gamma_dte_le2 += gdollar

        out.append(dict(
            date=pd.Timestamp(d), dte=int(dte_day), spot=S,
            n_total=n_total, n_band=n_band, n_fail=n_fail, n_penny=n_penny, n_mid_le0=n_mid_le0,
            fail_pct=(100.0 * n_fail / n_band) if n_band else np.nan,
            penny_pct=(100.0 * n_penny / n_band) if n_band else np.nan,
            mid_le0_pct=(100.0 * n_mid_le0 / n_band) if n_band else np.nan,
            gamma_total=gamma_total, gamma_dte_le2=gamma_dte_le2,
        ))
    return pd.DataFrame(out).set_index("date").sort_index()


def sawtooth(sym: str) -> dict:
    """frozen level_series net_gex (SADECE OKUNUR) → |Δnet_gex| varyansı: DTE≤5 vs DTE>5."""
    p = ROOT / "data" / "cache" / f"level_series_{sym.lower()}.parquet"
    ls = pd.read_parquet(p).sort_index()
    g = ls["net_gex"].astype(float)
    dgex = g.diff().abs()                       # gün-gün mutlak değişim
    dte = ls["dte"].astype(int)
    # |Δ| günü D'ye ait; D'nin expiry-DTE'siyle etiketle (roll/expiry haftası D'de mi?)
    df = pd.DataFrame({"dabs": dgex, "dte": dte}).dropna()
    exp_wk = df[df["dte"] <= 5]["dabs"]
    other = df[df["dte"] > 5]["dabs"]
    v_exp = float(exp_wk.var(ddof=1)) if len(exp_wk) > 1 else np.nan
    v_oth = float(other.var(ddof=1)) if len(other) > 1 else np.nan
    return dict(
        n_exp=int(len(exp_wk)), n_oth=int(len(other)),
        var_exp=v_exp, var_oth=v_oth,
        ratio=(v_exp / v_oth) if (v_oth and v_oth == v_oth and v_oth != 0) else np.nan,
        mean_exp=float(exp_wk.mean()) if len(exp_wk) else np.nan,
        mean_oth=float(other.mean()) if len(other) else np.nan,
    )


def report():
    print("=" * 78)
    print("D3 — IV/GREEKS KALİTE RAPORU  (DIAGNOSIS-ONLY; P&L üretilmez)")
    print("=" * 78)
    print("VERİ NOTU: md_{spy,qqq}.parquet'te iv/delta kolonları %100 BOŞ → IV daima mid'den BS-invert.")
    print("           Kolonlarda bid/ask YOK (tek 'mid') → bid-ask spread ÖLÇÜLEMEZ; mid≤0 hiç yok")
    print("           (MarketData mid'i 0.005'e tabanlıyor) → penny/instabil proxy = mid<0.05.")
    print("           Tarih başına TEK expiry (DTE 0-25, med 8) → 'DTE≤2 payı' günün ya 0 ya da tüm gamma$'ı.")
    print()

    for sym in ("spy", "qqq"):
        q = per_day_quality(sym)
        st = sawtooth(sym)
        # genel oranlar (satır-ağırlıklı = tüm band-içi satırlar üstünden)
        tot_band = int(q["n_band"].sum())
        tot_fail = int(q["n_fail"].sum())
        tot_penny = int(q["n_penny"].sum())
        tot_mid_le0 = int(q["n_mid_le0"].sum())
        # DTE≤2 gamma$ payı: HAVUZ (tüm günler toplamı) + günlük-koşullu
        g_all = float(q["gamma_total"].sum())
        g_le2 = float(q["gamma_dte_le2"].sum())
        n_days_le2 = int((q["dte"] <= 2).sum())

        print("#" * 78)
        print(f"## {sym.upper()}  ({len(q)} gün, {pd.Timestamp(q.index.min()).date()} → {pd.Timestamp(q.index.max()).date()})")
        print("#" * 78)
        print(f"### 1. bisection-fail (implied_vol None/≤0)")
        print(f"  satır-ağırlıklı: {tot_fail}/{tot_band} = %{100*tot_fail/tot_band:.2f}  (band-içi satırlar)")
        print(f"  gün-bazlı fail%: medyan %{q['fail_pct'].median():.2f}  ort %{q['fail_pct'].mean():.2f}  "
              f"min %{q['fail_pct'].min():.2f}  maks %{q['fail_pct'].max():.2f}")
        worst = q.nlargest(5, "fail_pct")[["dte", "fail_pct", "n_band"]]
        print("  en kötü 5 gün (fail%):")
        for dt_, row in worst.iterrows():
            print(f"    {pd.Timestamp(dt_).date()}  DTE {int(row['dte']):>2}  fail %{row['fail_pct']:.1f}  (n_band {int(row['n_band'])})")
        print()

        print(f"### 2. penny / instabil-mid")
        print(f"  mid<0.05 satır-ağırlıklı: {tot_penny}/{tot_band} = %{100*tot_penny/tot_band:.2f}")
        print(f"  mid≤0 satır-ağırlıklı   : {tot_mid_le0}/{tot_band} = %{100*tot_mid_le0/tot_band:.2f}  (md'de yok → 0 beklenir)")
        print(f"  gün-bazlı penny%: medyan %{q['penny_pct'].median():.2f}  ort %{q['penny_pct'].mean():.2f}  "
              f"maks %{q['penny_pct'].max():.2f}")
        # penny ve fail örtüşmesi: penny satırlar implied_vol'da None mı dönüyor?
        print(f"  NOT: penny (mid<0.05) çoğunlukla derin-OTM; bunların IV'si genelde aralık-dışı → fail'e katkı.")
        print()

        print(f"### 3. DTE≤2 toplam gamma$ payı")
        if g_all > 0:
            print(f"  HAVUZ (tüm 243-gün): Σ_DTE≤2 gamma$ / Σ_tüm gamma$ = "
                  f"{g_le2:.3e} / {g_all:.3e} = %{100*g_le2/g_all:.2f}")
        print(f"  DTE≤2 olan gün sayısı: {n_days_le2}/{len(q)} (%{100*n_days_le2/len(q):.1f})")
        # DTE≤2 günlerinde günlük gamma$ ortalaması vs DTE>2 (yoğunluk farkı)
        gm_le2 = q[q["dte"] <= 2]["gamma_total"]
        gm_gt2 = q[q["dte"] > 2]["gamma_total"]
        if len(gm_le2):
            print(f"  DTE≤2 günlerinde günlük gamma$ medyanı {gm_le2.median():.3e}  vs DTE>2 medyanı {gm_gt2.median():.3e}  "
                  f"(oran ×{gm_le2.median()/gm_gt2.median():.1f})" if len(gm_gt2) and gm_gt2.median() else "")
        print()

        print(f"### 4. net_gex SAWTOOTH — |Δnet_gex| varyansı (frozen level_series, SALT-OKUNUR)")
        print(f"  expiry-haftası (DTE≤5): n={st['n_exp']}  Var(|Δ|)={st['var_exp']:.4e}  ort|Δ|={st['mean_exp']:.4e}")
        print(f"  diğer-haftalar (DTE>5): n={st['n_oth']}  Var(|Δ|)={st['var_oth']:.4e}  ort|Δ|={st['mean_oth']:.4e}")
        print(f"  ORAN Var_exp / Var_oth = ×{st['ratio']:.2f}   "
              f"→ {'SAWTOOTH KANITI (roll-günü sıçraması)' if st['ratio'] and st['ratio'] > 1.5 else 'sawtooth zayıf/yok'}")
        print()

    print("=" * 78)
    print("ÖZET HİPOTEZ-BAĞI:")
    print("  H⑤ IV-from-mid instabilite: fail% + penny% birlikte → mid-tek-fiyat borusunun gürültü tabanı.")
    print("  H⑥ OPEX-sawtooth confound  : §4 oran»1 ise net_gex roll-günlerinde sıçrıyor (rejim ≠ saf bilgi).")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(report())
