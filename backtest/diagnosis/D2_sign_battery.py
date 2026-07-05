"""
backtest/diagnosis/D2_sign_battery — REJİM BAYRAĞI SAĞLAMLIK BATARYASI (TEŞHİS-ONLY).

Soru: net_gex-İŞARET'i (rejim bayrağı: +γ / −γ) ölçüm-seçimlerine ne kadar kırılgan?
Sadece SIGN serileri üretilir; P&L ÜRETİLMEZ (mevcut gamma_inv_pnl yalnız OKUNUR/recompute).

BASELINE = build_level_series._levels_for_day mantığının net_gex-işareti (front-expiry, ±15% band,
mid'den BS-IV, gamma_engine._greeks byte-eş). Buradan 5 VARYANT (yalnız İŞARET değişir, P&L'e bağlanmaz):
  V1 = penny-mid (mid<0.05) atılmış  — md'de bid YOK → bid=0/crossed proxy'si penny-mid. [NOT: gerçek bid/ask yok]
  V2 = IV winsorize [%5, %150] (0.05–1.50), aralık-dışı IV'ler clamp'lenir (None yerine).
  V3 = flat-ATM-IV — tüm strike'lara günün ATM-IV'ü (smile düzleştir).
  V4 = pure-OI-balance — gamma'sız Σ(±OI) işareti (call:+OI, put:−OI), band-içi.
  V5 = DTE≤2 hariç (0/1/2 DTE günleri DROP; o günler işaretsiz=NaN sayılır).

ÇIKTILAR:
  §1 variant başına baseline'a göre sign-flip-gün % (ortak günlerde).
  §2 FRAGILE-FLAG = ≥2 variant baseline ile çelişen günler (tarih listesi + sayı).
  §3 gamma_inv_pnl RECOMPUTE (block_robust.gamma_inv_pnl import; YENİ-varyant DEĞİL):
       holdout son-70g top-3 kazanç-günü FRAGILE listesinde mi? Toplam P&L'in %kaçı fragile-günlerden?
  §4 SqueezeMetrics agreement: net_gex-sign vs squeeze-gex-sign — GENEL + |net_gex|-tercile (alt/orta/üst).
       Bayrak sıfır-yakınında mı (alt-tercile) çuvallıyor (deadband-fixable) yoksa her-tercile'de mi (kök-rebuild)?

KURALLAR: teşhis-only. Mevcut P&L/level OKUNUR. V1-V5 = ölçüm-sağlamlık sign serileri, P&L'e BAĞLANMAZ.
  & <kader-macro venv python> backtest/diagnosis/D2_sign_battery.py
"""
from __future__ import annotations

import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "screen"))
sys.path.insert(0, str(ROOT / "backtest"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from _bsiv import implied_vol            # noqa: E402  (tek kanonik IV kaynağı)
from gamma_engine import _greeks         # noqa: E402  (byte-eş greeks)

M = 100
BAND = 0.15
PENNY = 0.05                              # V1: penny-mid eşiği (bid=0/crossed proxy)
IV_LO, IV_HI = 0.05, 1.50                # V2: winsorize aralığı


# ---------------------------------------------------------------- spot (build_level_series ile aynı)
def _daily_spot(sym):
    bars = pd.read_parquet(ROOT / "data" / "historical_bars" / f"alpaca_{sym.lower()}_1m.parquet")
    ts = pd.to_datetime(bars.index.get_level_values(-1))
    s = pd.Series(bars["close"].values, index=ts).resample("1D").last().dropna()
    s.index = s.index.date
    return s


def _prep_rows(g, S, *, drop_penny=False, winsor_iv=False, flat_iv=False, drop_short_dte=False):
    """Bir tarihin (tek expiry) zincirinden band-içi {K,oi,right,iv} satırları. İşaret-varyant bayrakları:
       drop_penny → mid<PENNY atılır; winsor_iv → IV [LO,HI] clamp; flat_iv → ATM-IV tüm strike'a;
       drop_short_dte → DTE≤2 ise satır-yok (gün işaretsiz). gamma _greeks ile baseline byte-eş hesaplanır."""
    dte = (pd.Timestamp(g["expiration"].iloc[0]) - pd.Timestamp(g["date"].iloc[0])).days
    if drop_short_dte and dte <= 2:
        return None, dte
    T = max(dte, 0.5) / 365.0
    rows = []
    for _, r in g.iterrows():
        K, oi, mid, right = r["strike"], r["open_interest"], r["mid"], r["right"]
        if K is None or oi is None or oi != oi or mid is None or mid <= 0:
            continue
        if drop_penny and float(mid) < PENNY:        # V1: bid=0/crossed proxy
            continue
        K = float(K)
        if abs(K / S - 1) > BAND:
            continue
        iv = implied_vol(float(mid), S, K, T, right)
        if winsor_iv:                                 # V2: None yerine aralığa clamp
            iv_raw = iv if iv is not None else _coarse_iv(float(mid), S, K, T, right)
            if iv_raw is None:
                continue
            iv = min(max(iv_raw, IV_LO), IV_HI)
        if not iv or iv <= 0:
            continue
        rows.append({"K": K, "oi": float(oi), "right": right, "iv": iv, "mid": float(mid)})
    if len(rows) < 4:
        return None, dte
    if flat_iv:                                       # V3: ATM-IV tüm strike'a
        atm = min(rows, key=lambda x: abs(x["K"] - S))
        for x in rows:
            x["iv"] = atm["iv"]
    return rows, dte


def _coarse_iv(price, S, K, T, right):
    """V2-yardımcısı: _bsiv None döndürdüğünde (IV aralık-dışı) clamp için ham bisection değeri.
       Sadece winsorize-clamp amaçlı; baseline/diğer varyantlar bunu KULLANMAZ."""
    if price is None or price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    from _bsiv import bs_price
    lo, hi = 0.001, 5.0
    for _ in range(60):
        m = (lo + hi) / 2
        if bs_price(S, K, T, m, right) > price:
            hi = m
        else:
            lo = m
    return (lo + hi) / 2


def _net_gex_sign(rows, S, T):
    """Σ ±gamma·OI·notional → işaret (+1 / −1). build_level_series ile aynı formül (call+, put−)."""
    sgn = lambda rt: 1.0 if rt == "C" else -1.0
    net = 0.0
    for x in rows:
        gg, *_ = _greeks(S, x["K"], T, x["iv"], x["right"])
        net += sgn(x["right"]) * gg * x["oi"] * M * S * S * 0.01
    return net


def _oi_balance_sign(rows):
    """V4: gamma'sız saf Σ(±OI) — call:+OI, put:−OI."""
    return sum((x["oi"] if x["right"] == "C" else -x["oi"]) for x in rows)


# ---------------------------------------------------------------- baseline + varyant sign serileri
def build_signs(sym):
    """Tarih→{net_gex(baseline), sign(baseline ±1), V1..V5 sign} DataFrame. abs_gex baseline'dan (tercile için)."""
    ch = pd.read_parquet(ROOT / "data" / "historical_chains" / f"md_{sym.lower()}.parquet")
    spot = _daily_spot(sym)
    recs = []
    for d, g in ch.groupby("date"):
        dd = pd.Timestamp(d).date()
        if dd not in spot.index:
            continue
        S = float(spot[dd])
        rec = {"date": pd.Timestamp(d)}

        # baseline (build_level_series._levels_for_day net_gex)
        rows, dte = _prep_rows(g, S)
        if rows is None:
            continue
        T = max(dte, 0.5) / 365.0
        ng = _net_gex_sign(rows, S, T)
        rec["net_gex"] = ng
        rec["abs_gex"] = abs(ng)
        rec["base"] = 1 if ng >= 0 else -1
        rec["dte"] = int(dte)

        # V1 penny-mid drop
        r1, _ = _prep_rows(g, S, drop_penny=True)
        rec["V1"] = (1 if _net_gex_sign(r1, S, T) >= 0 else -1) if r1 else np.nan
        # V2 winsorize IV
        r2, _ = _prep_rows(g, S, winsor_iv=True)
        rec["V2"] = (1 if _net_gex_sign(r2, S, T) >= 0 else -1) if r2 else np.nan
        # V3 flat ATM-IV
        r3, _ = _prep_rows(g, S, flat_iv=True)
        rec["V3"] = (1 if _net_gex_sign(r3, S, T) >= 0 else -1) if r3 else np.nan
        # V4 pure OI balance (baseline band rows, gamma'sız)
        rec["V4"] = 1 if _oi_balance_sign(rows) >= 0 else -1
        # V5 DTE<=2 hariç
        r5, dte5 = _prep_rows(g, S, drop_short_dte=True)
        rec["V5"] = (1 if _net_gex_sign(r5, S, T) >= 0 else -1) if r5 else np.nan

        recs.append(rec)
    df = pd.DataFrame(recs).set_index("date").sort_index()
    return df


# ---------------------------------------------------------------- raporlama
def section_1_2(sym, df):
    variants = ["V1", "V2", "V3", "V4", "V5"]
    print("=" * 100)
    print(f"  {sym} — §1 SIGN-FLIP %  (baseline n={len(df)} gün, |net_gex|>0 her gün)")
    print("=" * 100)
    flip_cols = {}
    for v in variants:
        ok = df[v].notna()
        n = int(ok.sum())
        flips = (df.loc[ok, v] != df.loc[ok, "base"])
        flip_cols[v] = flips.reindex(df.index, fill_value=False)
        print(f"    {v}: {int(flips.sum())}/{n} gün baseline'a göre işaret-flip  (%{100*flips.mean():.1f})  "
              f"[ortak-gün {n}]")

    # §2 FRAGILE: >=2 variant baseline ile çelişen günler
    flipmat = pd.DataFrame(flip_cols)
    disagree_cnt = flipmat.sum(axis=1)          # kaç variant baseline ile çelişiyor
    fragile = df.index[disagree_cnt >= 2]
    print()
    print(f"  §2 FRAGILE-FLAG (≥2 variant çelişiyor): {len(fragile)} gün / {len(df)} (%{100*len(fragile)/len(df):.1f})")
    if len(fragile):
        dates_str = ", ".join(d.strftime("%Y-%m-%d") for d in fragile)
        print(f"    tarihler: {dates_str}")
    # dağılım: kaç variant-çelişki kaç günde
    dist = disagree_cnt.value_counts().sort_index()
    print("    çelişen-variant-sayısı dağılımı: " +
          "  ".join(f"{int(k)}var→{int(v)}g" for k, v in dist.items()))
    return set(fragile), disagree_cnt


def section_3(sym, fragile_set, disagree_cnt):
    """Mevcut gamma_inv_pnl RECOMPUTE → holdout top-3 fragile mi + fragile-gün P&L payı."""
    from block_robust import gamma_inv_pnl
    pnl, _ = gamma_inv_pnl(sym)            # OKUNUR; YENİ varyant DEĞİL
    pnl.index = pd.to_datetime(pnl.index)
    print()
    print(f"  §3 gamma_inv_pnl RECOMPUTE (block_robust, YENİ-varyant DEĞİL) — n={len(pnl)}, "
          f"toplam {1e4*pnl.sum():+.0f}bps, full-Sharpe {pnl.mean()/pnl.std()*sqrt(252):+.2f}")

    ho = pnl.iloc[-70:]                    # holdout son-70g
    top3 = ho.nlargest(3)
    print(f"    holdout(son70) toplam {1e4*ho.sum():+.0f}bps; top-3 kazanç-günü:")
    n_frag_top3 = 0
    for d, val in top3.items():
        d0 = pd.Timestamp(d).normalize()
        is_frag = d0 in fragile_set
        n_frag_top3 += int(is_frag)
        nd = int(disagree_cnt.get(d0, 0)) if d0 in disagree_cnt.index else 0
        print(f"      {d0.strftime('%Y-%m-%d')}  {1e4*val:+.0f}bps  "
              f"{'FRAGILE' if is_frag else 'stabil '} (çelişen-variant={nd})")
    print(f"    → holdout top-3'ün {n_frag_top3}/3'ü FRAGILE listesinde.")

    # fragile-gün toplam P&L payı (tüm-örneklem)
    frag_list = list(fragile_set)
    frag_mask = np.asarray(pnl.index.normalize().isin(frag_list))
    tot = pnl.sum()
    frag_pnl = pnl.values[frag_mask].sum()
    n_frag = int(frag_mask.sum())
    print(f"    fragile-gün P&L payı (tüm-örneklem): {n_frag} fragile-gün toplam {1e4*frag_pnl:+.0f}bps "
          f"= toplam P&L'in %{100*frag_pnl/tot if tot else 0:.0f}  (toplam {1e4*tot:+.0f}bps)")
    # holdout-içi fragile payı
    ho_frag_mask = np.asarray(ho.index.normalize().isin(frag_list))
    ho_frag_pnl = ho.values[ho_frag_mask].sum()
    print(f"    fragile-gün P&L payı (holdout son70): {int(ho_frag_mask.sum())} fragile-gün "
          f"{1e4*ho_frag_pnl:+.0f}bps = holdout'un %{100*ho_frag_pnl/ho.sum() if ho.sum() else 0:.0f}")
    return n_frag_top3


def section_4(sym, df):
    """SqueezeMetrics agreement: net_gex-sign vs squeeze-gex-sign, genel + |net_gex|-tercile."""
    p = ROOT / "data" / "cache" / "squeeze_dix_gex.parquet"
    if not p.exists():
        print("\n  §4 SqueezeMetrics cache YOK → squeeze-agreement ÖLÇÜLEMEDİ.")
        return
    sq = pd.read_parquet(p)
    sq.index = pd.to_datetime(sq.index)
    sq_sign = pd.Series(np.where(sq["gex"] >= 0, 1, -1), index=sq.index.normalize())
    j = df.copy()
    j.index = j.index.normalize()
    j["sq"] = sq_sign.reindex(j.index)
    j = j.dropna(subset=["sq"])
    j["sq"] = j["sq"].astype(int)
    n = len(j)
    if n == 0:
        print("\n  §4 squeeze ile ortak gün YOK (tarih hizalanmadı) → ÖLÇÜLEMEDİ.")
        return
    agree = (j["base"] == j["sq"]).mean()
    sq_neg = (j["sq"] < 0).mean()
    print()
    print(f"  §4 SqueezeMetrics agreement (ortak n={n}) — squeeze ~%{100*sq_neg:.0f} gün NEGATİF "
          f"(SqueezeMetrics çoğu-zaman +GEX konvansiyonu)")
    print(f"    GENEL agreement (net_gex-sign == squeeze-sign): %{100*agree:.0f}")
    # naive-bias kontrolü: net_gex hep + olsaydı agreement ne olurdu (squeeze'in +oranı)
    base_neg = (j["base"] < 0).mean()
    print(f"    [bağlam] bizim net_gex %{100*base_neg:.0f} NEGATİF; squeeze %{100*sq_neg:.0f} negatif → "
          f"naive 'hep-+' agreement tavanı %{100*(j['sq']>0).mean():.0f}")

    # |net_gex|-tercile
    q = j["abs_gex"].quantile([1/3, 2/3]).values
    def terc(x):
        return "alt" if x <= q[0] else ("orta" if x <= q[1] else "üst")
    j["terc"] = j["abs_gex"].apply(terc)
    print(f"    |net_gex|-tercile agreement (alt=sıfır-yakını → deadband-test):")
    for t in ("alt", "orta", "üst"):
        sub = j[j["terc"] == t]
        if len(sub) == 0:
            continue
        ag = (sub["base"] == sub["sq"]).mean()
        negshare = (sub["base"] < 0).mean()
        print(f"      {t:>4}-tercile (n{len(sub):>3}, |gex| {sub['abs_gex'].min():.1e}–{sub['abs_gex'].max():.1e}): "
              f"agreement %{100*ag:.0f}  (bizim-neg %{100*negshare:.0f})")
    alt_ag = (j[j["terc"] == "alt"]["base"] == j[j["terc"] == "alt"]["sq"]).mean()
    ust_ag = (j[j["terc"] == "üst"]["base"] == j[j["terc"] == "üst"]["sq"]).mean()
    print(f"    → TEŞHİS: alt-tercile %{100*alt_ag:.0f} vs üst-tercile %{100*ust_ag:.0f}.  "
          + ("Üst-tercile'de DE düşükse=KÖK-REBUILD; yalnız alt'ta düşükse=DEADBAND-FIXABLE."))


def run(sym):
    df = build_signs(sym)
    frag_set, disagree = section_1_2(sym, df)
    section_3(sym, frag_set, disagree)
    section_4(sym, df)
    print()
    return df


def main():
    for sym in ("SPY", "QQQ"):
        run(sym)
        print()
    print("  NOT: V1-V5 = ölçüm-sağlamlık SIGN serileri (P&L'e BAĞLANMAZ). gamma_inv_pnl yalnız RECOMPUTE/OKUNUR.")
    print("  md'de gerçek bid YOK → V1 penny-mid<0.05 = bid=0/crossed PROXY'si (gerçek bid/ask veri lazım için ayrı kaynak).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
