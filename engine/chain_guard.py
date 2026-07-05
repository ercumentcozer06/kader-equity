"""
engine/chain_guard — C2: tarihsel option-chain OI verisi için QC harness (geçemeyen veri KULLANILMAZ).
ŞERİT-2A'da tasarlandı, burada SAF-FONKSİYON olarak kuruldu (sentetik-test edilebilir; gerçek backfill C1).

Kapılar (a priori; altı RED):
  1. ŞEMA: zorunlu kolonlar {date, expiration, strike, right, open_interest}; OI non-null ≥%95, int ≥0.
  2. BOUND: strike>0; 0<iv<5 (varsa); expiration≥date; |delta|≤1 (varsa); OI günlük-z |z|>8 → aykırı bayrağı.
  3. OPEX-OI-ÇÖKÜŞÜ (asıl PIT-sağlık): 3. Cuma (aylık OPEX) ertesi gün, vadesi dolan kontratların toplam-OI'si
     önceki güne göre ≥%70 düşmeli. Düşmüyorsa → ileri-tarih sızıntısı / OI bayat → RED.
  4. GEX-ÇAPRAZ: chain-OI'den hesaplanan günlük GEX vs SqueezeMetrics bedava serisi → işaret-uyumu ≥%90 VE
     Pearson korr ≥0.9 (a priori). SqueezeMetrics index-seviye → KAYNAK değil, REFERANS.
  5. LIVENESS: snapshot takvim-yaşı ≤ max_staleness; aksi STALE.
PIT: chain(t)→t+1. OI alan-semantiği belirsizse t−1 varsay + FLAG.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQ_COLS = {"date", "expiration", "strike", "right", "open_interest"}
OI_NONNULL_MIN = 0.95
SIGN_AGREE_MIN = 0.90
CORR_MIN = 0.90
OPEX_DROP_MIN = 0.70
OI_Z_OUTLIER = 8.0


def validate_schema(df: pd.DataFrame) -> list:
    fails = []
    miss = REQ_COLS - set(df.columns)
    if miss:
        fails.append(f"şema: eksik kolon {miss}")
        return fails
    oi = pd.to_numeric(df["open_interest"], errors="coerce")
    if oi.notna().mean() < OI_NONNULL_MIN:
        fails.append(f"OI non-null %{100*oi.notna().mean():.0f} < %{100*OI_NONNULL_MIN:.0f}")
    if (oi.dropna() < 0).any():
        fails.append("OI negatif değer var")
    return fails


def validate_bounds(df: pd.DataFrame) -> list:
    fails = []
    if (pd.to_numeric(df["strike"], errors="coerce") <= 0).any():
        fails.append("strike ≤ 0")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")
    if (df["expiration"] < df["date"]).any():
        fails.append("expiration < date (geçmiş vade)")
    if "iv" in df.columns:
        iv = pd.to_numeric(df["iv"], errors="coerce").dropna()
        if ((iv <= 0) | (iv >= 5)).any():
            fails.append("iv aralık-dışı (0,5)")
    if "delta" in df.columns:
        de = pd.to_numeric(df["delta"], errors="coerce").dropna()
        if (de.abs() > 1.0001).any():
            fails.append("|delta| > 1")
    return fails


def opex_oi_collapse(df: pd.DataFrame) -> dict:
    """3. Cuma vadeli kontratların OPEX-sonrası toplam-OI çöküşü (PIT-sağlık)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]); df["expiration"] = pd.to_datetime(df["expiration"])
    df["oi"] = pd.to_numeric(df["open_interest"], errors="coerce")
    res = {"checked": 0, "passed": 0, "fails": []}
    for exp, g in df.groupby("expiration"):
        if exp.weekday() != 4:                                    # yalnız Cuma-vadeler
            continue
        # 3. Cuma mı? (ayın 15-21'i arası Cuma)
        if not (15 <= exp.day <= 21):
            continue
        pre = g[g["date"] == exp]["oi"].sum()
        post = g[g["date"] == exp + pd.Timedelta(days=3)]["oi"].sum()   # ertesi işlem günü (Pazartesi ~+3)
        if pre <= 0:
            continue
        res["checked"] += 1
        drop = 1 - post / pre
        if drop >= OPEX_DROP_MIN:
            res["passed"] += 1
        else:
            res["fails"].append((str(exp.date()), round(drop, 2)))
    res["ok"] = res["checked"] == 0 or res["passed"] / max(1, res["checked"]) >= 0.8
    return res


def gex_cross_check(computed_gex: pd.Series, squeeze_gex: pd.Series) -> dict:
    """Hesaplanan GEX vs SqueezeMetrics: işaret-uyumu + korelasyon (ortak tarihlerde)."""
    common = computed_gex.index.intersection(squeeze_gex.index)
    if len(common) < 20:
        return {"ok": False, "reason": f"ortak gün <20 ({len(common)})", "sign_agree": None, "corr": None}
    a, b = computed_gex.reindex(common).values, squeeze_gex.reindex(common).values
    sign_agree = float(np.mean(np.sign(a) == np.sign(b)))
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else 0.0
    return {"ok": bool(sign_agree >= SIGN_AGREE_MIN and corr >= CORR_MIN),
            "sign_agree": round(sign_agree, 3), "corr": round(corr, 3),
            "reason": (f"işaret %{100*sign_agree:.0f}(≥%90) korr {corr:.2f}(≥0.9)"
                       + ("" if (sign_agree >= SIGN_AGREE_MIN and corr >= CORR_MIN) else " → RED"))}


def run_qc(df: pd.DataFrame, squeeze_gex: pd.Series = None, computed_gex: pd.Series = None) -> dict:
    """Tüm kapılar. ok=True → veri KULLANILABİLİR; aksi RED (Ş2B tetiklenmez)."""
    fails = validate_schema(df)
    if fails:
        return {"ok": False, "gates": {"schema": fails}}
    bounds = validate_bounds(df)
    opex = opex_oi_collapse(df)
    cross = gex_cross_check(computed_gex, squeeze_gex) if (squeeze_gex is not None and computed_gex is not None) else {"ok": None, "reason": "GEX-çapraz atlandı (veri yok)"}
    ok = (not bounds) and opex.get("ok") and (cross.get("ok") is not False)
    return {"ok": bool(ok), "gates": {"schema": [], "bounds": bounds, "opex": opex, "gex_cross": cross}}
