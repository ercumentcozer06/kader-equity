"""
backtest/gex_swing/gxs_config.py — SWING-GEX ÇALIŞMASI: TEK-GERÇEK-KAYNAK (pre-committed tanımlar).

EMİR'İN ÇERÇEVESİ (2026-06-12): GEX = SWING REJİM KADRANI. "flip üstü → tam oyna / flip altı → kaldıraç-kıs."
Intraday forced-buying cascade (call-wall→VU) = MANUEL (model değil). Model yalnız boyut-rejimini çevirir.
Üç test: T1 flip-gate/asimetrik-shield, T2 GEX-öncü-kırılganlık (stres-kapısı), T3 vanna/charm OPEX × rejim.

VERİ (read-only):
  squeeze : data/cache/squeeze_dix_gex.parquet — cols [price(=SPX), dix, gex]; 2011-05-02..2026-06-08, 3798g.
            gex = SqueezeMetrics dealer dollar-gamma; gex>0 = net LONG gamma (≈flip ÜSTÜ), gex<0 = net SHORT
            gamma (≈flip ALTI). price = SPX endeks seviyesi (standalone çok-rejim getiri zemini).
  frozen  : spine/frozen — SPX+NDX fiyat + module_scores(m0-m11) + vector; 2019-01..2026-05-22 (m9-çağı, tide var).
  tide    : spine.tide.tide_dir_series(tide_score_series(scores, vector))  (finalize_stack ile birebir).

PIT: sinyal[D] (EOD GEX/level) → getiri[D+1]; lag=1 (mevcut harness strat_ret ile birebir). Look-ahead yok.

REJİM TANIMLARI (kilitli):
  regime_sign = sign(gex)                                # +1 = flip-üstü (long gamma) / −1 = flip-altı (short)
  regime_z    = z(gex, 252g, min60)                      # modules/gex_shield.gex_zscore ile birebir
  "flip-altı" = gex < 0 (Emir'in kuralı); "derin-negatif" = z ≤ −1 (mevcut shield eşiği)

TRIM FORMLARI (T1; HEPSİ trim-only, ASLA short/add — model rebound-safe; Emir manuelde short alır, MODEL almaz):
  V0_zshield   : mevcut shield = (1 − 0.5·clip(−z−1,0,3)).clip(0.4,1)  [BASELINE, finalize_stack ile aynı]
  V1_flipbin   : gex<0 → floor ; gex≥0 → 1.0           [Emir'in ikili kuralı; floor ∈ {0.4,0.5,0.6} a-priori]
  V2_asym      : gex<0 & tide<0 → floor_lo (kaskad-tehlike, sert) ; gex<0 & tide>0 → floor_hi (squeeze-yardım,
                 hafif) ; gex≥0 → 1.0    [floor_lo=0.4 / floor_hi=0.7 a-priori; tide yönüne KOŞULLU]
  Not: A-priori eşikler; grid {floor} dışında tarama YOK. DSR muhasebesi: T1 ek-form sayısı raporun başında.

OPEX (T3): aylık 3.-Cuma (np.busday). pre-window = [OPEX−5 .. OPEX−1] iş-günü; opex-günü = OPEX; post-window =
  [OPEX+1 .. OPEX+5]. Vanna/charm tezi: pre-OPEX drift POZİTİF (vanna rüzgârı, dealer-long-gamma destek),
  post-OPEX "zayıflık penceresi" NEGATİF (charm unwind). REJİME KOŞULLU: pre-drift pozitif-gammada güçlü mü;
  post-zayıflık negatif-gammada keskin mi. (Mevsimsellik daha önce tide-EMİLDİ — yeni açı = rejim-koşulu.)

ALT-DÖNEMLER (stres, çok-rejim gücü): full 2011-26 + {2015-08-China, 2018Q4, 2020-COVID(2020-02..05),
  2022-bear(tüm-yıl), 2023-SVB(2023-03), 2025-tarife(2025-03..05)}. Her test sub-period kırılımı raporlar.

İSTATİSTİK: Sharpe(ann), maxDD, CVaR5, expo, turnover; oranlar Wilson %95; EV'ler t-stat; bootstrap-CI +
  paired-win-prob (screen._util) ablation'da; DSR (finalize_stack.dsr) ablation'da. n<full-coverage uyarısı.
ABLATION KURALI: model-WITH vs model-WITHOUT (Emir'in ablation disiplini). Standalone = sinyalin ham edge'i
  (2011-26 çok-rejim); ablation = OUR-model'e katkı (2019+ tide-limitli, finalize_stack harness'ı birebir).

ÇIKTI: backtest/gex_swing/results/T{1,2,3}.json + REPORT.md. Ana stack (tide×froth×shield) FROZEN — dokunma.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VENV_PY = "C:/Users/admin/Downloads/kader-macro/.venv/Scripts/python.exe"
RESULTS_DIR = ROOT / "backtest" / "gex_swing" / "results"

PIT_LAG = 1
Z_WIN, Z_MINP = 252, 60
DEEP_NEG_Z = -1.0                 # mevcut shield "derin-negatif" eşiği
FLIP_FLOORS = (0.4, 0.5, 0.6)     # V1 a-priori floor gridi
ASYM_FLOOR_LO, ASYM_FLOOR_HI = 0.4, 0.7   # V2 a-priori (tide<0 sert / tide>0 hafif)
OPEX_PRE, OPEX_POST = 5, 5        # iş-günü pencere yarıçapı

SUBPERIODS = {
    "full_2011_26": ("2011-05-02", "2026-06-08"),
    "2015_china":   ("2015-08-01", "2015-10-15"),
    "2018Q4":       ("2018-10-01", "2018-12-31"),
    "2020_covid":   ("2020-02-15", "2020-05-31"),
    "2022_bear":    ("2022-01-01", "2022-12-31"),
    "2023_svb":     ("2023-03-01", "2023-04-15"),
    "2025_tariff":  ("2025-03-01", "2025-05-31"),
}


def load_squeeze() -> pd.DataFrame:
    """SPX price + dix + gex, 2011+. Index = tarih (DatetimeIndex)."""
    g = pd.read_parquet(ROOT / "data" / "cache" / "squeeze_dix_gex.parquet")
    g.index = pd.to_datetime(g.index)
    return g.sort_index()


def gex_zscore(gex: pd.Series, win: int = Z_WIN, minp: int = Z_MINP) -> pd.Series:
    """modules/gex_shield.gex_zscore ile birebir (tek kaynak doğrulaması için kopyalandı)."""
    gex = gex.dropna()
    m = gex.rolling(win, min_periods=minp).mean()
    s = gex.rolling(win, min_periods=minp).std()
    return (gex - m) / s


def metrics(r: pd.Series) -> dict:
    """Günlük strateji getirisi → Sharpe/maxDD/CVaR5/expo/turnover-yok. finalize_stack._m ile aynı çekirdek."""
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return {"sharpe": 0.0, "maxdd": 0.0, "cvar5": 0.0, "expo": 0.0, "n": len(r)}
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    sh = float(r.mean() / r.std() * np.sqrt(252))
    k = max(1, int(0.05 * len(r)))
    cv = float(np.sort(r.values)[:k].mean())
    return {"sharpe": round(sh, 3), "maxdd": round(dd, 4), "cvar5": round(cv, 5),
            "expo": round(float((r != 0).mean()), 3), "n": len(r),
            "cum_ret": round(float(eq.iloc[-1] - 1), 4)}


def opex_dates(start, end) -> pd.DatetimeIndex:
    """Aylık 3.-Cuma OPEX tarihleri [start,end]."""
    days = pd.date_range(start, end, freq="D")
    fri = days[days.weekday == 4]
    out = []
    for (y, m), grp in pd.Series(fri).groupby([fri.year, fri.month]):
        if len(grp) >= 3:
            out.append(grp.iloc[2])
    return pd.DatetimeIndex(out)


def opex_offset_map(index: pd.DatetimeIndex, start, end) -> pd.Series:
    """Her iş-günü için en yakın OPEX'e iş-günü ofseti (negatif=önce, 0=opex, pozitif=sonra)."""
    ox = opex_dates(start, end)
    idx = pd.DatetimeIndex(index)
    pos = idx.get_indexer(ox, method="nearest")  # opex'in index'teki konumu
    off = pd.Series(np.nan, index=idx)
    arr = np.arange(len(idx))
    for p in pos:
        lo, hi = max(0, p - 10), min(len(idx), p + 11)
        for j in range(lo, hi):
            o = j - p
            if pd.isna(off.iloc[j]) or abs(o) < abs(off.iloc[j]):
                off.iloc[j] = o
    return off


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    g = load_squeeze()
    z = gex_zscore(g["gex"])
    print(f"squeeze {g.index.min().date()}..{g.index.max().date()} n={len(g)}")
    print(f"  gex<0 (flip-altı) gün payı: %{100*(g['gex']<0).mean():.0f} | z≤−1 derin-neg: %{100*(z<=-1).mean():.0f}")
    ox = opex_dates(g.index.min(), g.index.max())
    print(f"  OPEX tarih sayısı: {len(ox)} (ilk {ox[0].date()} son {ox[-1].date()})")
    off = opex_offset_map(g.index, g.index.min(), g.index.max())
    print(f"  ofset kapsama: %{100*off.notna().mean():.0f}; pre[-5..-1] gün %{100*off.between(-5,-1).mean():.0f}"
          f" post[1..5] %{100*off.between(1,5).mean():.0f}")
