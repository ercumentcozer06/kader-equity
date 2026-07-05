"""
backtest/remeasure/config.py — FAZ-R TAMAMLAMA TEK-GERÇEK-KAYNAK (single source of truth).
TÜM RC-ajanları bunu import eder; sabit hardcode YASAK. Bu dosyanın sha256'sı her çıktıya metadata yazılır.
PRE-REGISTRATION AMENDMENT bu dosyayla kilitlenir (battery unblinding ÖNCESİ; bkz. AMENDMENT bölümü).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # kader-equity kökü
VENV_PY = "C:/Users/admin/Downloads/kader-macro/.venv/Scripts/python.exe"

# ---- semboller / havuz ----
SYMS = ["SPY", "QQQ", "SPX", "NDX"]                  # ham-capture evreni
TRADE_SYMS = ["SPY", "QQQ"]                          # P&L panel sembolleri (SPX/NDX getirisi = SPY/QQQ delta-one)
INDEX_FLAG_MAP = {"SPY": "SPX", "QQQ": "NDX"}        # AMENDMENT: INDEX-FLAG kaynağı (FULL-SURFACE index bayrağı)

# ---- maliyet / PIT ----
COST_DAILY = 0.00015            # 1.5bps round-trip, işlem-günü başına düz (D-FAZ disentangle ile birebir)
PIT_LAG_DAYS = 1                # level[D] (D-EOD) → D+1 seansı open→close; gap = D+1-open/D-close − 1

# ---- panel penceresi / train-holdout (D-FAZ pozisyon-150 sınırının TARİH karşılığı; SPY=QQQ aynı çıktı) ----
PANEL_START = "2025-06-13"
PANEL_END = "2026-06-08"        # pre-registered pencere SONU (06-09/06-10 yalnız RC2.8 canlı-uyum için, panel-DIŞI)
TRAIN_END = "2026-01-30"        # train = D ≤ TRAIN_END   (eski panel idx 0-149)
HOLDOUT_START = "2026-02-02"    # holdout = D ≥ HOLDOUT_START (eski panel idx 150+)

# ---- level-build (D0 kod-gerçeği; gamma_engine birebir) ----
M_CONTRACT = 100
BAND = 0.15                     # ±%15 strike bandı
N_EXP_LIVE = 5                  # LIVE-MATCH = front 5 expiry (gamma_engine.py:38 N_EXP)
SCAN_LO, SCAN_HI, SCAN_N = -0.06, 0.06, 13   # flip zero-gamma taraması
R_RATE, Q_DIV = 0.04, 0.013

# ---- IV hijyeni (D2 ile BİREBİR; yeni ayar YOK) ----
HYG_V1_DROP_BID0_CROSSED = True   # V1: bid≤0 veya bid>ask → kontrat DROP (ham bid/ask artık mevcut)
HYG_V2_IV_LO = 0.05               # V2: IV winsorize alt  (D2 [%5,%150])
HYG_V2_IV_HI = 1.50               # V2: IV winsorize üst
HYG_V5_DTE_FLAG = 2               # V5: DTE≤2 günleri FLAG (drop DEĞİL; dte2_share kolonu)

# ---- battery (KİLİTLİ KÜME — ekleme/çıkarma YASAK) ----
# P&L üyeleri (günlük): pozisyon kuralları D-FAZ disentangle.py'den birebir; sg = sign(gap)
#   gamma_txt : reg>0 → −sg (fade) / reg<0 → +sg (follow)   [ders-kitabı]
#   gamma_inv : tersi
#   vol_only  : atm_iv > panel-medyanı → −sg / değilse +sg  [bayrak-bağımsız; atm_iv kaynağı: own LIVE-MATCH]
#   hep_mom   : +sg ; hep_rev: −sg                           [bayrak-bağımsız]
#   M3_setup  : event-günü: reg>0 & duvar-dokunuş → MR (fade) ; reg<0 & dokunuş → −MR (breakout)
#               MR = spine_diagnostic.mean_reversion_return tanımı (call: (cw−c1)/cw; put: (c1−pw)/pw; leg-ort.)
# ÖLÇÜM üyeleri (P&L değil; spine_diagnostic M1/M2 tanımları): M1 = corr(gap, intraday) rejim-başına;
#   M2 = duvar-MR ortalaması rejim-başına (t, n ile). [D-FAZ'da M1/M2 korelasyon/koşullu-ortalama idi; tradeable
#   karşılıkları gamma_txt/gamma_inv (M1) ve M3_setup (M2-M3) — burada AYNI eşleme korunur, yeni üye üretilmez.]
BATTERY_PNL = ["gamma_txt", "gamma_inv", "vol_only", "hep_mom", "hep_rev", "M3_setup"]
BATTERY_MEAS = ["M1_corr", "M2_wallMR"]
FLAG_DEPENDENT = ["gamma_txt", "gamma_inv", "M3_setup", "M1_corr", "M2_wallMR"]
FLAG_FREE = ["vol_only", "hep_mom", "hep_rev"]       # tek koşum (replacement), flag="none"
# bayrak setleri: own-LIVE-MATCH / own-FULL-SURFACE / INDEX-FLAG (amendment)
FLAG_SETS = ["livematch_own", "fullsurface_own", "index_flag"]
# index_flag: regime = FULL-SURFACE index (SPX→SPY, NDX→QQQ); duvar/atm = own FULL-SURFACE (amendment'ın
# doğal okuması: yalnız BAYRAK index-türevi, level'lar own; ekstra kombinasyon YOK)

# ---- metrikler ----
N_BLOCKS = 6
ROLL_WIN = 63
BOOT_N = 3000
BOOT_SEED = 7
TOPK_CONC = 3

# ---- DSR / trial muhasebesi ----
K_PRIOR = 10                    # D6'nın K'sı
K_AMENDMENT = 10                # INDEX-FLAG: flag-bağımlı 5 üye × 2 sembol = +10 nominal trial
K_CURRENT = K_PRIOR + K_AMENDMENT   # = 20; battery DSR'ı bu K ile

# ---- PRE-REGISTRATION AMENDMENT (kilitli; battery unblinding ÖNCESİ declare) ----
AMENDMENT = {
    "id": "FAZR-AMD-1",
    "declared_utc": "2026-06-11T13:10:00Z",   # config yazım anı; dosya-mtime + her çıktının metadata'sı kanıt
    "what": "INDEX-FLAG bayrak ailesi: FULL-SURFACE SPX-türevi bayrak SPY-trade, NDX-türevi QQQ-trade",
    "why": "instrument-quality: R2-prelim SPX-full vs SqueezeMetrics sign-agreement genel %79 / büyük-|gex| %96",
    "k_delta": K_AMENDMENT,
    "pit": "index D-EOD flag → ETF D+1 open→close, +1g lag (değişiklik yok)",
}

# ---- yollar ----
RAW_DIR = ROOT / "data" / "raw_chains"
CACHE = ROOT / "data" / "cache"
ARCHIVE_157 = CACHE / "archive_157g"
REMEASURE_DIR = ROOT / "backtest" / "remeasure"
TRIAL_LEDGER = REMEASURE_DIR / "trial_ledger.csv"
SAFETY_CREDITS = 200            # R0: kalan kredi bu eşiğin altına inince dur


def level_path(mode: str, sym: str, hygiene: bool = True) -> Path:
    """mode: 'livematch'|'fullsurface'."""
    tag = "" if hygiene else "_nohyg"
    return CACHE / f"level_series_{mode}_{sym.lower()}{tag}.parquet"


def config_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


if __name__ == "__main__":
    print(f"config_sha={config_sha()}  K={K_CURRENT}  train≤{TRAIN_END}  holdout≥{HOLDOUT_START}  panel≤{PANEL_END}")
