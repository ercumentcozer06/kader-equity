"""
backtest/karsan/k_config.py — KARSAN MEKANİZMA VALİDASYONU: PRE-REGISTERED kilitli tanımlar.
Spec: tüm window/eşik/FOMC-tarih THEORY-GIVEN ve ex-ante sabit. Grid-search YOK. Çalıştırmadan önce kilitli.
Faz 1 = footprint validasyonu (edge DEĞİL). Faz 2 = GATED (Emir onayı). Bu dosya HER iki fazın tek kaynağı.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KDIR = ROOT / "backtest" / "karsan"
KDATA = KDIR / "data"
KRESULTS = KDIR / "results"
VENV_PY = "C:/Users/admin/Downloads/kader-macro/.venv/Scripts/python.exe"

SEED = 20260612                    # determinizm: tüm bootstrap bu seed'le
BOOT_BLOCK = 20                    # block bootstrap blok uzunluğu (gün) — overlap/otokorelasyon için
BOOT_N = 5000                      # bootstrap replikasyon
FDR_ALPHA = 0.05                   # BH-FDR hedef
EXEC_LAG = 1                       # t+1 execution lag (her sinyal gün T, exec T+1)

# --- kalite kapısı eşikleri (Faz 0) ---
QG_MAX_ABS_LOGRET = 0.25          # tek-gün |log-return| > 0.25 → FLAG (split/bad-tick şüphesi)
QG_ETF_INDEX_MINCORR = 0.98       # index vs ETF günlük return korelasyonu < 0.98 → veri sorunu FLAG
QG_MAX_FFILL = 1                  # forward-fill ≤ 1 bar; fazlası = gap FLAG

# --- C1 OpEx: 3rd Friday (monthly), quad-witch (Mar/Jun/Sep/Dec 3rd Fri) ---
QUAD_MONTHS = (3, 6, 9, 12)
OPEX_PRE_WIN = 5                   # OpEx haftası ~ 3rd-Fri'den geriye 5 işgünü (theory: into-expiry ramp)
OPEX_POST_WIN = 5                 # OpEx ertesi hafta ~ 5 işgünü (theory: charm/vanna unwind release)
COVID_OPEX = {"feb2020": "2020-02-21", "mar2020": "2020-03-20"}   # H1c kalitatif case-check (STAT DEĞİL, n=1)

# --- C2 pinning/grind: düşük-IV rejimi = VIX < rolling median ---
C2_VIX_MED_WIN = 252              # rolling median penceresi (1y, theory-given değil ama standart; tek değer, taranmadı)
C2_FWD_RV_WIN = 21               # gelecek realized-vol penceresi (1ay)

# --- C4 FOMC: scheduled decision (statement) günleri. 2015-2026 (VIX9D-çağı + çok-rejim:
#     2015-liftoff/2018Q4/2020-COVID/2021-22-hikes/2023-SVB). Pre-2015 ELLE-GİRİŞ-RİSKİ diye DIŞARIDA.
#     2020-03 SCHEDULED meeting iptal+emergency (Mar15 Pazar) → IRREGULAR, C4'ten ÇIKARILDI. ---
FOMC_DATES = [
    "2015-01-28","2015-03-18","2015-04-29","2015-06-17","2015-07-29","2015-09-17","2015-10-28","2015-12-16",
    "2016-01-27","2016-03-16","2016-04-27","2016-06-15","2016-07-27","2016-09-21","2016-11-02","2016-12-14",
    "2017-02-01","2017-03-15","2017-05-03","2017-06-14","2017-07-26","2017-09-20","2017-11-01","2017-12-13",
    "2018-01-31","2018-03-21","2018-05-02","2018-06-13","2018-08-01","2018-09-26","2018-11-08","2018-12-19",
    "2019-01-30","2019-03-20","2019-05-01","2019-06-19","2019-07-31","2019-09-18","2019-10-30","2019-12-11",
    "2020-01-29","2020-04-29","2020-06-10","2020-07-29","2020-09-16","2020-11-05","2020-12-16",  # 2020-03 IRREGULAR çıkarıldı
    "2021-01-27","2021-03-17","2021-04-28","2021-06-16","2021-07-28","2021-09-22","2021-11-03","2021-12-15",
    "2022-01-26","2022-03-16","2022-05-04","2022-06-15","2022-07-27","2022-09-21","2022-11-02","2022-12-14",
    "2023-02-01","2023-03-22","2023-05-03","2023-06-14","2023-07-26","2023-09-20","2023-11-01","2023-12-13",
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31","2024-09-18","2024-11-07","2024-12-18",
    "2025-01-29","2025-03-19","2025-05-07","2025-06-18","2025-07-30","2025-09-17","2025-10-29","2025-12-10",
]
FOMC_PRE_WIN = 5                  # FOMC öncesi 5 işgünü IV seviyesi
FOMC_POST_WIN = 3                 # FOMC sonrası 3 işgünü (crush + drift)
ZERODTE_FLAG_FROM = "2023-01-01"  # 0DTE bozulma bayrağı: VIX9D 2023+ ayrı işaretlenir

# --- C5/C6 intraday (1-min) ---
RTH_OPEN = ("09:30", "10:00")
RTH_MID = ("11:30", "13:30")
RTH_CLOSE = ("15:30", "16:00")

# --- yfinance fetch hedefleri (Faz 0 cache) ---
YF_TICKERS = {"SPX": "^GSPC", "NDX": "^NDX", "VIX9D": "^VIX9D"}


def boot_rng():
    return np.random.default_rng(SEED)
