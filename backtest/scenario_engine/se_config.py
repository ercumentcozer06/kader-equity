"""
backtest/scenario_engine/se_config.py — SENARYO MOTORU ÇALIŞMASI: TEK-GERÇEK-KAYNAK (pre-committed tanımlar).

AMAÇ (Emir, 2026-06-11): InSillico "My Free Gamma Levels" indikatörünün senaryolarını SENTETİK kopya ile
tarihsel olarak puanlamak — GÜNLÜK MANUEL kullanım kararı için. Bu bir strateji-arama DEĞİL, indikatörün
KENDİ kullanım kılavuzundaki iddiaların betimsel ölçümü. Grid/eşik taraması YASAK; tüm tanımlar burada,
çalışma koşulmadan ÖNCE kilitli. Tüm hücreler raporlanır (seçicilik yok).

VERİ:
  seviyeler  : data/cache/level_series_livematch_{spy,qqq}.parquet  (D-EOD, FAZ-R onarılmış enstrüman,
               front-5 expiry = canlı gamma_engine birebir; FINDING-16'da indikatöre ~%0-1 kalibre)
  rejim-duy. : data/cache/level_series_fullsurface_{spx,ndx}.parquet (INDEX bayrak duyarlılığı)
  barlar     : data/historical_bars/alpaca_{spy,qqq}_1m.parquet (IEX, UTC; RTH 09:30-16:00 ET)
  VU/VD      : ham zincirden (data/raw_chains/) front-5 expiry, OI>0, band ±%15:
               VU = call_wall ÜSTÜNDEKİ ilk listeli call-strike; VD = put_wall ALTINDAKİ ilk listeli put-strike
               (indikatör tanımı: "next strike step above the Call Wall and below the Put Wall")

PIT: level[D] (D-EOD zinciri) → D+1 RTH seansında kullanılır. Look-ahead yok. (Canlı pratikte OI[D]
D+1 sabah ~09:30'da yayınlanır → bu hizalama indikatörün "open'da çizilen harita"sıyla aynı saat sınıfı.)

SEVİYE TÜRETMELERİ (indikatör kılavuzu birebir):
  mid_up = (flip + call_wall)/2      # "MID — between the GEX Flip and Call Wall"
  mid_dn = (put_wall + flip)/2       # "MID — between the Put Wall and GEX Flip"
  rejim  = sign(net_gex)  PRIMARY = own-livematch; SENSITIVITY = index fullsurface (SPY→SPX, QQQ→NDX)

OLAY/İŞLEM TANIMLARI (kilitli):
  touch(level, yukarı)  : 1-dk high ≥ level ;  touch(level, aşağı): 1-dk low ≤ level
  BREAK-CONFIRM         : 09:30-anchored 15-dk barın KAPANIŞI seviyenin ötesinde ("closing through")
  REJECT                : touch var ama seans sonuna kadar 15-dk close-through yok
  giriş                 : confirm 15-dk barın kapanış fiyatı; giriş yalnız bar-başlangıcı ≤ 14:45 ET
  TP/SL fill            : girişten SONRAKİ 1-dk barlarla (TP: low≤tp (short)/high≥tp (long); SL: 15-dk
                          close geri-dönüş = "soft stop", seviye-referanslı — indikatör "wall as stop reference")
  aynı barda TP&SL      : SL önce sayılır (muhafazakâr)
  seans sonu            : ne TP ne SL → 15:59 close'ta çık
  volume-confirm split  : confirm 15-dk bar hacmi > o günün 15-dk medyan hacmi (betimsel, ikili split)
  maliyet               : COST_RT = 2.0 bps round-trip (delta-one; gross AYRICA raporlanır)

T1 — GHOST GAP-FADE (indikatör: "open GHOST'un üstündeyse mekanik çekim aşağı GHOST'a; trade with the
bias, target GHOST; quick fill ≤30dk = temiz mekanik gün; never fills = trend günü, ertesi gün devam"):
  olay-günü : |open(D+1) − ghost[D]| ≥ 0.25 × em1[D]   (indikatörün "flat overnight → N/A" filtresi;
              alternatif eşik 0.15% × spot AYRICA raporlanır)
  ölçüler   : P(touch ghost), time-to-touch dağılımı (≤30dk/≤120dk/gün-içi), MAE-before-touch (em1 birimi),
              EV: open'da gir → TP ghost → yoksa close'ta çık (stop'suz; + 1×em1 ters-stop varyantı)
  KONTROL   : eşleşik-mesafe plasebo — her olay günü mesafe d=|open−ghost|/em1 kovasında ({0.25-0.5,
              0.5-1, 1-2, 2+}), TÜM günlerde open'dan aynı d×em1 mesafedeki hedefin koşulsuz touch-oranı.
              Ghost-koşullu oran plaseboyu GEÇMELİ ki "mıknatıs" gerçek olsun.
  never-fill: dolmayan günlerde ertesi-gün open→close devam getirisi (indikatör iddiası: devam olasılığı ↑)

T2 — DUVAR HOLD/BREAK + VU/VD KASKAD (rejim-koşullu; Emir'in çekirdek senaryoları):
  CW-REJECT-FADE  : CW touch + dokunan 15-dk bar CW ALTINDA kapanırsa → bar-kapanışında SHORT;
                    TP = ghost (ghost < giriş×(1−0.001) değilse mid_up; o da değilse SKIP, sayısı raporlanır);
                    SL = 15-dk close ≥ VU; yoksa seans sonu.
  CW-BREAK-MOM    : ilk 15-dk close > CW → bar-kapanışında LONG; SL = 15-dk close geri CW altı;
                    TP-varyant-A: VU'da çık; varyant-B: VU'ya değerse seans-sonuna tut (kaskad/convexity).
  PW-REJECT-BOUNCE: ayna (PW touch + 15-dk close PW üstünde → LONG; TP = mid_dn yoksa flip; SL = 15-dk close ≤ VD)
  PW-BREAK-MOM    : ilk 15-dk close < PW → SHORT; SL = 15-dk close geri PW üstü; TP-A: VD; B: VD-sonrası tut.
  HÜCRELER        : sym{SPY,QQQ} × rejim{+γ,−γ} × setup ×{tüm, volume-confirm} — HEPSİ raporlanır.
  ölçüler         : P(touch), P(break|touch), P(VU/VD reach|break), her setup EV bps (gross+net), isabet,
                    ort-MFE/MAE (em1), n + Wilson %95 CI.

T3 — FLIP RECLAIM REVERSAL (indikatör: "breakdown sonrası GEX Flip'in geri alınması en yüksek olasılıklı
dönüşlerden biri"):
  olay : 15-dk close < flip[D] OLDUKTAN SONRA aynı seansta 15-dk close > flip[D] (reclaim)
  işlem: reclaim bar-kapanışında LONG → seans-sonu close; ertesi-gün open→close de raporlanır
  KONTROL: koşulsuz baz — TÜM günlerde aynı saat-dilimlerindeki 15-dk bar-kapanışından seans-sonuna LONG EV
  rejim-split + n + Wilson CI. Ek bağlam: rejime göre gerçekleşen gün-içi range/σ (bilinen vol-etkisi, sanity).

İSTATİSTİK: oranlar Wilson %95 CI ile; EV'ler ort ± t-stat (n≥10 hücrelerde); n<10 hücre "YETERSİZ-N"
etiketi (yorumsuz). ÇOK-TEST NOTU: bu çalışma betimsel senaryo-ölçümü; yine de tüm hücre sayısı ve
"kaç hücre test edildi" raporun başına yazılır (seçim-yanlılığı şeffaflığı).

ÇIKTI: backtest/scenario_engine/results/T{1,2,3}_{spy,qqq}.json + REPORT.md (sentez).
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENV_PY = "C:/Users/admin/Downloads/kader-macro/.venv/Scripts/python.exe"

SYMS = ["SPY", "QQQ"]
INDEX_FLAG_MAP = {"SPY": "SPX", "QQQ": "NDX"}

# pencere: eldeki tüm seviye-günleri (D+1 seansı barlarda olmalı)
PANEL_START = "2025-06-13"
PANEL_END = "2026-06-09"          # D=06-09 → D+1=06-10 son bar günü

COST_RT = 0.0002                  # 2.0 bps round-trip (net); gross ayrıca
EM1_MIN_DIST = 0.25               # T1 olay eşiği: |open−ghost| ≥ 0.25×em1
ALT_DIST_PCT = 0.0015             # T1 alternatif eşik: 0.15% × spot
QUICK_FILL_MIN = 30               # "ilk 30 dk" hızlı-doluş
ENTRY_CUTOFF = "14:45"            # giriş yapılabilen son 15-dk bar başlangıcı (ET)
BAR15_ANCHOR = "09:30"
TP_MIN_GAP = 0.001                # TP girişten en az %0.1 uzakta olmalı (yoksa sıradaki seviye / SKIP)
T1_STOP_EM1 = 1.0                 # T1 stop-varyantı: 1×em1 ters hareket
DIST_BUCKETS = [(0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 99.0)]   # em1 birimi

PANEL_PATH = ROOT / "backtest" / "scenario_engine" / "panel_{sym}.parquet"
RESULTS_DIR = ROOT / "backtest" / "scenario_engine" / "results"


def level_series(sym: str, mode: str = "livematch"):
    import pandas as pd
    return pd.read_parquet(ROOT / "data" / "cache" / f"level_series_{mode}_{sym.lower()}.parquet")


def panel_path(sym: str) -> Path:
    return Path(str(PANEL_PATH).format(sym=sym.lower()))
