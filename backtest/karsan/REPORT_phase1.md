# Karsan Mekanizma Validasyonu — FAZ 0 + FAZ 1 (PRE-REGISTERED, footprint testi)

**Tarih:** 2026-06-12. **Disiplin:** k_config.py kilitli (window/eşik/FOMC ex-ante sabit, grid-search yok).
**Footprint ≠ mekanizma:** hiçbir verdict "dealer yaptı" demez; "mekanizmanın öngördüğü fiyat/vol izi
veride tutarlı mı". **Toplam 28 trial, Benjamini-Hochberg FDR (α=0.05).** Determinizm: seed 20260612,
block bootstrap (block=20g, 5000 rep). Faz 2 = GATED (Emir onayı). Scriptler: k_config/k_stats/k_data/k_phase1.

## FAZ 0 — veri & kalite
- **Daily OHLC** ^GSPC + ^NDX **1990+** (n=9177) — GFC/2011/2017-lowvol/2018Q4/2020/2022/2023/2025 = **MULTI-REGIME**.
- VIX 1990+, SKEW 1990+, VVIX 2006+, VIX3M+VXN 2007+, COR1M 2006+, VIX9D 2011+ (0DTE-flag 2023+ ayrı).
- 1-min SPY/QQQ ~2020-09..2026-06 (~5.8y) = **SINGLE-REGIME** (C5/C6).
- FOMC 87 scheduled decision 2015-2026 (elle-giriş public takvim; 2020-03 irregular çıkarıldı).
- Kalite: SPX/NDX temiz; index↔ETF corr SPX/SPY 0.985, NDX/QQQ 0.980 (OK>0.98). VIX9D'deki 196 |logret|>0.25
  = **gerçek vol hareketleri** (vol endeksi, split değil) — flag açıklandı, atılmadı.

## FAZ 1 — verdict tablosu

| Claim | Mekanizma (1 cümle) | Test edilebilir? | Effect size | raw p | BH p | Verdict | Pencere & caveat |
|---|---|---|---|---|---|---|---|
| **C1** OpEx charm/vanna ramp | vade-öncesi vol/skew yükselir, alımı zorlar, piyasayı destekler | evet (daily) | into-OpEx vol HAFİF DÜŞÜK (−2…−4bps), drift ~0 | 0.035–0.85 | hepsi ≥0.19 | **DESTEKLENMİYOR** (hatta ters: pin/compression) | multi-regime SPX+NDX 1990+ |
| **C2a** pinning (reflexive comp.) | düşük-IV → reflexive vol sıkışması | evet (IV-PROXY) | vix×low etkileşim −0.00014, anlamlı | **0.005** | **0.035 ✓** | **FOOTPRINT DESTEKLENİYOR** (zayıf; küçük etki, IV-proxy) | multi-regime; PROXY |
| **C2b** grind asimetrisi | high-IV→yavaş/sığ, low-IV→sert | evet | KARIŞIK: low-IV trough'a daha hızlı (32 vs 49g ✓) ama worst-1d high-IV'de daha sert (−2.95 vs −2.33% ✗) | 0.059/0.44 | ≥0.27 | **ZAYIF/karışık** | 149 episode; PROXY |
| **C3** skew-slide | down-day'de VIX↑ = mekanik, IV artışı DEĞİL | evet | down-day ΔVIX'in **~%100'ü** ret-terimi; β=−116, t=−28, R²=0.62 | **0.000** | **0.000 ✓** | **FOOTPRINT DESTEKLENİYOR** (güçlü; bilinen ilişkinin büyüklük-teyidi) | multi-regime 1990+ |
| **C4** FOMC vol-crush | event-vol çıktıdan bağımsız crush → destek | evet (VIX9D) | VIX9D −3.0% (crush %67) ama t=−1.6; VIX-30g ~0 | 0.115 | 0.359 | **ZAYIF** (yön doğru, FDR geçmez) | 87 FOMC 2015-26; 0DTE-flag 2023+ |
| **C5** open/close konsantrasyon | charm/vanna hedge gün-başı/sonu | evet (1-min) | vol konsantrasyonu **güçlü** (+0.8/+1.2bps, t18-19) AMA drift YOK + OpEx-etkileşim YOK | 0.000 | 0.000 ✓ | **KISMEN** (genel intraday U-şekli var; Karsan-spesifik yön/OpEx izi YOK) | SINGLE-REGIME ~5.8y |
| **C6** GHOST imbalance | açılış→GHOST mekanik fade | yalnız PROXY | fade EV ~0 (P(touch) 0.54-0.57); gap-down hafif fill / gap-up ters | 0.92/0.53 | ≥0.77 | **VERİMİZLE TEST EDİLEMEZ** (proxy=önceki-close, gerçek GHOST değil) | SINGLE-REGIME; PROXY |

**H1c COVID case-check (n=1, STAT DEĞİL, illustrative):** Feb-2020 OpEx ertesi 5g **−11.5%** (düşüş OpEx
ertesi başladı ✓); Mar-2020 OpEx (−15% önce) ertesi 5g **+10.3%** (dip ~OpEx günü ✓). Karsan'ın anekdotu
KALİTATİF tutuyor — ama tek gözlem, istatistik değil.

## Düz-dil okuma (evenhanded)
Karsan'ın 6 mekanik iddiasından veri, esasen **bilinen fiyat/vol özdeşliklerini** doğruluyor (C3 skew-slide
= VIX zaten SPX'in tersi; C5-vol = evrensel intraday U-şekli; C2a hafif vol-sıkışma) ve **daha spesifik /
tradeable footprint'leri DOĞRULAMIYOR**: OpEx vol-rampası (C1) yok — hatta ters; FOMC-crush (C4) zayıf;
GHOST (C6) proxy'yle test edilemez; intraday'in yön/OpEx-spesifik kısmı (C5 drift+H5b) yok. **BH-FDR'ı geçen
4 trial'ın 3'ü (C3, C5×2) zaten-bilinen özdeşlik;** gerçek sürpriz/tradeable footprint çıkmadı.

**Faz 2 (TIDE üzerine incremental edge) = Emir onayına bağlı. DURULDU.**
