# KADER-EQUITY — GEX DIRECTIONAL — FAZ-R RE-MEASUREMENT (FINAL)

**RC3 SENTEZ — 2026-06-11.** Tek-gerçek-kaynak: `backtest/remeasure/config.py`, `config_sha = c80e6558181015a6`.
Bu rapor TEŞHİS-ONLY: yeni strateji/eşik/parametre üretilmedi; tüm sayılar gerçek script çıktısı.

## 0. ASSERT + ZİNCİR BÜTÜNLÜĞÜ

- **config_sha ASSERT: PASS** — 7 `RC2_*.json` + 16 `level_series_{livematch,fullsurface}_*.parquet.meta.json`
  dosyasının HEPSİNDE `config_sha = c80e6558181015a6` = `config.config_sha()`.
  Script: `backtest/remeasure/RC3_assert.py` → `backtest/remeasure/RC3_assert.json` (dosya-dosya liste).
- **RC1 (determinizm):** 16 seri 244–245g REBUILT; determinizm **16/16, max|Δ| = 0.0** (157g-overlap byte-exact);
  sağlık kapısı 15/16 PASS + **1 belgelenmiş istisna**: `fullsurface_ndx` flip_found **%84.8** (< %95 gate) —
  kök-neden yapısal: 37 eksik günün 35'i derin-pozitif-gamma (net_gex med +4.65bn), zero-crossing ±%6 scan-bandı
  DIŞINDA; deterministik (max|Δ|=0); **RC2-etkisiz** (hiçbir RC2 ölçümü bu serinin flip'ini kullanmıyor; NDX-full
  yalnız QQQ INDEX-FLAG rejim kaynağı). Kaynak: `backtest/remeasure/RC1_determinism.json`.
- **RC0 (kapsama):** ham full-chain SPY/QQQ 245/245 gün, SPX/NDX 244 gün (2025-12-12 kaynakta yok); eksik=0.
  Kaynak: `backtest/remeasure/RC0_gapcheck.json`.
- **Trial muhasebesi:** `backtest/remeasure/trial_ledger.csv` (46 satır = 10 prior-nominal + 26 replacement +
  10 amendment). **K_CURRENT = 20** (K_PRIOR 10 + K_AMENDMENT 10; replacement K'ya eklenmez = instrument-fix
  re-run). Amendment FAZR-AMD-1 declared_utc = 2026-06-11T13:10:00Z (config mtime 13:06:05Z), battery unblinding
  run_utc = 2026-06-11T17:45:41Z → **declare < unblinding kanıtlı**. Tüm battery DSR'ları K=20 ile.

## 1. HİPOTEZ TABLOSU ①–⑦: D-FAZ → FAZ-R FINAL → HÜKÜM

| # | hipotez | D-FAZ sayısı (kaynak: `backtest/DIAGNOSIS.md`) | FAZ-R FINAL sayısı (kaynak script) | HÜKÜM |
|---|---|---|---|---|
| ① | truncation (tek-expiry sansürü) | tarih-başına tek-expiry %100; görünen gamma$ **%11 SPY / %10 QQQ** (`diagnosis/D1_truncation.py`) | ham full-chain 244–245g rebuild; panel-medyan gamma$ **SPY-full 36.01bn / QQQ-full 17.79bn / SPX-full 328.5bn / NDX-full 12.36bn** (`remeasure/RC3_synthesis.py`; D1'in bağımsız bugün-ölçümü ~33.8bn ile tutarlı). Eski D-FAZ serisinde gamma_dollar kolonu yok → eski medyan ölçülemedi; D1 tek-gün kıyası $3.61bn/$1.71bn | **KAPANDI** (bedava; re-backfill `expiration=all`) |
| ② | sign (naive +call/−put) | SqueezeMetrics agreement üst-tercile **%30 SPY / %49 QQQ** (`diagnosis/D2_sign_battery.py`) | **SPX-full genel %72.7 / üst-tercile %92.6 / top10 10/10** (n=242); NDX-full %84.7 / %100; SPY-full %48.6 / %46.9; QQQ-full %57.6 / %49.4; SPY-livematch %53.5 / %59.3 (`remeasure/RC2_sign.py`) | **büyük-\|gex\| kolunda KAPANDI** (doğru havuz = index full-surface); **genel-ayak AÇIK** (%72.7 < %75 pre-committed eşik; alt-tercile %50.6 = sıfır-yakını-gamma koini). ETF-havuz proxy REDDEDİLDİ |
| ③ | havuz (index vs ETF) | ÖLÇÜLEMEDİ (yfinance ^SPX front-OI=0; 0.04×/0.52× artefakt) | **SPX/SPY med 9.06×** (p5 7.32 / p95 11.54), **NDX/QQQ med 0.70×** (p5 0.57 / p95 0.92), n=242; INDEX-vs-OWN bayrak uyumu %72.7 / %73.1 (`remeasure/RC2_pool.py`) | **KAPANDI (ölçüldü)** — S&P gamması index'te (~9×); NDX kompleksinde ETF baskın |
| ④ | canlı-uyum (5-vade vs 1-vade) | örtüşen-gün = 0, ölçülemedi (`diagnosis/D1_truncation.py` §5) | LIVE-MATCH serisi tanım-eş (front-5, `gamma_engine` birebir; RC1 max\|Δ\|=0); **ilk canlı örtüşme: as_of-join 3/3 agree, chain-EOD-join 5/5 agree** (n küçük = tutarlılık-kontrolü, güç-iddiası değil) (`remeasure/RC2_live.py`) | **KAPANDI (yapısal)**; istatistiksel güç forward-collector ile birikir (+1 örtüşme/gün) |
| ⑤ | IV-from-mid instabilite | bisection-fail %6.7/%9.1 (DTE0-2'de %32–37); ham bid/ask YOK → V1 proxy; flag-flip V1 %0.8 (`diagnosis/D2,D3`) | ham bid/ask %100 dolu → **gerçek V1 (bid≤0/crossed drop) flip %3.3 SPY / %2.5 QQQ**; V3 flat-IV %6.6/%6.6; **V5 DTE≤2-hariç %24.7/%20.2** (front-5'te 0-2DTE artık işaret-taşıyıcı; eski tek-expiry'de %0 idi); V4 OI-balance %46.9/%49.8 (ayrı sinyal); fragile-gün (≥2 varyant) **%14.8 SPY / %15.6 QQQ** (D2'de %2.1/%2.1); gamma_inv fragile-PnL payı SPY %10.5 / **QQQ %33.7** (D2'de %2/%3) (`remeasure/RC2_fragile.py`) | ölçüm-hijyeni **KAPANDI** (V1/V3 küçük); **kırılganlık D-FAZ'dan YÜKSEK çıktı → kısmen AÇIK** (DTE≤2 ve OI-bileşimi duyarlılığı) |
| ⑥ | confound (vol/OPEX/trend) | sign-flag acc %78–79 (taban+23pp); trend corr +0.467/+0.493; \|gex\| gamma-specific %72–74 (`diagnosis/D5_confound.py`) | livematch: SPY acc 0.72 (+22.0pp), trend corr 0.308; QQQ 0.69 (+13.6pp), 0.301. fullsurface: SPY 0.74 (+19.5pp); QQQ 0.77 (+22.5pp), corr 0.475. **SPX index-flag acc 0.92 (taban 0.69, +22.6pp), trend corr 0.631 / phi 0.62**; NDX 0.86 (taban 0.81, +5.1pp). \|gex\| büyüklük gamma-specific: **livematch %97.3/%98.0** (eski %72.2/%74.3); fullsurface %85.8/%95.8; SPX %89.1 / NDX %85.2 (`remeasure/RC2_confound.py`) | **AÇIK (yapısal)** — büyüklük temizlendi ama sign-flag hâlâ confound-tahmin-edilebilir (+5 ila +23pp); en-güvenilir SPX bayrağı aynı zamanda en trend-yüklü. Tek-rejim pencere sınırı sürüyor |
| ⑦ | istatistik (güç / best-of-K) | SPY t=+1.24 / DSR 0.387 (K=10); QQQ t=+0.84 / 0.233; gamma_inv Sharpe +1.29/+0.87; gereken-N 605/1331 (`diagnosis/D6_power.py`) | **24 üyede pozitif yönde hiçbir \|t\| ≥ 1.5 yok**; en iyi pozitif üye SPY hep_rev t=+0.79 (flag-BAĞIMSIZ kontrol); flag-bağımlı en iyi QQQ-full gamma_inv t=+0.69 / DSR(K=20) 0.115, SPY-full gamma_inv t=+0.55 / 0.090. \|t\|≥1.5 yalnız NEGATİF: QQQ-index M3_setup **t=−2.28**, SPY-index M3_setup −1.89, SPY hep_mom −1.52. **D-FAZ gamma_inv 'edge'i onarılmış flag'le çöktü: SPY +1.29 → +0.14, QQQ +0.87 → −0.56** (`remeasure/RC2_battery.py`) | **D-FAZ aday-edge'i REDDEDİLDİ (enstrüman-artefaktı)**; güç-sorusu güncel kalan-N ile AÇIK (bkz. §3) |

Eski-bayrak hata-oranı teyidi (re-measurement gerekçesi): eski tek-expiry işaret vs LIVE-MATCH flip
**SPY %28.4 / QQQ %24.6** (236g; `remeasure/RC2_fragile.py` a_flag_stability; 157g-prelim %32/%24 ile tutarlı)
→ D-FAZ'ın tüm downstream sayıları günlerin ~%25–28'inde yanlış bayrakla hesaplanmıştı.

## 2. OKUMA ANAHTARI (pre-committed; koşullar SAYIYLA — yorum değil)

| yol | koşul | ölçüm | sonuç |
|---|---|---|---|
| **A** | SPX-full sign genel ≥ %75 | **%72.7** (n=242; `RC2_sign.json`) | ✗ (−2.3pp) |
| **A** | büyük-\|gex\| ≥ %70 | **%92.6** (üst-tercile n=81); top10 10/10 | ✓ |
| **B** | event-edge textbook-yönüne döndü | HAYIR — own-livematch +γ 4/4 hücre negatif (SPY +g call −13.6bps t=−1.75; QQQ +g call **−33.4bps t=−3.33**); index_flag −γ hücreler pozitif ama tez −γ için MR<0 bekler → o da tez-dışı (QQQ −g put +30.6bps t=+1.29) (`RC2_events.json`) | ✗ |
| **B** | battery'de herhangi-üye \|t\| ≥ 1.5 | yalnız NEGATİF yönde (−2.28 / −1.89 / −1.52); pozitif max t = +0.79 (`RC2_battery_results.json`) | ✗ (pozitif yönde) |
| **C** | uyum kapanmadı VE edge'ler ters/null | uyum: YOL-A formel düştü (genel %72.7 < 75; üst-ayak %92.6 kapandı); edge'ler: pozitif max t +0.79, +γ event'ler negatif | **formel ✓** |

**Formel okuma: YOL-C tetiklendi → "bedava yol bitti".**
**SINIR-DURUM beyanı (sayı, karar değil):** YOL-A'nın genel-ayağı 2.3pp ile düştü; büyük-|gex| ayağı güçlü geçti
(%92.6); 157g-prelim'de A geçiyordu (%79/%96, `R2_PRELIM.md`) — 243g'de genel'i düşüren alt-tercile %50.6
(sıfır-yakını-gamma günleri koin). NDX-full %84.7/%100 iki ayağı da geçiyor ama anahtar SPX'e kilitli.
Fiilen A-sınırında olduğumuz için pre-committed talimat gereği ÜÇ seçenek SAYILARLA hazırlandı, **KARAR VERİLMEDİ**:

**(i) Forward-birikim + paralı çoklu-rejim derinlik (maliyet/kazanç):**
- Forward (bedava): flag-bağımlı en-yakın üyeler QQQ-full gamma_inv **6.9 yıl**, SPY-full gamma_inv **11.4 yıl**
  (t≥2 & DSR>0 K=20; §3). Flag-bağımsız kontrol SPY hep_rev 5.2 yıl (GEX-içerikli DEĞİL).
- Paralı tarih (D7 vendor tablosu, `backtest/DIAGNOSIS.md` §D7): **ORATS $99–299/ay** (EOD 2007+, ~19y →
  kalan-N'i veri-olarak anında kapatır + ⑤'i çözer); **ThetaData $40–160/ay** (4–12y); **Cboe Open-Close
  teklif-bazlı** (kanonik imza, SPX+SPY 2005+); SqueezeMetrics $720/ay (2011+). Örnek: ORATS-birey 12 ay =
  $1,188–3,588. Satın alınan şey = çoklu-rejim TEST (2018Q4/2020/2022), t≥2 garantisi DEĞİL — battery FINAL
  Sharpe'ları (max +0.72) tarihsel pencerede de bu büyüklükteyse t≥2 için yine ~2,000–3,100 gün gerekir (§3).
- Sign-verisi özelinde: büyük-|gex| uyumu %92.6 → paralı KANONİK imzanın (Cboe) marjinal değeri artık esas
  olarak alt-tercile (%50.6) ve genel-ayak (%72.7→?) içindir.

**(ii) Tez-revizyonu tedavi-adayı NOTU (karar değil):** directional günlük-EOD edge 24 üyenin hiçbirinde yok;
GEX'in ZATEN validated kullanımı vol/rejim-KOŞULLANDIRMA: `overlays/gex_shield.py` LOCKED 2026-06-09
(dealer short-gamma drawdown-shield; gex↔vol korr **−0.45**, kısmi **−0.30**; maxDD −6/−7pp, DSR 0.985/0.994 —
kader-equity model kaydı). Event-edge'in tutarlı NEGATİFLİĞİ (+γ'da duvar KIRILIYOR: −13.6/−33.4bps) bir
'anti-tez' gözlemdir; yeni üye türetmek bu fazda YASAK → yalnız not.

**(iii) Kill-criteria masaya:** şablon §6 (BOŞ; Emir doldurup kilitler).

## 3. D6 GÜNCELLEME — battery FINAL t'leriyle "t≥2 & DSR>0 (K=20)" kalan-N

Formüller D6 ile birebir (`diagnosis/D6_power.py`: t = SR_daily·√N → N=(2/SR_daily)²; DSR Bailey-LdP ikili-arama;
K = config.K_CURRENT = 20). Üye-P&L'leri `RC2_battery.build_panel/member_pnl` import (tek kaynak).
Script: `backtest/remeasure/RC3_synthesis.py` → `backtest/remeasure/RC3_d6_update.json`. Bağlayıcı kısıt her
satırda **t≥2** (N_t2 > N_dsr). Sharpe≤0 üyelerde bu yönde N tanımsız.

| sym | flag | member | SR_ann | t (N=235–236) | N(t≥2) | N(DSR>0,K=20) | kalan-N | forward-yıl | paralı-tarih-yıl |
|---|---|---|---|---|---|---|---|---|---|
| SPY | livematch_own | gamma_inv | +0.14 | +0.14 | 48,600 | 43,645 | 48,364 | 191.9 | 191.9 |
| SPY | none | vol_only | +0.72 | +0.70 | 1,935 | 1,696 | 1,699 | 6.7 | 6.7 |
| SPY | none | hep_rev | +0.81 | +0.78 | 1,534 | 1,339 | 1,298 | **5.2** | 5.2 |
| SPY | livematch_own | M3_setup | +0.32 | +0.31 | 9,545 | 8,423 | 9,309 | 36.9 | 36.9 |
| SPY | fullsurface_own | gamma_inv | +0.57 | +0.55 | 3,117 | 2,712 | 2,881 | **11.4** | 11.4 |
| SPY | fullsurface_own | M3_setup | +0.31 | +0.30 | 10,639 | 9,380 | 10,403 | 41.3 | 41.3 |
| QQQ | fullsurface_own | gamma_inv | +0.71 | +0.69 | 1,973 | 1,739 | 1,737 | **6.9** | 6.9 |
| QQQ | fullsurface_own | M3_setup | +0.10 | +0.10 | 100,019 | 89,283 | 99,783 | 396.0 | 396.0 |
| QQQ | index_flag | gamma_txt | +0.34 | +0.33 | 8,868 | 7,953 | 8,633 | 34.3 | 34.3 |
| diğer 15 üye | — | — | ≤0 | −2.28 … +0.01 | — | — | — | yön-ters/sıfır: bu yönde N tanımsız | — |

- **Sembol-başına özet:** SPY flag-bağımlı en-yakın = fullsurface gamma_inv **2,881 gün ≈ 11.4 yıl** forward
  (paralı-tarih: 11.4 yıl geçmiş; ORATS 2007+ ~19y bunu kapsar). QQQ = fullsurface gamma_inv **1,737 gün ≈ 6.9
  yıl** (ORATS kapsar; ThetaData 4–12y plana göre kapsar/kapsamaz). Flag-bağımsız SPY hep_rev 5.2y — GEX değil,
  gap-fade kontrolü.
- **D-FAZ kıyası:** eski gereken-N (SPY 605 / QQQ 1,331; `DIAGNOSIS.md` §D6) kırık-enstrüman Sharpe'larıyla
  (+1.29/+0.87) hesaplanmıştı; onarılmış Sharpe'lar küçüldüğü için **gereken-N BÜYÜDÜ** (en iyi GEX-üye için
  ~2,000–3,100 gün). Tek-rejim/IID varsayımı sürüyor → bu sayılar ALT-SINIR.

## 4. D4 FEED-TIMING HÜKMÜ (deploy dependency — `backtest/DIAGNOSIS.md` §D4'ten AYNEN)

> **HÜKÜM (OI[D]+mid[D], D+1 09:30 ET ÖNCESİ çekilebilir mi?): HAYIR (mekanik).** Kanıt: D-akşamı bile 402 →
> same-day yok; OI ertesi sabah açılış-civarı yayınlanır → mevcudiyet açılışLA çakışır, açılışTAN ÖNCE değil.
> → **vol-rejim CANLI sinyali için free MarketData = paid-feed dependency** (D+1 açılışta-en-erken/muhtemelen-
> sonra). Tek doğrudan-EVET testi D+1 09:25 ET tekrar-koşusu; bu run o pencerede değildi (dürüstçe: açılış-öncesi
> pencere doğrudan-ölçülmedi, ama D-akşamı-402 + OCC-mekaniği HAYIR'ı destekliyor).

**RC0 snapshot-job kanıtı (forward PIT-zinciri kuruldu):**
- `schtasks /query /tn "kader-equity-chain-snapshot"`: Schedule **Daily 18:00**, Status **Ready**,
  Last Run 2026-06-11 18:59:08 **Result 0**, Next Run 2026-06-12 18:00.
- `data/raw_chains/pit_ledger.csv`: 13 kayıt; 2026-06-10 zincirleri **SPY 13,390 / QQQ 11,076 / SPX 29,854 /
  NDX 12,822 kontrat** status=ok (fetch 2026-06-11T13:34–35Z), sonraki koşum status=exists (idempotent, append-only).

## 5. S1–S5 (DIAGNOSIS.md §4 soruları) — FAZ-R FINAL sayılarıyla

- **S1 (veri-bütçesi):** ① **0$ ile ÇÖZÜLDÜ** (rebuild 244–245g; `RC0_gapcheck.json`). ② paralı-sign ihtiyacı
  küçüldü: SPX-full %72.7 genel / %92.6 büyük-|gex| (`RC2_sign.json`) — paralı yol artık "imza onarımı" değil
  "tarih-derinliği" sorusu: ORATS $99–299/ay, ThetaData $40–160/ay, Polygon $29–199/ay, Cboe teklif-bazlı,
  SqueezeMetrics $720/ay (D7). Kalan-N tablosu §3.
- **S2 (havuz):** ÖLÇÜLDÜ — SPX/SPY **9.06×**, NDX/QQQ **0.70×** (n=242; `RC2_pool.json`). S&P'de asıl gamma
  index'te; NDX'te ETF baskın. INDEX-flag'in P&L değeri ise NEGATİF çıktı (QQQ-index M3_setup t=−2.28,
  SPY-index M3_setup t=−1.89; `RC2_battery_results.json`) → "doğru havuz" sign-uyumunu düzeltiyor ama
  directional edge üretmiyor.
- **S3 (sign-yöntemi):** naive +call/−put, doğru havuz+tam yüzeyle büyük-|gex| günlerde SqueezeMetrics ile
  **%92.6 (SPX) / %100 (NDX üst-tercile)** uyumlu (`RC2_sign.json`) → kanonik-imza (Cboe Open-Close) alımının
  marjinal değeri alt-tercile (%50.6) + genel-ayak (%72.7) ile sınırlı.
- **S4 (horizon):** D4 hükmü DEĞİŞMEDİ (yuk. §4; free-feed ile gün-içi canlı sinyal mekanik kurulamaz; snapshot-
  job 18:00 EOD forward-PIT biriktiriyor). Event-edge FINAL'de de tez-dışı: QQQ +g call **−33.4bps t=−3.33**
  (n=46), SPY +g call −13.6bps t=−1.75 (`RC2_events.json`) → daily-EOD'de "duvar tutuyor" yok; intraday-event
  reformülasyonu kararı Emir'de (1-dk bar: `data/historical_bars/alpaca_{spy,qqq}_1m.parquet` mevcut).
- **S5 (tarih-derinliği):** pencere hâlâ **~1.0 yıl TEK-rejim** (2025-06-13→2026-06-08; 2018Q4/2020/2022 yok).
  Güncel minimum-N: GEX-üyeleri için **SPY ~2,881g / QQQ ~1,737g** ek veri (t≥2&DSR>0 K=20; §3) — D-FAZ'ın
  605/1,331'inden BÜYÜK çünkü onarılmış Sharpe'lar küçüldü. "≥2018 kapsama" kabulü = ORATS/Cboe/IvyDB backfill
  (S1 bütçesi); ret = forward-biriktirme (≥6.9–11.4 yıl).

## 6. KILL-CRITERIA ŞABLONU (BOŞ — Emir dolduracak + pre-registration olarak kilitleyecek)

- [ ] Koşul → directional-GEX programı KAPANIR: ______
- [ ] Koşul → forward-birikim DEVAM (bedava, kill-tarihli): ______
- [ ] Koşul → paralı tarih-derinliği AÇILIR (vendor + bütçe): ______
- [ ] Koşul → tez-revizyonu (directional → koşullandırma) gündeme: ______
- [ ] Gözden-geçirme tarihi / tetikleyici: ______

---
*Üretim zinciri: RC0 `RC0_gapcheck.py` → RC1 `RC1_determinism.py` → RC2 `RC2_{sign,events,battery,pool,live,fragile,confound}.py` → RC3 `RC3_assert.py` + `RC3_synthesis.py` (bu rapor + `trial_ledger.csv`). Tüm çıktılar `config_sha=c80e6558181015a6` damgalı. Ham cache append-only; eski 157g seriler `data/cache/archive_157g/` (provenance). TIDE/OVERLAYS frozen; kader-macro READ-ONLY.*
