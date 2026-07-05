# SWING-GEX ÇALIŞMASI — GEX'i rejim-kadranı olarak modele katmak (FINAL)

**Tarih:** 2026-06-12. **Çerçeve (Emir):** GEX = swing rejim kadranı ("flip üstü tam oyna / flip altı kıs"),
intraday cascade manuel. **Veri:** SqueezeMetrics dealer-GEX + SPX, **2011-05 → 2026-06, 3798 gün, çok-rejim**
(2015/2018Q4/2020/2022/2023/2025 hepsi içinde) — intraday çalışmanın 1-yıllık verisinin tersine gerçek güç.
3 test, 3 paralel ajan, her biri bağımsız adversarial doğrulayıcı: **T1 20/20 birebir, T2/T3 CONFIRMED.**
Baseline stack byte-sadık üretildi (SPX 1.636 / NDX 1.773 = finalize_stack hedefi → harness doğru).

## TEK CÜMLE
**GEX'in swing değeri gerçek ama SAVUNMA amaçlı, ve modelin mevcut z-shield'ı bunu ZATEN yakalıyor.**
Üç iyileştirmenin hiçbiri modele konunca pozitif çıkmıyor — model GEX-swing sınırına çoktan yakın.

## T1 — Senin flip kuralın (flip-altı kıs) — verdict: TIDE_ABSORBED

| | standalone SPX 2011-26 | modele konunca (2019+) |
|---|---|---|
| B&H | Sharpe 0.74, maxDD −34% | — |
| Mevcut z-shield (V0) | 0.76, −27% | **stack SPX 1.636 / maxDD −13.2% / DSR 0.985** |
| Flip-bin 0.5 (senin kuralın) | **0.79, −22%** | stack 1.593 / maxDD **−16.5%** / DSR 0.977 |
| Asimetrik (tide-koşullu) | 0.80, −22% | stack 1.585 / maxDD −16.8% / DSR 0.978 |

- **Standalone'da senin kuralın HAKLI:** 15 yılda mevcut z-shield'dan daha iyi (Sharpe + maxDD), 6/7 alt-dönemde, 2022 dahil.
- **AMA modele konunca TERSİNE DÖNÜYOR:** stack'in Sharpe'ını, maxDD'sini ve DSR'ını hem SPX hem NDX'te BOZUYOR.
- **Neden:** ikili `gex<0` kuralı günlerin ~yarısında kısıyor (z-shield ise sadece derin-negatif ~%14'te); froth katmanıyla üst üste binince iyi günleri çift-kısıyor ve modelin bilerek bindiği rebound'ları kesiyor. Model zaten rebound-safe → ağır ikili kapı yanlış yer.
- **Pratik:** modeldeki z-shield kalsın (zaten daha iyi in-stack). Flip-gate senin **MANUEL risk kadranın** olarak değerli (standalone gerçekten drawdown kesiyor) — modeli değiştirmesin.

## T2 — GEX öncü-kırılganlık mı? — verdict: MARGINAL

- GEX, gerçekleşen oynaklığın **ÖTESİNDE** ileriye-dönük drawdown bilgisi taşıyor ama **çok küçük**: kısmi-korr +0.17 (örtüşmesiz +0.23), OLS gz katsayısı HAC t=4.7–6.0, **ek R² sadece ~%2**.
- **Erken uyarı:** en kötü 20 drawdown'ın **20'sinde de** GEX derin-negatife geçti (medyan 26 gün önce); oynaklık 16/20'sinde (medyan 23 gün). GEX biraz daha **güvenilir** (hepsini yakaladı) ama dramatik daha erken değil (her ikisinin yandığı olaylarda +3 gün).
- **Kapı kıyası:** vol-only Sharpe 0.84/maxDD −21.7%; gex-only 0.79/−24%; **kombine 0.76/−20.4%** → kombine en iyi maxDD ama Sharpe maliyeti. Vol tek başına daha iyi.
- **Sonuç:** GEX gerçek ama küçük erken-uyarı katıyor; modele yeni kablo değmez (z-shield zaten GEX kullanıyor). **Manuel "kırılganlık alarmı"** olarak değerli: GEX derin-negatife geçince drawdown penceresi bekle, boyut düşür.

## T3 — Vanna/charm OPEX × rejim (en umutlu olduğun) — verdict: TIDE_ABSORBED

- **Tezin yarısı doğru, yarısı TERS:**
  - OPEX-öncesi yukarı drift, **pozitif gammada** → DOĞRU (+0.059%/gün, t=2.04). OPEX−4 günü +0.24% t=3.28 (eski mevsimsellik tekrar çıktı).
  - OPEX-sonrası "zayıflık", **negatif gammada** → **TERS**: en güçlü pozitif hücre bu (+0.530%/gün, t=2.25) = zayıflık değil **sıçrama** ("korkuyu fade et").
- **Standalone overlay KAYBEDİYOR:** Sharpe 0.71→0.68 (post∩neg-gamma'yı kısmak yanlış, orası pozitif).
- **Modele konunca:** rejim-koşullu OPEX overlay SPX 1.636→1.667 / NDX 1.773→1.806 — **minik artış** ama anlamlı değil (P>frozen sadece %84-86, gereken %95+), maxDD değişmiyor, ve mevcut shield'la **ağır örtüşüyor** (aynı derin-negatif-gamma günleri).
- **Sonuç:** tradeable edge çıkmadı; tek gerçek olgu pozitif-gammadaki zayıf öncesi-drift (betimsel) ve negatif-gamma OPEX-sonrası sıçrama (manuel "fade etme" notu). Mevsimsellik yine tide/shield tarafından emilmiş.

## KARAR — modele ne girer, sana ne kalır

- **Modele YENİ HİÇBİR ŞEY girmiyor.** Mevcut z-shield zaten GEX-swing değerinin en iyi in-stack halini taşıyor; üç iyileştirme de ya in-model bozuyor (T1) ya marjinal (T2) ya tide-emilmiş (T3). Bu iyi haber: **model GEX-swing sınırına yakın, overfit'e gerek yok.**
- **Senin işbölümün DOĞRULANDI:** model rejim kadranını (z-shield) tutar; sen **flip-gate'i manuel risk kadranı** olarak kullanırsın (standalone gerçekten drawdown kesiyor, T1) + **GEX-derin-negatif = kırılganlık alarmı** (T2, 20/20 olay) + **negatif-gamma OPEX-sonrası = korkuyu fade et** (T3).
- Intraday forced-buying cascade zaten manuelde — model ona dokunmaz.

## DOSYALAR
`gxs_config.py` (kilitli tanımlar) · `T1_flipgate.py` / `T2_leading.py` / `T3_opex.py` ·
`results/T{1,2,3}.json` + `verify_T{1,2,3}*` (T1 20/20 birebir, T2/T3 CONFIRMED). Ana stack FROZEN, dokunulmadı.
