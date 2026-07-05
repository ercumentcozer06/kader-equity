# FAZ-R / R2 ÖN-SONUÇ (157/243 gün, 2026-06-11 gece)

**Statü:** ham capture 157/243 gün (2025-06-13→2026-02-04, %65); kalan ~78 gün (en yeni, düşmüyor) + tam DSR/t
battery yarın kredi-reset sonrası. Script: `backtest/remeasure/R2_keydiag.py`.

## ASIL BULGU: ② sign-sorunu onarımla ÇÖZÜLÜYOR (YOL-A sağlandı)

**Kritik apples-to-apples test — SPX-FULL-SURFACE (gerçek SPX-market gamma) vs SqueezeMetrics (SPX-market):**

| seri | gamma$ med | sign-agr GENEL | tercile alt / orta / **üst (büyük-\|gex\|)** |
|---|---|---|---|
| SPY OLD (kırık, tek-expiry) | — | %48 | %61 / %54 / **%30** ← D-FAZ'ın çöküşü |
| SPY LIVE-MATCH (front-5) | 14.04bn | %56 | %58 / %56 / %53 |
| SPY FULL-SURFACE | 35.82bn | %49 | %55 / %46 / %45 |
| QQQ FULL-SURFACE | 17.83bn | %59 | %66 / %65 / %46 |
| SPX LIVE-MATCH | 56.28bn | %64 | %56 / %63 / %73 |
| **SPX FULL-SURFACE** | **316.90bn** | **%79** | %60 / %80 / **%96** |

**YOL-A eşiği (önceden kilitli): genel ≥%75 VE büyük-\|gex\| ≥%70 → naive-sign kurtuldu.**
SPX-FULL: genel **%79 ✅**, üst-tercile **%96 ✅** → **YOL-A SAĞLANDI.**

**Yorum:** D-FAZ'ın "%30/%49 sapması = naive-sign bozuk, paralı Cboe gerek" korkusu YANLIŞTI. Doğru havuzu (SPX index)
+ tüm-yüzeyi ölçünce, naive call-long/put-short işareti SqueezeMetrics'in profesyonel sınıflamasıyla **büyük-gamma
günlerinde %96 örtüşüyor.** Sapma tamamen KIRIK ENSTRÜMANDI (tek-front-expiry + ETF-havuz). → **paralı sign-verisi
ŞİMDİLİK GEREKSİZ; doğru pool = SPX index full-surface (SPY-ETF tek başına yetmiyor → ③ doğrulandı, asıl gamma index'te).**

## Kapsama (①) — onarım doğrulandı
- SPY FULL $35.82bn (D1'in bağımsız bugünkü ölçümü ~$33.8bn ile tutarlı); LIVE-MATCH $14bn; eski-tek-expiry ~%11.
- SPX FULL $316.9bn ≈ SPY-FULL'ün ~9×'i → ③ index-havuzu ETF'in ~9 katı (asıl S&P gamma index kompleksinde).

## Bayrak istikrar (③/④) — eski flag yanlıştı
- Eski-işaret vs LIVE-MATCH-işaret flip: **SPY %32, QQQ %24.** → D-FAZ'ın rejim-bayrağı günlerin %24-32'sinde YANLIŞTI
  → tüm D-FAZ downstream sayıları (event-edge, gamma_inv P&L) o yanlış flag'le hesaplanmıştı → re-measurement HAKLI.

## Event-edge 8-hücre (⑦) — YOL-B HENÜZ değil
Ort MR bps (+>0 duvar-tuttu / <0 kırıldı). Tez: +γ→pozitif, −γ→negatif.

| | +γ-call | +γ-put | −γ-call | −γ-put |
|---|---|---|---|---|
| SPY OLD | −5 | −31 | +7 | +3 |
| SPY LIVE-MATCH | −23 | −16 | −11 | −14 |
| QQQ OLD | −23 | −18 | −21 | +12 |
| QQQ LIVE-MATCH | −19 | −17 | −1 | −15 |

**Onarılmış flag'le duvarlar HER İKİ rejimde de kırılıyor (hepsi ~negatif)** — tez-yönüne (+γ pozitif) DÖNMEDİ.
AMA: (a) ön-sonuç, hücre-başı n=14-39 küçük; (b) SPY/QQQ-ETF flag+wall'la, en-güvenilir SPX-rejimiyle DEĞİL;
(c) eski-157-gün (gamma_inv edge'inin yoğun olduğu son-dönem HARİÇ). → **YOL-B (directional tez yaşıyor) HENÜZ
doğrulanmadı; tam battery (243g + SPX-rejimi) yarın karar verir.**

## PRE-COMMITTED OKUMA ANAHTARI — şu anki konum
- **YOL-A (sign kurtuldu): ✅ SAĞLANDI** (SPX-FULL genel %79 / üst %96) → paralı-sign gereksiz, doğru-pool=SPX-index.
- **YOL-B (edge düz, tez yaşıyor): ⏳ HENÜZ DEĞİL** (event-edge tez-yönüne dönmedi; ama ön-sonuç + yanlış-pool-rejimi).
- **YOL-C (tükendi/kill): ❌ DEĞİL** (uyum kapandı, enstrüman onarıldı).

## YARIN (kredi-reset sonrası)
1. R0 resume → kalan ~78 gün (2026-02-05→06-08).
2. R1 full build (243g, NDX-full dahil).
3. R2 FULL BATTERY: ⑥confound + ⑦pre-registered RE-RUN (gamma_inv/txt/vol_only/M1-2-3, DSR aynı-K) + event-edge
   243g **SPX-rejimiyle** (en-güvenilir flag) + ③pool günlük + ④canlı-snapshot-uyum.
4. R3 REMEASURE.md + S1-S5'e FAZ-R sayıları → Emir kill-criteria kilitler.
