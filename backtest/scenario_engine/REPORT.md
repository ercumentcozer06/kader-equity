# SENARYO MOTORU ÇALIŞMASI — InSillico gamma-indikatörü senaryolarının sentetik puanlaması (FINAL)

**Tarih:** 2026-06-11/12. **Amaç:** Emir'in TradingView'da canlı kullandığı "My Free Gamma Levels"
indikatörünün KENDİ kılavuzundaki işlem senaryolarını, bizim sentetik kopyamızla (FAZ-R onarılmış
enstrüman, front-5 expiry = canlı gamma_engine birebir) tarihsel olarak puanlamak.
**Karar sorusu:** günlük manuel kullanıma alınmalı mı, hangi senaryoları ciddiye almalı?

Tanımlar ÖNCEDEN kilitlendi: `se_config.py` (grid/eşik taraması yok, tüm hücreler raporlandı).
Veri: 237 seans (2025-06-13 → 2026-06-10), SPY+QQQ, seviye[D] (gün-sonu) → D+1 seansı (1-dk barlar).
Üç çalışma 3 paralel ajanla koşuldu, HER BİRİ bağımsız adversarial ajanla yeniden hesaplandı:
T1 19/19, T2 27/27 karşılaştırma birebir tuttu; T3 tüm denetimler geçti (PIT, giriş-saati, kontrol-tabanı).

## SONUÇ TABLOSU (indikatör iddiası → ölçüm → hüküm)

| İddia | Ölçüm (SPY / QQQ) | Hüküm |
|---|---|---|
| **GHOST = mıknatıs** (açılış boşluğu GHOST'a dolar) | P(dokunma) %42 / %36; **plasebo testi: aynı-mesafe koşulsuz oran 0.5-1 em1 kovasında %51-64 — koşullu oran plasebonun ALTINDA** | **RED — mıknatıs yok; doluş saf mesafe mekaniği** |
| GHOST trade EV (open→ghost, TP ghost) | net +1.3bps t=0.35 / −2.1bps t=−0.40; 1×em1 stoplu daha kötü (−1.3 / −5.3) | **RED — sıfır/negatif** |
| "Quick fill ≤30dk = temiz mekanik gün" | dokunanların %26.5 / %38'i ≤30dk; medyan 60/52dk | zayıf |
| "Hiç dolmazsa = trend günü, yarın devam" | ertesi-gün devam isabeti %48.7 / %44.7 (yazı-tura), ort. negatif | **RED** |
| **Duvar tutar** (Call/Put Wall yapısal tavan/taban) | dokununca 15-dk kapanışla kırılma: CW %80 / %89, PW %88 / %83 (rejimden bağımsız %78-93; gün-içi-yaklaşım ayrıştırılınca da %75-91) | **RED — duvarlar bu pencerede TUTMUYOR** |
| Duvar kırıldı → VU/VD'ye kaskad (yol) | P(VU\|kırılma) %86 / %90; P(VD\|kırılma) %87 / %94 | **DOĞRULANDI (yol olarak)** — "kırılmayı fade etme" uyarısı haklı |
| Kırılma-momentum girişi (15-dk teyit, stop=duvar, TP=VU/VD) | S2A: −15.6 t=−3.2 / −22.1 t=−4.7; S4A: −20.0 t=−6.6 / −31.9 t=−7.0 net bps | **RED (mekanik haliyle)** — giriş geç, TP dar, stop whipsaw'u yiyor. NOT: bizim VU/VD = literal "sonraki strike" ($1 ≈ 0.3 em1); indikatörün gerçek VU'su daha uzak (NAS100'de 200pt) → A-varyantı onların geometrisini birebir temsil etmez |
| Kırılma-kaskad (VU sonrası tut, seans sonu çık) | S2B/S4B ~düz: tek pozitif hücre **QQQ +γ CW-kırılma-kaskad +9.0 net t=1.32 (n=43)**; −γ'da CW-kaskad NEGATİF (−21/−16) | edge yok; +γ-yukarı-kırılma izlenebilir, −γ'da yukarı kırılmalar geri dönüyor |
| Duvar-reject fade (S1/S3) | reject zaten nadir (kırılma hâkim); n=13-21, EV ~0 | ölçülemez-küçük örneklem, edge görünmüyor |
| **Flip geri-alma = "en yüksek olasılıklı dönüş"** | SPY: −5.4 net t=−1.25 (taban −0.2'nin ALTINDA, ertesi gün de −12.6); **QQQ: +9.2 net t=1.02, saat-eşli tabanı +10.7bps aşıyor, ertesi gün +13.6** | **SPY'da RED, QQQ'da umut-verici-ama-kanıtsız** (n=36, t~1) → forward-izle, işlem yok |
| Flip altında kalış = hızlanma | reclaim'siz kırılma günleri kapanışa −9.6 / −14.5 bps (t≈−1.3) | yönü destekler, zayıf |
| Rejim = oynaklık bağlamı (−γ geniş range) | gün-içi range −γ'da %1.08 vs %0.77 (SPY), %1.42 vs %1.12 (QQQ) | **DOĞRULANDI** (zaten validated: gex_shield) |

## NE AYAKTA KALDI (yapıcı sonuç)

1. **Rejim = risk kadranı.** Negatif gamma günleri ~%40 daha geniş salınıyor. Zaten modeldeki
   gex_shield bunu kullanıyor; manuel işlemde de boyut-küçült/stop-genişlet kuralı olarak geçerli.
2. **"Teyitli kırılmayı fade etme" disiplini.** Duvar 15-dk kapanışla kırıldıysa VU/VD %86-94 geliyor.
   İndikatörün bu uyarısı verinin en sağlam yol-bulgusu. (Ama bunu GİRİŞ olarak para kazanmaya çevirmek
   mekanik gecikmeyle mümkün olmadı — uyarı evet, sinyal hayır.)
3. **QQQ flip-breakdown→reclaim long** = tek izleme-listesi adayı (taban-üstü +10.7bps, t~1, SPY'da yok).
   Forward'da ~20-30 olay birikmeden işlem boyutu verilmez.
4. Harita (seviyeler) geometrik olarak sağlam (FINDING 16: indikatöre ~%0-1) — bağlam/harita olarak
   kullanılabilir; ama seviyelerin kendisi mıknatıs/bariyer DEĞİL (plasebo + kırılma oranları).

## DÜRÜST SINIRLAR

- **237 seans, tek rejim (boğa, 2025-26).** "Duvarlar kırılır" bulgusu boğa-pencere ürünü olabilir;
  ayı/yatay rejimde duvar-tutma oranı farklı çıkabilir. Stres dönemi (2020/2022 tipi) yok.
- **Mekanik puanlama ≠ insan eli.** 15-dk kapanış teyidi yapısal olarak geç giriş; Emir'in canlı akış
  okuyarak seviyede/öncesinde girmesi farklı sonuç verebilir — ama bu ancak forward manuel kayıtla ölçülür.
- Sentetik CW indikatörün CW'sinden ~%1 sapabilir; VU/VD geometrisi farklı (yukarıda).
- Çok hücre ölçüldü (T1 120 + T2 ~90 + T3 ~20); düzeltme uygulanmadı (betimsel çalışma). Tek tük
  "yeşil" hücre (QQQ reclaim, QQQ +γ kaskad) tam da seçim-yanlılığının üreteceği şey → forward şart.

## DOSYALAR

`se_config.py` (kilitli tanımlar) · `se_panel.py` (panel+VU/VD) · `T1_ghost.py` / `T2_walls.py` /
`T3_flip.py` (çalışmalar) · `results/T{1,2,3}_{spy,qqq}.json` · `results/verify_T{1,2,3}*.{py,json}`
(bağımsız doğrulamalar: T1 19/19, T2 27/27 birebir; T3 denetimler tam geçti).
