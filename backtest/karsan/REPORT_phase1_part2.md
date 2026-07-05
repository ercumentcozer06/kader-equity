# Karsan Validasyonu — PART 2 (vol-surface adayları) — AYRI FDR FAMILY

**Tarih:** 2026-06-12. **POST-HOC batch:** bu adaylar Part-1'in çoğunlukla null çıktığı görüldükten
SONRA seçildi → her anlamlı sonuç **zayıf kanıt** (null-sonrası fishing). **Part-1'in 28 trial'ına POOL
EDİLMEDİ;** ayrı BH-FDR. **Part-2 = 12 trial.** Aynı disiplin (PIT, t+1, block bootstrap, pre-registered).

## Verdict tablosu (Part-2 family)

| Claim | Mekanizma | Effect | raw p | BH p | Verdict | Etiket + caveat |
|---|---|---|---|---|---|---|
| **C7** RV up/down asimetrisi | aşağı daha hızlı → vol↑ | down-day fwd-5g vol **+9.4 / +12.2 bps** (SPX/NDX, t~10) | 0.000 | **0.000 ✓** | **DOĞRULANDI ama IDENTITY** | leverage effect — bilinen özdeşlik, **tradeable DEĞİL** (R²=0.30) |
| **C8a** term-structure / monthly | vade-öncesi slope elevated | VIX/VIX3M −0.005 (t−1.4); VIX9D/VIX ~0 | 0.15–0.97 | ≥0.30 | **DESTEKLENMİYOR** | monthly = null (C1-monthly ile tutarlı) |
| **C8a** term-structure / **quarterly** | quad-witch'e slope elevated | **VIX9D/VIX +0.038 (t4.22)** CI[+0.021,+0.056]; VIX/VIX3M null | **0.000** | **0.000 ✓** | **FOOTPRINT DESTEKLENİYOR** (tek non-identity hit) | ön-uç slope (9g/1ay) **quad-witch'e yükseliyor**; ~61 olay 2011-26; **0DTE-flag 2023+**, post-hoc → zayıf-orta |
| **C8b** post-OpEx normalize | slope geri döner | quarterly +0.015 (t1.2), diğerleri null | 0.22–0.42 | ≥0.33 | **DESTEKLENMİYOR** | "yüksel-sonra-normalize" döngüsünün yalnız YÜKSELME yarısı tuttu |
| **C9** skew = f(trend) | up-run → skew steepen | SKEW ~ trailing-ret **β +36/+32** (t4.8/5.5) | 0.000 | **0.000 ✓** | **DESTEKLENİYOR ama EXPLORATORY** | gerçek ama zayıf (R²=0.02/0.04), gürültülü; COR1M-froth'a tamamlayıcı froth-gauge |

## Düz-dil okuma (evenhanded)
- **C7** beklendiği gibi anlamlı — ama leverage-effect özdeşliği, sürpriz değil, edge değil.
- **C8** (en yüksek değerli, term-structure): **monthly hiçbir şey vermiyor** (C1 monthly null ile aynı).
  **Tek non-identity footprint = VIX9D/VIX ön-uç slope'unun QUAD-WITCH'e yükselmesi** (t4.2, ~61 olay,
  2011-26 çok-rejim-ish). Bu Part-2'nin yegâne hem-non-identity hem-FDR-geçen sonucu. AMA: yalnız quarterly,
  yalnız 9g/1ay slope, küçük-n, 2023+ 0DTE-bozulması, post-hoc batch → **zayıf-orta kanıt, güçlü DEĞİL**.
  Post-OpEx normalizasyon (H8b) tutmadı → tam döngü doğrulanmadı.
- **C9** gerçek ama zayıf/exploratory: trailing pozitif getiri → SKEW steepen (froth göstergesi). COR1M-froth'la
  kavramsal örtüşür.

## Kapı (spec gereği DUR)
Emir'in pre-registered kuralı: "C8 anlamlı çıkarsa, o **tek başına** Faz 2'ye taşımaya değer tek aday
(term-structure = Tier-1 vol-surface sinyali)." **C8-quarterly geçti** → Faz 2'ye değer **tek aday = OpEx-
koşullu quad-witch ön-uç vol-slope**. Ama dikkat: bu bir vol-MAGNITUDE/term-structure sinyali, **yön DEĞİL**
— yani Faz 2'yi geçse bile "expression-timing + risk-sizing + manuel context" sonucunu pekiştirir, otomatik
**directional** engine üretmez. Kalan tüm okuma (C7 identity, C8-monthly null, C9 exploratory) bu sonucu doğruluyor.
**Faz 2 = Emir onayı.**
