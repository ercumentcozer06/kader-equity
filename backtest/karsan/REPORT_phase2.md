# Karsan Validasyonu — FAZ 2 (C8 survivor → TIDE incremental edge)

**Tarih:** 2026-06-12 (Emir onayı: "Faz 2'yi koş, iyimser ol"). **Kriter:** TIDE/stack üzerine **incremental
Sharpe** (standalone Sharpe DEĞİL), t+1 lag, PIT-clean. **9 Phase-2 trial, ayrı BH-FDR, DSR cumulative-N=48.**
İYİMSER kuruldu: directional (quad-witch support) + contrarian + vol-regime okumalarının HEPSİ denendi.

## Standalone directional içerik (2011-26, max güç)
- **slope tercile → forward ret:** high-slope (backwardation) → daha YÜKSEK forward (fwd21 +1.61% vs low +0.67%)
  = contrarian-bullish eğilim (low-GEX gibi: korku→sıçrama). AMA fwd21 t2.49 **raw p 0.013 → BH 0.114 GEÇMEZ**;
  fwd5 anlamsız. **Yön var ama FDR-robust DEĞİL.**
- **Karsan quad-witch support/weakness:** into-quad −5.1bps (t−0.87), post-quad −2.9bps (t−0.40) — **NULL** (into
  pozitif-support DEĞİL). into-quad ∩ slope-elevated = **−20.8bps (t−1.87) = TERS** (vanna-support tezi tutmuyor).

## Incremental ablation (2019+, asıl kriter)

| Stack | SPX Sharpe / maxDD | NDX Sharpe / maxDD | Δ vs stack |
|---|---|---|---|
| stack (base: tide×froth×shield) | 1.636 / −13% | 1.773 / −16% | — |
| **stack × slope-trim** (backwardation→0.5) | **1.693 / −11%** | **1.806 / −11%** | **+0.057 / +0.033, maxDD −2pp** |
| stack + quad-vanna-override (directional) | 1.612 / −14% | 1.753 / −15% | **−0.024 / −0.020 (ZARAR)** |
| tide + into-quad-LONG (directional boost) | 1.370 (tide 1.424'ten DÜŞTÜ) | 1.526 | ZARAR |

**Anlamlılık:** slope-trim ΔSharpe paired-bootstrap **P(>stack) ~%60-64 / BH 0.599 → GEÇMEZ** (gereken %95).
DSR 0.990/0.997 (yüksek ama in-sample). **0/9 trial BH-FDR geçti.**

## Verdict
- **Directional Karsan okuması = ÖLÜ:** quad-witch support null/ters, into-quad-long Sharpe'ı düşürüyor,
  vanna-override zarar. C8 footprint gerçek ama **yön taşımıyor** (vol-magnitude sinyali).
- **Vol-regime okuması (slope-trim) = mild AMA robust değil:** ön-uç backwardation'da kısmak SPX/NDX maxDD'yi
  −2pp iyileştiriyor + küçük Sharpe (+0.03-0.06), DSR ~0.99. **AMA paired-P ~%60 (FDR-geçmez) VE mevcut
  gex_shield'la kavramsal-örtüşür** (ikisi de "yakın-vade vol-stresinde kıs"). In-sample bump, robust edge değil.
- **Tek dürüst olumlu:** slope-trim maxDD'yi iki varlıkta da düşürüyor — yeni edge DEĞİL ama **forward-watch /
  ileride savunma-sertleştirme adayı** (deploy değil; shield'ı zaten var, bu onun hızlı-front-end versiyonu).

## SONUÇ — Karsan/GEX katmanı (mühür)
İyimser, çok-yönelimli, FDR/DSR-disiplinli test sonrası: **C8 dahil hiçbir Karsan footprint'i TIDE üzerine
robust incremental directional edge vermedi.** Katman = **expression-timing + risk-sizing + manuel context**;
otomatik **directional** engine DEĞİL. Bu, Part-1/Part-2 ön-değerlendirmesinin sayılarla doğrulanmış kapanışı.
Frozen TIDE/OVERLAYS dokunulmadı.
