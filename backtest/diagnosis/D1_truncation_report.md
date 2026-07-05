# D1 — VERİ SANSÜRÜ (truncation) teşhisi

## 1. Tarihsel chain yapısı — expiry sayısı / monthly-OPEX testi / DTE konvansiyonu

### SPY
- unique tarih: 243; unique expiration (tüm seri): **20**; tarih başına expiry: min 1 / med 1 / max 1 → **tarih başına TEK expiry** (100%)
- 3.-Cuma (weekday=Cuma & ay-3.-haftası) olan expiry: **12/20** (60%) → KARIŞIK (weekly dahil)
- expiry/yıl ≈ **20.3** (span 0.99y) → monthly-12 DEĞİL
- DTE **calendar-day** (expiration−date): min 0 / med **8** / max 25
- DTE **trading-day** (np.busday_count): min 0 / med **6** / max 19
- → 'DTE 0-25 med8' = **calendar-day** konvansiyonu (trading-day med=6)

  örnek expiry: 2025-06-20(wd4,hf3✓3F), 2025-07-18(wd4,hf3✓3F), 2025-07-31(wd3,hf5), 2025-08-15(wd4,hf3✓3F), 2025-08-29(wd4,hf5), 2025-09-19(wd4,hf3✓3F)

### QQQ
- unique tarih: 243; unique expiration (tüm seri): **20**; tarih başına expiry: min 1 / med 1 / max 1 → **tarih başına TEK expiry** (100%)
- 3.-Cuma (weekday=Cuma & ay-3.-haftası) olan expiry: **12/20** (60%) → KARIŞIK (weekly dahil)
- expiry/yıl ≈ **20.3** (span 0.99y) → monthly-12 DEĞİL
- DTE **calendar-day** (expiration−date): min 0 / med **8** / max 25
- DTE **trading-day** (np.busday_count): min 0 / med **6** / max 19
- → 'DTE 0-25 med8' = **calendar-day** konvansiyonu (trading-day med=6)

  örnek expiry: 2025-06-20(wd4,hf3✓3F), 2025-07-18(wd4,hf3✓3F), 2025-07-31(wd3,hf5), 2025-08-15(wd4,hf3✓3F), 2025-08-29(wd4,hf5), 2025-09-19(wd4,hf3✓3F)

## 3. BUGÜN full chain (yfinance, tüm vade) → toplam gamma$ + DTE-bucket + monthly dağılımı

### SPY  spot 725.43  (33 vade kullanıldı, ±15% bant, mid-IV)
- toplam gamma$ (yön-bağımsız Σ|γ·OI·100·S²·0.01|): **$33.81bn**  (toplam OI 10.33M)
- DTE-bucket: 0-1 $5.05bn (15%) | 2-5 $1.66bn (5%) | 6-21 $14.62bn (43%) | >21 $12.48bn (37%)
- monthly (3.-Cuma) $9.26bn (27%) vs non-monthly $24.55bn (73%)
- **TEK SAYI**: tarihsel-veri (tek-front-monthly $3.61bn) bugünkü toplam gamma$'ın **%11**'ini görüyordu  (kalan %89 sansürlü)

### QQQ  spot 693.69  (30 vade kullanıldı, ±15% bant, mid-IV)
- toplam gamma$ (yön-bağımsız Σ|γ·OI·100·S²·0.01|): **$17.56bn**  (toplam OI 7.50M)
- DTE-bucket: 0-1 $2.72bn (15%) | 2-5 $0.59bn (3%) | 6-21 $8.19bn (47%) | >21 $6.07bn (35%)
- monthly (3.-Cuma) $5.10bn (29%) vs non-monthly $12.46bn (71%)
- **TEK SAYI**: tarihsel-veri (tek-front-monthly $1.71bn) bugünkü toplam gamma$'ın **%10**'ini görüyordu  (kalan %90 sansürlü)

### Index chain (^SPX / ^NDX) — ④ havuz için
- ^SPX spot 7266.99 toplam gamma$ **$1.48bn** (27 vade)
- ^NDX spot 28508.03 toplam gamma$ **$9.21bn** (42 vade)

## 4. Havuz oranı — index (SPX/NDX) vs ETF (SPY/QQQ) gamma$

- ^SPX/SPY ham gamma$ oranı = **0.04×** (index $1.48bn / ETF $33.81bn, index-OI 107k vs ETF 10332k) — **ölçülemedi/ARTEFAKT** (index gamma$ ETF'in altında = yfinance index-OI eksik; gerçek SPX/NDX havuzu için CBOE/ORATS/SpotGamma OI lazım)
- ^NDX/QQQ ham gamma$ oranı = **0.52×** (index $9.21bn / ETF $17.56bn, index-OI 94k vs ETF 7497k) — **ölçülemedi/ARTEFAKT** (index gamma$ ETF'in altında = yfinance index-OI eksik; gerçek SPX/NDX havuzu için CBOE/ORATS/SpotGamma OI lazım)

## 5. Canlı snapshot bayrağı vs backtest level_series bayrağı uyumu

### SPY: 3 canlı snapshot, level_series'le örtüşen gün: 0
  2026-06-09: canlı net_gex -3.73bn (işaret -1) vs backtest regime örtüşme-yok → —
  2026-06-10: canlı net_gex -6.96bn (işaret -1) vs backtest regime örtüşme-yok → —
  2026-06-11: canlı net_gex -3.78bn (işaret -1) vs backtest regime örtüşme-yok → —
  → örtüşen gün YOK (canlı snapshot'lar level_series son tarihinden (2026-06-08) sonra) → ④ uyum ileriye-dönük ölçülür

### QQQ: 2 canlı snapshot, level_series'le örtüşen gün: 0
  2026-06-10: canlı net_gex -1.88bn (işaret -1) vs backtest regime örtüşme-yok → —
  2026-06-11: canlı net_gex -1.39bn (işaret -1) vs backtest regime örtüşme-yok → —
  → örtüşen gün YOK (canlı snapshot'lar level_series son tarihinden (2026-06-08) sonra) → ④ uyum ileriye-dönük ölçülür
