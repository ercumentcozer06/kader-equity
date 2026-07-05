# D7 — VERİ PAZARI MATRİSİ (TEŞHİS-ONLY)

Üreten script: `backtest/diagnosis/D7_market.py` (canonical venv ile koşuldu, gerçek çıktı).
Vendor olguları = 2026-06 WebSearch/WebFetch snapshot'ı (kaynaklar §5). Fiyat/kapsam
zamanla değişir → **as-of 2026-06** etiketli. Ölçülemeyen alan = `belirsiz` (tahmin yok).

## 1. AÇIK problemlerin repo-içi kanıtı (gerçek ölçüm, D7_market.py §A)

| problem | kanıt (md_spy / md_qqq, 243 gün, 2025-06-13→2026-06-08) | durum |
|---|---|---|
| ① truncation | `uniq_expiry=20`, **vade/gün med=1 max=1** → tarih başına TEK vade. D1: tek-front-monthly = bugünkü toplam gamma$'ın **%11 (SPY) / %10 (QQQ)** | **AÇIK** |
| ② sign | `buy/sell-open/close imza kolonu = YOK` (md yalnız OI). Dealer-envanter naive +call/−put proxy'sine mahkûm | **AÇIK** |
| ③ havuz | D1: SPX/SPY gamma$ **0.04×**, NDX/QQQ **0.52×** → SPX dealer-pool'u neredeyse-tamamı index'te, ETF-only veri kaçırıyor | **AÇIK (SPX'te ağır)** |
| ④ canlı-uyum | canlı 5-vade snapshot vs backtest 1-vade; D1: örtüşen-gün=0 (canlı tarihler level_series sonrası) → ileriye-dönük ölçülür | **ileri-ölçüm** |
| ⑤ IV-from-mid | `iv var=False, delta var=False` → IV/greeks YOK, mid'den türet zorunlu (deep-ITM/penny-mid instabilitesi) | **AÇIK** |

**ÖNCELİK** (görevdeki kural): **②sign > ①all-expiry > ③havuz** — işaret yanlışsa truncation/havuz düzeltmesi anlamsız.

## 2. ② SIGN katmanı (EN YÜKSEK ÖNCELİK) — dealer-envanter

| vendor | kapsam (sembol/expiry/yıl) | PIT | format | fiyat (as-of 2026-06) | çözer |
|---|---|---|---|---|---|
| **Cboe Open-Close Volume Summary** (DataShop/LiveVol) | TÜM Cboe-borsa serileri; **SPX index (C1 birincil) + SPY ETF**; tüm-vade | **YÜKSEK** (borsa-birincil; EOD overnight / intraday +15dk; look-ahead yok) | EOD özet **veya** intraday 1-dk/10-dk CSV; satır = participant-type(customer/pro-cust/broker-dealer/MM) × buy/sell × open/close, customer ayrıca contract-size | **fiyat-belirsiz** (sayfa basmaz; SEC 'LiveVol Fees'; teklif). EOD özet ucuz uçta | **② KANONİK** — customer buy/sell-to-open vs close = dealer-envanter rekonstrüksiyonunun doğrudan girdisi |
| ThetaData (Standard/Pro tick) | OPRA tüm; SPY/QQQ+index; tick trade+quote | YÜKSEK (zaman-damgalı tick; sign sınıflama KULLANICIDA) | REST/stream tick; greeks+IV per-contract | **$40 Value / $80 Standard / $160 Pro** aylık (4/8/12y) | ② **dolaylı** (quote-rule/Lee-Ready ile buy/sell) + ① bonus |
| Polygon.io / Massive | tüm US; tam zincir+tick | YÜKSEK (tick; greeks self-compute) | REST+WS+flat-file | **$29 / $79(opt) / $199** aylık; geçmiş **2–4y (sığ)** | ② dolaylı + ① ama uzun-backtest zayıf |
| Cboe LiveVol intraday Trades/Quotes | OPRA; SPX+SPY+QQQ; çok-yıllık tick | YÜKSEK (trade-condition ID map) | tick trade + 1-dk quote interval | **fiyat-belirsiz** (teklif) | ② için en temiz tick (DIY; Open-Close zaten signed verir) |

**Not (②):** Open-Close = "cevap hazır" signed-flow. ThetaData/Polygon/LiveVol-tick = aynı şeyin
trade-classification ile DIY'ı (Lee-Ready/quote-rule kullanıcı tarafında).

## 3. ① ALL-EXPIRY (truncation) katmanı — full-chain EOD

| vendor | kapsam | PIT | format | fiyat (as-of 2026-06) | çözer |
|---|---|---|---|---|---|
| Cboe DataShop **Option EOD Summary** / Open-Close OI | tüm-vade EOD OI+vol+OHLC; Calcs add-on=IV+greeks; SPX+SPY+QQQ; **2005+** | YÜKSEK | günlük CSV tüm-vade | **fiyat-belirsiz** (teklif) | ① — md %10-11'i görüyordu, kalan %89-90'ı geri getirir |
| **ORATS** | 5000+ sembol; tüm-vade EOD **2007+** + 1-dk **2020+**; 98 gösterge | YÜKSEK (smoothed quotes) | API EOD/1-dk; **greeks+IV hazır** | **$99/ay birey, $299/ay pro** (+$50 RT) as-of 2026-04 | ① + **⑤** (IV/greeks hazır→mid-IV bypass). İmza YOK |
| **OptionMetrics IvyDB US** (WRDS, akademik) | tüm US equity+index opt; **1996+** (her opsiyon/gün); SPX/SPY/QQQ+3000 underlying | **ÇOK YÜKSEK** (akademik altın-standart) | WRDS flat; günlük tüm-vade | **fiyat-belirsiz** (kurumsal/akademik WRDS; birim yayınlanmaz) | ① + ⑤; EN UZUN temiz geçmiş. İmza YOK; erişim kuruma-bağlı |
| iVolatility / HistoricalOptionData.com | iVol 20y; HistOptData **SPX 1990+ (35y)** aylık | YÜKSEK (EOD) | iVol pay-per-use; HistOptData satın-al CSV | iVol **pay-per-usage** (perakende-düşük, sabit-belirsiz); HistOptData 'Bloomberg'in kesri' (belirsiz) | ① + ⑤; perakende-ucuz uzun SPX. İmza YOK |

## 4. ③ index-havuz + DEALER-POSITIONING vendor (hazır-GEX, metodoloji şeffaflığı kritik)

| vendor | kapsam | PIT | format | fiyat (as-of 2026-06) | çözer / uyarı |
|---|---|---|---|---|---|
| **③ Index chain ^SPX/^NDX** (Cboe/ORATS/IvyDB/Theta hepsi taşır) | native index options (ETF değil); IvyDB 1996/Cboe 2005/ORATS 2007 | YÜKSEK | ek-sembol, taşıyıcı fiyatına dahil | dahil | **③** — havuz 0.04×/0.52× → SPX sinyali index-havuzda; ETF-only kaçırır |
| **SqueezeMetrics** (DIX+GEX) | 4500+; S&P GEX+DIX; **2011+** | YÜKSEK (~05:30 master-spreadsheet gün-sonrası) | CSV/spreadsheet+API; curl-endpoint YOK | **$720/ay** ($8,640/yıl) | ② **kısmen** (GEX=dealer-net-gamma proxy); metodoloji **YARI-ŞEFFAF** (2017 white-paper, formül kapalı) |
| **SpotGamma** (GEX/HIRO/levels) | SPX/SPY/QQQ+tek-isim; dealer-pano | **ORTA** (canlı pano; tarihsel-export zayıf-belgeli) | tarayıcı panosu; **API-first DEĞİL** | **$89/$99/$129/$299/$1,999+** aylık | ② kısmen (HIRO=sıralı-flow) ama **KARA-KUTU + tarihsel-zayıf → backtest-girdi DEĞİL**, canlı-overlay uygun |
| **MenthorQ** (gamma levels) | tüm-zincir GEX+0DTE; index+fut+ETF | ORTA (canlı; tarihsel belgeli-değil) | pano/seviye | **fiyat-belirsiz** | ② kısmen; **KARA-KUTU + tarihsel-belirsiz → backtest-girdi değil** |
| **Tier1Alpha** (daily GEX models) | günlük GEX, IV-ranges, MBAD; index | ORTA (haftalık rapor+günlük model; PIT-export belgeli-değil) | web/app+webcast | **fiyat-belirsiz** (kurumsal) | ② kısmen (türetilmiş); **KARA-KUTU+tarihsel-belirsiz → araştırma/teyit katmanı** |

**Hazır-GEX vendor sıralaması (backtest-girdi için):** SqueezeMetrics (yarı-şeffaf+2011-seri+CSV) >
SpotGamma/MenthorQ/Tier1Alpha (kara-kutu + tarihsel-seri belirsiz → backtest için riskli;
canlı-overlay/teyit için uygun). **Hiçbiri ham-imza vermez** → ② kanonik çözümü Cboe Open-Close.

## 5. Sonuç / öneri (öncelik sırasıyla)

1. **②sign (kanonik) = Cboe Open-Close Volume Summary.** SPX(C1)+SPY kapsar, 2005+ EOD /
   2019+ 1-dk, PIT-yüksek, customer buy/sell-to-open/close → gerçek dealer-envanter. **Fiyat
   teklif-bazlı (belirsiz)** ama akademik referanslarda EOD özet ucuz. Bu, naive +call/−put
   proxy'sinin (squeeze'le ~%50 örtüşür hipotezi ②) doğrudan ikamesidir.
2. **①all-expiry** zaten ② alındığında çoğu vendor full-chain taşır; en uzun temiz = **IvyDB 1996+**
   (kurumsal-erişim), perakende-pratik = **ORATS 2007+ ($99/ay, IV+greeks hazır → ⑤ de çözer)**.
3. **③havuz** = native **^SPX/^NDX** sembolünü eklemek (taşıyıcı fiyatına dahil); SPX'te
   0.04× oranı yüzünden ZORUNLU, NDX'te 0.52× ile yarı-kritik.
4. **Hazır-GEX vendor** (SpotGamma/Menthor/Tier1) backtest-GİRDİSİ olarak ÖNERİLMEZ
   (kara-kutu+tarihsel-belirsiz); SqueezeMetrics yarı-şeffaf+2011-seri ile tek istisna.

## Kaynaklar (WebSearch/WebFetch, 2026-06)
- Cboe Open-Close: datashop.cboe.com/cboe-options-open-close-volume-summary ; .../cboe-open-close-volume-summary-subscription
- Cboe EOD/Trades/Quotes: datashop.cboe.com/{option-eod-summary, option-trades, option-quote-intervals}
- ThetaData: thetadata.net/pricing
- Polygon/Massive: polygon.io/pricing?product=options
- ORATS: orats.com/data-api ; orats.com/intraday-data-api
- OptionMetrics IvyDB: optionmetrics.com ; wrds-www.wharton.upenn.edu/.../optionmetrics
- iVolatility / HistoricalOptionData: ivolatility.com/historical-options-data ; historicaloptiondata.com
- SqueezeMetrics: squeezemetrics.com/monitor/{dix,plans,docs}
- SpotGamma: spotgamma.com/subscribe-to-spotgamma ; support.spotgamma.com
- MenthorQ: menthorq.com/guide/* ; Tier1Alpha: tier1alpha.com
- Karşılaştırma: flashalpha.com/articles/options-data-pricing-comparison-... ; .../best-options-data-apis-2026
