# KADER-EQUITY — GEX DIRECTIONAL — TEŞHİS RAPORU (D-FAZ)

## 1. YÖNETİCİ ÖZETİ

| hipotez | doğrulandı-mı (SAYI/kanıt) | şiddet | mevcut-veriyle-çözülür-mü | değilse-ne-lazım | tahmini-maliyet |
|---|---|---|---|---|---|
| ① truncation (monthly-only / tek-vade) | **DOĞRULANDI (kısmi)** — tarih başına %100 tek-expiry (med=1/max=1, 20 unique expiry); backtest bugünkü toplam gamma$'ın yalnız %11 SPY / %10 QQQ'sünü görüyor (≈%89-90 sansür). Zincir saf-monthly DEĞİL (%60 3.-Cuma + weekly; DTE 0-25 med-8). Kök-neden = `marketdata_backfill.py:42` `expiration` param omit (default=sonraki-monthly) | **kritik** | **EVET (bedava)** — historical = 1-kredi/1000-sembol; `expiration=all&to=` ile ≤35DTE ~5 kredi/gün ×243 = ~1215 kredi (free 100/gün → ~13 gün; yüksek-plan token'da tek-oturum) | — (yalnız re-backfill kodu) | ~0$ (MarketData free-tier yeter) |
| ② sign (naive call-long/put-short) | **DOĞRULANDI kod-gerçeği** (`net_gex` sgn=+C/−P, `build_level_series.py:66` / `gamma_engine.py:127`); md'de buy/sell-open/close imza kolonu YOK → naive proxy'ye mahkûm. SqueezeMetrics agreement üst-tercile büyük-\|gex\| günlerinde %30 SPY / %49 QQQ'ye ÇÖKÜYOR (alt-tercile %61/%63). En yüksek-\|net_gex\| günler (2026-03-20 −1.8e10, 2025-08-29 −1.2e10, 2026-05-15 −8e9) bizde güçlü-negatif, squeeze'de pozitif | **kritik** | **HAYIR** — ham buy/sell imzası repoda yok | Cboe Open-Close Volume Summary (SPX+SPY, 2005+, PIT-yüksek), VEYA DIY trade-classification (ThetaData / Polygon / LiveVol-tick) | Cboe Open-Close fiyat-belirsiz (teklif-bazlı, EOD-özet ucuz uçta); ThetaData $40-160/ay; Polygon $29-199/ay (geçmiş 2-4y sığ) |
| ③ havuz (ETF SPY/QQQ vs index SPX/NDX) | **ÖLÇÜLEMEDİ — yfinance index-OI eksik** (^SPX front-OI=0; ^NDX 2642 = QQQ'nun %1.5'i). Ham oran SPX/SPY 0.04×, NDX/QQQ 0.52× ama fiziksel-olanaksız (index < ETF) = artefakt | **orta** (SPX'te ağır) | **HAYIR** — yfinance ile ölçülemez | MarketData index-chain (^SPX/^NDX serviste var) / CBOE / ORATS / SpotGamma index-OI | taşıyıcı-vendor fiyatına dahil (MarketData index-chain free-tier; ORATS $99-299/ay) |
| ④ canlı-uyum (canlı 5-vade vs backtest 1-vade) | **DOĞRULANDI kod-gerçeği** (`gamma_engine.py:38` N_EXP=5 vs `build_level_series.py:48` tek-expiry; kod `:9-10` açıkça flag'liyor). Doğrudan örtüşme-% ÖLÇÜLEMEDİ — canlı snapshot'lar (gamma_spy 06-09/10/11, gamma_qqq 06-10/11) level_series bitiminden (06-08) SONRA, örtüşen-gün=0. Dolaylı: ②'deki üst-tercile çöküşü (%30/%49) bu uyumsuzluğun imzası | **kritik** | **KISMEN (ileriye-dönük)** — backfill'i 06-08 ötesine taşı VEYA canlı snapshot biriktir | forward 5-expiry-matched net_gex serisi (kütüphanede yok) | ~0$ (re-backfill + forward-collect; mevcut feed yeter) |
| ⑤ IV-from-mid instabilite | **KISMEN DOĞRULANDI** — üretim IV-None drop gerçek-spot'la SPY %6.7 / QQQ %9.1 (strike-median proxy'deki %35 artefakt); DTE0/1/2'de %32-37'ye fırlıyor. Penny-mid (≤$0.05) KORUNAN kontratların SPY %17.9 / QQQ %13.6'sı (ham havuzda %28.5/%24.7). AMA atm_iv dağılımı temiz (<0.02 ya da >2.0 outlier SIFIR) → "kirli-IV level'a sızdı" kolu REDDEDİLDİ; "noisy-but-clamped" var. md'de ham bid/ask YOK → gerçek spread/crossed-quote ÖLÇÜLEMEDİ (yalnız mid<0.05 proxy) | **orta-düşük** | **KISMEN** — instabilite-zemini ölçüldü; gerçek bid-ask spread ölçülemez | md parquet'te ham bid/ask kolonları (şu an yalnız mid). ORATS/IvyDB = hazır greeks+IV (⑤'i kapatır) | bid/ask için re-collect (MarketData mid-only); ORATS $99-299/ay |
| ⑥ confound (vol / OPEX-sawtooth / trend) | **DOĞRULANDI** — '%72 örtüşme'nin tamamı gamma değil: phi(flag, düşük-vol) +0.44/+0.45 (taban %50→%72'yi açıklıyor). En güçlü confound = **TREND** (corr flag~trail20g +0.467 SPY / +0.493 QQQ; +γ neredeyse hiç düşüş-trendinde yok). VIX (QQQ phi +0.32) > OPEX-DTE sawtooth (\|Δnet_gex\| var expiry-haftası ×1.9; gamma$'ın ~%30'u DTE≤2). KAPANIŞ: sign-flag confound'larla %78-79 doğrulukla tahmin (taban+23pp); \|net_gex\| büyüklüğün ~%72-74'ü gamma-özgü kalan | **kritik** | **KISMEN** — confound-decompose yapıldı; çoklu-rejim ayrımı yapılamaz | >236 gün / 2018Q4+2020+2022 stres içeren çoklu-rejim pencere | veri-derinliği (D7 vendor'larından tarihsel-seri) |
| ⑦ istatistik (güç / best-of-K) | **DOĞRULANDI** — mevcut N'de SPY t=+1.24/DSR=0.387, QQQ t=+0.84/DSR=0.233 (ikisi de eşik-altı; +1.29/+0.87 Sharpe K=10 null max'ını aşmıyor). Koşulsuz-günlük directional iddia için min N: SPY 605g (~2.4y, eksik ~1.5y), QQQ 1331g (~5.3y, eksik ~4.4y). Wall-touch event-edge'ler TEZİN TERSİ (+γ MR ≈ −20bps = duvar kırılıyor). Tek-rejim 0.92y → tüm gereken-N ALT-SINIR | **kritik** | **HAYIR** — veri yetmiyor | SPY ≥2.4y / QQQ ≥5.3y günlük çoklu-rejim VEYA intraday-event yoğunlaştırma | veri-derinliği + zaman (forward-biriktirme) |

---

## 2. KARAR MATRİSİ

Öncelik D7-notuyla: **②sign > ①all-expiry > ③havuz** (işaret yanlışsa gerisi anlamsız; truncation işareti de çevirir; havuz büyüklük-ölçeği).

| sorun | çözüm | maliyet | ne-kazandırır (hangi SAYIyı düzeltir) |
|---|---|---|---|
| ② naive sign (üst-tercile agreement %30/%49) | Cboe Open-Close Volume Summary (SPX+SPY, 2005+) ile gerçek buy/sell-open/close imzası → dealer-envanteri imzadan kur | teklif-bazlı (EOD-özet ucuz uçta); alt: ThetaData $40-160/ay | naive +call/−put proxy'sini KANONİK imzayla değiştirir; ②'nin SqueezeMetrics-divergence'ını (%30/%49 → ?) doğrudan kapatır |
| ① truncation (gamma$'ın %89-90'ı sansürlü) | `marketdata_backfill.py:42`'ye `expiration=all&to=<≤35DTE>` ekle, re-backfill | ~0$ (free-tier; ~1215 kredi historical-1-kredi/1000-sembol) | backtest'in gördüğü gamma$'ı %11→~%100'e taşır (SPY $3.61bn→$33.81bn brüt); ④'ün de zeminini düzeltir |
| ③ havuz (yfinance index-OI=0) | MarketData index-chain (^SPX/^NDX) ile gerçek index-OI çek | taşıyıcı-vendor fiyatına dahil (free-tier) | SPX/SPY 0.04× artefaktını gerçek-orana çevirir; SPX-havuzu ETF'ten büyük olmalı |
| ④ canlı-uyum (örtüşen-gün=0) | backfill'i 06-08 ötesine + 5-expiry-matched serisi üret; canlı snapshot biriktir | ~0$ | canlı 5-vade ↔ backtest 1-vade örtüşme-%'sini ÖLÇÜLEBİLİR yapar; ②'deki üst-tercile çöküşün ④-kaynaklı olduğunu doğrular/çürütür |
| ⑤ IV-from-mid (penny %14-18, DTE0/1/2 fail %32-37) | md re-collect'te ham bid/ask sakla; VEYA ORATS/IvyDB hazır-greeks | re-collect ~0$; ORATS $99-299/ay | gerçek bid-ask spread + crossed-quote ölçümünü açar (şu an proxy); DTE≤2 instabilitesini kantifiye eder |
| ⑥ confound (trend corr +0.47/+0.49; sign %78-79 tahmin-edilebilir) | çoklu-rejim pencere (2018Q4+2020+2022) ile trend/vol-orthogonalize | veri-derinliği (D7) | sign-flag'in confound-üstü gamma-özgü içeriğini izole eder (şu an in-sample) |
| ⑦ istatistik (t=1.24/0.84, DSR<0.5) | veri-derinliği (SPY ≥2.4y / QQQ ≥5.3y çoklu-rejim) VEYA intraday-event | veri + zaman | t≥2 / DSR>0 eşiğine taşır; best-of-K seçim-yanlılığını kırar |

---

## 3. TEŞHİS DETAYLARI (D0-D7)

### D0 — KOD GERÇEĞİ RAPORU (kader-equity, satır-referanslı)

**Scriptler (canonical-venv ile koşuldu, gerçek çıktı):**
- `C:/Users/admin/Downloads/kader-equity/backtest/diagnosis/D0_probe.py` (mtime/hash forensics)
- `C:/Users/admin/Downloads/kader-equity/backtest/diagnosis/D0_data_probe.py` (chain içerik gerçeği)
- `C:/Users/admin/Downloads/kader-equity/backtest/diagnosis/D0_truedrop.py` (gerçek-spot IV-drop)

Repo **git DEĞİL** (`fatal: not a git repository`) → hüküm `os.path.getmtime`+`sha256` ile verildi.

#### 1. (a) IV bisection fail-policy — DROP (clamp/default DEĞİL)

`screen/_bsiv.py::implied_vol` (satır 33-45) fail edince **`None` döner**, kontrat çağıran tarafından **DÜŞÜRÜLÜR**. Clamp yok, default yok, ham-yahoo-IV'ye fallback yok.

- **İki fail-kapısı, satır-referanslı:**
  - `_bsiv.py:35` — girdi-guard: `price is None or price<=0 or T<=0 or S<=0 or K<=0` → `None`.
  - `_bsiv.py:45` — **range-clamp-as-reject**: `return iv if 0.02 < iv < 2.9 else None`. Bisection [lo=0.01, hi=3.0] 60-iter koşar (satır 37-44); sonuç [0.02, 2.9] dışındaysa (deep-ITM intrinsic-pegli mid, ya da uçuk mid) **DROP**.
- **Tüketici tarafı DROP'u (satır-referans):**
  - `build_level_series.py:58-60` — `iv = implied_vol(...); if not iv or iv<=0: continue` (kontrat atlanır).
  - `gamma_engine.py:116-117` — `if iv is None or iv<=0: continue`.
  - `mid_iv_from_row` (`_bsiv.py:48-53`) — `bid/ask` yok ya da ≤0 → `None` (yfinance yolu; backfill-parquet yolu `mid`'i doğrudan `implied_vol`'a verir).
- **Üretim DROP-oranı (D0_truedrop.py, GERÇEK Alpaca-spot + gerçek band):** band-içi kontratların **SPY %6.7 (4.570/68.500), QQQ %9.1 (5.396/59.548)** `iv=None` ile düşüyor. (Not: `D0_data_probe.py`'deki %35, spot=strike-medyan PROXY artefaktı — gerçek-spot'la %6-9.)
- **`<4-strike` ikincil-drop:** `build_level_series.py:63-64` `if len(rows)<4: return None`. Gerçekte **0 gün** bu yüzden düştü (D0_truedrop). 243 chain-günün **236'sı** level_series'e girdi; eksik 7 gün `<4` değil, **Alpaca günlük-spot eksikliği** (`build_level_series.py:114` `if dd not in spot.index: continue`).

#### 2. (b) bid=0 / crossed-quote handling — md_parquet'te HAM bid/ask YOK

- **D0_data_probe.py çıktısı:** `md_{spy,qqq}.parquet` kolonları = `date,expiration,strike,right,open_interest,iv,delta,mid`. **Ham `bid`/`ask` kolonu YOK** (`has_bidask=False`). `mid`, MarketData server-tarafı mid (`marketdata_backfill.py:58` `"mid": col("mid")` — API'nin `j["mid"]` alanı, yerel bid/ask'tan türetilMİYOR).
- **Sonuç:** `bid=0`/`bid>ask` kontrolü backtest borusunda **yapılMIYOR ve yapılAMIYOR** — ham quote yok. `build_level_series.py:53` yalnız `mid is None or mid<=0` eler. **mid≤0 ya da NaN: SPY 0 / QQQ 0 satır** (tüm 117.712 SPY + 101.280 QQQ satırında `mid>0`, min 0.0050).
- **Canlı (yfinance) yolu FARKLI:** `_bsiv.py:50-53 mid_iv_from_row` ham `bid`/`ask` kullanır, `not bid or not ask or bid<=0 or ask<=0 → None`. Yani **bid=0 yalnızca CANLI motorda** ele alınır; **crossed-quote (bid>ask) HİÇBİR yolda** açıkça reddedilmez (negatif mid çıkmaz çünkü `(bid+ask)/2` yine pozitif → `implied_vol` range-clamp'ine bırakılır).
- **mid üretimi (3 yol):** ① backtest = parquet `mid` direkt (`build_level_series.py:52`). ② canlı yfinance = `(bid+ask)/2` (`_bsiv.py:53`). ③ MarketData backfill = API `mid` alanı (`marketdata_backfill.py:58`).
- **Penny-mid instabilite (hipotez ⑤ için):** band-içi **KORUNAN** kontratların **SPY %19.2 (12.279), QQQ %15.2 (8.251)** penny-mid (0<mid≤0.05). Ham havuzda penny-mid oranı daha da yüksek: SPY %28.5, QQQ %24.7. Penny-mid'de bisection ±$0.005 mid-gürültüsüne IV-hassas → **hipotez ⑤ (penny-mid/IV-from-mid instabilite) GERÇEK ZEMİNE oturuyor** (ama ölçülen drop düşük; instabilite "drop" değil "noisy-but-kept" biçiminde).

#### 3. (c) net_gex / call_wall / put_wall / flip — birebir kod-gerçeği

Backtest (`build_level_series.py`) ve canlı (`gamma_engine.py`) **aynı formül** (byte-eş `_greeks`, `gamma_engine.py:46-57`):

- **net_gex** (`build_level_series.py:65-66` / `gamma_engine.py:126-127`):
  `sgn(r)=+1 if "C" else −1`; `net_gex = Σ sgn·g·oi·M·S²·0.01`, M=100. **Naive-dealer konvansiyonu (call-long/put-short)** — yani hipotez ②'nin tam tarifi. `regime = 1 if net_gex≥0 else −1` (`build_level_series.py:103`).
- **flip (zero-gamma)** (`build_level_series.py:68-79` / `gamma_engine.py:133-146`): `net_g_at(hs)` hypothetical-spot'ta net-gamma'yı yeniden hesaplar; `SCAN=linspace(−0.06,0.06,13)` ±%6/13-nokta grid; ilk işaret-değiştiren komşu çiftte **lineer interpolasyon** `s0+(s1−s0)·(0−g0)/(g1−g0)` (`:78` / `:145`).
- **call_wall** (`build_level_series.py:86-87, 92`): **HAM-OI tepesi** — yalnız `K≥S` call'lar, `call_oi[K]+=oi`, `max(call_oi, key=oi)`. (gamma-ağırlık DEĞİL; `gamma_engine.py:158-162` aynı.) `gamma_engine.py:154-156` yorumu kritik: eski "call_wall" aslında GHOST'tu, şimdi **call_wall=raw-OI peak, ghost=gamma-peak** ayrımı yapıldı.
- **put_wall** (`build_level_series.py:88-89, 93`): **GAMMA-notional tepesi** — `K≤S` put'lar, `by_k_put[K]+=g·oi`, `max(by_k_put, key=g·oi)`. (call_wall ham-OI ama put_wall gamma — **asimetrik tanım**, `gamma_engine.py:163` "Emir-eşleşti" notu.)
- **ghost** (`:91`): `K≥S` call gamma-peak. **hvl** (`:90,94`): `max |g·oi|` (call+put, yön-bağımsız). **max_pain** (`:95-100`): standart pain-min.

#### 4. (d) Canlı N_EXP=5 vs backtest 1-vade — satır-referanslı fark

- **Canlı:** `gamma_engine.py:38` `N_EXP=5`; `:100-101` `exps=sorted(...)[:N_EXP]` → **ön 5 expiry**. `net_gex` (`:127`) 5-vadenin TÜM kontratlarını toplar (vade-ağırlığı YOK — düz toplam, her kontrat kendi T'siyle greekslenir, `:104-105` `T=max(dte,0.5)/365`). flip-scan (`:135-140`) 5-vadeyi hypothetical-spot'ta yeniden toplar.
- **Backtest:** `build_level_series.py:48` `dte=(expiration−date).days` **TEK** expiry; `_levels_for_day` (`:46`) `g["expiration"].iloc[0]` tek-T. **D0_data_probe teyidi: per-day expiry nunique = min1/med1/max1** (MarketData-free tarih-başına-1-expiry sınırı), DTE med 8 (0-25).
- **Fark = ağırlıklama DEĞİL, KAPSAM:** canlı 5-vade-toplam, backtest 1-vade (front). İkisi de düz-toplam (term-weight yok). `build_level_series.py:9-10` bunu açıkça flag'liyor: *"front-expiry-ONLY (canlı N_EXP=5) → seviyeler canlıyla term-structure açısından TUTARSIZ olabilir"*. → **Hipotez ④ (canlı-uyum) KOD-DOĞRULANDI: backtest level'ları canlı motorla term-structure açısından uyumsuz; forward-revalidate gerekiyor.** spine_diagnostic/disentangle/block_robust **hepsi bu 1-vade level_series'i okur** (`spine_diagnostic.py:47`), canlı 5-vade GEX'i değil.

#### 5. (e) PIT hizalama — D-level → D+1 session (satır-referanslı)

`spine_diagnostic.py::build_panel` (satır 45-64):
- `lv` = `level_series` (D-EOD seviyeleri, index=D), `rth` = D+1 RTH OHLC (`daily_rth`, `:31-42`, ET 09:30-16:00, `tz_convert("America/New_York")`).
- **Eşleme:** `:51` her `D in lv.index` için → `:54` `nxt=[s for s in sess if s>D]`, `:56` `N=nxt[0]` (**D'den SONRAKİ ilk seans**). `:58` `c0=rth.loc[D,"c"]` (D-kapanış), `:59` `o1,h1,l1,c1=rth.loc[N,...]` (D+1 OHLC). `gap=o1/c0−1`, `intraday=c1/o1−1` (`:62`).
- **Look-ahead temiz:** D-EOD seviyesi (OI=EOD) → D+1 seansta trade. `block_robust.py:9` özeti: `net_gex[D] + gap[D+1-open] → intraday[D+1]`. **Sızıntı yok** (level D'de hesaplanmış, D+1'de kullanılmış). D0_data_probe teyidi: level-index ilk-3 `2025-06-13/06-16/06-17`, son-3 `2026-06-04/06-05/06-08`.

#### 6. HÜKÜM: level_series, IV-mid-fix'ten ÖNCE mi SONRA mı? → **SONRA (kesin)**

D0_probe.py mtime/hash (repo git-değil → getmtime hükmü):

| dosya | mtime (local) | sha16 |
|---|---|---|
| `_bsiv.py` (canonical IV-mid) | **2026-06-10 04:20:24** | 2daf2aec8579eafd |
| `gamma_engine.py` | 2026-06-11 01:07:51 | d92dc8fc495a84a3 |
| `build_level_series.py` | 2026-06-11 03:48:21 | 9887b8cb04168ee7 |
| `level_series_spy.parquet` | **2026-06-11 03:48:42** | 656a93d016038379 |
| `level_series_qqq.parquet` | **2026-06-11 03:48:55** | 0ceda4a1e5078fb9 |

- **level_series − _bsiv = +23.47 saat (SPY) / +23.48 saat (QQQ)** → level_series, `_bsiv.py` canonical-fix'ten **23.5 saat SONRA** üretildi. build_level_series'ten yalnız +0.01 saat sonra (aynı build koşusunun çıktısı).
- **İçerik teyidi (mid-IV imzası):** `atm_iv` SPY med 0.1390 (min 0.0208/max 0.3427), QQQ med 0.1954 (min 0.0211/max 0.5092); **<0.02 ya da >2.0 outlier SIFIR**. Ham-yahoo-IV'nin tipik %37-56 saçma-front-IV imzası YOK → level_series **temiz mid-IV ile üretilmiş**. **Hipotez ⑤'in "kirli-IV level'a sızdı" kolu REDDEDİLDİ** (instabilite penny-mid'de var ama clamp+median bunu massediyor; atm_iv dağılımı sağlıklı).

**Hipotez hükümleri (D0'ın değdiği kadarıyla):**
- **① truncation (monthly-only?):** REDDEDİLDİ kısmen — per-day expiry med=1 ama DTE 0-25 med-8 (haftalık dahil, monthly-only DEĞİL); tek-vade sınırı VAR.
- **② sign (naive call-long/put-short):** DOĞRULANDI kod-gerçeği (`net_gex` `sgn=+C/−P`, `:66`/`:127`) — ölçüm D2'de.
- **④ canlı-uyum (5-vade vs 1-vade):** DOĞRULANDI (N_EXP=5 vs tek-expiry; kod açıkça flag'liyor).
- **⑤ IV-from-mid instabilite:** KISMEN — penny-mid %15-19 (kept) gerçek instabilite-zemini; ama üretim IV-drop düşük (%6.7/9.1) ve atm_iv dağılımı temiz → "level'a sızan kirlilik" yok, "noisy-but-clamped" var.
- ③/⑥/⑦ bu D0'ın kapsamı dışı (pool/confound/istatistik = D1+ işi).

---

### D1 — VERİ SANSÜRÜ (truncation): kanıt/ret

**Script:** `C:/Users/admin/Downloads/kader-equity/backtest/diagnosis/D1_truncation.py` (canonical venv ile koşuldu, gerçek çıktı; rapor `backtest/diagnosis/D1_truncation_report.md`'ye de yazıldı)

#### 1. Tarihsel chain yapısı — expiry / monthly-OPEX / DTE konvansiyonu
Kaynak: `D1_truncation.py` §1, `md_{spy,qqq}.parquet`.
- **SPY + QQQ (ikisi de aynı):** 243 unique tarih, 20 unique expiration (tüm seri), tarih başına expiry **min 1 / med 1 / max 1 → %100 tarih başına TEK expiry**. ① doğrulandı: tarihsel veri tek-vade.
- **3.-Cuma testi (weekday=Cuma & ay-3.-haftası):** 20 expiry'nin **12'si (%60)** 3.-Cuma monthly; kalan 8 weekly/ay-sonu (ör. 2025-07-31 wd3-hf5, 2025-08-29 wd4-hf5). → **"monthly-only" hipotezi KISMEN RET**: zincir saf-monthly değil, ama her tarih için seçilen expiry o tarihe en yakın olan (MarketData default = "date'e göre sonraki monthly" → çoğu zaman 3.-Cuma, OPEX yakınında bir sonraki).
- **expiry/yıl ≈ 20.3** (span 0.99y) → 12-monthly DEĞİL (weekly'ler karışmış).
- **DTE konvansiyonu (iki kolon):** calendar-day (expiration−date) min 0 / **med 8** / max 25; trading-day (np.busday_count) min 0 / **med 6** / max 19. → **"DTE 0-25 med8" = calendar-day konvansiyonu** (trading-day med=6).

#### 2. MarketData.app — default vade / param / free-tier (docs + AMPİRİK)
Docs (marketdata.app/docs/api/options/chain, /account/plan-limits) + 3 ampirik ölçüm-çağrısı (token .env'den, ekrana basılmadı).
- **`expiration` param VAR.** Omit edilirse default = *"the next monthly expiration ... relative to the `date` parameter for historical quotes"* → **backfill script param'sız çağırdığı için TEK monthly-front vade alıyor.** ① ROOT CAUSE bu. `expiration=all` keyword'ü tüm zinciri verir; `to=` ile DTE-kırpılabilir.
- **AMPİRİK kanıt (tarih 2026-05-15):**
  - Param'sız (`date` only) → **1 expiry / 538 sembol, 1 kredi.** (backfill'in aldığı şey)
  - `expiration=all&to=2026-06-19` (≤~35 DTE) → **13 expiry / 4602 sembol, 5 kredi.** → param'sız çağrı sembollerin **%12'sini** alıyor.
- **Kredi muhasebesi (docs):** historical = **1 kredi / 1000 sembol** (real-time'ın 1-kredi/sembol'ünden FARKLI). Free-tier = **100 kredi/gün** (9:30 ET reset). [Not: bu .env token'ı daha yüksek planda — remaining 9500/10000 döndü, 100 değil.]
- **243 gün × tüm ≤30DTE-vade sığar mı:** ≤35DTE full chain ≈ 4600 sembol/gün = **5 kredi/gün × 243 = ~1215 kredi.** Free 100/gün → **~13 günde** (ya da yüksek-plan token'da tek-oturum) tamamlanır. **Tarihsel-1-kredi/1000-sembol sayesinde ① BEDAVA çözülebilir** — backfill'in "1 istek=1 kredi, 504 istek/5 gün" varsayımı YANLIŞ (gerçek maliyet sembol-bazlı, çok daha ucuz).
- **SPX/NDX index chain serviste VAR** (am/pm param'ları sadece index için). → kader-equity ETF yerine doğrudan index-OI çekebilir (③'ün gerçek-veri yolu MarketData, yfinance değil).

#### 3. BUGÜN full chain (yfinance, tüm vade, mid-IV) → toplam gamma$ + sansür-oranı
`D1_truncation.py` §3, yfinance KENDİ verisi, `_bsiv.mid_iv_from_row` (bid/ask MID), `gamma_engine._greeks`, ±15% bant. gamma$ = Σ|γ·OI·100·S²·0.01|.
- **SPY** (spot 725.43, 33 vade): toplam **$33.81bn** (OI 10.33M). DTE-bucket: 0-1 %15 / 2-5 %5 / 6-21 %43 / >21 %37. monthly %27 vs non-monthly %73.
  → **TEK SAYI: tarihsel-veri (tek-front-monthly $3.61bn) bugünkü toplam gamma$'ın %11'ini görüyordu (≈%89 sansürlü).**
- **QQQ** (spot 693.69, 30 vade): toplam **$17.56bn** (OI 7.50M). DTE-bucket: 0-1 %15 / 2-5 %3 / 6-21 %47 / >21 %35. monthly %29 vs non-monthly %71.
  → **TEK SAYI: tarihsel-veri (tek-front-monthly $1.71bn) bugünkü toplam gamma$'ın %10'unu görüyordu (≈%90 sansürlü).**
- **① KESİN DOĞRULANDI:** backtest level_series gamma$'ın yalnız ~%10-11'ini görüyor; gamma'nın %78-80'i tek-front-monthly DIŞINDA (6-21 + >21 DTE bucket'ları + non-monthly weekly'ler). Sansür gerçek ve büyük.

#### 4. Havuz oranı (index SPX/NDX vs ETF SPY/QQQ) — ③
- **^SPX/SPY** ham oran 0.04× (index $1.48bn / ETF $33.81bn), **^NDX/QQQ** ham oran 0.52× (index $9.21bn / ETF $17.56bn).
- **İKİSİ DE ÖLÇÜLEMEDİ / ARTEFAKT:** index gamma$ ETF'in ALTINDA çıkıyor — bu fiziksel olarak olanaksız (SPX havuzu SPY'den büyük olmalı). Ayrı probe: **^SPX front-expiry total-OI = 0**, ^NDX front-OI 2642 (QQQ 173665'in %1.5'i); yfinance index-OI bid/ask kapsamı %33-62 (ETF %63-66). → **yfinance index-OI eksik/sıfır.** ③ havuz oranı yfinance ile ölçülemez; gerçek SPX/NDX gamma$ için **MarketData index-chain (§2'de var) / CBOE / ORATS / SpotGamma OI** lazım.

#### 5. Canlı snapshot vs backtest bayrak uyumu — ④
`D1_truncation.py` §5. Canlı snapshot'lar: gamma_spy 3 gün (06-09/10/11), gamma_qqq 2 gün (06-10/11). level_series son tarihi **2026-06-08**.
- **Örtüşen gün = 0** (tüm canlı snapshot'lar level_series bitiminden SONRA). Uyum-% hesaplanamadı.
- Yani **④ uyum ileriye-dönük ölçülür** — backfill'i 06-08'in ötesine taşıyıp ya da canlı snapshot biriktikçe ölçülecek.
- Yan-gözlem (P&L'e bağlanmaz): canlı 5-vade net_gex 3/3 SPY + 2/2 QQQ günü **negatif (SHORT GAMMA)**; SPY −3.7/−7.0/−3.8bn, QQQ −1.9/−1.4bn. Canlı motorun gördüğü gamma$ ölçeği (§3'te SPY $33.8bn brüt havuz) tarihsel tek-vade ölçeğinin ~10×'i — yani ④ uyumu ölçülünce truncation'ın bayrağı çevirip-çevirmediği test edilebilir.

**Hipotez kararları:**
- **① truncation (monthly-only?):** **DOĞRULANDI** (kısmi düzeltmeyle). Tarih başına tek-vade %100; backtest bugünkü gamma$'ın yalnız **%10-11**'ini görüyor (%89-90 sansür). Kök-neden = MarketData `expiration` param'ı omit edilmiş (default=sonraki-monthly). Zincir saf-monthly DEĞİL (%60 3.-Cuma + weekly'ler) ama her tarih TEK front-vade. **Bedava düzeltilebilir** (historical 1-kredi/1000-sembol; `expiration=all&to=` ile ≤35DTE ~5 kredi/gün).
- **③ havuz (ETF vs index):** **ÖLÇÜLEMEDİ** — yfinance index-OI eksik (^SPX front-OI=0). Gerçek oran için MarketData/CBOE/ORATS index-OI gerekiyor.
- **④ canlı-uyum (5-vade vs 1-vade):** **ÖLÇÜLEMEDİ (henüz)** — canlı snapshot'lar level_series penceresiyle örtüşmüyor; ileriye-dönük ölçülecek.

İlgili dosyalar: script `C:/Users/admin/Downloads/kader-equity/backtest/diagnosis/D1_truncation.py`; veri `data/historical_chains/md_{spy,qqq}.parquet`, `data/cache/level_series_{spy,qqq}.parquet`, `data/cache/gamma_{spy,qqq}/*.json`; kök-neden kodu `screen/marketdata_backfill.py` satır 42 (`params={"date": d, "token": token}` — `expiration` yok).

---

### D2 — REJİM BAYRAĞI SAĞLAMLIK BATARYASI

**Script:** `C:/Users/admin/Downloads/kader-equity/backtest/diagnosis/D2_sign_battery.py`
Canonical-venv (`C:/Users/admin/Downloads/kader-macro/.venv/Scripts/python.exe`) ile koştu, temiz çıktı. TEŞHİS-ONLY: V1–V5 yalnız SIGN serisi, P&L'e bağlanmadı; `gamma_inv_pnl` yalnız import+recompute (yeni varyant yok).

#### 1. Sign-flip-gün % (baseline = build_level_series net_gex işareti, n=236)

| Variant | SPY flip% | QQQ flip% | yorum |
|---|---|---|---|
| V1 penny-mid(<0.05) atılmış | %0.8 (2/236) | %0.8 (2/236) | **bayrak penny-mid'e DUYARSIZ** — md'de gerçek bid YOK, bu bir proxy |
| V2 IV winsorize [%5,%150] | %2.1 (5) | %1.3 (3) | IV-clamp'e duyarsız |
| V3 flat-ATM-IV (smile düzlet) | %13.1 (31) | %6.8 (16) | smile orta-etkili; bayrak IV-yapısına biraz bağlı |
| V4 pure-OI-balance (gamma'sız Σ±OI) | **%44.9 (106)** | **%50.4 (119)** | gamma-ağırlık olmadan işaret ~yazı-tura → gamma load-bearing |
| V5 DTE≤2 hariç (n=183) | %0.0 (0) | %0.0 (0) | **0DTE/1DTE/2DTE günleri işareti HİÇ çevirmiyor** |

Çekirdek sonuç: bayrak ölçüm-gürültüsüne (penny/IV/DTE) **SAĞLAM**; tek gerçek bağımlılık gamma-ağırlığın kendisi (V4, beklenen — V4 zaten farklı bir sinyal, ölçüm-hatası değil).

#### 2. FRAGILE-FLAG (≥2 variant baseline ile çelişiyor)
- **SPY: 5 gün / 236 (%2.1)** → 2025-06-20, 2025-07-15, 2025-09-09, 2025-11-21, 2026-02-27
- **QQQ: 5 gün / 236 (%2.1)** → 2025-10-30, 2025-12-12, 2025-12-16, 2025-12-18, 2026-02-27
- Dağılım: 0-çelişki ~%42 gün, 1-çelişki ~%56, 2-çelişki yalnız ~%2. **Tek günde 3+ variant çelişen YOK.**

#### 3. gamma_inv P&L (block_robust.gamma_inv_pnl RECOMPUTE) — holdout-tartışmasını kapatan sayı
- **SPY** full +1170bps / Sharpe +1.29; **QQQ** +1108bps / +0.87.
- **Holdout son-70g top-3 kazanç-günü: HER İKİ sembolde de 0/3'ü FRAGILE.** Top-3 günler (SPY 06-04/03-06/04-01; QQQ 06-04/03-06/06-08) hepsi 0–1 çelişkili (stabil).
- **Fragile-gün P&L payı tüm-örneklem: SPY %2 (+29bps/+1170), QQQ %3 (+36bps/+1108)** → kazanç fragile-günlerden GELMİYOR.
- Holdout-içi: tek fragile-gün düşüyor ve o gün NEGATİF (SPY −114bps = holdout'un −%18; QQQ −154bps = −%15) → fragile-günler kazandırmıyor, hafifçe kaybettiriyor. **Holdout-kazancı işaret-kırılganlığına bağlı DEĞİL.**

#### 4. SqueezeMetrics agreement (net_gex-sign vs squeeze-gex-sign, ortak n=236)
SqueezeMetrics ~%3 gün negatif (çoğu-zaman +GEX konvansiyonu; naive 'hep-+' agreement tavanı %97).

| Tercile (|net_gex|) | SPY agreement | QQQ agreement |
|---|---|---|
| alt (sıfır-yakını) | %61 | %63 |
| orta | %54 | %67 |
| **üst (büyük \|gex\|)** | **%30** | **%49** |

**Bu KÖK-REBUILD sinyali, deadband DEĞİL.** Deadband hipotezi alt-tercile'de düşük agreement bekler; tam tersi — agreement **büyük-|gex| günlerinde ÇÖKÜYOR**. Üst-tercile'de bizim net_gex SPY'da %80 / QQQ %57 NEGATİF iken squeeze pozitif. En yüksek |net_gex| 8 günün doğrulaması: 2026-03-20 (−1.8e10), 2025-08-29 (−1.2e10), 2026-05-15 (−8.0e9) bizde güçlü-negatif, squeeze'de pozitif.

**Hipotez kararları (sayıyla):**
- **① truncation (monthly-only?) → REDDEDİLDİ.** Tarih-başına TEK front-expiry; DTE 0–25, medyan 8 (haftalık+aylık karışık), monthly-only DEĞİL. Ama "tek-expiry"nin kendisi ④'ün kökü.
- **④ canlı 5-vade vs backtest 1-vade → DOĞRULANDI, ana divergence kaynağı.** Üst-tercile büyük-|gex| günleri OpEx/expiry-konsantrasyon günleri; tek-front-expiry band ±%15 düşen-spotta put-ağır → negatife dönüyor, oysa tam yüzey (squeeze, çok-vade) long-gamma kalıyor. Agreement %30/%49 buradan.
- **② naive sign (~%50 squeeze ile) → KISMEN.** Genel agreement SPY %48 / QQQ %60; ama bu konvansiyon-farkı değil, üst-tercile-çöküşünden (④) geliyor.
- **③ havuz (ETF SPY/QQQ band vs index SPX çok-strike) → ④ ile aynı kök** (band ±%15 + tek-expiry vs market-wide).
- **⑤ IV-from-mid instabilite → REDDEDİLDİ (önemsiz).** V1 %0.8, V2 %2.1, V3 %13 → işaret IV-ölçüm gürültüsüne sağlam; penny-mid %28.5 satır olmasına rağmen flip %0.8.

**Ölçülemeyen:** Gerçek bid/ask (md'de yok) → V1 sadece penny-mid<0.05 PROXY; gerçek crossed/bid=0 ayrımı için ayrı bid/ask veri lazım. Forward 5-expiry-matched net_gex serisi (kütüphanede yok) ④'ü doğrudan kapatmak için gerekli — şu an yalnız squeeze-proxy ile çıkarıldı.

---

### D3 — IV/GREEKS KALİTE RAPORU

**Script:** `C:/Users/admin/Downloads/kader-equity/backtest/diagnosis/D3_iv_quality.py`
**Çalıştırıldı:** kanonik venv (`C:/Users/admin/Downloads/kader-macro/.venv/Scripts/python.exe`) — GERÇEK çıktı, uydurma yok. DIAGNOSIS-ONLY: yeni P&L/strateji üretilmedi; `level_series_{sym}.parquet` SALT-OKUNUR; IV-invert çağrısı `build_level_series` ile byte-aynı (`_bsiv.implied_vol` + `gamma_engine._greeks`).

#### 0. Veri kısıtı (ölçüm öncesi, KANIT)
- `md_{spy,qqq}.parquet`'te `iv` ve `delta` kolonları **%100 BOŞ** → IV daima `mid`'den BS-invert ediliyor (bisection 60-iter, aralık `0.02<iv<2.9`).
- Kolonlarda **bid/ask YOK** (tek `mid`) → **bid-ask spread ÖLÇÜLEMEZ** ("md'de bid-yok" doğrulandı). `mid≤0` hiç görülmüyor (MarketData mid'i `0.005`'e tabanlıyor) → penny/instabil proxy = `mid<0.05`.
- Tarih başına **TEK expiry** (DTE 0-25, med 8) → "DTE≤2 payı" günün ya tüm gamma$'ı ya 0'ı.

#### 1. bisection-fail % (`implied_vol` None/≤0)
| | satır-ağırlıklı | gün medyan | gün ort | gün maks |
|---|---|---|---|---|
| **SPY** | 4570/68500 = **%6.67** | %0.32 | %6.41 | **%37.21** |
| **QQQ** | 5396/59548 = **%9.06** | %3.87 | %8.68 | **%37.30** |

En kötü 5 gün İSTİSNASIZ **DTE 0/1/2**'de (SPY: 2025-08-15 DTE0 %37, 2025-08-14 DTE1 %35; QQQ: 2026-04-17 DTE0 %37, 2025-10-30 DTE1 %36). → fail, kısa-DTE expiry günlerine kümeleniyor.

#### 2. penny / instabil-mid (`mid<0.05`)
| | satır-ağırlıklı | gün medyan | gün maks | mid≤0 |
|---|---|---|---|---|
| **SPY** | 12267/68500 = **%17.91** | %11.64 | %50.00 | %0.00 |
| **QQQ** | 8066/59548 = **%13.55** | %6.08 | %50.00 | %0.00 |

Penny satırlar çoğunlukla derin-OTM; IV'leri aralık-dışı → fail'in büyük kısmını besliyor (fail% ile penny% birlikte hareket ediyor).

#### 3. DTE≤2 toplam gamma$ payı (Σ gamma·OI·100·S²·0.01)
- **SPY: %29.72** (5.431e+11 / 1.827e+12); DTE≤2 olan gün 53/236 = %22.5; bu günlerde günlük gamma$ medyanı DTE>2'ye göre **×1.4**.
- **QQQ: %29.16** (2.883e+11 / 9.886e+11); 53/236 günde; medyan yoğunluk **×1.3**.
→ Gamma$'ın ~%30'u son 2-DTE'de yoğunlaşıyor; bu satırlar aynı zamanda en yüksek fail%'li olanlar (§1).

#### 4. net_gex SAWTOOTH — |Δnet_gex| varyansı (expiry-haftası DTE≤5 vs diğer DTE>5)
| | Var(\|Δ\|) expiry-wk | Var(\|Δ\|) diğer | **ORAN** | ort\|Δ\| exp / diğer |
|---|---|---|---|---|
| **SPY** | 1.1145e+19 (n=87) | 5.8036e+18 (n=148) | **×1.92** | 2.57e+9 / 1.32e+9 |
| **QQQ** | 2.2594e+18 (n=87) | 1.2028e+18 (n=148) | **×1.88** | 1.12e+9 / 5.61e+8 |

→ **SAWTOOTH KANITI**: expiry-haftalarında günlük net_gex sıçramasının varyansı ~**1.9×** ve ortalama |Δ| de ~2×. net_gex'in günlük değişimi roll/expiry takvimine bağlı testere-dişi içeriyor.

**Hipotez bağı:**
- **⑤ IV-from-mid instabilite: DOĞRULANDI** — fail %6.7-9.1 (band-içi), DTE0/1/2'de %32-37'ye fırlıyor; penny %18/%14. Tek-mid boru (bid yok → spread ölçülemiyor) kısa-DTE'de gürültülü.
- **⑥ OPEX-sawtooth confound: DOĞRULANDI** — expiry-haftası |Δnet_gex| varyansı ~1.9× yüksek; gamma$'ın ~%30'u DTE≤2'de. net_gex sinyali takvim-kaynaklı sıçrama taşıyor (saf rejim bilgisi değil).
- **ÖLÇÜLEMEDİ:** gerçek bid-ask spread / quote-stale oranı → veri lazım: `md`'de **bid/ask kolonları** (şu an sadece `mid`). Onsuz mid-instabilitesi yalnızca penny-proxy + bisection-fail üzerinden dolaylı ölçülebiliyor.

---

### D4 — TIMING / PIT / FEED

**Script:** `C:/Users/admin/Downloads/kader-equity/backtest/diagnosis/D4_timing.py`
**Venv ile koşuldu:** `C:/Users/admin/Downloads/kader-macro/.venv/Scripts/python.exe` — 3 bölüm de geçti, sayılar GERÇEK çıktıdan. kader-macro'ya dokunulmadı; level/PnL okunmadı bile (bu görev saf timing/PIT).

#### 1. FEED-TIMING (MarketData free, ampirik canlı probe)
Token `.env`'den (ekrana basılmadı). Probe zamanı: **2026-06-10 21:40 ET** (seans ~5.6 saat önce kapandı).

| istek_tarihi | gün | HTTP | s | n_strike | updated | yaş |
|---|---|---|---|---|---|---|
| 2026-06-10 (D, dün-seans) | Wed | **402** | error | 0 | — | 0 |
| 2026-06-09 | Tue | 203 | ok | 528 | **16:00 ET** | 1 |
| 2026-06-08 | Mon | 203 | ok | 528 | 16:00 ET | 2 |
| 2026-06-05 → 06-02 | — | 203 | ok | 528 | 16:00 ET | 5-8 |

- En yeni veri dönen tarih = **2026-06-09**, ölçülen gecikme = **1 işlem günü**.
- Bütün `updated` damgaları **16:00 ET** = EOD settle-snapshot (gün-içi taze değil).
- **BUGÜN (D=06-10) verisi, seans kapandıktan saatler sonra HÂLÂ 402** → veri same-day DEĞİL, gece OCC-batch.
- Docs + OCC mekaniği (web): OCC open-interest'i seans-sonrası batch'te hesaplar, **ertesi sabah ~09:30-09:45 ET** yayınlar, gün-içi güncellemez. MarketData free token "Historical (1 day old)" / 15-dk-delayed sınıfında.

**HÜKÜM (OI[D]+mid[D], D+1 09:30 ET ÖNCESİ çekilebilir mi?): HAYIR (mekanik).** Kanıt: D-akşamı bile 402 → same-day yok; OI ertesi sabah açılış-civarı yayınlanır → mevcudiyet açılışLA çakışır, açılışTAN ÖNCE değil. → **vol-rejim CANLI sinyali için free MarketData = paid-feed dependency** (D+1 açılışta-en-erken/muhtemelen-sonra). Tek doğrudan-EVET testi D+1 09:25 ET tekrar-koşusu; bu run o pencerede değildi (dürüstçe: açılış-öncesi pencere doğrudan-ölçülmedi, ama D-akşamı-402 + OCC-mekaniği HAYIR'ı destekliyor).

#### 2. DST ASSERT (alpaca 1-dk bar → ET; 1402 RTH-günü, 2020-09-01 → 2026-06-10)
- **İlk-RTH-bar UTC dağılımı: 13:30 → 903g (EDT/yaz), 14:30 → 499g (EST/kış)** — ikisi de ET'de 09:30. **DST doğru çevriliyor: first≠09:30 ihlali SPY 0 / QQQ 1.** (Tek QQQ ihlali 2024-09-13 ilk=09:31 = veri-boşluğu, saat-hatası değil.)
- ÖLÇÜM-NOTU: Alpaca bar'ı bar-BAŞLANGICI ile damgalı → son minute-bar normal-günde 15:59 (15:59→16:00), yani {15:59,16:00} ikisi de DOĞRU. (İlk naive testimde 805 "ihlal" çıkmıştı = yanlış-eşik; düzeltildi.)
- **normal-gün son-likit-bar ∉ {15:59,16:00}: SPY 1 / QQQ 1** — her ikisi de 2024-12-23 (likit-son 10:21/10:16, n=47) = Alpaca **veri-gap'i**, saat-hatası değil.
- **Yarım-günler (12:59/13:00 likit-kapanış):** hacim-cliff tam 13:00 ET'de doğrulandı (12:59 bar'ı 33K-43K pay, sonrası ≤birkaç-yüz pay odd-lot/geç-print). DOĞRU: SPY 8 / QQQ 10. "YANLIŞ" kalanlar (SPY 4: 2020-11-27, 2020-12-24, 2021-11-26, 2024-12-24; QQQ 2: 2021-11-26, 2025-11-28) = düşük-likidite tatil-seanslarında 13:00-sonrası 1000-pay-üstü tek-tük print → likit-son 13:07-13:52'ye kayıyor; **eşik-artefaktı, DST-hatası değil** (hepsi resmî 13:00-kapanış günleri, kapanış dakikalar içinde).

**İhlal listesi (gerçek saat-hatası): YOK.** Tüm "ihlaller" ya veri-gap (2024-12-23) ya da tatil-thin-tape eşik-artefaktı. EST (Kas-Mar) dahil çevrim doğru.

#### 3. PIT re-verify (expiry-geçişlerinde OI-drop)
- **Ham PIT-leak (satırda expiration < date = vadesi-geçmiş kontrat): SPY 0 / QQQ 0 satır.**
- **OI-drop ≥%70: SPY 19/19, QQQ 19/19** (tüm expiry-geçişleri, drop %100 her birinde).
- Geçiş anı DTE: **19/19 vade-günü (DTE=0)** — erken-roll yok; her expiry tam vade-gününe kadar tutuluyor, sonra zincirden tamamen çıkıyor (`still_present=False`).
- **PIT TEMİZ:** vadesi-geçmiş kontrat sonraki snapshot'ta taşınmıyor → look-ahead yok.

**Hipotez hükümleri (bu görevin değdikleri):**
- **① truncation (monthly-only?) = DOĞRULANDI.** Chain yapısı: tarih başına TEK expiry, toplam **20 distinct expiry, hepsi 3.-Cuma monthly + ay-sonu Cuma OpEx** (exp DOW: Cuma 219 / Per 24). DTE 0→25 testere-dişi (med 8). Yani backtest **tek-en-yakın-monthly** tüketiyor, full multi-expiry surface DEĞİL → ④ (canlı 5-vade vs backtest 1-vade) için doğrudan zemin.
- **④ canlı-uyum:** feed D+1-açılışta-en-erken (Bölüm 1) + veri tek-monthly-vade (Bölüm 2'deki chain yapısı) → backtest'in beslemesi ile canlı çok-vade/açılış-öncesi tetik HİZASIZ; canlı sinyal free-feed ile gün-içi kurulamaz.
- Bölüm 2/3'teki ölçüm-düzeltmeleri (15:59-eşiği, yarım-gün hacim-cliff, tek-expiry PIT) = ölçüm-sağlamlık; hiçbir P&L/parametre üretilmedi.

**Ölçülemeyenler (dürüst):** D+1 09:25 ET açılış-öncesi pencerede OI[D]'nin DOĞRUDAN mevcudiyeti (run o saatte değildi — mekanik+akşam-402 kanıtıyla HAYIR'a işaret ama doğrudan-gözlem için 09:25 ET tekrar-koşu lazım).

Kaynaklar: [MarketData Plan Limits](https://www.marketdata.app/docs/account/plan-limits/), [MarketData Option Chain docs](https://www.marketdata.app/docs/api/options/chain/), [OCC Open Interest](https://www.theocc.com/market-data/market-data-reports/volume-and-open-interest/open-interest)

---

### D5 — CONFOUND HARİTASI

**Script:** `C:/Users/admin/Downloads/kader-equity/backtest/diagnosis/D5_confound.py` (canonical kader-macro venv ile koşuldu, gerçek çıktı; TEŞHİS-ONLY, P&L üretilmedi).

Panel: 236 gün (2025-06-13→2026-06-08), tek-rejim. Bayrak = `level_series.regime` = sign(net_gex); büyüklük = |net_gex|. Tüm confound regressor'ları PIT (D-EOD): RV20 (alpaca günlük-getiri std, yıllık), VIX (yerel `vixcls.parquet` cache, ağ yok), DTE (zincir front-expiry), trailing-20g getiri. Intraday = D+1 RTH c1/o1−1 (spine_diagnostic ile aynı).

**0. '%72 örtüşme' decompose (disentangle.py'nin tam reprodüksiyonu).** Bu %72, `gamma_inv` ve `vol_only` pozisyonlarının (ikisi de ±gap) örtüşmesi; gap ortak çarpan olduğu için sadeleşir → örtüşme = flag-kovası ile `atm_iv`-vol-kovasının yön-eşleşmesi. SPY: 170/236 = **%72** (örtüşen = +γ&düşük-vol 79 + −γ&yüksek-vol 91). QQQ: 171/236 = **%72.5** (93+78). phi(flag, düşük-vol): SPY **+0.44**, QQQ **+0.45**. Yani %72'nin tamamı gamma-yön bilgisi değil: rastgele-coin baz çizgisi zaten %50; flag↔atm_iv-vol ilişkisi (phi ~+0.44) %50→%72'yi açıklıyor. **Hipotez ⑥ (vol-confound) KISMEN DOĞRULANDI — flag güçlü-orta vol-bağlı, ama tam-determinist değil.**

**1. Bayrak ~ vol.** RV20 tek başına zayıf (logistic pseudo-R²: SPY 0.000, QQQ 0.030) ama VIX güçlü (SPY 0.149, QQQ 0.256); RV20+VIX birlikte SPY 0.202 / QQQ 0.261. 2x2 (+γ↔düşük-vol): SPY phi +0.12 (zayıf, RV20-tabanlı), QQQ phi +0.32 (güçlü). |net_gex|~RV20 OLS R²: SPY 0.028, QQQ 0.036 (büyüklük RV20'den neredeyse bağımsız). **Vol — özellikle implied (VIX) — sign-flag'i kayda değer açıklıyor; realized (RV20) zayıf.**

**2. Bayrak ~ OPEX-takvim (sawtooth DOĞRULANDI).** +γ-oranı DTE düştükçe artıyor (SPY: 0-2g→%60, 15+g→%31; corr DTE~flag **−0.240**). |net_gex| büyüklüğü DTE'ye daha güçlü bağlı (SPY 0-2g ort 3.75B$ vs 15+g 1.50B$; QQQ corr DTE~log|net_gex| **−0.339**, OLS R² 0.115). **Sawtooth gerçek: kısa-DTE'de hem +γ-eğilimi hem büyük-magnitude → takvim-artefaktı mevcut, özellikle büyüklükte.**

**3. Bayrak ~ trend (EN GÜÇLÜ confound).** trailing-20g getiri vs flag korr: SPY **+0.467**, QQQ **+0.493**. 2x2 (+γ↔yukarı-trend): SPY phi +0.37 (+γ&up 102 / +γ&dn sadece 4), QQQ phi +0.41 (+γ&up 122 / +γ&dn 11). **Tez doğrulandı: yukarı-trend → call-wall kırılmaz → +γ; +γ neredeyse hiç düşüş-trendinde görülmüyor.** Büyüklük-bağı orta (OLS R² SPY 0.126 / QQQ 0.072).

**4. KAPANIŞ — çoklu-regresyon decompose ([rv20+vix+dte+tr20]).**
- **A) |net_gex| büyüklük (log, OLS):** SPY R² **0.278** → gamma'ya-özgü kalan **%72.2**; QQQ R² **0.257** → kalan **%74.3**. En büyük tek-katkı: SPY vix (0.213), QQQ dte (0.115).
- **B) sign-flag (logistic, McFadden):** SPY pseudo-R² **0.315** (accuracy %78 vs taban %55, +23pp); QQQ **0.360** (accuracy %79 vs taban %56, +23pp). Yani üç confound flag yönünü **%78-79** doğrulukla tahmin ediyor.

**KAPANIŞ SAYISI:** |net_gex| büyüklüğünün **~%72-74'ü** vol+takvim+trend ile açıklanmıyor (gamma-özgü kalan). Ama **sign-flag** confound'larla **%78-79 doğrulukla** tahmin edilebiliyor (taban-üstü +23pp) → yön-bayrağı büyüklükten çok daha fazla confound-yüklü. Dominant confound = **TREND** (corr +0.47/+0.49, phi +0.37/+0.41), ardından **implied-vol/VIX** (QQQ'da phi +0.32), sonra **OPEX-DTE sawtooth** (büyüklükte daha belirgin).

**Hipotez verdict'leri:** ⑥ confound — vol %56-66 örtüşme orta-güçlü (DOĞRULANDI ama tek-başına değil); OPEX-sawtooth NET MEVCUT (DOĞRULANDI); tek-rejim pencere (236g, 2025-26) → tüm R²'ler in-sample, forward'da değişebilir (SINIR). Trend, spec'te listelenmemiş olmasına rağmen **en güçlü confound** çıktı.

**Ölçülemeyenler (tahmin edilmedi):** (a) `iv`/`delta` kolonları zincirlerde BOŞ → opsiyon-seviye greeks confound'u ölçülemedi, level_series'in türettiği `atm_iv` kullanıldı. (b) Canlı 5-vade GEX (hipotez ④) bu panelde yok (backtest 1-vade) → canlı-uyum confound'u için canlı gamma_engine snapshot'larıyla paralel build lazım. (c) Çoklu-rejim ayrımı (boğa/ayı) için >236 gün / kriz-içeren pencere lazım.

---

### D6 — İstatistiksel Güç & Minimum Veri

**Script:** `C:/Users/admin/Downloads/kader-equity/backtest/diagnosis/D6_power.py` (canonical kader-macro venv ile koşuldu, gerçek çıktı; TEŞHİS-ONLY, P&L üretilmedi — `block_robust.gamma_inv_pnl` + `spine_diagnostic.mean_reversion_return` YALNIZ okundu).

#### §0 — SE formülleri ve varsayımlar (açık)
1. **Günlük-Sharpe t:** `t = SR_daily·√N`, `SR_daily = SR_ann/√252`. `t≥2 ⇒ N = (2/SR_daily)²`.
2. **DSR (Bailey-López de Prado, K=10 trial):** `SR_0 = √Var[SR]·[(1−γ)Φ⁻¹(1−1/K) + γΦ⁻¹(1−1/(Ke))]` (γ=Euler), `Var[SR]=(1−γ3·SR+(γ4−1)/4·SR²)/(N−1)`. `DSR>0 ⇔ DSR>0.5 ⇔ SR_daily > SR_0(N)`; N büyüdükçe Var↓→SR_0↓, gereken-N ikili-aramayla bulundu.
3. **Event-edge:** `SE = σ_event/√N_event`, `t = mean/SE`, `N(t≥2) = (2σ/mean)²`.
4. **Varsayımlar:** getiriler/event'ler IID; K=10 bağımsız trial; momentler (γ3,γ4) gözlenen örnekten; edge-yönü doğru. IID-ihlali (otokorelasyon/kümeleme) veya forward rejim-kayması ⇒ gereken-N **daha büyük** (yani aşağıdaki sayılar ALT-SINIR).

#### §1 — (a) Koşulsuz-günlük gereken-N
Gözlenen full Sharpe **SPY +1.29 (mevcut N=233), QQQ +0.87 (N=234)** — block_robust ile teyit.

| sym | SR_ann | SR_daily | γ3 | γ4 | N(t≥2) | N(DSR>0,K=10) | **N_gerekli** | mevcut-N'de t / DSR | sonuç |
|-----|--------|----------|-----|-----|--------|---------------|---------------|---------------------|-------|
| SPY | +1.29 | +0.0813 | +1.02 | 6.37 | 605 | 349 | **605** | t=+1.24 / DSR=0.387 | **GEÇMEZ** (t≥2 ✗, DSR>0 ✗) |
| QQQ | +0.87 | +0.0548 | +0.51 | 6.37 | 1.331 | 807 | **1.331** | t=+0.84 / DSR=0.233 | **GEÇMEZ** (t≥2 ✗, DSR>0 ✗) |

Mevcut-N'de her iki bağlayıcı eşik de düşüyor; bağlayıcı kısıt **t≥2** (DSR'den daha katı).

#### §2 — (b) Rejim-koşullu gereken-N (branch başına)
Günlük P&L +γ/−γ dallarına ayrıldı:

| sym | dal | mevcut-N | SR_ann | N(t≥2) | N(DSR>0) | **N_gerekli** | eksik |
|-----|-----|----------|--------|--------|----------|---------------|-------|
| SPY | +γ | 106 | +0.10 | 93.285 | 57.613 | **93.285** | ~370 yıl (dal ~sıfır-edge → pratikte sonsuz) |
| SPY | −γ | 127 | +2.00 | 251 | 140 | **251** | ~0.5 yıl daha |
| QQQ | +γ | 133 | +0.69 | 2.126 | 1.332 | **2.126** | ~7.9 yıl daha |
| QQQ | −γ | 101 | +1.06 | 894 | 530 | **894** | ~3.1 yıl daha |

Edge'in büyük kısmı **−γ dalında** yoğunlaşmış (SPY −γ SR +2.0, +γ ~0). Branch'a bölmek N'i yarıya düşürdüğü için her dal tek başına mevcut N ile geçmiyor; en yakın olan SPY −γ (251 gerek vs 127 mevcut).

#### §3 — (c) Wall-touch event-bazlı gereken-N
Event-edge = `mean_reversion_return` (duvardan kapanışa geri-dönüş; >0 duvar-tuttu, <0 kırıldı):

| sym | dal | n-event | ort-MR | σ | t | **N(t≥2)** | yön |
|-----|-----|---------|--------|-----|-----|-----------|-----|
| SPY | +γ | 30 | −19.0bps | 57.7 | −1.81 | **37** | tez-DIŞI (tez +γ→MR>0 bekliyor, gerçek negatif) |
| SPY | −γ | 34 | +2.5bps | 73.6 | +0.20 | **3.538** | tez-dışı + neredeyse sıfır-edge |
| QQQ | +γ | 40 | −20.4bps | 79.2 | −1.63 | **60** | tez-DIŞI |
| QQQ | −γ | 45 | +1.3bps | 92.0 | +0.10 | **18.668** | tez-dışı + sıfır-edge |

Kritik bulgu: gözlenen event-edge'ler **tezin TERSİ yönünde** (+γ'da duvar tutmuyor, MR negatif — spine_diagnostic'in NO-GO verdiği bulguyla tutarlı). En güçlü ölçülebilir sinyal (+γ MR ≈ −20bps) "duvar **kırılıyor**" diyor; bunu t≥2 anlamlılığa taşımak için SPY 37 / QQQ 60 event yeter (mevcut 30/40 — yakın). Ama bu, modelin iddia ettiği yönün AKSİ.

##### Mevcut veride wall-touch event sayımı (sembol × rejim × duvar-tipi)
| sym | rejim | duvar | event-n | ort-MR | tez-yönü |
|-----|-------|-------|---------|--------|----------|
| SPY | +γ | call | 16 | −5.4bps | tez-dışı |
| SPY | +γ | put | 15 | −30.2bps | tez-dışı |
| SPY | −γ | call | 12 | +22.3bps | tez-dışı |
| SPY | −γ | put | 23 | −4.8bps | **tez-içi** |
| QQQ | +γ | call | 22 | −22.5bps | tez-dışı |
| QQQ | +γ | put | 18 | −17.8bps | tez-dışı |
| QQQ | −γ | call | 15 | −20.6bps | **tez-içi** |
| QQQ | −γ | put | 30 | +12.3bps | tez-dışı |

Sayım mutabakatı (spine_diagnostic ile): SPY 64 dokunuş-GÜN = 28 call + 38 put − 2 aynı-gün-her-iki = 66 dokunuş-EVENT (8 hücrenin 4'ü tez-dışı); QQQ 85 dokunuş-gün = 85 event. 8 hücreden yalnız 2'si tez-yönünde → cell-başına n çok düşük (12–30), hiçbiri tek başına anlamlı değil.

#### §4 — ÇEVİRİ: minimum veri ve mevcut-N kıyası
**Mevcut veri:** ~233 işlem-günü ≈ **0.92 yıl, TEK likidite-rejimi** (2025-06→2026-06). Stres pencereleri 2018Q4 + 2020 + 2022 **kapsam-dışı**.

- **Koşulsuz-günlük directional iddia için minimum:** SPY **605 gün ≈ 2.4 yıl** çoklu-rejim günlük seri (eksik 372 gün / ~1.5 yıl); QQQ **1.331 gün ≈ 5.3 yıl** (eksik 1.097 gün / ~4.4 yıl).
- **Rejim-koşullu:** en ulaşılabilir dal SPY −γ = 251 dal-günü (~1 yıl o-rejimde, eksik ~0.5 yıl); QQQ −γ 894 (~3.5 yıl); +γ dalları pratikte ulaşılamaz (edge≈0).
- **Wall-touch event yolu:** event-hızı SPY 64/236 = 0.27 (~1 event/3.7 gün), QQQ 85/236 = 0.36 (~1 event/2.8 gün). +γ event-edge için SPY 37 event (~135 işlem-günü / 0.5 yıl), QQQ 60 event (~168 gün / 0.7 yıl) — mevcut 30/40'a EN YAKIN yol bu, **ama tez-dışı yönde**. Intraday-bar ile event-yoğunluğunu artırmak (günde 1 yerine birden çok dokunuş ölçmek) bu yolu daha hızlı doldurur.

**Ne kadar eksik (özet):** En iyi senaryoda (event-yolu, en güçlü sinyal) sayısal-anlamlılık eşiğine **~6–8 event uzakta**; ama koşulsuz-günlük directional iddia için **SPY ~1.5 yıl, QQQ ~4.4 yıl daha günlük çoklu-rejim veri** gerekiyor. Kapsam-şartı (sayıdan bağımsız): N ne kadar büyürse büyüsün, non-stationary / best-of-K riski **2018Q4+2020+2022 stres-rejimleri görülmeden** düşmez — yani minimum = "≥2.4 yıl (SPY) / ≥5.3 yıl (QQQ) GÜNLÜK ÇOK-REJİMLİ" VEYA intraday-event yoğunluğunu artırma.

#### §5 — Hipotez kararları (D6'nın görebildiği kadarıyla)
- **⑦ istatistik (DOĞRULANDI):** mevcut N'de SPY t=+1.24/DSR=0.387, QQQ t=+0.84/DSR=0.233 — ikisi de eşik-altı. +1.29 best-Sharpe, K=10 null'un beklenen-maksimumunu aşmıyor → best-of-N seçim-yanlılığı tezi sayısal olarak destekleniyor.
- **⑥ confound / tek-rejim (KISMEN — D5/D6 ortak):** D6 yalnız güç-ekseninde; tek-rejim (0.92 yıl) → her gereken-N alt-sınır.
- Diğer hipotezler (①truncation, ②sign, ③havuz, ④canlı-uyum, ⑤IV-mid) **D6 kapsamı dışı** — D1/D2/D3/D5 scriptleri ölçer; D6 bunları ölçmedi, varsaymıyorum.

---

### D7 — VERİ PAZARI MATRİSİ (TEŞHİS-ONLY)

**Üreten script:** `C:/Users/admin/Downloads/kader-equity/backtest/diagnosis/D7_market.py` (canonical venv ile koşuldu, gerçek çıktı — UYDURMA YOK)
**Rapor:** `C:/Users/admin/Downloads/kader-equity/backtest/diagnosis/D7_market.md`
Vendor olguları = 2026-06 WebSearch/WebFetch snapshot → **as-of 2026-06** etiketli; ölçülemeyen alan `belirsiz`.

#### 1. AÇIK problemlerin repo-içi kanıtı (D7_market.py §A, gerçek ölçüm)
md_spy/md_qqq, 243 gün, 2025-06-13→2026-06-08:
- **① truncation AÇIK** — `uniq_expiry=20`, **vade/gün med=1 max=1** (tek-vade). D1: tek-front-monthly = bugünkü toplam gamma$'ın **%11 SPY / %10 QQQ** (kalan ~%89-90 sansürlü).
- **② sign AÇIK** — `buy/sell-open/close imza kolonu = YOK` (md yalnız OI); dealer-envanter naive +call/−put proxy'sine mahkûm.
- **③ havuz AÇIK (SPX'te ağır)** — D1: SPX/SPY gamma$ **0.04×**, NDX/QQQ **0.52×**.
- **⑤ IV-from-mid AÇIK** — `iv var=False, delta var=False` → mid'den türet zorunlu.

#### 2. ② SIGN (EN YÜKSEK ÖNCELİK; işaret yanlışsa gerisi anlamsız)
- **Cboe Open-Close Volume Summary** = **②'nin KANONİK çözümü.** Kapsam: TÜM Cboe-borsa serileri, **SPX index (C1 birincil) + SPY ETF**, tüm-vade; başlangıç **C1 EOD 2005+**, intraday 10-dk 2011 / 1-dk 2019. PIT **YÜKSEK** (borsa-birincil, look-ahead yok). Format: satır = participant-type(customer/pro-cust/broker-dealer/MM) × buy/sell × open/close. **Fiyat-belirsiz** (sayfa basmaz; SEC 'LiveVol Fees'; teklif; EOD özet ucuz uçta).
- DIY-proxy alternatifleri (trade-classification): **ThetaData** $40/$80/$160-ay (4/8/12y), **Polygon/Massive** $29/$79/$199-ay (geçmiş **2–4y sığ**), **Cboe LiveVol intraday tick** (fiyat-belirsiz). Hepsi sign'ı kullanıcıda hesaplatır.

#### 3. ① ALL-EXPIRY (full-chain EOD)
- **Cboe Option EOD Summary** — tüm-vade 2005+, IV+greeks Calcs add-on (fiyat-belirsiz).
- **ORATS** — tüm-vade EOD **2007+** + 1-dk 2020+, **greeks+IV hazır (⑤'i de çözer)**, **$99/ay birey / $299/ay pro** (+$50 RT).
- **OptionMetrics IvyDB US (WRDS)** — **1996+** her opsiyon/gün, akademik altın-standart, **fiyat-belirsiz** (kurumsal WRDS).
- **iVolatility / HistoricalOptionData** — SPX **1990+ (35y)**, perakende-ucuz; iVol pay-per-use, HistOptData 'Bloomberg'in kesri' (belirsiz).
İmza YOK → bunlar ②'yi çözmez.

#### 4. ③ index-havuz + hazır-GEX vendor
- **③** = native **^SPX/^NDX** sembolü (taşıyıcı vendor fiyatına dahil); havuz 0.04×/0.52× yüzünden SPX'te ZORUNLU.
- **SqueezeMetrics** DIX+GEX **2011+**, **$720/ay**, CSV+API, metodoloji **YARI-ŞEFFAF** (2017 white-paper, formül kapalı) → hazır-GEX'ler içinde backtest-girdisi için tek savunulabilir.
- **SpotGamma** ($89/$99/$129/$299/$1,999+-ay), **MenthorQ** (fiyat-belirsiz), **Tier1Alpha** (fiyat-belirsiz, kurumsal) → **KARA-KUTU + tarihsel-seri belirsiz → backtest-GİRDİSİ DEĞİL** (canlı-overlay/teyit katmanı). Hiçbiri ham-imza vermez.

#### 5. Öncelik sonucu
1. ②sign → **Cboe Open-Close** (SPX+SPY, 2005+, PIT-yüksek; fiyat teklif-bazlı). naive +call/−put proxy'sinin doğrudan ikamesi.
2. ① → pratik **ORATS 2007+** (⑤'i de kapatır), en-uzun **IvyDB 1996+** (kurumsal).
3. ③ → ^SPX/^NDX ekle (SPX'te zorunlu).
4. Hazır-GEX (SpotGamma/Menthor/Tier1) backtest-girdisi olarak ÖNERİLMEZ; SqueezeMetrics istisna.

**Ölçülemeyenler (tahmin edilmedi):** Cboe Open-Close / LiveVol-tick / IvyDB / MenthorQ / Tier1Alpha **kesin $ fiyatı** (teklif-bazlı veya yayınlanmamış); SpotGamma/Menthor/Tier1'in **tarihsel-seri PIT-export derinliği** (belgeli-değil) — bunlar için satıcı-teklifi veya kurumsal-erişim gerekir.

---

## 4. EMİR'İN CEVAPLAYACAĞI SORULAR (tedavi-promptu bunlarsız yazılmaz)

**S1. Aylık veri-bütçesi bandı: 0 / ~50$ / 50-200$ / üstü?**
İlgili sayılar: ① truncation **bedava** çözülebilir (MarketData free-tier, ~1215 kredi historical-1-kredi/1000-sembol — D1.2). ②sign için Cboe Open-Close fiyatı teklif-bazlı (D7.2); DIY-proxy alt-bandı ThetaData $40-160/ay veya Polygon $29-199/ay (geçmiş 2-4y sığ, D7.2). ① pratik-vendor ORATS $99-299/ay (⑤'i de kapatır, D7.3). Hazır-GEX SqueezeMetrics $720/ay (D7.4). → Bütçe bandı hangi sorunların çözülebileceğini doğrudan belirler (0$ → yalnız ① + forward-④; ~50$ → +DIY-proxy-sign-sığ; 50-200$ → ORATS full-chain+IV; üstü → Cboe-kanonik-sign / IvyDB / SqueezeMetrics).

**S2. Havuz: SPX+NDX mi, SPY/QQQ mu, agregat mı?**
İlgili sayılar: ③ havuz oranı yfinance ile ÖLÇÜLEMEDİ ama SPX'te ağır artefakt var (SPX/SPY 0.04×, NDX/QQQ 0.52× — D1.4); ^SPX front-OI=0 (D1.4). MarketData index-chain SPX/NDX serviste var (D1.2). Vendor-fiyatı genelde sembole-dahil (D7.4). → "SPY/QQQ yeter" derseniz ③ büyük-ölçüde kapanır (ETF-havuzu kapsanıyor); "SPX+NDX gerek" derseniz native index-OI feed (MarketData/Cboe/ORATS) zorunlu, ekstra ③-iş-yükü.

**S3. Sign-yöntemi: Cboe open-close mu, intraday trade-classification mı, vendor mı?**
İlgili sayılar: ②sign'in en yüksek öncelik olmasının nedeni üst-tercile agreement çöküşü %30 SPY / %49 QQQ (D2.4); md'de buy/sell-open/close imza YOK (D7.1). Cboe Open-Close = kanonik, SPX+SPY 2005+, PIT-yüksek, fiyat teklif-bazlı (D7.2). Trade-classification DIY = ThetaData/Polygon/LiveVol, sign'ı kullanıcı hesaplar, geçmiş 2-4y sığ (D7.2). → Yöntem seçimi hem maliyeti (S1) hem tarih-derinliğini (S5) belirler.

**S4. Horizon: daily-EOD ısrarı mı, intraday-event reformülasyonu kabul mü?**
İlgili sayılar: EOD-OI'nin yapısal-tavanı = feed D+1-açılışta-en-erken (D4.1: D-akşamı 402, OCC ertesi-sabah ~09:30-09:45 yayın) → free-feed ile gün-içi canlı-sinyal MEKANİK kurulamaz (D1.3 anlamında EOD-OI 0DTE-çağı tavanı). Event-yolu en-yakın anlamlılığa (~6-8 event uzak, D6.4) ama tez-DIŞI yönde (+γ MR ≈ −20bps, D6.3). Intraday-bar event-yoğunluğunu artırır (D6.4). → "daily-EOD ısrarı" → ⑦ için SPY ~1.5y/QQQ ~4.4y daha veri (D6.4); "intraday-event reformülasyon" → forward-OI/1-dk-bar feed gerekir (paid/forward).

**S5. Tarih-derinliği: D6 minimum-N'i kabul mü (≥2018 kapsama)?**
İlgili sayılar: mevcut veri 0.92 yıl tek-rejim (D6.4); 2018Q4+2020+2022 stres kapsam-dışı (D6.4). Koşulsuz-günlük min: SPY 605g ≈2.4y, QQQ 1331g ≈5.3y (D6.1). Best-of-K risk N büyüse de stres-rejim görülmeden düşmez (D6.4). Vendor tarih-başlangıçları: Cboe 2005+, ORATS 2007+, IvyDB 1996+, SqueezeMetrics 2011+ (D7.2-7.4). → "≥2018 kapsama kabul" → ORATS/Cboe/IvyDB ile geriye-backfill gerekir (S1 bütçesine bağlı); "kabul değil" → forward-biriktirme (yıllar).

---

## 5. KILL-CRITERIA TASLAĞI (BOŞ — Emir dolduracak + kilitleyecek; pre-registration)

- [ ] Koşul → directional-GEX programı KAPANIR: ______
- [ ] Koşul → DEVAM: ______

*(sadece başlık/şablon — Emir doldurup pre-registration olarak kilitleyecek.)*
