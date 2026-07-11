# KADER-EQUITY — Mimari

> **Durum: BUILT.** MODEL = **tide × COR1M-froth × GEX-shield** → SPX **1.64** / NDX **1.77** @2019+ frozen
> (honest forward ~1.0-1.3, DSR 0.985/0.994). §1-9 = TIDE ayağı (spine, DONMUŞ); **§10 = as-built overlay stack**.
> tide spine §1-9'da, 2 overlay modülü §10'da. pytest 11/11. Yazım: 2026-06-08 (seed) → 2026-06-09 (as-built §10).

---

## 0. Konum & felsefe

- **kader-macro = FİNAL, DOKUNULMAZ.** Sadece modül skorlarını üreten "skor fabrikası" olarak read-only tüketilir. Onun kendi canlı stance'i (full-history SPX ~0.84 / NQ ~0.99) **bu modelin parçası değildir** — ayrı bir şeydir.
- **Bu tide = equities'e ÖZEL ağırlıklandırma.** Genel macro stance m2-eşit ağırlık kullanır; equities ise **fiskal-dominant** (m9 = 0.56) ister. Sebep: post-2019 equity rejimi fiskal-sürücülü (açık finansmanı, teşvik, issuance).
- **Çıktı:** günlük **yön (LONG/FLAT)** + **sürekli kompozit skor**. Downstream bunun üstüne equity-özel **alfa katmanları** (pozisyonlama, akış) ekler.
- **Ayrı repo** (`kader-equity/`), kader-macro'yu import-only kullanır → makro temiz kalır, bu model bağımsız evrilir (kader-btc şablonu).

---

## 1. Katmanlı akış

```
╔═══════════════════════════════════════════════════════════════════╗
║  KADER-MACRO  (final, dokunulmaz — skor fabrikası)                  ║
║  her modül → Carver-capped  −20 .. +20  GÜNLÜK skor (PIT-temiz)     ║
╚═══════════════════════════════════════════════════════════════════╝
            │  read-only: modül skorları + yayın-lag'leri
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  KATMAN 1 — GİRDİ SEÇİMİ  (equities-özel: 8 modül)                    │
│  KULLAN:  m9 m5 m2 m0 m3 m6 m8 m4                                     │
│  AT:      m1 (CB-tide), m10 (USD), m11 (collateral) → equity-kontam.  │
└─────────────────────────────────────────────────────────────────────┘
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  KATMAN 2 — AĞIRLIKLANDIRMA  (sweep-optimize vektör, SABİT)          │
│  TIDE_SCORE = 0.563·m9 +0.214·m5 +0.118·m2 +0.061·m0                  │
│             +0.025·m3 +0.010·m6 +0.006·m8 +0.002·m4                   │
└─────────────────────────────────────────────────────────────────────┘
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  KATMAN 3 — SİNYAL  (combine + execution lag)                        │
│  TIDE_DIR = LONG  eğer TIDE_SCORE > 0   else FLAT                     │
│  +1 GÜN exec lag  (sinyal[t] → t+1'de uygulanır; look-ahead-free)    │
└─────────────────────────────────────────────────────────────────────┘
            ▼
╔═══════════════════════════════════════════════════════════════════╗
║  ÇIKTI SÖZLEŞMESİ  →  kader-equity downstream                       ║
║  { tide_dir, tide_score, alt_modül_skorları, freshness, lag_ok }    ║
╚═══════════════════════════════════════════════════════════════════╝
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  KADER-EQUITY DOWNSTREAM  (inşa edilecek)                            │
│  final = w_tide·TIDE  +  w_pos·POZİSYON  +  w_flow·AKIŞ              │
│          (≈35-60%)        (ana alfa?)        (akış/breadth)          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Girdi katmanı — modül modül (ağırlık sırasıyla)

| Modül | Ne ölçer | Kaynak | Freq / lag | Ağırlık | Equities'te rolü |
|---|---|---|---|---|---|
| **m9** | Fiskal impuls/duration (issuance + MTS; LEVEL-bazlı 60ay self-cal + Fed prong; stres-koşullu) | Treasury / FRED | aylık / +lag | **0.563** | **EN BÜYÜK sürücü** — fiskal-dominant rejim |
| **m5** | Vol rejimi (MOVE birincil) | FRED / yfinance | günlük | **0.214** | risk-on/off motoru |
| **m2** | Net-liq = WALCL−TGA−RRP (**RAW! smart-RRP DEĞİL** — bkz. §9 parite) | FRED | haftalık / +1g | **0.118** | Fed plumbing tide + m9-hedge |
| **m0** | Büyüme/enflasyon rejim kompoziti | FRED | aylık | **0.061** | makro zemin |
| m3 | UST açık-artırma talebi (bid/cover, indirect) | TreasuryDirect | ihale günü | 0.025 | minik |
| m6 | Kredi spreadi (Moody's Baa-Aaa veya ICE HY OAS) | FRED | günlük | 0.010 | minik |
| m8 | Sistem kaldıracı | NY Fed | haftalık | 0.006 | ihmal |
| m4 | Primary dealer pozisyon | NY Fed | haftalık / +8g | 0.002 | ihmal |
| ~~m1~~ | CB gross tide (4-CB) | — | — | **0** | equity-kontaminant |
| ~~m10~~ | USD (DTWEXBGS) | — | — | **0** | equity-kontaminant |
| ~~m11~~ | Collateral/repo | — | — | **0** | redundant |

> **8 modülün HEPSİNİ tut — SADELEŞTİRME.** İlk 4 ağırlığın %95'i olsa da, minik m3/m6/m8/m4 sign-eşiği (>0) yakınında **tie-breaker** rolü oynar: top-4'e indirmek SPX 1.43→1.16 / NQ 1.49→1.28 kaybettiriyor (~0.25). Parite testi kanıtladı (§9).

---

## 3. Skorlama katmanı (her modül → −20..+20)

kader-macro'nun kendi `forecast/per_module.forecast_mX` fonksiyonları üretir (Carver per-module ±20 cap — DOKUNULMAZ). Tide bu skorları **olduğu gibi** alır (yeniden hesaplama yok → makro ile birebir tutarlı):

- **m2** = `clip(net-liq 4-hafta-ort değişim × 0.2, ±20)`
- **m6** = `clip(−z(kredi spreadi) × 8, ±20)`
- **m3** = `clip((ihale-percentile − 0.5) × 40, ±20)`
- **m9** = LEVEL-bazlı fiskal-z + Fed prong (stres-koşullu: m5<0 VEYA m6<0 ise dışlanır)
- **m5 / m0** = rejim skorları, ±20 capped

---

## 4. Ağırlıklandırma katmanı

```
w = { m9: 0.563, m5: 0.214, m2: 0.118, m0: 0.061,
      m3: 0.025, m6: 0.010, m8: 0.006, m4: 0.002, m1/m10/m11: 0 }
```
- **Kaynak:** kader-macro 4000-clone Dirichlet/sparse/random-sign sweep, 2019+ FULL Sharpe ile seçilmiş en iyi vektör. TR1.41 ≈ TE1.45 (train≈test, overfit-yok kanıtı).
- **DSR DÜZELTMESİ (denetim 2026-06-13):** Eski "DSR 0.965" YANILTICI — o ~60-form OVERLAY-grid deflasyonundan; SPINE-VEKTÖR seçimi 4000-aday argmax'ı (yalnız 1/4000 Sharpe>1.40, medyan 0.71). Gerçek N=4000 ile deflate edilince **spine DSR ≈ 0.62** (0.965 değil). HAFİFLETİCİ (gerçek genelleme, DSR'nin iid varsayımının kredite edemediği): SPX-kazananı AYNI ANDA NDX argmax'ı, %93 çapraz-asset Sharpe korelasyonu; 272/4000 vektör Sharpe>1.0 geçiyor. Yani forward ~1.0-1.3 beklentisi DÜRÜST ve iyi-destekli; yalnız tekil "0.965" rakamı fazla cömertti.
- **Neden m9-dominant:** sweep equities'in fiskal-impulse'a en duyarlı olduğunu buldu (post-2019). Genel macro stance'in m2-eşit ağırlığından bilinçli olarak FARKLI.
- **Sabit vektör** — canlıda yeniden optimize EDİLMEZ (overfit kapısı). Forward-ledger ile izlenir, gerekirse sürüm yükseltilir.

---

## 5. Sinyal matematiği

```
TIDE_SCORE(t) = Σ_i  w_i · score_i(t)              # sürekli, ~±20 aralığı
TIDE_DIR(t)   = 1 (LONG)  if TIDE_SCORE(t) > 0  else  0 (FLAT)
pozisyon(t)   = TIDE_DIR(t − 1)                     # +1 gün lag → look-ahead-free
```
**+1 gün exec lag ZORUNLU:** SPX/NQ global işlem görür, sinyal US-EOD verisinden gelir → 0-lag look-ahead şişirir (metallerde 1.51→0.98 yaşandı). Lag'siz deploy = YASAK.

---

## 6. Çıktı sözleşmesi (downstream'in tükettiği)

```json
{
  "tide_dir":   1,
  "tide_score": 6.4,
  "module_scores": {"m9": 9, "m5": 3, "m2": 5, "m0": 1, "m3": 0, "m6": 2, "m8": 0, "m4": 1},
  "freshness":  {"as_of": "2026-06-08", "max_staleness_days": 5, "stale": false},
  "lag_applied": true
}
```
- `tide_dir` = birincil sinyal (LONG/FLAT).
- `tide_score` = sürekli kompozit → downstream sizing/conviction veya pozisyon-katmanıyla **interaction** için.
- `module_scores` = downstream alt-modüllerle reflexive etkileşim isterse.
- `freshness` = **KRİTİK.** Bayat veri → `stale: true` → CALL VERME. (kader-btc'de donmuş-snapshot'ı canlı-call diye sunma hatası yaşandı, Emir çok kızdı — asla tekrar.)

---

## 7. Validation & dürüst caveat'lar (downstream'i buna göre boyutla)

**2019+ (m9-çağı):** SPX **1.43** / NQ **1.49**. Buy&hold'u 3 metrikte de yener:

| | tide ALPHA | buy&hold BETA |
|---|---|---|
| SPX Sharpe | 1.43 | 0.84 |
| SPX maxDD | −19% | −34% |
| SPX PnL | +325% | +198% |
| NQ Sharpe | 1.49 | 0.97 |
| NQ maxDD | −22% | −35% |
| NQ PnL | +568% | +367% |
| Jensen α | +11% / +13% / yıl | — |
| β | 0.53 / 0.56 | 1.0 |

**Caveat'lar (irreducible — bilerek deploy et):**
1. **m9-çağı tek-rejim:** m9 yalnız 2019-08'den var → ~6 yıllık, GFC-yok, tek-fiskal-rejim sayısı. **Forward gerçekçi ~1.0-1.3** (fiskal rejim sürerse); 1.43/1.49 GARANTİ DEĞİL.
2. **m9-dominant = genç-sinyale bahis:** fiskal rejim değişirse (sıkılaşma) kenar solabilir.
3. **best-of-4000 seçim:** spine-DSR ≈ 0.62 (N=4000 deflate; eski "0.965" overlay-grid'di, §En İyi Vektör'e bkz). Seçim-iyimserliği GERÇEK ama çapraz-asset genellemesi (SPX-kazananı=NDX-argmax, %93 korr) hafifletiyor → forward ~1.0-1.3 dürüst beklenti.

---

## 8. Equities modeline entegrasyon (≈%35-60)

```
final_equity_signal = w_tide · TIDE  +  w_pos · POZİSYON  +  w_flow · AKIŞ
```
- **w_tide'ı VERİ belirler** (35? 50? 60?) — kader-btc'deki gibi **ablation** ile: tide-tek vs tide+pozisyon vs hepsi → her katmanın **incremental Sharpe**'ı.
- Tide = **defansif yön + düşük-beta (≈0.5) zemin + drawdown-yarı.** Asıl **alfa muhtemelen POZİSYON katmanından** gelir (kader-btc'de funding/OI/COT ana alfaydı).
- **Disiplin:** her downstream katmanı tide-üstü incremental alfasını **ABLATION** ile kanıtlamalı (orthogonality lensi DEĞİL — kader-btc'de bunu öğrendik).

### Önerilen downstream katmanları (kader-btc analojisi)
| Katman | kader-btc'de | kader-equity'de aday |
|---|---|---|
| Tide | net-liq+gate (1.41 spine) | **bu doküman** (equities tide) |
| Pozisyon (ana alfa) | perp funding / OI / COT | equity-index COT + opsiyon (put/call, dealer gamma/GEX) |
| Akış | ETF / stablecoin | ETF / fon akışları + breadth (A/D, % MA-üstü) |

> Sentiment-anketi (NAAIM/AAII) = Bible-ruhuna aykırı, KULLANMA. COT/put-call/gamma = pozisyon verisi, sorun yok.

---

## 9. Deploy notları (canlı-parite testi sonrası — `kader-macro/backtest/revalidation/_tide_liveparity.py`)

**Parite testi 2 KRİTİK şey buldu (deploy bunlara göre):**

| bacak | SPX Sh | NQ Sh | karar |
|---|---|---|---|
| A raw-winner (raw m2) | 1.43 | 1.49 | ✅ HEDEF |
| B winner + smart-RRP m2 (macro'da canlı) | 1.02 | 1.19 | ❌ smart-m2 bozuyor |
| C top-4 (raw m2) | 1.16 | 1.28 | ❌ sadeleştirme −0.25 |

1. **m2 = RAW net-liq (WALCL−TGA−RRP), smart-RRP DEĞİL.** kader-macro'nun m2'si artık smart-RRP → onu OKUMA. Equities tide kendi **raw** net-liq'ini hesaplar (FRED'den + `forecast_m2` transform). Sebep: m9 ile kombinasyonda raw-m2 dekorele bir 2021-risk-off (m9-hedge) verir; smart-m2 bu hedge'i kaldırır → 2021-tepe/2022 düşüşe fazla-bullish (maxDD −19%→−33%). (BTC spine'da AYNI mekanizma.)
2. **8 modülü de tut** (top-4 sadeleştirme ~0.25 kaybettirir; minik mod'lar sign-eşiği tie-breaker'ı).
3. **Reversible + flag'li**, +1g lag SABİT, freshness-gate ZORUNLU.
4. **Forward-ledger:** canlı sinyalleri kaydet, m9-çağı caveat'ını ileriye doğru izle; rejim değişirse vektörü gözden geçir.

> **Not (kader-macro tarafı):** smart-RRP standalone net-liq timer'da valide edildi ama m9-KOMBİNE modellerde (tide + BTC + muhtemelen stance) raw kazanıyor. kader-macro donmuş; smart-RRP flag'li/geri-alınabilir → ileride gözden geçirilebilir.
```

---

## 10. AS-BUILT — overlay stack (2026-06-09)

§1-9 tide spine'ı donmuş skeleton olarak tanımlar (`spine/frozen/`, kader-macro-free + ağsız reprodüksiyon).
Üstüne **2 overlay** eklendi — ikisi de **trim-only / rebound-safe** (ASLA short/add; pozisyon yalnız KISILIR).
Aday tarama (`screen/`, strict BH-FDR {SPX,NDX} İKİSİ + episode round-trip) free-backtestlenebilir TÜM
ekseni (vol-surface/SKEW/VVIX/VIX-TS/implied-corr/RV/COT-5yol/breadth/re-entry/flows/seasonality/fundamental/
econ) taradı → SADECE 2 mekanistik-ortogonal opsiyon-mikroyapı sinyali ekledi; gerisi güçlü tide ile çift-sayım.

```
nihai_pozisyon(t) = TIDE_DIR(t)  ×  COR1M-froth-factor(t)  ×  GEX-shield-factor(t)
                    (LONG/FLAT)      (∈[0,1] alfa)            (∈[0.4,1] kalkan)
```

| Overlay | Modül | Sinyal | Form | Rol | Kanıt |
|---|---|---|---|---|---|
| **COR1M-froth** | `modules/cor1m_froth.py` | CBOE 1ay implied-corr DÜŞÜK = single-stock call-froth/complacency | ramp: COR1M≤lo(8)→floor, ≥hi(11)→1.0, arası lineer | **ALFA #1** (de-risk) | 20yr bucket monotonik (<8→fwd-21g SPX −1.5%/NDX −3.2%), strict-FDR PASS 5 form (+0.15/+0.19), 3 episode (2024/25/26) |
| **GEX-shield** | `modules/gex_shield.py` | SqueezeMetrics dealer GEX z(252g) DERİN-DÜŞÜK = dealer short-gamma/kırılgan | (1−k·clip(−zg−thr,0,3)).clip(floor,1); k.5/thr1/fl.4 | **KALKAN** (alfa değil) | standalone sub-strict-FDR ama STACK'te öder (kader-btc B8): maxDD SPX −17→−13/NDX −20→−16, CVaR↓, P(stack>tide) %100 |

**Kümülatif (`screen/finalize_stack.py`, @2019+ frozen):**

| | tide | ×COR1M-froth | ×GEX-shield | DSR |
|---|---|---|---|---|
| SPX Sharpe / maxDD | 1.42 / −19% | 1.55 / −17% | **1.64 / −13%** | 0.985 |
| NDX Sharpe / maxDD | 1.49 / −23% | 1.66 / −20% | **1.77 / −16%** | 0.994 |

2 ortogonal katman → **+0.22/+0.28 Sharpe AND −6/−7pp maxDD**, P(stack>tide) %100 ikisinde.

**Veri & path:** frozen path sinyali `data/cache/{corr_pc,squeeze_dix_gex}.parquet`'ten tide as-of'ta hesaplar
(ağsız, byte-aynı); live path CBOE COR1M + SqueezeMetrics GEX CSV'sini canlı çeker. Veri yok/bayat → factor 1.0
(nötr, asla agresif). `config.yaml overlays.{cor1m_froth,gex_shield}` ile flag'li; OFF → çıktı == tide_dir (invariant test).

### 10.1 DEPLOY 2026-07-08 — COR1M-froth → **dispersion_ensemble** (HALEF, Emir-onaylı)

Froth ekseni **3-way tam-konstitüent ensemble**'a yükseltildi: `froth_pct = mean( pit(VIXEQ−VIX spread),
pit(DSPX), 1−pit(COR1M) )` [eşit-ağırlık, **FIT YOK**, PIT trailing-756g]; `factor = ramp(froth_pct; lo.70/hi.95/fl0)`.
COR1M zaten 1/3 bileşen → `cor1m_froth` SUPERSEDED (`config.enabled:false`, kod+testler intact = **geri-alınabilir**).
`modules/dispersion_ensemble.py` + `screen/fetch_dispersion.py` → `data/cache/dispersion.parquet` (VIXEQ/DSPX/VIX 2014+).

**DÜRÜST ETİKET — bu bir Sharpe-alfası DEĞİL, maxDD/TAIL upgrade'i.** Deploy-gate = MC forward-dağılım
(`mc_implied_distribution` block-bootstrap 10k, eşleştirilmiş, 3-way vs canlı):

| | ΔmaxDD p50 | P(3-way daha sığ) | ΔSharpe | P(Sh≥canlı−0.1) |
|---|---|---|---|---|
| **NDX** | **+2.1pp daha sığ** | %88-90 (blok 10/21/42) | p50 +0.07..+0.11 | %93-97 |
| **SPX** | +1.5-1.8pp daha sığ | %84-87 | p50 ≈ 0 (nötr) | %85-90 |

Nokta: SPX 1.64/−13% → **1.62/−13%** (Sharpe-nötr); NDX 1.77/−16% → **1.83/−12%**. Overfit-bataryası (T1-T4):
Sharpe-gain FWER-fail + param-hassas (bu yüzden Sharpe-alfası SAYILMAZ) AMA maxDD faydası param-robust (tüm
grid) + MC-robust (blok-bağımsız) → drawdown-kısıtlı book için (Alpha Swing bariyer) meşru. **MEKANİK:** NDX =
Mag-7 ~%50 = tekil-isim dispersion'ın ta kendisi (SPX geniş) → NDX>SPX asimetrisi artefakt DEĞİL. **ÜRÜN-NOTU:**
VIXEQ/DSPX yeni ürün → 2014+ backfill; kazanç 2022-26 konsantrasyon-rejimine yaslı = ürünün doğası (post-2020
mega-cap dispersion yapısal). Fail-closed: 3 kaynaktan biri bayat → bloke. Bugün canlı: froth_pct 0.99 → factor 0.0 = FLAT.
reproduce spine 1.43/1.49 byte-exact (overlay değişimi spine'a dokunmaz); 182 test PASS. Detay: [[kader_equity_vixeq_vix_spread_tested]].

**CAVEAT:** 2019+ m9-çağı tek-rejim (1.64/1.77 ceiling değil hedef; honest forward ~1.0-1.3). COR1M-froth
genç (2024+ rejim, 3 episode) → forward-watch. Finer gamma (flip-distance/vanna/charm) FORWARD-only —
`screen/gamma_engine.py` + `collect_daily.py` günlük topluyor. **Free-backtestlenebilir arama TÜKETİLDİ**
(detay: [[kader_equity_model]] FINDING 1-9). Kalan kenarlar PAID-gated (fwd-EPS-revisions / single-stock
skew-rank=SpotGamma / ETF-daily-flows / true-CESI) ya da FORWARD-only.

**SONRAKİ (task 3):** canlı spine rekonstrüktörü (`spine/reconstruct.py`) — kader-macro JSON `per_module.*.capped`
(m0/m3/m4/m5/m6/m8/m9) + FRED'den RAW m2 (smart-RRP DEĞİL) → frozen yerine canlı tide skoru.

---

## FINDING 24 — Constan arkı: koşullu-Noel + hisse net-arzı (2026-06-13, Emir-talebi)

Constan 101 Canon'dan iki edge test edilip modele BAĞLAM-BANDI olarak girdi (OpEx-etiket emsali;
pozisyon etkisi SIFIR; frozen stack dokunulmadı, reproduce 1.64/1.77 intact, 72/72 test).

**Koşullu Noel (santa_window.py + santa_context flag):** STANDALONE GERÇEK — 1 Kas YTD≥+10% →
Kas-Ara 1928-2025 n=43 ort +4.54% isabet %88 min −2.1%, 3 dönem stabil, perm-p<0.001; koşulsuz
yıllar ort ~0 min −22.7% (candidate_santa_conditional). STACK'E ABSORBED — nitelikli pencerede
stack zaten ~tam-long (poz 0.816, tide %86 long); V1/V2 boost + V3 ters-trim hepsi FDR-FAIL,
2024 froth-trim alfasını (+4.2pp, Aralık zayıflığı atlatması) boost SİLERDİ; V3 güvenilir zararlı
(candidate_santa_incremental, 2 adversarial denetçi 0-bloklayıcı). → Etiket: QUALIFYING_ACTIVE /
NON_QUALIFYING (kuyruk-uyarısı) / Ekim ön-izleme bantları.

**Hisse net-arzı (net_supply_context.py + fetch_net_equity_supply.py + net_supply_context flag):**
Z.1/FRED NFC net hisse ihracı (NCBCEBQ027S; anılan NCBEILQ027S LEVEL çıktı — flow değil; fin-bacağı
ETF-yaratımıyla KONTAMİNE → NFC ana motor), pub-lag +165g PIT, rolling-4Q %NGDP + z10y →
data/cache/net_equity_supply.parquet + output/net_supply_panel.txt (Constan-grafiği ikizi).
YÖN-SİNYALİ DEĞİL: tam-örneklem p=0.006 SAHTE-TREND (sinyal~zaman −0.726; Granger-Newbold);
1984+ içerik ~0, 2005+ işaret TERS, 2019+ tide-üstü 8 kural FDR-FAIL (adversarial denetim 4
bloklayıcı buldu → onarımcı doğrulayıp raporu dürüstleştirdi). → Betimsel panel + uç-kuyruk
izleme bayrağı (z10y≥+2.5 = 2000/2021 ihraç-çılgınlığı imzası; n=2 + 2020 karşı-örnek → kural değil).
Canlı haber-değeri: 2026Q1 NFC tek-çeyrek ihracı 2021Q2'den beri İLK POZİTİF (+$124bn SAAR),
z10y +1.29 yükseliyor → arz-daralması rejimi zayıflıyor; panel bu geçişi izlemek için doğru enstrüman.
Sonraki faz (opsiyonel): bileşen ayrıştırması (IPO/SPAC/buyback ayrı — 2020-vs-2021 niyet ayrımı).

Ayrıca doğrulandı (Emir sorusu): pozisyon testleri TFF kullanmış (fetch_cot=gpe5-46if lev+am;
legacy yalnız Williams-endeksi için bilinçli) — Constan'ın "TFF kullanın" önerisiyle zaten uyumlu;
TFF modelde DEĞİL (4 test tide'a FDR-FAIL) — pozisyon alfası COR1M+GEX'te.

---

## FINDING 25 — Net-arz BİLEŞEN ayrıştırması (2026-06-13): PANEL-ONLY, bayrak-yükseltme RED

Hipotez: toplam-netin yandığı yerde NİYET ayrıştırması sinyali kurtarır (2020 kurtarma-ihracı vs
2021 spekülatif arz). Workflow: veri-kurucu → ön-kayıtlı test → 2 adversarial denetçi (onarımcı
oturum-limitine takıldı → onarımı ana oturum yaptı).

**Veri (screen/fetch_supply_components.py → supply_components.parquet, 265Ç×45 kolon 1960Q1+):**
Z.1 bulk'ta bileşen YOK (Fed yalnız NET yayınlıyor — kalıcı keşif: kimse Z.1'den ayrıştırma aramasın);
SIFMA modern form-kapılı → Wayback'ten yıllık 1990-2013; Ritter aylık IPO sayısı 1960+ + SPAC yıllık
1990-2025 (PDF tablo); S&P 500 buyback çeyreklik 2008Q3-2024Q4 = 23 Wayback snapshot DİKİŞİ.
**DİKİŞ-BUG (adversarial denetim yakaladı, onarıldı):** kolon-tarama 'son BUYBACK-eşleşen' aldığı
için 2008Q3-2016Q2 birleşik buyback+temettü kolonundan gelmişti (%55-120 şişik) → saf-kolon önceliği
+ (birleşik−temettü) kimliği; düzeltilmiş değerler denetçi tahminleriyle birebir (89.7/159.3/144.1).

**Test (screen/candidate_supply_components.py, düzeltilmiş veride yeniden koşuldu):**
- H1 spekülatif-ihraç: FAIL (merdiven ~0; tek anlamlı hücre S3 2005+ −0.37 ama 1980-2004 +0.05 →
  dönem-stabil değil). H2 buyback-yield: desteklenmedi (tek nominal sonuç TERS yönde). H3 makas: toplamdan farksız.
- ANA SINAV (2000-yakala + 2021-yakala + 2020-ateşleme): HİÇBİR sinyal geçemedi. Sayım-z 2000-KÖR
  (90'ların tamamı sıcaktı → 10y taban maniayı yutuyor; çılgınlık dolarda/fiyatlamadaydı); dolar-z
  NİYET-KÖR (2009 kurtarma-ihraçlarında da ateşledi). TEK gerçek ayrışma: S3 opco-IPO sayımı
  2020-temiz (z +0.30) / 2021-yakala (z +3.26) — toplam-z'nin 2020Q3 yanlış-ateşlemesini düzeltiyor
  ama 2000-FAIL → üçlü yine düşük. Incremental 2019+ 6 kural HEPSİ negatif, FDR-PASS sıfır.
- **VERDICT: PANEL-ONLY** — bayrak/pozisyon kablosu YOK; bileşen bilgisi betimsel panele girdi.

**Kablo:** net_supply_context._components() — kaynak-bazlı PIT kapısıyla opco-IPO 4Ç sayım+z /
SPAC yıllık / SPX-buyback $+%NGDP+z (ön-değer bayraklı, evren-etiketi dürüst); run.py bileşen satırı.
Bugünkü okuma: opco-IPO 4Ç 90 adet (z −0.38, soğuk) | SPAC 2025: 144 (yeniden ısınıyor; 2021: 613) |
buyback 2024Q4 $243B = %3.22 NGDP (tarihi tepe bandı, z +0.12). 75/75 test; frozen stack dokunulmadı.

---

## FINDING 26 — Constan Q3-Q4 tezi izleme boruları (2026-06-13): buyback-tamiri + EDGAR dev-arz dedektörü

Emir talebi ("kur tabi amk çok önemli"): tezin iki bacağına izleme altyapısı. İki workflow-hattı +
4 adversarial denetçi (buyback 0-bloklayıcı; IPO 0-bloklayıcı; ilk IPO-koşusu soket hatasıyla düştü,
resume'la tamamlandı). Pozisyon etkisi SIFIR — bağlam-bandı sınıfı.

**1. Buyback 2025 tamiri (fetch_supply_components manuel-katman):** S&P DJI çeyreklik basın
bültenlerinden (prnewswire bot-açık) 2025Q1/Q2/Q3 kaynak-URL'li + çapraz-teyitli eklendi
(Q1 $293.451B REKOR / Q2 $234.57B tarife-şoku −%20 / Q3 $249.004B prelim); denetçi her rakamı
kaynak sayfasında gözüyle doğruladı + merge-önceliğini canlı sahte-satır testiyle sınadı
(xlsx kazanır, manuel yalnız boşluk doldurur). 12-aylık bülten-toplamları bizim rolling-4Q ile
kuruş-kuruş tutuyor. --check-buyback çeyreklik kontrol komutu (Wayback+bülten tarar; şu an doğru
şekilde 'manuel giriş bekliyor: 2025Q4' diyor — resmi Q4 bülteni ~3 AY GECİKMİŞ, kendisi ilginç).
NOT (verify-nb): net_supply_context buyback-yaşı bayrağı yok — pit-yaşı sessiz büyür; düşük-öncelik.
CANLI OKUMA: 12-aylık buyback $1.02tn nominal REKOR ama %NGDP %3.36'da plato (2022 zirvesi %4.04
altında) → capex-kayışı tezi HENÜZ gerçekleşmede görünmüyor; test 2025Q4+2026'da.

**2. EDGAR S-1/F-1 boru-hattı (fetch_ipo_pipeline + modules/ipo_pipeline_context):** 2001Q1-2026Q2
102/102 çeyrek-indeksi, 110k dosyalama; SB-2 yapısal-kırılması KEŞFEDİLDİ ve düzeltildi (SEC 33-8876,
2008-02: SB-2 iptali küçük kayıtçıları S-1'e göçürdü — ham seri 2008 çöküşünü gizliyordu; adj-seri
NET gösteriyor). Ritter çapraz-yön-uyumu +0.37 (yalnız yön; dosyalama=NİYET etiketi sabit).
Mega-izleme: iki-kademe ad-eşleşme (strict sayılır; tarihsel 41/41 loose-isabet yanlış-pozitifti —
tasarım kendini kanıtladı). Fee-eki parse: kayıt-tavanı $ çıkarımı çalışıyor (tavan≠arz etiketi).
**DEDEKTÖR İLK KOŞUDA ATEŞLENDİ: SpaceX (CIK 1181412) S-1 2026-05-20 + 2 değişiklik;
2026-06-03 değişikliğinde kayıt-tavanı $86,249,999,880 ($135/hisse) — Constan'ın dev-arz tezi
hipotez değil, EDGAR'da duran olgu. OpenAI/Anthropic/Stripe henüz dosyalamadı.**
Boru-hattı SAYISI ılık (2026Q1 4Ç 1523, z +0.69 — 2021 seli z +3.49 değil): tez ADET değil DOLAR
tarafında ateşliyor, onu da mega-izleme yakalıyor. Cache-fix (denetçi): kapanmış-çeyrek idx'i ancak
çeyrek-bitişi+5g sonrası indirildiyse kalıcı (sessiz-undercount önlendi).

Kablo: run.py S-1-bandı + DEV-ARZ-alarm satırı + bileşen-satırında buyback 2025Q3'e uzadı; config
ipo_pipeline_context flag; 78/78 test; frozen stack dokunulmadı. Tazeleme: fetch_ipo_pipeline
(SEC nazik-limitli) + --check-buyback çeyreklik. NOT: FINDING 25'teki 'bugünkü okuma' satırı artık
eski (buyback 2024Q4→2025Q3 uzadı) — güncel okuma her koşunun panelinde.

---

## FINDING 27 — Arz-talep dengesi (K1) + koşullu de-risk (K2): K2 POZİSYONA BAĞLI (2026-06-13)

Emir direktifi: "model toplam arzı VE talebi takip etsin + bunların pozisyona ETKİSİ olmalı". 2 builder +
2 adversarial denetçi + 1 onarımcı. İki ZIT verdict çıktı, ikisi de değerli.

**K1 — arz-talep denge kadranı (modules/supply_demand_balance.py + fetch_supply_demand_balance.py):**
Tam panel: ARZ-z (NFC net-ihraç + SIFMA + SPAC + IPO-boru + buyback-NEGATİF) vs TALEP-z (NGDP-büyüme +
buyback-POZİTİF + tide), %NGDP normalize. **ADVERSARIAL REDDETTİ (3 bloklayıcı, onarıldı):** look-ahead
(tam-örneklem z → genişleyen-z PIT'e çevrildi); ve KRİTİK — PIT-dürüst onarımdan sonra 2020-rali/2021-tepe
AYRILMIYOR (ayraç +0.59→−0.08, işaret döner; her ikisi güçlü-bearish). Bearish kayma TALEP-zayıflamasından
DEĞİL ARZ-froth'tan (SPAC+IPO patlaması). → K1 = BETİMSEL kadran, yön-AYRACI DEĞİL (etiket her çıktıda).
Bugün: 2026Q1 baskı +0.60z (arz +0.99/talep +0.39, arza kayıyor). Pozisyon etkisi YOK.

**K2 — koşullu de-risk (modules/supply_demand_derisk.py + candidate_conditional_derisk.py): POZİSYONA BAĞLI.**
Gerçek 2020/2021 ayracı denge-z değil, ham tide-DÜZEY kapısı: z10y_nfc>=+1 (arz-aşırı) VE [tide_dir=0 VEYA
(tide<=+2 VE 63g-düşüşte)]. **3 ön-kayıtlı kriter GEÇTİ + 2 doğrulayıcı ham-veriden onayladı:** 2020 H2
SUSAR (tide +5.9..+11.7 düzey-kapısı üstü → +%27 rali KESİLMEZ = tek-arz sinyalinin batı hatası giderildi);
2021 H2 ATEŞLER (tide +1.1..−0.5). DÜRÜST SINIR (verify-2): incremental ≈ 0 (kümülatif hafif negatif
SPX −0.13%/NDX −0.54%; Sharpe +0.005 kozmetik, lag-2'de döner) — çünkü tide 2021-22'yi ZATEN FLAT'a geçip
de-risk ediyor, trim'e az maruziyet kalıyor. **ALFA DEĞİL, REJİM-DEĞİŞİM SİGORTASI:** backtest'in göremediği
arz-şoku rejimi (Constan Q3-Q4: $800B-capex-kayması/dev-IPO arz dönerken tide HENÜZ dönmemişse) → trim
koruma sağlar, prim ≈ 0. **KABLO:** asset_deploy katmanı (OpEx emsali) ×0.85 — frozen position_target
DOKUNULMADI → **reproduce SPX 1.64/NDX 1.77 byte-exact teyit edildi.** config position_effect:true (Emir
"etkisi olmalı"+gate-geçti; false=gözcü modu, tek satır). Bugün SUS (arz-z 0.78<1.0). 84/84 test.

NOT: K2 forward-only/n-az/tek-rejim (2021-22) — FDR-PASS yok, beklenmiyordu; canlı forward-ledger ile
kendini kanıtlar/çürütür. Lag-fragility açık-disclosed. Emir-onaylı tasarım, gate'i geçti.

---

## FINDING 28 — TAM OTOMASYON: auto-buyback parser + run_daily Constan-refresh (2026-06-13)

Emir: "modelle ilgili manuel hiçbir şey kalmasın, ne kadar dinamik o kadar iyi." Workflow (2 builder +
2 adversarial). Son manuel adım (buyback elle-giriş) öldürüldü + Constan bantları günlük orkestratöre bağlandı.

**auto-buyback parser (screen/fetch_supply_components.py auto_pull_buyback + buyback_selfcheck):**
press.spglobal.com/prnewswire bülten listesini tarar, yeni çeyrek bültenini çeker, ham-HTML'den sektör-
TOTAL satırını parse eder (WebFetch-özeti GÜVENİLMEZ — 'Top-20 $123B' tuzağı; \bS&P 500\b kelime-sınırı +
≥4-dolar regex ile elenir). --auto-buyback CLI + run_daily'ye bağlı; --check-buyback (sadece-rapor) korundu.
**SESSİZ-BOZULMA ÖNLEYİCİ self-check (saf-fonksiyon buyback_selfcheck):** BAĞIMSIZ çapa = bültenin bastığı
önceki-çeyrek (col1) ≈ bizim CSV'deki önceki-çeyrek (±$5B; iki AYRI bülten = gerçekten bağımsız) + 12mo-
aritmetik (≥3 prior, ±$1B). Geçmezse YAZMAZ → loud-log + verify_failed (yanlış sayı asla panele girmez).
**LATENT DELİK KAPATILDI (adversarial-parser buldu):** eski <3-prior dalı aynı-kaynak anlatı-yedeğine
düşüyordu (999/999 uyumlu ama TTM=2000 çelişik → yazıyordu); col1-bağımsız-çapayla değiştirildi, 0-prior
artık REFUSE. 6 regresyon testi (Top-20-tuzağı/birim-hatası/0-prior-refuse/sahte-tek-prior hepsi BLOKLANIR).
Bugün --auto-buyback: 'güncel 2025Q3' (Q4 bülteni yok → uydurmaz). Q3 parse=249.004 birebir.

**run_daily Constan-refresh (run_daily.py _refresh_constan + _substep):** collect_daily SONRASI / brief
ÖNCESİ 4 alt-fetch: IPO-pipeline (EDGAR S-1 GÜNLÜK-taze → yeni dev-IPO aynı gün), auto-buyback, net-supply
(FRED cache-TTL 7g-gated, çeyreklik veriye günlük ağ-israfı yok), balance-derive. **BEST-EFFORT/NON-FATAL
(adversarial fault-injection ile kanıtlandı):** bir kaynak düşerse loud-log + bant graceful-stale, run_daily
ÖLMEZ exit 0; collect_daily fail-loud KORUNDU (levels-ledger time-decay). YENİ SCHEDULER TASK GEREKMEDİ —
aynı KaderEquity_RunDaily orkestratörüne bindi. reproduce SPX 1.64/NDX 1.77 byte-exact, 90/90 test.

SONUÇ: kader-equity tam-otomatik — Emir hiçbir şey koşmadan her gün (kapanış sonrası) tide tazelenir,
dev-IPO/buyback/arz-talep bantları güncellenir, dev-arz alarmı ekrana düşer, K2 de-risk canlı değerlendirilir.

---

## FINDING 29 — Mega-IPO ANLIK arz tetigi (5.5ay gecikme kapatma) + lock-up/endeks dalga izleyici (2026-06-13)

Emir: "5.5 ay kabul edilemez." K2'nin ceyreklik net-arz-z'si (Z.1 ~5.5ay gecikmeli) tek-dev-IPO anlik
sokunu gec goruyordu. Workflow (2 builder + 2 adversarial; 0 bloklayici, kritik invariant bagimsiz teyit).

K2 mega-IPO anlik kolu (modules/supply_demand_derisk.py): supply_hi = (z10y_nfc>=1.0) OR
(mega_ceiling_usd>=50B); fired = supply_hi AND demand_weak (kapi DEGISMEDI). Mega tavan
data/cache/mega_ipo_hits.json'dan PIT-okunur (MAX proposed_max_aggregate, date_filed<=as_of, 120g
pencere, gelecek-sizintisi yok). 50B esigi = tarihsel rekor ~26B Aramco -> acikca-rejim-disi tek-arz.
KRITIK INVARIANT KORUNDU (adversarial bagimsiz dogruladi): mega-IPO yalniz ARZ kolunu acar,
TALEP-ZAYIF kapisini ASLA atlamaz -> 2020-tipi guclu-talepte gelen dev IPO SUSAR (rali kesilmez).
Dogrulama: SpaceX 86.25B + tide-zayif (2021-12-13 tide -0.53) -> ATESLER (anlik); SpaceX 86.25B +
tide-guclu (2020-09-14 +8.8 / 2020-12-14 +5.9) -> SUSAR; mega tek-basina ASLA tetiklemez; mega=0 ->
eski z-only bire bir. CANLI BUGUN: SpaceX 86B kayitli ama tide +10.4 -> de-risk SUSUYOR (2020-koruma
canli is basinda). 5.5AY GECIKME KAPANDI: dev-IPO dosyalandigi gun (tide-zayifsa) ~1 gunde pozisyona etki.

Lock-up/endeks dalga izleyici (modules/ipo_supply_waves.py, BETIMSEL — pozisyon etkisi SIFIR):
Ilk-arz 86B sadece baslangic. Her mega-IPO icin gelecek arz dalgalari: (1) LOCK-UP — prospektusten
parse (SpaceX: kurucu 366-gun kilidi) -> 2027-06-04 (bugunden 356g sonra), serbest kalacak ~7.8 MILYAR
hisse = sirketin >%63'u (prospektusten birebir dogrulandi); (2) ENDEKS-DAHIL — 50B+ -> S&P/Nasdaq
~3-12ay, forced pasif talep ~13-17B (float x %15-20). Durust etiket: lock-up=arz-yukari overhang,
endeks=talep-yukari, NET zamanlama-bagimli; tahmin, kesin degil.

reproduce SPX 1.64/NDX 1.77 byte-exact (frozen position_target/asset_deploy trim DOKUNULMADI). 108 test.
NON-BLOCKING forward-rafine: mega PIT-penceresi 120g; IPO fiyatlanirsa 424B'de yeniden-atesler, 'IPO-
pending boyunca kalsin' istenirse pencere uzatilir — dusuk oncelik.

### FINDING 29-EK (2026-06-13): mega-IPO KÖRLEŞME deliği KAPATILDI (Emir "körleşmesin")

FINDING 29'da "düşük öncelik" dediğim 120g pencere GERÇEK delikti, iki katmanlıydı: (1) derisk
penceresi 120g + (2) kaynak mega_ipo_hits.json'ın recent_120d listesi de zaten 120g tutuyor →
SpaceX 120 gün sonra KAYNAKTAN düşüp tetik körleşiyordu. ÇÖZÜM: KALICI dev-IPO defteri
(data/cache/mega_active_ipos.json) — strict-watchlist dev-IPO'lar şirket-bazlı, HER dosyalama
(tarih,tavan) ayrı saklanır; 120g pencereyle SİLİNMEZ. _mega_ceiling_pit defterden okur:
PIT-max (yalnız as_of'a kadar görünen dosyalamaların tavanı; $86B 06-03'te bilindi → 06-02'de
GÖRÜNMEZ, sızıntı yok) + arm-süresi BÜYÜK dosyalamadan MEGA_ARM_MAX_DAYS=540g (bekleyen+fiyatlama+
listeleme kapsar). Doğrulandı: SpaceX 06-04→armed, 180g/455g sonra HÂLÂ armed (körleşme YOK),
573g sonra (540g cap) elenir; 06-02 PIT-temiz. 110 test, reproduce 1.64/1.77 byte-exact.
(424B/RW pricing/withdrawal tespiti eklenince arm 'listelenene kadar'a döner; 540g o zamana dek cap.)
