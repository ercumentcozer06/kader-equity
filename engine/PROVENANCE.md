# kader-equity ENGINE — Karar Provenance

Katman 3 (engine/ledger/execution) parametrelerinin kaynağı. Katman 1-2 (frozen winner vektörü + COR1M-froth/
GEX-shield overlay'leri) AYRI ve dokunulmaz (bkz. `spine/frozen/provenance.json`).

Etiketler:
- **a priori policy, analysis-derived** = frozen-log analizinden TÜRETİLDİ, getiriye FIT EDİLMEDİ → DSR trial sayacına YAZILMAZ.
- **fitted** = bir grid/sweep'te arandı → DSR-muhasebesinde (bkz. `screen/finalize_stack.py` N_TRIALS notu).

## FAZ GEÇİŞİ kararları (2026-06-10) — a priori policy, analysis-derived

| Parametre | Değer | Kaynak / gerekçe | Nerede |
|---|---|---|---|
| **min_dte** | 21 | GÖREV 2 horizon-analizi: TIDE LONG-run ort 35g, ACF21=0.79, gün-1 getiri −0.22% (NEGATİF), edge g2-45'te, intraday-2DTE run'ın %6'sını + negatif-gün1'i yakalar. Rejim YAPIYI seçer, vadeyi 21 altına indiremez. | `config_accounts.yaml` options.min_dte; `engine/trade.py` |
| **max_dd_halt** | SPX −19.8 / NDX −23.4 / 50-50 −21.6 (%) | GÖREV 4: %2-per-event gap bütçesi KALDIRILDI (0.97/5.8≈0.17 anlamsız). Yerine pre-registered halt = in-sample (frozen stack) maxDD × 1.5. Frozen stack maxDD: SPX −13.2/NDX −15.6/50-50 −14.4. Gap riski = bilinen yapısal kör nokta. | `config_accounts.yaml` live_book.max_dd_halt_pct |
| **live_book = delta_one** | SPX→SPLG, NDX→QQQM | RİSK-1: $1000'da opsiyon friction %86-120 → ekonomik değil. Canlı kitap delta-one ETF directional; horizon-analizi de haftalarca-tut diyor (directional buna uyar). Opsiyon ENGINE'de paper-forward. | `config_accounts.yaml` live_book |
| **options_unlock_min_account** | $10,000 | friction.py: "beklenen friction ≤ risk × 0.15" koşulunu sağlayan min hesap. $5k: %17-24 RED; $10k: %8.6-12 OK. O eşiğe dek canlıda opsiyon yok. | `config_accounts.yaml` live_book.options_unlock_min_account_usd |

## GÖREV 5 (önceki faz) eşik etiketleri — özet
- COR1M lo=8 A PRİORİ (SpotGamma), hi=11/floor=0 FITTED. GEX thr=1.0/floor=0.4 A PRİORİ, k=0.5 FITTED, win=252 A PRİORİ-hiç-taranmadı.
- DSR N=60 İYİMSER (sadece COR1M+GEX grid); dürüst N≈150-200 → SPX ~0.96 / NDX ~0.98 (0.985/0.994 DEĞİL). Detay: `screen/finalize_stack.py`.

## prop_sim / playbook (FAZ GEÇİŞİ)
- prop_sim pozisyon taraması 0.4x–1.0x = **sizing taraması, sinyal-fit DEĞİL** (a priori etiketli; DSR'ye yazılmaz).
- gex_playbook kural-1 "max 3 gün erteleme" = a priori SABİT (taranmaz).
- G3 toplamı DSR trial sayacına **1 analiz** olarak işlenir.
- FTMO Swing 2-Step kuralları (2026-06, ftmo.com/trading-objectives doğrulandı): P1 +10% / P2 +5% / günlük −5% (00:00 CET) / toplam −10% statik / min 4 işlem günü / swing hafta-sonu+haber serbest, consistency yok.
