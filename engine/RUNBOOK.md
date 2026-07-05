# KADER-EQUITY — GÜNLÜK RUNBOOK (Emir, ~10 dk ritüel)

FTMO eval/funded hesabı için günlük operasyon. Sistematik EOD + tutarlı boyut. Brief `run_daily.py` ile otomatik üretilir (hafta-içi kapanış sonrası, Task Scheduler).

## Adımlar (sırayla)

1. **Brief'i aç** (`output/kader_equity_brief_YYYYMMDD_<ticker>.json` ya da terminal) → oku:
   - rejim / model exposure / **lot-delta** (translator satırı)
2. **Delta emrini gir** (varsa): brief'teki `order_str` (ör. `US100: 0.03 SAT`). Yoksa pozisyon sabit.
3. **Felaket-stopu güncelle** (A1 EVET ise — funded fazda): brief'teki `stop_level` endeks fiyatını platforma stop olarak gir. (Eval fazda stop opsiyonel/HAYIR — A1: marjinal + %76 whipsaw.)
4. **Platform-equity'yi tracker'a gir** (30 sn): hesabın güncel equity'sini `prop_tracker.append_day(..., platform_equity=...)` ile kaydet → execution-drag ölçülür.
5. **Alarm varsa İŞLEM YOK:** brief'te `VERİ ÇÖP` / `VERİ BAYAT` / `OVERLAY FAIL-SAFE` / STALE → **trade etme**, alarm notunu oku, ertesi gün tekrar.

## KURALLAR (pre-registered — ihlal = sistemi bozar)

- **Mid-eval boyut/kural değişikliği YASAK.** Sim'lenmemiş hiçbir override yok (Niederhoffer: sim-dışı discretionary müdahale = felaket). config'te `eval_pos` değişimi **"pre-registered policy ihlali" FLAG** basar.
- **Haber günü pozisyon kapatma YOK.** Swing hesabı tam bunun için seçildi (hafta-sonu/haber serbest). Panikle kapatma yok.
- **FTMO Kural 7.3 uyumu:** sistematik EOD + tutarlı boyut + tek strateji = sorun değil (HFT/arbitraj/copy-trade değil). Davranış tutarlı kalsın.
- **Payout politikası (pre-register):** funded'da **ilk uygun tarihte payout al** (counterparty riskini minimize et — parayı çek). İlk payout earmark:
  1. **€155 FTMO fee iadesi** (ilk payout'ta gelir),
  2. **$599 ORATS** tarihsel chain (kârdan) → Ş2B retroaktif TAM açılır (intraday GEX sleeve + swing kural 2-3 gerçek-veriyle test).

## Asimetrik politika (özet)
- **Eval:** agresif (yüksek-pos, sadece fee riski; fee iade-edilir → kill ucuz). Stop opsiyonel.
- **Funded:** **0.6 sabit** + felaket-stop EVET (survival %94→%100). Payout'u koru.

## Alarm referansı (brief'te görürsen)
| Alarm | Anlam | Aksiyon |
|---|---|---|
| VERİ ÇÖP | dataguard kapısı patladı | işlem yok |
| OVERLAY FAIL-SAFE | GEX-z bayat >5g | işlem yok |
| VERİ BAYAT / STALE | snapshot/makro eski | temkinli, tercihen işlem yok |
| coarse_flag (translator) | lot-adımı kaba (cs büyük) | exposure hedefe oturmuyor — kabul/US500-düşün |
