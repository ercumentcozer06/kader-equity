# kader-equity ince-tarama denetimi 2026-07-11 — KAPANDI (Emir emriyle aynı gün fix'lendi)

3 kapsamlı ajan (macro/deira ile aynı yöntem; canlı yol tam okundu, screen/backtest yalnız sınır-taraması),
36 bulgu: **1 P0, 6 P1 (→ 3 kök), 16 P2, 13 P3**. Üç kök elle satır satır doğrulandı.
Tam liste: seans arşivi `witma6cxe.output` / `scratchpad/equity_findings.json`.

## KÖK A — P0: signal_pnl kısmi-bar ile mühürleniyor (validation/ledger.py:98,135)

Tek zamanlanmış koşu 09:31 ET (pre-open+1dk). O saatte `yf.history(period="2y")` bugünün DEVAM EDEN
barını içerir → `fwd1[T-1] = close[T-1] → fiyat@09:31(T)` ve **dünkü satırın signal_pnl'i kısmi
getiriyle FINAL gibi yazılır**, gün boyu öyle kalır (aynı gün ikinci koşu yok). dei-ra bu kolonu
risk-parity paneline biriktiriyor → equity accrual'ı günün tüm seans hareketini kaçırır, ertesi sabah
sessizce değişir (yeniden-üretilemez panel vintage). `computed_at` damgaları seans-içi koşuyu teyit
ediyor. Fix yönü: bugüne ait bar kapanmadan mark'a girmesin (`cl = cl[cl.index < bugün]`).

## KÖK B — P1: price_stale en kötü durumda False'a döner (ledger.py:155)

`price_stale = (not has_next) and src_stale_bd>2 and as_of <= clmax+10g`. yfinance ölüp frozen-fallback'e
düşülünce (`clmax=2026-05-22` sabit) 10 günden derin kesintide **yeni satırların hepsi False alır** —
deira'nın gate'lediği alan tam feed öldüğünde "temiz" okunur; H4 alarmı yalnız print. Ek: append sonrası
mark çökerse son satır `price_stale=None` (falsy) kalır → deira temiz sanır. Test yalnız ≤10g dalını
kapsıyor. Fix yönü: `as_of >= clmax` olan işaretlenemeyen satırlar koşulsuz True + notify.alert.

## KÖK C — P1: degraded tide sessizce "current" (run.py:206 + spine/tide.py:53)

`tide.decide()` eksik modülü fail-VISIBLE bayraklıyor (`degraded`, JSON'da `tide_degraded` var) ama bu
bayrak **stale/call_status hesabına, ledger kolonlarına ve `_alert_if_degraded` push'una girmiyor**.
Senaryo: kader-macro grid'inin son satırında m9 NaN (vektör ağırlığının %56'sı) → tide 0'a çöker →
position_target=0 "current" damgasıyla deftere iner, kitabın en büyük sleeve'i hayalet sinyalle
sıfırlanır, push yok. (46-günlük gölge olayıyla aynı sınıf drift kanalı.) Fix yönü: missing_weight_frac
eşiği aşınca stale'e katla + alert + ledger kolonu.

## P2 (16; öne çıkanlar)

forward_ledger.parquet non-atomik (deira'nın okuduğu dosya, crash=bozulma); aynı-gün re-run satırı
sessizce yeniden yazıyor (deira'nın tükettiği değer değişir, iz yok); 2026-06-19 Juneteenth hayalet-satırı
hâlâ defterde (forward getiri çift sayım); `run.py` CLI append yolu mark'sız → son satır price_stale=None;
reconstruct canlı-panel subprocess'i timeout'suz (run_daily süresiz asılabilir); gün-cache 09:31 panelini
tüm UTC-günü pinliyor; 07-07 gölge-fix'inin regresyon testi YOK; dispersion CBOE CSV kolonu pozisyonla
seçiliyor (şema kayarsa sessiz yanlış); notify token-sızıntı (macro'dakiyle aynı sınıf).

## P3 (13; öne çıkanlar)

heartbeat ölü-adam anahtarı hiç yazılmayan `output/latest.json`'ı okuyor (asıl dosya kader_equity_latest.json)
ve kendisi zamanlanmamış; K2 de-risk evaluate her exception'ı yutuyor (pozisyon-etkili trim sessiz devre dışı);
UTC-takvimle market_open/OpEx sınıflaması (geç-akşam ET koşusu yanlış gün); collect_daily best-effort
subprocess'leri timeout'suz; H4 testi >10g dalını hiç sürmüyor; wilson_lo(0,0) totolojik assert.

## Yöntem notu
Bilinen-kapalı sınıflar (07-07 gölge-fix mevcudiyeti — ama regresyon testi eksik bulundu; DIX/VRP red;
dispersion_ensemble tasarımı; alpaca kapanışı; hafta-sonu yazmama davranışı) yeniden raporlanmadı.
Sınır temiz: canlı yol screen/backtest/lab-script import etmiyor.


## KAPANIŞ (07-11 gece, Emir emri: "bütün hataları fixle")
36/36 uygulanabilir bulgu kapatıldı; 7 yeni FAIL-dalı testi (`tests/test_audit_2026_07_11.py`), paket **189/189**.
- KÖK A (P0): seans açıkken bugünün devam-eden barı marka girmiyor (`cl = cl[< bugün]` @ ET<17:00) →
  dün dürüstçe BEKLEMEDE; kapanış sonrası koşu final barla skorlar. Enjekte-closes (test) yolu etkilenmez.
- KÖK B (P1): price_stale koşulu `(not has_next) and src_stale_bd>2` — clmax+10g tavanı kalktı (derin
  kesinti artık STALE); append default'u fail-closed True (mark çökerse None-temiz deliği kapalı);
  H4 alarmı notify.alert push'una çıkarıldı.
- KÖK C (P1): degraded tide (kayıp ağırlık >%5) → stale zinciri ("current" basamaz, F8 defteri keser)
  + push-alarm sebebi + log.error.
- P2/P3: forward_ledger + tüm kayıtlar atomik (tmp+replace); aynı-gün re-run yeniden-yazımı BAĞIRIYOR;
  tatil-hayalet satırlar skorlanmıyor (Juneteenth çift-sayımı kapandı — legacy satır artık None);
  yıkıcı re-mark koruması (>21g dolu signal_pnl korunur); CLI append sonrası best-effort mark;
  reconstruct + collect_daily subprocess'leri timeout'lu; gün-cache 4 saat TTL (akşam deira koşusu
  sabah panelini yemez); heartbeat doğru dosyayı okuyor (kader_equity_latest.json); K2 de-risk
  hatası ERROR-görünür; CBOE kolonu isim-öncelikli + band-kontrollü; notify token-maskeli;
  market_open/OpEx ET-gününe göre; 07-07 gölge-fix regresyon-kilidi + wilson totolojisi düzeltildi.
Davranış notu: hepsi fail-closed/görünürlük sınıfı — Sharpe-etkili mimari değişiklik yok.
Kalıntı: heartbeat görevinin zamanlayıcıya KAYDI hâlâ yok (OS-level, Emir'in schtasks adımı).
