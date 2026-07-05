"""
run_preopen — ABD piyasası AÇILDIKTAN 1 dk sonra (09:31 ET) modeli koşturan ZAMAN KAPISI.
(Tarihsel ad "preopen"; 2026-06-16 Emir talebiyle 09:00 ET pre-open → 09:31 ET açılış+1dk'ya alındı:
canlı gamma/greek'ler ancak piyasa AÇIKKEN dolar — 09:00 ET koşusunda zincir NaN geliyordu.)

Neden kapı: Windows Task Scheduler tetiği YEREL saatle (Türkiye, sabit UTC+3) düşer; ama ABD açılışı
DST ile kayar (yazın EDT=UTC-4, kışın EST=UTC-5). 09:31 ET = yazın 16:31 TSİ, kışın 17:31 TSİ. Bu yüzden
görev İKİ tetikle çağrılır (16:31 + 17:31 TSİ); bu kapı GERÇEK ABD/Doğu saatini hesaplar ve YALNIZ 09:31 ET
penceresinde run_daily'i koşar. Yanlış tetik (yazın 17:31 TSİ=10:31 ET, kışın 16:31 TSİ=08:31 ET) sessizce
atlanır → DST otomatik halledilir, elle saat değiştirmek YOK. run_daily idempotent (gün-içi 2. koşu zararsız).

Elle: `python run_preopen.py` (kapı şu an 09:31 ET değilse ATLAR; modeli elle koşmak için run_daily.py kullan).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OPEN_HOUR_ET = 9          # ABD açılış saati (ET); açılış tam 09:30 ET
RUN_MINUTE_ET = 31        # 09:31 ET = açılıştan +1 dk (Emir 2026-06-16: canlı greek'ler için piyasa AÇIK olmalı)
WINDOW_END_MIN = 59       # 09:31–09:59 ET arası tetiği kabul et (geç düşen / StartWhenAvailable telafi tamponu)


def _us_eastern(utc: datetime) -> tuple[int, int, int]:
    """UTC → (ABD-Doğu hafta-günü 0=Pzt, saat, dakika). zoneinfo (DST-doğru) birincil; manuel kural fallback
    (tzdata yoksa). Manuel: ABD DST = Mart 2. Pazar 07:00 UTC → Kasım 1. Pazar 06:00 UTC (2007+ kuralı)."""
    try:
        from zoneinfo import ZoneInfo
        et = utc.astimezone(ZoneInfo("America/New_York"))
        return et.weekday(), et.hour, et.minute
    except Exception:
        y = utc.year
        mar1 = datetime(y, 3, 1, tzinfo=timezone.utc)
        dst_start = (mar1 + timedelta(days=(6 - mar1.weekday()) % 7 + 7)).replace(hour=7)   # 2. Pazar 02:00 EST
        nov1 = datetime(y, 11, 1, tzinfo=timezone.utc)
        dst_end = (nov1 + timedelta(days=(6 - nov1.weekday()) % 7)).replace(hour=6)          # 1. Pazar 02:00 EDT
        offset = -4 if (dst_start <= utc < dst_end) else -5
        et = utc + timedelta(hours=offset)
        return et.weekday(), et.hour, et.minute


def should_run(utc: datetime) -> tuple[bool, str]:
    wd, h, m = _us_eastern(utc)
    tag = f"ET {['Pzt','Sal','Çar','Per','Cum','Cmt','Paz'][wd]} {h:02d}:{m:02d}"
    if wd >= 5:
        return False, f"hafta sonu ({tag})"
    # F6 (denetim 2026-07-05): hafta-içi NYSE TATİLİNDE de atla — tatil farkındalığı yalnız missed_weekday'de
    # (bayrak-bastırıcı) vardı; 2026-06-19 Juneteenth'te run_daily 09:31 ET'de TAM koşmuş (kapalı piyasanın
    # donmuş zinciri snapshot'lanmış + tatil çağrısı deftere girmişti). missed_weekday tatili zaten bayraklamaz
    # → bu atlama 'beklenen'; kaçırılmış AÇIK işgünü bekçisi (EQ-5) aynen fail-loud kalır.
    try:
        from run import market_closed_reason              # NYSE tatil takvimi (tek kaynak run.py→opex_calendar)
        if market_closed_reason(_et_date(utc)) is not None:
            return False, f"NYSE tatili — beklenen atlama ({tag})"
    except Exception:
        pass                                              # run import edilemezse muhafazakâr: koşmaya devam (eski davranış)
    if h == OPEN_HOUR_ET and RUN_MINUTE_ET <= m <= WINDOW_END_MIN:
        return True, f"{tag} = açılıştan +1dk (≥09:31 ET) → KOŞ"
    return False, f"yanlış pencere ({tag}; yalnız 09:31 ET koşar)"


def _et_date(utc: datetime):
    """UTC → ABD-Doğu takvim günü (tatil kontrolü + bayrak dosya adı). zoneinfo yoksa kaba EST fallback'i
    (bayrak amaçlı; saat-hassas karar _us_eastern'da)."""
    try:
        from zoneinfo import ZoneInfo
        return utc.astimezone(ZoneInfo("America/New_York")).date()
    except Exception:
        return (utc - timedelta(hours=5)).date()


def _ran_today(utc: datetime) -> bool:
    """run_daily bugün (UTC log damgası) zaten koştu mu — output/run_daily.log işaretinden (EQ-5)."""
    log = ROOT / "output" / "run_daily.log"
    if not log.exists():
        return False
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return False
    day = f"[{utc:%Y-%m-%d}"
    return any(ln.startswith(day) and "run_daily BAŞLADI" in ln for ln in lines[-800:])


def missed_weekday(utc: datetime, ran_today: bool) -> bool:
    """EQ-5 (denetim 2026-07-04) SAF karar: bugünün koşusu KAÇTI mı? True yalnız şu üçü birden:
    (1) 09:31–09:59 ET penceresi GEÇMİŞ, (2) run_daily bugün hiç koşmamış, (3) işgünü (hafta sonu /
    NYSE tatili DEĞİL). Pencere-ÖNCESİ atlama (örn. kışın 16:31 TSİ erken tetiği = 08:31 ET) beklenen →
    False (aynı günün ikinci tetiği gelecek). Hafta sonu / tatil atlaması beklenen → False (bayrak YOK)."""
    if ran_today:
        return False
    wd, h, m = _us_eastern(utc)
    if wd >= 5:
        return False
    if (h, m) <= (OPEN_HOUR_ET, WINDOW_END_MIN):          # pencere henüz geçmedi (erken tetik) → beklenen atlama
        return False
    try:
        from run import market_closed_reason              # NYSE tatil takvimi (tek kaynak run.py'de)
        if market_closed_reason(_et_date(utc)) is not None:
            return False
    except Exception:
        pass                                              # run import edilemezse muhafazakâr: işgünü varsay (bayrak yaz)
    return True


def main() -> int:
    now = datetime.now(timezone.utc)
    run, why = should_run(now)
    stamp = f"[{now:%Y-%m-%d %H:%M:%SZ}] run_preopen"
    if not run:
        print(f"{stamp}: ATLA — {why}")
        # EQ-5 (denetim 2026-07-04): kaçırılmış İŞGÜNÜ sessizce yeşil (exit 0) görünüyordu ve gamma-ledger'ın
        # kaçan günü geri gelmez (time-decay). Exit kodu DEĞİŞMEZ (scheduler retry semantiği aynı kalır);
        # yalnız görünür bayrak dosyası + yüksek sesli satır. Hafta sonu / NYSE tatili atlaması beklenen → bayrak YOK.
        try:
            if missed_weekday(now, _ran_today(now)):
                out = ROOT / "output"
                out.mkdir(exist_ok=True)
                flag = out / f"preopen_missed_{_et_date(now):%Y%m%d}.flag"
                flag.write_text(
                    f"{stamp}: KAÇIRILMIŞ İŞGÜNÜ — 09:31 ET penceresi geçti, run_daily bugün hiç koşmadı "
                    f"({why}); gamma-ledger günü geri gelmez, elle telafi: python run_daily.py\n",
                    encoding="utf-8")
                print(f"{stamp}: ⛔ KAÇIRILMIŞ İŞGÜNÜ — run_daily bugün koşmadı ve pencere geçti → "
                      f"{flag.name} yazıldı (elle telafi: python run_daily.py)")
        except Exception as e:
            print(f"{stamp}: ⚠ kaçırılmış-gün bayrağı yazılamadı ({type(e).__name__}: {e})")
        return 0
    print(f"{stamp}: {why}")
    import time

    import run_daily
    rc = run_daily.main()
    if rc != 0:
        # Geçici veri hıçkırığına (yfinance boş zincir vb.) karşı TEK yeniden deneme — 'mutlaka' garantisi.
        print(f"{stamp}: run_daily rc={rc} → 120sn bekleyip TEK yeniden deneme...")
        time.sleep(120)
        rc = run_daily.main()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
