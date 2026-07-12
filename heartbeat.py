"""heartbeat — DEAD-MAN switch (P0-B, denetim 2026-07-07). run_daily'nin KENDİSİ hiç koşmazsa yakalar.

En sinsi bayatlık modu: run_daily.py sessizce çökerse / Task Scheduler devre-dışıysa / makine planlı-saatte kapalıysa
→ hiçbir şey koşmaz → dünkü latest.json "current" diye durur, kimse fark etmez. run_daily-içi alarm (P0-A) bunu
yakalayamaz çünkü run_daily HİÇ çalışmadı. Bu yüzden AYRI, hafif, bağımsız bir bekçi gerekir.

Bu script AYRI bir Task Scheduler görevinden koşar (günde birkaç kez). Kontrol: son BAŞARILI run_daily ne kadar
eski? > MAX_SILENCE_H (varsayılan 36s = hafta-sonu+1-outage toleransı) ise → notify.alert (push) + exit 1.
Sağlıklıysa sessiz + exit 0. Piyasa-kapalı günleri (hafta-sonu/tatil) hoş görür.

  & <venv python> heartbeat.py
  Task Scheduler: register_task.ps1 içine ayrı 'KaderEquity_Heartbeat' (S4U, 09:00/13:00/18:00).
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MAX_SILENCE_H = 36.0        # (ESKİ/kullanılmıyor) ham-saat eşiği hafta-sonu boşluğunu (~64-72s) SAHTE-tetikliyordu
MISS_TRIGGER = 2            # 07-12 FIX: 2+ İŞGÜNÜ kaçınca ALARM (hafta-sonu/tatil sayılmaz = gerçek "hafta-sonu+1 outage")


def _trading_days_missed(last: "datetime", now: "datetime") -> int:
    """Son başarıdan bu yana KAÇIRILAN işgünü (bugün HARİÇ; hafta-sonu+tatil sayılmaz). Piyasa-tatil
    takvimi varsa kullanır, yoksa hafta-içi'ne düşer (hafta-sonu sahte-tetiği yine de kapanır)."""
    try:
        from modules.opex_calendar import is_market_holiday
        _hol = is_market_holiday
    except Exception:
        def _hol(_d):
            return False
    from datetime import timedelta
    n, d, today = 0, last.date() + timedelta(days=1), now.date()
    while d < today:
        if d.weekday() < 5 and not _hol(d):
            n += 1
        d += timedelta(days=1)
    return n


def _last_success_utc() -> datetime | None:
    """run_daily.log'daki son '✓ run_daily BİTTİ' satırının UTC zaman-damgası."""
    p = ROOT / "output" / "run_daily.log"
    if not p.exists():
        return None
    last = None
    pat = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})Z\].*run_daily BİTTİ — tüm adımlar OK")
    try:
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = pat.search(line)
            if m:
                last = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None
    return last


def _market_open_today() -> bool:
    try:
        from modules.opex_calendar import is_market_holiday
        import numpy as np
        today = datetime.now(timezone.utc).date()
        return today.weekday() < 5 and not is_market_holiday(today)
    except Exception:
        return datetime.now(timezone.utc).weekday() < 5


def main() -> int:
    now = datetime.now(timezone.utc)
    last = _last_success_utc()
    latest = ROOT / "output" / "kader_equity_latest.json"   # Denetim 07-11 P3 ([23]): eski 'latest.json' HIC yazilmiyordu — olu-adam anahtari koru okuyordu
    call_status = None
    if latest.exists():
        try:
            call_status = json.loads(latest.read_text(encoding="utf-8")).get("call_status")
        except Exception:
            pass

    if last is None:
        _fire("run_daily HİÇ başarıyla koşmamış (log'da 'BİTTİ OK' yok)", now)
        return 1
    silence_h = (now - last).total_seconds() / 3600.0
    missed = _trading_days_missed(last, now)
    print(f"heartbeat: son başarılı run_daily {last:%Y-%m-%d %H:%MZ} ({silence_h:.1f}s / "
          f"{missed} işgünü önce) | latest.call_status={call_status} | piyasa-bugün-açık={_market_open_today()}")

    # 07-12 FIX: yalnız 2+ İŞGÜNÜ kaçınca ateşle — hafta-sonu/tatil boşluğu artık SAHTE-tetiklemez
    if missed >= MISS_TRIGGER:
        _fire(f"run_daily {missed} İŞGÜNÜ koşmadı (son başarı {last:%Y-%m-%d %H:%MZ}, "
              f"{silence_h:.0f}s; call_status={call_status}) — koşu düşmüş olabilir", now)
        return 1
    return 0


def _fire(reason: str, now: datetime) -> None:
    try:
        import notify
        notify.alert("DEAD-MAN: run_daily sessiz", reason)
    except Exception as e:
        print(f"[{now:%Y-%m-%d %H:%MZ}] DEAD-MAN ALARM (notify başarısız {e}): {reason}")


if __name__ == "__main__":
    raise SystemExit(main())
