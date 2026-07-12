# -*- coding: utf-8 -*-
"""Anti-spam testleri (2026-07-12; Emir 'telegramda saçma mesajlar'): heartbeat hafta-sonu
SAHTE-tetiklemez + notify.alert aynı-konuyu cooldown içinde dedup'lar (KANONİK sürüm; 5 repo eş)."""
from datetime import datetime, timezone

import notify
from heartbeat import _trading_days_missed


def _utc(y, m, d, h=12):
    return datetime(y, m, d, h, tzinfo=timezone.utc)


def test_heartbeat_no_false_fire_over_weekend():
    # Cuma başarı → Pazartesi sabahı: hafta-sonu SAYILMAZ → 0 kaçırılan işgünü → ATEŞLEMEZ
    assert _trading_days_missed(_utc(2026, 7, 10), _utc(2026, 7, 13, 9)) == 0   # Cuma→Pzt
    assert _trading_days_missed(_utc(2026, 7, 10), _utc(2026, 7, 12)) == 0      # Cuma→Pazar
    # Cuma başarı → Çarşamba: Pzt+Sal = 2 işgünü kaçtı → ATEŞLER (gerçek arıza)
    assert _trading_days_missed(_utc(2026, 7, 10), _utc(2026, 7, 15)) == 2
    # tek-outage toleransı: 1 işgünü kaçış eşik-altı (MISS_TRIGGER=2)
    assert _trading_days_missed(_utc(2026, 7, 10), _utc(2026, 7, 14)) == 1      # Cuma→Salı = yalnız Pzt


def _isolate(tmp_path, monkeypatch):
    """notify'ın ROOT + cooldown-durum dosyasını izole tmp'ye yönlendir; SAHTE-kanal kur (cooldown
    ancak GERÇEK kanal-gönderiminden sonra devreye girer → _post'u True'ya sabitle + dummy webhook)."""
    (tmp_path / "output").mkdir()
    monkeypatch.setattr(notify, "ROOT", tmp_path)
    monkeypatch.setattr(notify, "_ALERT_STATE", tmp_path / "output" / ".alert_state.json")
    monkeypatch.setattr(notify, "_cfg_alert", lambda: {})           # config-webhook'u devre-dışı
    monkeypatch.setattr(notify, "_post", lambda url, payload, timeout=10: True)  # kanal "başardı"
    monkeypatch.setenv("KADER_ALERT_WEBHOOK", "https://example.test/hook")        # bir kanal mevcut
    for k in ("KADER_ALERT_TELEGRAM_TOKEN", "KADER_ALERT_DISCORD"):
        monkeypatch.delenv(k, raising=False)


def test_notify_dedup_suppresses_repeat_but_not_new(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(notify, "_COOLDOWN_H", 72.0)
    r1 = notify.alert("VERİ BAYAT", "SPX stale 3g")
    r2 = notify.alert("VERİ BAYAT", "SPX stale 4g")     # gövde yaş değişti, KONU aynı → dedup
    r3 = notify.alert("YENİ ARIZA", "NDX feed down")    # farklı konu → geçer
    assert r1["suppressed"] is False
    assert r2["suppressed"] is True                     # günlük yaş-artışı SPAM YAPMAZ
    assert r3["suppressed"] is False                    # gerçek yeni sorun yine bildirilir


def test_notify_clear_alert_resets_cooldown(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(notify, "_COOLDOWN_H", 72.0)
    assert notify.alert("VERİ BAYAT", "x")["suppressed"] is False
    assert notify.alert("VERİ BAYAT", "x")["suppressed"] is True   # cooldown içinde susar
    notify.clear_alert()                                            # düzelme → durum sıfır
    assert notify.alert("VERİ BAYAT", "x")["suppressed"] is False  # nüks ANINDA yeniden alarmlar


def test_notify_cooldown_zero_disables_dedup(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(notify, "_COOLDOWN_H", 0.0)
    assert notify.alert("X", "y")["suppressed"] is False
    assert notify.alert("X", "y")["suppressed"] is False   # <=0 → her zaman gönder
