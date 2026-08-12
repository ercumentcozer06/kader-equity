# -*- coding: utf-8 -*-
"""TEST-KACAGI BEKCISI regresyon testi (2026-08-11) — KANONIK, 4 repoda es.

BULGU (Emir 2026-08-11'de canli alarm dosyasini gorup "fiyat kaynagi 2 aydir bayat mi?"
diye sordu; DEGILDI): modellerin alarm kolu TESTLERDE de gercekten calisiyordu.
Testler ledger/run_daily'nin ROOT'unu tmp'ye monkeypatch'liyor, ama notify'in ROOT'u
`__file__` tabanli oldugu icin bundan ETKILENMIYOR -> her `pytest` kosusu GERCEK repoya
output/STALE_ALERT.json yaziyordu (webhook doluysa telefona GERCEK push).

Olculen zarar:
  * kader-equity/output/STALE_ALERT.json = "fiyat-kaynagi BAYAT 43 isgunu (son 2026-06-11)"
    -> 2026-06-11 bir test FIXTURE'inin sentetik serisiydi; canli besleme SAGLAMDI
    (yfinance 5/5 taze, cache 2026-08-10, ledger price_stale hepsi False).
  * ayni sinif .claude/worktrees/*/output/ icinde 2026-08-06'da da olusmustu (tekrarlayan).
Sahte alarm en pahali arizadir: alarm kanalinin guvenilirligini yok eder.

KURAL: pytest altindayken ROOT hala GERCEK repo koku ise notify YAZMAZ/GONDERMEZ.
Notify'in KENDISINI test eden dosyalar ROOT'u tmp'ye tasidigi icin ETKILENMEZ.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import notify


def test_guard_active_under_pytest_with_real_root():
    """Su an pytest altindayiz ve notify.ROOT gercek repo koku -> bloke OLMALI."""
    assert Path(notify.ROOT).resolve() == notify._REAL_ROOT
    assert notify._test_leak_blocked() is True


def test_alert_does_not_touch_real_repo(capsys):
    """REGRESYON: test icinden alert() gercek output/STALE_ALERT.json YAZMAMALI."""
    real = notify._REAL_ROOT / "output" / "STALE_ALERT.json"
    before = real.exists()
    res = notify.alert("REGRESYON-TESTI (yazilmamali)", "bu dosya olusursa kacak geri geldi")
    assert res.get("test_blocked") is True and res["fired"] == []
    if not before:
        assert not real.exists(), f"TEST KACAGI: gercek repoya alarm yazildi -> {real}"


def test_clear_alert_does_not_delete_real_artifacts(tmp_path):
    """Ters yon: test, GERCEK bir alarmi silip Emir'in gozunden kacirmamali."""
    real = notify._REAL_ROOT / "output" / "STALE_ALERT.json"
    real.parent.mkdir(exist_ok=True)
    created = False
    if not real.exists():
        real.write_text('{"ts":"probe"}', encoding="utf-8")
        created = True
    try:
        notify.clear_alert()
        assert real.exists(), "TEST KACAGI: test gercek alarmi SILDI (sessiz kor nokta)"
    finally:
        if created:
            real.unlink()


def test_redirected_root_still_works(tmp_path, monkeypatch):
    """ESCAPE HATCH: ROOT'u tmp'ye tasiyan testler (notify'in kendi testleri) CALISMAYA devam eder."""
    monkeypatch.setattr(notify, "ROOT", tmp_path)
    monkeypatch.setattr(notify, "_ALERT_STATE", tmp_path / "output" / ".alert_state.json")
    monkeypatch.setattr(notify, "_cfg_alert", lambda: {})
    assert notify._test_leak_blocked() is False
    notify.alert("izole", "tmp'ye yazilmali")
    assert (tmp_path / "output" / "STALE_ALERT.json").exists()


def test_production_path_unaffected(monkeypatch):
    """Uretimde pytest yuklu degil -> bekci KAPALI, davranis birebir eski."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    assert notify._test_leak_blocked() is False
