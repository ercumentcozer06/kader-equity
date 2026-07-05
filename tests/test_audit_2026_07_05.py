"""
Denetim 2026-07-05 fix testleri — takvim-farkındalıklı tazelik (yanlış-STALE alarmlarını bitir,
fail-closed'ı KORU: gerçek donma yine YÜKSEK SESLE STALE).

  • F1 — MTS (MTSO/MTSR133FMS) release-aware: ay-başı damga + ertesi-ay-8.-işgünü yayın takvimi;
    düz 45bd tavanı her ay ~5 işgünü yanlış-STALE atıyordu (ilk ateşleme 2026-07-04/05, aylık defter
    deliği). Yeni: bayat ⇔ bugün > beklenen-yayın+2bd; + FRED last_updated donma-dedektörü (>45 takvim g).
  • F2 — WDTGAL haftalık H.4.1 (Çarşamba seviyesi, Perşembe yayın): 'günlük' 5bd → 8bd (kardeşi WALCL).
  • F3 — GEX-shield kaynak-donma kapısı: CSV tümüyle donarsa (son satır geçerli z) sessiz 'canlı' YOK →
    işlem-günü yaşı >4 ⇒ fail-loud blok.
  • F4 — SPECS yaş sayacı ABD-federal-tatil farkındalıklı (2026-07-03 işgünü SAYILMAZ).
  • F5 — NYSE tatil takvimi tek kaynak opex_calendar (sabit 2025-27 frozenset'in 2027+ sessiz sona
    ermesi bitti; MLK 2027 / Good Friday 2027 doğru).
  • F6 — run_preopen.should_run hafta-içi NYSE tatilinde koşmaz (Juneteenth-tipi tatil koşusu biter).
  • F7 — collect_daily kapalı günde (hafta sonu/tatil) defterlere HAYALET satır yazmaz.
  • F8 — forward-ledger kapısı run.ledger_eligible (TEK kaynak): current + overlay_block yok + piyasa AÇIK.

Hepsi saf/mock — ağ yok.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import spine.reconstruct as R  # noqa: E402


# ── F1: MTS release-aware takvim ─────────────────────────────────────────────

def test_f1_mts_next_release_schedule():
    """Yayın takvimi: obs+2 ayının 8. federal-işgünü (Mayıs→13 Tem: 3 Tem gözlenen-4-Temmuz tatili
    dahil; Nisan→10 Haz = gerçek takvimle birebir; Kasım→13 Oca 2027: 1 Oca tatili + yıl taşması)."""
    assert R._mts_next_release(date(2026, 5, 1)) == date(2026, 7, 13)
    assert R._mts_next_release(date(2026, 4, 1)) == date(2026, 6, 10)   # Mayıs MTS gerçekte 06-10'da geldi
    assert R._mts_next_release(date(2026, 11, 1)) == date(2027, 1, 13)  # yıl taşması + Yılbaşı tatili


def test_f1_mts_on_schedule_not_stale():
    """Yayın-öncesi pencere (2026-07-04..07-13 + grace 07-15): obs 2026-05-01 TAZE sayılır.
    Eski düz-45bd kural 07-04/05'te '46bd' ile ateşlemişti (canlı yanlış alarm) — artık None."""
    lu = "2026-06-10 14:04:38-05"                          # gerçek FRED meta (Mayıs MTS tam takviminde)
    for day in (4, 5, 6, 8, 10, 13, 14, 15):               # 13=yayın günü, 14-15=grace (2bd)
        assert R._check_spec("MTSO133FMS", "m9", None, date(2026, 5, 1), date(2026, 7, day),
                             last_updated=lu) is None, f"2026-07-{day:02d} yanlış-STALE"
        assert R._check_spec("MTSR133FMS", "m9", None, date(2026, 5, 1), date(2026, 7, day),
                             last_updated=lu) is None


def test_f1_mts_past_grace_goes_stale_loud():
    """FAIL-CLOSED korunur: beklenen yayın (07-13) + 2bd grace (07-15) GEÇTİ → yüksek sesle bayat."""
    v = R._check_spec("MTSO133FMS", "m9", None, date(2026, 5, 1), date(2026, 7, 16),
                      last_updated="2026-06-10 14:04:38-05")
    assert v is not None and v["series"] == "MTSO133FMS"
    assert v["expected_release"] == "2026-07-13" and "yayın" in v["reason"]


def test_f1_mts_frozen_meta_detector():
    """Donma-dedektörü: takvim penceresi içinde bile FRED last_updated >45 takvim günü eskiyse
    kaynak GERÇEKTEN donmuş → STALE (sessiz geçiş yok). last_updated yoksa takvim kapısı tek başına."""
    v = R._check_spec("MTSO133FMS", "m9", None, date(2026, 5, 1), date(2026, 7, 10),
                      last_updated="2026-05-20 14:00:00-05")          # 51 takvim günü eski
    assert v is not None and "DONMUŞ" in v["reason"] and v["meta_age_cal"] == 51
    # meta yok (eski cache dosyası) → takvim kapısı karar verir, yanlış alarm yok
    assert R._check_spec("MTSO133FMS", "m9", None, date(2026, 5, 1), date(2026, 7, 10),
                         last_updated=None) is None


def test_f1_mts_after_release_lands_fresh():
    """Haziran MTS 07-13'te düşünce (obs 2026-06-01, last_updated 07-13) → bir sonraki pencereye kadar taze."""
    assert R._check_spec("MTSO133FMS", "m9", None, date(2026, 6, 1), date(2026, 7, 14),
                         last_updated="2026-07-13 14:00:00-05") is None


# ── F2: WDTGAL haftalık H.4.1 ────────────────────────────────────────────────

def test_f2_wdtgal_thursday_prerelease_fresh():
    """Perşembe 09:31 ET yayın-öncesi (2026-07-09, obs 2026-07-01 Çarşamba) = takvimin İZİN VERDİĞİ
    en yaşlı an → TAZE (eski 'günlük' 5bd tavanı burada her hafta yanlış-STALE atacaktı)."""
    assert R._check_spec("WDTGAL", "m2", 8, date(2026, 7, 1), date(2026, 7, 9)) is None


def test_f2_wdtgal_frozen_still_fires():
    """2+ kaçık haftalık yayın (obs 2026-06-17, bugün 07-09) → yüksek sesle bayat (fail-closed)."""
    v = R._check_spec("WDTGAL", "m2", 8, date(2026, 6, 17), date(2026, 7, 9))
    assert v is not None and v["age_bd"] == 14 and v["max_bd"] == 8


def test_f2_specs_pinned():
    """SPECS kalibrasyonu regresyon-kilidi: WDTGAL=8 (haftalık, WALCL kardeşi), MTS satırları
    release-aware (max_bd=None), diğerleri sweep-doğrulanmış değerlerinde."""
    d = {sid: mx for sid, _mod, mx in R.SPECS}
    assert d["WDTGAL"] == 8 and d["WALCL"] == 8
    assert d["MTSO133FMS"] is None and d["MTSR133FMS"] is None
    assert d["RRPONTSYD"] == 4 and d["VIXCLS"] == 4 and d["DTWEXBGS"] == 9
    assert d["DBAA"] == 4 and d["DAAA"] == 4


# ── F4: federal-tatil farkındalıklı yaş sayacı ───────────────────────────────

def test_f4_holiday_week_age_counts_trading_days():
    """2026-07-03 (gözlenen 4 Temmuz) işgünü SAYILMAZ: Per 07-02 → Pzt 07-06 = 1 işgünü (2 değil)."""
    assert R._bd_age(date(2026, 7, 2), date(2026, 7, 6)) == 1
    # VIXCLS tatil haftası marjı geri geldi: obs 07-01, bugün 07-06 → yaş 2 ≤ 4 (tatil-kör sayaçta 3'tü)
    assert R._check_spec("VIXCLS", "m5", 4, date(2026, 7, 1), date(2026, 7, 6)) is None


def test_f4_real_freeze_still_fires():
    """Sayaç yumuşadı diye gerçek donma kaçmaz: VIXCLS 06-24'te donmuş, bugün 07-06 → 7bd > 4 → bayat."""
    v = R._check_spec("VIXCLS", "m5", 4, date(2026, 6, 24), date(2026, 7, 6))
    assert v is not None and v["age_bd"] == 7


# ── F3: GEX-shield kaynak-donma kapısı ───────────────────────────────────────

class TestGexFrozenSource:
    CFG = {"overlays": {"gex_shield": {"enabled": True, "k": 0.5, "thr": 1.0, "floor": 0.4, "win": 252}}}

    @staticmethod
    def _series(end_date):
        idx = pd.bdate_range(end=pd.Timestamp(end_date), periods=300)
        return pd.Series(1e9 * (1.0 + 0.2 * np.sin(np.arange(300) / 9.0)), index=idx)

    def test_frozen_csv_blocks_loud(self, monkeypatch):
        """Endpoint ölü ama eski dosyayı servis ediyor (son satır GEÇERLİ z, kuyruk-NaN yok) →
        eski kod 'available: True' ile bayat kalkanı CANLI servis ederdi; artık fail-loud blok."""
        from modules import gex_shield as G
        monkeypatch.setattr(G, "fetch_gex_live", lambda *a, **k: self._series("2026-06-12"))  # 3+ hafta önce
        r = G.evaluate(self.CFG)
        assert r["fail_safe_block"] is True and r["available"] is False
        assert r["factor"] == 1.0 and "DONMUŞ" in r["error"]

    def test_fresh_source_passes(self, monkeypatch):
        """Son işlem gününde biten seri (normal takvim: dünkü kapanış CSV'de) → blok YOK, kalkan canlı."""
        from modules import gex_shield as G
        from modules.opex_calendar import _prior_trading_day
        end = _prior_trading_day(datetime.now(timezone.utc).date())
        monkeypatch.setattr(G, "fetch_gex_live", lambda *a, **k: self._series(end))
        r = G.evaluate(self.CFG)
        assert not r.get("fail_safe_block") and r["available"] is True
        assert r["src_age_td"] <= 1                       # görünürlük alanı (0-1 normal)


# ── F5: NYSE tatil takvimi 2027+ dayanıklı ───────────────────────────────────

class TestDurableNyseCalendar:
    def test_2027_holidays_closed(self):
        """Eski sabit frozenset 2027-01-01'de sessizce sona eriyordu → MLK/Good Friday 2027 'açık'
        okunur, run_preopen SAHTE kaçırılmış-gün bayrağı yazardı. opex_calendar ile kapalı."""
        from run import market_closed_reason
        assert market_closed_reason(date(2027, 1, 18)) == "NYSE tatili"    # MLK 2027
        assert market_closed_reason(date(2027, 3, 26)) == "NYSE tatili"    # Good Friday 2027

    def test_2026_behavior_unchanged(self):
        from run import market_closed_reason
        assert market_closed_reason(date(2026, 7, 3)) == "NYSE tatili"     # gözlenen 4 Temmuz
        assert market_closed_reason(date(2026, 7, 4)) == "hafta sonu"      # Cumartesi
        assert market_closed_reason(date(2026, 7, 6)) is None              # Pazartesi açık
        assert market_closed_reason(date(2027, 1, 19)) is None             # MLK ertesi Salı açık


# ── F6: run_preopen tatilde koşmaz ───────────────────────────────────────────

class TestPreopenHolidaySkip:
    def test_thanksgiving_0935_et_skipped(self):
        """2026-11-26 Per (Şükran Günü) 14:35 UTC = 09:35 EST = pencere İÇİ ama tatil → KOŞMA."""
        import run_preopen as P
        ok, why = P.should_run(datetime(2026, 11, 26, 14, 35, tzinfo=timezone.utc))
        assert ok is False and "tatil" in why

    def test_normal_monday_still_runs(self):
        """2026-07-06 Pzt 13:35 UTC = 09:35 EDT → KOŞ (kapı tatil-dışında değişmedi)."""
        import run_preopen as P
        ok, _ = P.should_run(datetime(2026, 7, 6, 13, 35, tzinfo=timezone.utc))
        assert ok is True

    def test_thanksgiving_no_missed_flag(self):
        """Tatil atlaması 'beklenen' → missed_weekday bayrağı YOK (EQ-5 bekçisi açık işgünlerinde aynen)."""
        import run_preopen as P
        assert P.missed_weekday(datetime(2026, 11, 26, 16, 0, tzinfo=timezone.utc), ran_today=False) is False


# ── F8: forward-ledger kapısı (tek kaynak) ───────────────────────────────────

class TestLedgerEligible:
    BASE = {"call_status": "current", "overlay_block": False, "market_open": True,
            "market_closed_reason": None}

    def test_current_open_eligible(self):
        import run
        ok, _ = run.ledger_eligible(dict(self.BASE))
        assert ok is True

    def test_market_closed_not_eligible(self):
        """Tatil/hafta-sonu koşusunun çağrısı deftere GİRMEZ (Juneteenth çift-sayım dersi)."""
        import run
        d = dict(self.BASE, market_open=False, market_closed_reason="NYSE tatili")
        ok, why = run.ledger_eligible(d)
        assert ok is False and "piyasa kapalı" in why

    def test_stale_not_eligible(self):
        import run
        ok, why = run.ledger_eligible(dict(self.BASE, call_status="STALE"))
        assert ok is False and "STALE" in why

    def test_overlay_block_not_eligible(self):
        import run
        ok, why = run.ledger_eligible(dict(self.BASE, overlay_block=True))
        assert ok is False and "overlay_block" in why

    def test_old_dict_without_market_open_backcompat(self):
        """Eski karar dict'i (market_open alanı yok) → eski davranış (eklenir); sessiz kırılma yok."""
        import run
        ok, _ = run.ledger_eligible({"call_status": "current", "overlay_block": False})
        assert ok is True


# ── F7: collect_daily kapalı günde hayalet satır yazmaz ──────────────────────

class TestCollectClosedDay:
    def test_is_trading_day(self):
        import collect_daily as CD
        assert CD._is_trading_day(date(2026, 7, 3)) is False    # NYSE tatili (gözlenen 4 Temmuz)
        assert CD._is_trading_day(date(2026, 7, 4)) is False    # Cumartesi
        assert CD._is_trading_day(date(2026, 7, 6)) is True     # Pazartesi

    @staticmethod
    def _wire(monkeypatch, CD, today, calls):
        monkeypatch.setattr(CD, "_snap", lambda t: {"spot": 1.0, "atm_iv": 0.2, "flip": 1.0, "gex": 1.0})
        monkeypatch.setattr(CD, "_gamma_levels", lambda t: {"as_of": today.isoformat(), "ticker": t})
        monkeypatch.setattr(CD, "_written_this_run", lambda k, t, s: True)
        monkeypatch.setattr(CD, "_append_surface",
                            lambda rec: calls.__setitem__("surface", calls["surface"] + 1) or 1)
        monkeypatch.setattr(CD, "_append_levels",
                            lambda rows: calls.__setitem__("levels", calls["levels"] + 1) or 1)

        class _FakeDate(date):
            @classmethod
            def today(cls):
                return today
        monkeypatch.setattr(CD, "date", _FakeDate)

    def test_sunday_run_no_ledger_rows(self, monkeypatch):
        """2026-07-05 Pazar koşusu (gerçek örnek: hayalet 07-05 satırı) → exit 0, defterlere SIFIR yazım."""
        import collect_daily as CD
        calls = {"surface": 0, "levels": 0}
        self._wire(monkeypatch, CD, date(2026, 7, 5), calls)
        assert CD.main() == 0
        assert calls == {"surface": 0, "levels": 0}

    def test_holiday_run_no_ledger_rows(self, monkeypatch):
        """2026-06-19 Juneteenth sınıfı (hayalet tatil satırı) → defterlere yazım YOK."""
        import collect_daily as CD
        calls = {"surface": 0, "levels": 0}
        self._wire(monkeypatch, CD, date(2026, 6, 19), calls)
        assert CD.main() == 0
        assert calls == {"surface": 0, "levels": 0}

    def test_trading_day_run_writes(self, monkeypatch):
        """Açık işgünü → davranış DEĞİŞMEDİ: surface + levels defterlerine yazılır."""
        import collect_daily as CD
        calls = {"surface": 0, "levels": 0}
        self._wire(monkeypatch, CD, date(2026, 7, 6), calls)
        assert CD.main() == 0
        assert calls == {"surface": 1, "levels": 1}
