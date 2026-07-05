"""
test_opex — OpEx takvim kapısı (FINDING 23: NDX monthly OpEx −14bps p=0.001 anomali).
Saf takvim + NDX-özel flat + ≥3 işgünü uyarı; frozen stack DEĞİŞMEZ (position_target etkilenmez).
TATİL FARKINDALIĞI (2026-06-16): 3.-Cuma NYSE tatiliyse fiili vade bir önceki işlem gününe (Perşembe) kayar.
"""
from __future__ import annotations
import datetime as dt
from modules import opex_calendar as OC


def test_third_friday_raw_calendar():
    # third_friday = HAM takvim 3.-Cuma (tatil-kaymasız; geriye-uyum + screen)
    assert OC.third_friday(2026, 6) == dt.date(2026, 6, 19)     # Cuma (Juneteenth — kapalı)
    assert OC.third_friday(2026, 3) == dt.date(2026, 3, 20)
    assert OC.third_friday(2026, 1) == dt.date(2026, 1, 16)


def test_quad_witch_is_effective_day():
    # Quad-witch = çeyrek-sonu ayının FİİLİ (tatil-kaymalı) OpEx günü
    assert OC.is_quad_witch(dt.date(2026, 6, 18)) is True       # fiili Haziran quad (Per — Juneteenth kayması)
    assert OC.is_quad_witch(dt.date(2026, 6, 19)) is False      # kapalı Cuma quad-witch DEĞİL
    assert OC.is_quad_witch(dt.date(2026, 3, 20)) is True       # Mart quad (tatil yok)
    assert OC.is_quad_witch(dt.date(2026, 1, 16)) is False      # Ocak çeyrek-sonu değil


def test_is_opex_and_next_holiday_shift():
    # Juneteenth (2026-06-19, NYSE kapalı) → fiili OpEx 06-18 Perşembe
    assert OC.is_opex_day(dt.date(2026, 6, 18)) is True
    assert OC.is_opex_day(dt.date(2026, 6, 19)) is False        # kapalı Cuma OpEx DEĞİL
    assert OC.next_opex(dt.date(2026, 6, 1)) == dt.date(2026, 6, 18)
    assert OC.next_opex(dt.date(2026, 6, 19)) == dt.date(2026, 7, 17)   # Haziran OpEx geçti → Temmuz (tatilsiz Cuma)
    assert OC.next_opex(dt.date(2026, 12, 20)) == dt.date(2027, 1, 15)  # yıl sınırı (tatilsiz Cuma)


def test_juneteenth_pre_2022_no_shift():
    # NYSE Juneteenth'i ilk 2022'de tatil yaptı → 2020-06-19 (Cuma, 3.-Cuma) KAYMAZ
    assert OC.third_friday(2020, 6) == dt.date(2020, 6, 19)
    assert OC.opex_day(2020, 6) == dt.date(2020, 6, 19)         # 2022 öncesi → kayma yok


def test_good_friday_shift():
    # 2014-04-18 Good Friday VE Nisan'ın 3.-Cuma'sı (NYSE kapalı) → fiili OpEx 04-17 Perşembe
    assert OC.third_friday(2014, 4) == dt.date(2014, 4, 18)
    assert OC.is_market_holiday(dt.date(2014, 4, 18)) is True
    assert OC.opex_day(2014, 4) == dt.date(2014, 4, 17)


def test_trading_days_until_excludes_holidays():
    assert OC.trading_days_until(dt.date(2026, 6, 16), dt.date(2026, 6, 18)) == 2   # Sal→Per = Sal,Çar
    assert OC.trading_days_until(dt.date(2026, 6, 16), dt.date(2026, 6, 22)) == 3   # Juneteenth (19) ÇIKARILDI
    assert OC.trading_days_until(dt.date(2026, 6, 18), dt.date(2026, 6, 18)) == 0


def test_evaluate_opex_day_ndx_flat():
    cfg = {"warn_days": 3, "flat_assets": ["NDX"]}
    e = OC.evaluate(dt.date(2026, 6, 18), cfg)                  # FİİLİ OpEx günü (Per)
    assert e["is_opex_today"] is True
    assert e["asset_overrides"]["NDX"]["deploy"] == 0.0
    assert "SPX" not in e["asset_overrides"]                    # SPX dokunulmaz (asimetri)
    assert e["warn"] is True
    # kapalı Cuma OpEx günü DEĞİL → override yok
    e_fri = OC.evaluate(dt.date(2026, 6, 19), cfg)
    assert e_fri["is_opex_today"] is False
    assert e_fri["asset_overrides"] == {}


def test_evaluate_warning_window_holiday_aware():
    cfg = {"warn_days": 3, "flat_assets": ["NDX"]}
    e3 = OC.evaluate(dt.date(2026, 6, 16), cfg)                 # fiili OpEx (06-18) 2 işgünü sonra → UYARI
    assert e3["warn"] is True and e3["is_opex_today"] is False
    assert e3["trading_days_until"] == 2
    assert e3["next_opex"] == "2026-06-18"
    assert e3["holiday_shifted"] is True
    assert e3["next_opex_weekday"] == "Thursday"
    assert e3["is_quad_witch_next"] is True                     # Haziran = quad
    far = OC.evaluate(dt.date(2026, 6, 1), cfg)                 # uzak → uyarı yok
    assert far["warn"] is False


def test_run_output_opex_block_stack_unchanged():
    """run.py çıktısında opex_gate + asset_deploy var; frozen stack (position_target) ETKİLENMEZ (trim-only invariant)."""
    from config import load_config
    import run
    d = run.build_decision(load_config())
    assert "opex_gate" in d and d["opex_gate"] is not None
    assert "asset_deploy" in d
    assert d["position_target"] <= float(d["tide_dir"]) + 1e-9   # OpEx kapısı stack'i değiştirmedi (per-asset only)
    assert 0.0 <= d["position_target"] <= 1.0
