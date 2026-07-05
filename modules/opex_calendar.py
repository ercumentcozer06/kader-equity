"""
modules/opex_calendar — monthly OpEx (3.-Cuma) takvimi + NDX-özel OpEx-günü FLAT taktik kapısı.

GEREKÇE (FINDING 23 / k_opex_avoid.py, 2026-06-12): NDX monthly OpEx günü 1990-2026 boyunca −14.4 bps/gün
(diğer günler +7.7), block-bootstrap t=−3.35 p=0.001, OpEx-SPESİFİK (OpEx-Cuma vs diğer-Cumalar t−2.70),
DÖRT ON YILDA DA negatif (kalıcı anomali). Standalone "flat NDX OpEx günü" → NDX Sharpe 0.635→0.711 (+0.076).
SPX OpEx-günü zayıflığı ANLAMSIZ (t−1.04) ve modele konunca ZARAR → bu kapı YALNIZ NDX (asimetrik, veri-temelli).

TATİL FARKINDALIĞI (2026-06-16): vade/expiration ayın 3.-Cuma'sıdır; o gün NYSE tatiliyse fiili vade bir
önceki işlem gününe (genelde Perşembe) KAYAR. Örn. 2026-06-19 Juneteenth (borsa kapalı) → fiili OpEx 06-18 Perşembe.
Eski saf-tarih takvimi bunu göremiyordu (kapalı Cuma'ya OpEx der, gerçek vade Perşembe'yi kaçırırdı). Artık
`opex_day()` NYSE tatil takvimine göre kayar; `third_friday()` HAM takvim Cuması olarak kalır (geriye-uyum + screen).

NYSE takvimi (Columbus/Veterans Day HARİÇ; Good Friday + Juneteenth[2022+] DAHİL) — saf pandas, ağ YOK, deterministik.

DİSİPLİN: frozen tide×froth×shield stack DEĞİŞMEZ. Bu, stack'in ÜSTÜNE per-asset ifade-override'ı (NDX OpEx
günü deploy→0). Modelin 1.64/1.77 reprodüksiyonu etkilenmez (position_target aynı; yalnız NDX sleeve o gün flat).
DÜRÜST ETİKET: standalone-gerçek (+0.076 NDX) ama 2019+ model-incremental ~nötr → bu bir ALFA iddiası DEĞİL,
Emir-talebiyle eklenen veri-doğrulanmış (p=0.001) taktik NDX de-risk + takvim uyarısı.
"""
from __future__ import annotations
import datetime as _dt
import functools
import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, Holiday, nearest_workday,
    USMartinLutherKingJr, USPresidentsDay, USMemorialDay,
    USLaborDay, USThanksgivingDay, GoodFriday,
)


class _NYSEHolidayCalendar(AbstractHolidayCalendar):
    """NYSE borsa-tatilleri. Columbus/Veterans Day borsada açık → HARİÇ. Good Friday + Juneteenth(2022+) DAHİL."""
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date=pd.Timestamp("2022-01-01"), observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


_HOL_START = pd.Timestamp("1990-01-01")   # FINDING 23 tarihçesini (1990+) ve geleceği kapsar
_HOL_END = pd.Timestamp("2035-12-31")


@functools.lru_cache(maxsize=1)
def _holiday_index() -> pd.DatetimeIndex:
    return _NYSEHolidayCalendar().holidays(_HOL_START, _HOL_END)


@functools.lru_cache(maxsize=1)
def _holiday_set() -> frozenset:
    return frozenset(ts.date() for ts in _holiday_index())


@functools.lru_cache(maxsize=1)
def _holiday_np() -> np.ndarray:
    return _holiday_index().values.astype("datetime64[D]")


def _to_date(as_of) -> _dt.date:
    if isinstance(as_of, _dt.date) and not isinstance(as_of, _dt.datetime):
        return as_of
    return pd.Timestamp(as_of).date()


def is_market_holiday(d) -> bool:
    """d bir NYSE borsa-tatili mi (hafta sonu DEĞİL — yalnız tatil)."""
    return _to_date(d) in _holiday_set()


def _prior_trading_day(d: _dt.date) -> _dt.date:
    """d AÇIK işlem günüyse d; değilse (hafta sonu / NYSE tatili) geriye doğru ilk açık işlem günü."""
    while d.weekday() >= 5 or d in _holiday_set():
        d -= _dt.timedelta(days=1)
    return d


def third_friday(year: int, month: int) -> _dt.date:
    """Ayın 3.-Cuma'sı — HAM takvim (tatil-kaymasız). Geriye-uyum + screen için bozulmadan kalır."""
    d = _dt.date(year, month, 1)
    first_fri = d + _dt.timedelta(days=(4 - d.weekday()) % 7)   # Cuma=4
    return first_fri + _dt.timedelta(days=14)


def opex_day(year: int, month: int) -> _dt.date:
    """Fiili monthly OpEx/vade günü: 3.-Cuma; o Cuma NYSE tatiliyse bir önceki işlem gününe (genelde Perşembe) kayar."""
    return _prior_trading_day(third_friday(year, month))


def next_opex(as_of) -> _dt.date:
    """as_of'tan sonraki (veya bugünkü) ilk FİİLİ OpEx günü (tatil-kaymalı)."""
    a = _to_date(as_of)
    ox = opex_day(a.year, a.month)
    if ox < a:                                                 # bu ayınki geçti → gelecek ay
        y, m = (a.year + 1, 1) if a.month == 12 else (a.year, a.month + 1)
        ox = opex_day(y, m)
    return ox


def is_quad_witch(d) -> bool:
    """d, çeyrek-sonu ayının (3/6/9/12) FİİLİ OpEx (quad-witch) günü mü."""
    d = _to_date(d)
    return d.month in (3, 6, 9, 12) and d == opex_day(d.year, d.month)


def is_opex_day(as_of) -> bool:
    """as_of, o ayın FİİLİ OpEx günü mü (tatil-kaymalı; kapalı Cuma'ya OpEx DEMEZ)."""
    a = _to_date(as_of)
    return a == opex_day(a.year, a.month)


def trading_days_until(as_of, target: _dt.date) -> int:
    """as_of → target arası TAM işgünü (Mon-Fri, NYSE tatilleri ÇIKARILIR). target günü=0."""
    a = _to_date(as_of)
    if a >= target:
        return 0
    return int(np.busday_count(a, target, holidays=_holiday_np()))


def evaluate(as_of, cfg: dict | None = None) -> dict:
    """OpEx takvim kapısı. cfg: {enabled, warn_days, flat_assets}. Döndürür uyarı + per-asset override (tatil-kaymalı)."""
    cfg = cfg or {}
    warn_days = int(cfg.get("warn_days", 3))
    flat_assets = list(cfg.get("flat_assets", ["NDX"]))
    a = _to_date(as_of)
    nx = next_opex(a)
    tf_raw = third_friday(nx.year, nx.month)                   # nx'in ayının HAM 3.-Cuma'sı
    shifted = (nx != tf_raw)                                   # tatil yüzünden Perşembe'ye kaydı mı
    tdu = trading_days_until(a, nx)
    today_is = is_opex_day(a)
    overrides = {}
    if today_is:
        for asset in flat_assets:
            overrides[asset] = {"deploy": 0.0, "reason": f"monthly OpEx günü → {asset} FLAT (anomali p=0.001, NDX)"}
    return {
        "next_opex": str(nx),
        "next_opex_weekday": nx.strftime("%A"),
        "third_friday": str(tf_raw),
        "holiday_shifted": shifted,
        "is_quad_witch_next": is_quad_witch(nx),
        "trading_days_until": tdu,
        "is_opex_today": today_is,
        "warn": (0 <= tdu <= warn_days),
        "warn_days": warn_days,
        "flat_assets": flat_assets,
        "asset_overrides": overrides,
        "note": ("veri-doğrulanmış (FINDING 23: NDX OpEx −14bps p=0.001, 4 on yıl); taktik NDX de-risk + uyarı; "
                 "3.-Cuma NYSE tatiliyse fiili vade Perşembe'ye kayar (Juneteenth/Good Friday vb.)"),
    }
