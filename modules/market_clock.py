"""Canonical New York market-date helpers.

All snapshot filenames, ledger ``as_of`` values, and freshness checks must use
the same calendar.  The host can run in Istanbul while UTC is still on the
previous day, so ``date.today()`` is not a safe market-date source.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")


def market_date(now: datetime | None = None) -> date:
    """Return the calendar date in New York for an aware/UTC datetime."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(NEW_YORK).date()

