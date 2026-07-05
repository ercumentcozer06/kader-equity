"""
backtest/remeasure/wave0_credit_probe.py — WAVE-0 MarketData kredi-kapisi probe'u.

TESHIS-ONLY: tek 1-kredi probe atar (SPY chain, date=2026-06-08), header'dan
X-Api-Ratelimit-Remaining + Reset okur. BACKFILL BASLATMAZ, baska istek atmaz.

Esikler (WAVE-0 gorev-spesifikasyonundan; config.py'de karsiligi olmayan
orkestrasyon sabitleri — model/olcum sabiti DEGIL, o yuzden config'e eklenmedi):
  ihtiyac ~ (78 md-gunu + 2 uzatma-gunu) x 57 kredi/gun (SPY9+QQQ9+SPX23+NDX16) ~ 4560-4700
  PASS esigi = remaining >= 5200  (ihtiyac + SAFETY tamponu)
Token .env'den okunur ve ASLA stdout'a basilmaz.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402  (tek-gercek-kaynak; config_sha metadata'ya yazilir)

# --- WAVE-0 gorev sabitleri (orkestrasyon; yukaridaki docstring'e bakiniz) ---
PROBE_URL = "https://api.marketdata.app/v1/options/chain/SPY/"
PROBE_DATE = "2026-06-08"          # config.PANEL_END ile ayni gun (pre-registered pencere sonu)
PASS_THRESHOLD = 5200              # ihtiyac (~4700) + SAFETY tamponu
CREDITS_PER_DAY = {"SPY": 9, "QQQ": 9, "SPX": 23, "NDX": 16}   # toplam 57/gun
NEED_DAYS = 78 + 2                 # kalan md-gunleri + 2 uzatma-gunu

OUT_PATH = config.REMEASURE_DIR / "wave0_credit_probe_result.json"


def _sanitize(msg: str, token: str) -> str:
    """Hata mesajlarinda token sizmasin (requests URL'i exception'a koyabilir)."""
    return msg.replace(token, "***TOKEN***") if token else msg


def main() -> int:
    load_dotenv(config.ROOT / ".env")
    token = os.environ.get("MARKETDATA_TOKEN", "")
    if not token:
        print(json.dumps({"error": "MARKETDATA_TOKEN .env'de bulunamadi",
                          "config_sha": config.config_sha()}))
        return 2

    result: dict = {
        "probe": {"url": PROBE_URL, "date": PROBE_DATE, "symbol": "SPY"},
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pass_threshold": PASS_THRESHOLD,
        "need_estimate": {
            "days": NEED_DAYS,
            "credits_per_day": CREDITS_PER_DAY,
            "total": NEED_DAYS * sum(CREDITS_PER_DAY.values()),
        },
        "config_sha": config.config_sha(),
    }

    try:
        r = requests.get(PROBE_URL, params={"date": PROBE_DATE, "token": token},
                         timeout=60)
    except Exception as e:  # noqa: BLE001
        result["error"] = _sanitize(f"{type(e).__name__}: {e}", token)
        print(json.dumps(result, indent=2))
        return 2

    h = {k.lower(): v for k, v in r.headers.items()}

    def _int(name: str):
        v = h.get(name)
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    remaining = _int("x-api-ratelimit-remaining")
    limit = _int("x-api-ratelimit-limit")
    consumed = _int("x-api-ratelimit-consumed")
    reset_epoch = _int("x-api-ratelimit-reset")

    result["http_status"] = r.status_code
    result["ratelimit"] = {
        "remaining": remaining,
        "limit": limit,
        "consumed_this_call": consumed,
        "reset_epoch": reset_epoch,
    }
    if reset_epoch is not None:
        secs = reset_epoch - time.time()
        result["ratelimit"]["reset_utc"] = datetime.fromtimestamp(
            reset_epoch, tz=timezone.utc).isoformat(timespec="seconds")
        result["ratelimit"]["hours_to_reset"] = round(secs / 3600.0, 2)

    # govdeden minimal dogrulama (chain gercekten geldi mi) — sayilar, veri degil
    try:
        body = r.json()
        result["body_s"] = body.get("s")
        opt = body.get("optionSymbol") or []
        result["n_contracts_returned"] = len(opt)
    except Exception:  # noqa: BLE001
        result["body_s"] = None

    result["pass"] = bool(remaining is not None and remaining >= PASS_THRESHOLD)

    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
