"""notify — ANINDA ALARM kanalı (P0-A, denetim 2026-07-07). Emir'in emri: "veri bayat = ANINDA haberim olsun".

Gate bayatı YAKALIYOR ama şimdiye dek yalnız kimsenin okumadığı run_daily.log'a yazıyordu → Emir brief'i elle
açana kadar bilmiyordu ("model asla bayat çalışmaz" değişmezinin YARISI eksikti: yakalama var, HABER-VERME yok).

alert() bir bayatlık/degradasyon olayını mevcut HER kanaldan iletir:
  • Discord webhook   (config alert.discord_webhook | env KADER_ALERT_DISCORD)  → POST {"content": msg}
  • Telegram bot      (alert.telegram_token+chat_id | env KADER_ALERT_TELEGRAM_TOKEN/_CHAT)
  • Generic webhook   (alert.webhook_url | env KADER_ALERT_WEBHOOK)              → POST {"text": msg}
  • YEREL fallback (HER ZAMAN): output/STALE_ALERT.json + log.error → kanal yoksa bile kalıcı iz kalır.

Kanal yoksa CRASH ETMEZ (best-effort; alarm bir run'ı asla öldürmez). Telefonuna push için Emir bir Telegram/Discord
webhook'u (ücretsiz, ~2dk) verip config'e koymalı; o güne kadar yerel STALE_ALERT.json + log iz tutar.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_LOG = logging.getLogger("kader_equity.notify")


def _cfg_alert() -> dict:
    """alert bloğunu model-arayüzünden BAĞIMSIZ oku (equity/gold/silver: `config.load_config`; oil: `run.load_config`;
    son çare: config.yaml / config/config.yaml doğrudan). Hiçbiri tutmazsa {} → env-var + yerel fallback devrede."""
    for loader in ("config", "run"):
        try:
            mod = __import__(loader)
            if hasattr(mod, "load_config"):
                return (mod.load_config().get("alert", {}) or {})
        except Exception:
            pass
    for rel in ("config.yaml", "config/config.yaml"):
        try:
            import yaml
            p = ROOT / rel
            if p.exists():
                return ((yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("alert", {}) or {})
        except Exception:
            pass
    return {}


def _post(url: str, payload: dict, timeout: int = 10) -> bool:
    try:
        import requests
        r = requests.post(url, json=payload, timeout=timeout)
        return 200 <= r.status_code < 300
    except Exception as e:  # noqa: BLE001 — alarm asla run'ı öldürmez
        # Denetim 07-11 P2 ([20]): url[:40] token-prefix'i, exception metni tam URL'yi basabiliyordu
        try:
            from urllib.parse import urlparse
            _host = urlparse(url).netloc or "?"
        except Exception:
            _host = "?"
        _err = str(e).replace(url, f"https://{_host}/<masked>")
        _LOG.error("notify kanal POST başarısız (%s): %s: %s", _host, type(e).__name__, _err[:160])
        return False


def alert(subject: str, body: str = "") -> dict:
    """Bayatlık/degradasyon alarmı → tüm yapılandırılmış kanallar + yerel iz. Hangi kanalların ateşlediğini döndürür."""
    a = _cfg_alert()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    msg = f"🔴 KADER-EQUITY [{ts}] {subject}" + (f"\n{body}" if body else "")
    fired = []

    disc = a.get("discord_webhook") or os.environ.get("KADER_ALERT_DISCORD")
    if disc and _post(disc, {"content": msg}):
        fired.append("discord")

    tg_tok = a.get("telegram_token") or os.environ.get("KADER_ALERT_TELEGRAM_TOKEN")
    tg_chat = a.get("telegram_chat_id") or os.environ.get("KADER_ALERT_TELEGRAM_CHAT")
    if tg_tok and tg_chat and _post(f"https://api.telegram.org/bot{tg_tok}/sendMessage",
                                    {"chat_id": tg_chat, "text": msg}):
        fired.append("telegram")

    gen = a.get("webhook_url") or os.environ.get("KADER_ALERT_WEBHOOK")
    if gen and _post(gen, {"text": msg}):
        fired.append("webhook")

    # YEREL fallback — HER ZAMAN (kanal olsun olmasın kalıcı iz)
    try:
        (ROOT / "output").mkdir(exist_ok=True)
        rec = {"ts": ts, "subject": subject, "body": body, "channels_fired": fired}
        with open(ROOT / "output" / "STALE_ALERT.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    _LOG.error("ALARM: %s | %s | kanallar=%s", subject, body, fired or "YALNIZ-YEREL (webhook yapılandırılmadı)")
    return {"fired": fired, "message": msg}


def clear_alert() -> None:
    """Sağlıklı koşuda önceki STALE_ALERT.json'ı temizle (bayat-alarm artefaktı kalmasın)."""
    try:
        p = ROOT / "output" / "STALE_ALERT.json"
        if p.exists():
            p.unlink()
    except Exception:
        pass
