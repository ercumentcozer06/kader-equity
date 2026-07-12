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

FLOOD-FRENİ (2026-07-12): aynı alarm her koşuda kanallara YENİDEN gidiyordu (dedup yok) → Telegram "her şeye
stale" bombardımanı. Kanal-gönderimi KONU-imzası başına COOLDOWN penceresinde (varsayılan 72s; env
KADER_ALERT_COOLDOWN_H) susturulur — kalıcı bir arıza günde ~1 kez pinglenir, her koşuda değil. Gövdedeki
gün-sayacı (ör. "21 işgünü" → "22 işgünü") imzayı bozmaz (konu aynı = tek arıza). Yerel STALE_ALERT.json izi
HER ZAMAN yazılır (görünürlük kaybolmaz). Sağlıklı koşuda clear_alert() cooldown-durumunu sıfırlar → düzelme
sonrası nüks ANINDA yeniden alarmlar (susturulmaz).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Model adı repo dizininden türetilir (port-güvenli): notify.py modeller arası kopyalanırken
# etiketin equity'de sabitlenip macro alarmını yanlış modele yazması bu şekilde tekrar edemez.
_MODEL = ROOT.name.upper().replace("_", "-")
_LOG = logging.getLogger(f"{ROOT.name.replace('-', '_')}.notify")

_ALERT_STATE = ROOT / "output" / ".alert_state.json"       # konu-imzası → son kanal-gönderim zamanı (cooldown)
_COOLDOWN_H = float(os.environ.get("KADER_ALERT_COOLDOWN_H", "72"))   # 07-12: 12→72s (12s çok-günlük
# bayat koşulda GÜNLÜK re-push'a izin veriyordu = spam); 72s günlük tekrarı öldürür; env-override; <=0 KAPALI


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
        # P2 07-11 ([33]/[61]): url[:40] bot-token prefix'ini, exception metni tam URL'yi
        # loglayabiliyordu -> host'a indirgenir, hata metninde path maskelenir (token ASLA basilmaz).
        try:
            from urllib.parse import urlparse
            _host = urlparse(url).netloc or "?"
        except Exception:
            _host = "?"
        _err = str(e)
        if "/" in url:
            _err = _err.replace(url, f"https://{_host}/<masked>")
        _LOG.error("notify kanal POST başarısız (%s): %s: %s", _host, type(e).__name__, _err[:160])
        return False


def _sig(subject: str) -> str:
    return hashlib.sha1(subject.encode("utf-8", "replace")).hexdigest()[:16]


def _channels_due(subject: str) -> bool:
    """Bu konu son COOLDOWN saatinde zaten kanala gitti mi? Gittiyse kanal-gönderimini ATLA (yerel iz yine yazılır).
    Konu-imzası bazlı → gövdedeki gün-sayacı değişse de aynı arıza tek alarm/pencere. Cooldown<=0 → her zaman gönder."""
    if _COOLDOWN_H <= 0:
        return True
    try:
        st = json.loads(_ALERT_STATE.read_text(encoding="utf-8")) if _ALERT_STATE.exists() else {}
        last = st.get(_sig(subject))
        if last:
            age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds() / 3600.0
            if age_h < _COOLDOWN_H:
                return False
    except Exception:
        pass                                                 # durum okunamazsa fail-open (alarmı kaçırma > spam)
    return True


def _mark_sent(subject: str) -> None:
    try:
        st = json.loads(_ALERT_STATE.read_text(encoding="utf-8")) if _ALERT_STATE.exists() else {}
    except Exception:
        st = {}
    st[_sig(subject)] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        _ALERT_STATE.parent.mkdir(exist_ok=True)
        _ALERT_STATE.write_text(json.dumps(st), encoding="utf-8")
    except Exception:
        pass


def alert(subject: str, body: str = "") -> dict:
    """Bayatlık/degradasyon alarmı → tüm yapılandırılmış kanallar + yerel iz. Hangi kanalların ateşlediğini döndürür.
    Kanal-gönderimi cooldown ile spam-korumalı (konu başına); yerel STALE_ALERT.json izi her zaman yazılır."""
    a = _cfg_alert()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    msg = f"🔴 {_MODEL} [{ts}] {subject}" + (f"\n{body}" if body else "")
    fired = []
    due = _channels_due(subject)          # cooldown içinde aynı konu → kanal SPAM'i kesilir (yerel iz kalır)

    if due:
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

        if fired:                          # en az bir kanal başardıysa cooldown penceresini başlat
            _mark_sent(subject)

    # YEREL fallback — HER ZAMAN (kanal gitsin/sustursun kalıcı iz)
    try:
        (ROOT / "output").mkdir(exist_ok=True)
        rec = {"ts": ts, "subject": subject, "body": body, "channels_fired": fired,
               "suppressed": (not due)}
        with open(ROOT / "output" / "STALE_ALERT.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    _LOG.error("ALARM: %s | %s | kanallar=%s%s", subject, body,
               fired or "YALNIZ-YEREL (webhook yapılandırılmadı)",
               " [cooldown-susturuldu]" if not due else "")
    return {"fired": fired, "message": msg, "suppressed": (not due)}


def clear_alert() -> None:
    """Sağlıklı koşuda önceki STALE_ALERT.json + cooldown-durumunu temizle (bayat-alarm artefaktı kalmasın;
    düzelme sonrası nüks ANINDA yeniden alarmlar — cooldown susturmaz)."""
    for p in (ROOT / "output" / "STALE_ALERT.json", _ALERT_STATE):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass
