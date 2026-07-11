"""
run_daily — ŞERİT-3.2 GÜNLÜK orkestratör (tek giriş noktası). Windows Task Scheduler ile hafta-içi kapanış
sonrası (≈23:30 TSİ) koşar. İDEMPOTENT (aynı gün ikinci koşu zarar vermez: reconstruct gün-cache'li, snapshot
+ ledger as_of-dedup'lı). FAIL-LOUD (kritik adım patlarsa nonzero exit + log + brief'te alarm; SESSİZ atlama YOK
— levels ledger'ın kaçan günü geri gelmez, time-decay).

Akış: (1) collect_daily → gamma+surface+levels snapshot/ledger (SPY+QQQ); (2) engine.brief SPY+QQQ → canlı tide
tazele + JSON/terminal; (3) forward-ledger: current çağrıyı kaydet + mark_to_market; (4) aylık forward-watch
hook (m9 21g-corr alarmı + COR1M episode sayacı → tek satır). Saat parametrik (Task Scheduler XML).

  & <kader-macro venv python> run_daily.py            (elle)
  Task Scheduler: register_task.ps1 (aşağıda) ile otomatik.
"""
from __future__ import annotations

import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
PY = sys.executable
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _log(msg: str):
    line = f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%SZ}] {msg}"
    print(line)
    try:
        (ROOT / "output").mkdir(exist_ok=True)
        with open(ROOT / "output" / "run_daily.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _step(name: str, fn) -> tuple[bool, str]:
    try:
        fn(); _log(f"OK   {name}"); return True, ""
    except Exception as e:
        tb = f"{type(e).__name__}: {e}"
        _log(f"FAIL {name} → {tb}")
        _log(traceback.format_exc().splitlines()[-1])
        return False, tb


def _substep(name: str, fn) -> bool:
    """BEST-EFFORT alt-adim: fail'i LOUD-log'lar ama EXCEPTION FIRLATMAZ (cagirana hata sizmaz).
    _step'ten farki: bu hep True'ya benzer akista kalir — bir Constan kaynagi dususe (FRED/EDGAR/
    press down) bant graceful-stale olur (moduller zaten bayat-etiketler), daily-run OLMEZ.
    Donus: True=OK, False=bu alt-adim patladi (ust _step yine de yesil kalir)."""
    try:
        fn(); _log(f"  OK   constan: {name}"); return True
    except Exception as e:
        _log(f"  ⚠ FAIL constan: {name} → {type(e).__name__}: {e} (best-effort, daily-run'i OLDURMEZ; "
             f"bant graceful-stale, sonraki kosu yeniden dener)")
        _log("  " + traceback.format_exc().splitlines()[-1])
        return False


def _collect():
    import collect_daily
    if collect_daily.main() != 0:
        raise RuntimeError("collect_daily nonzero")


FRED_TTL_DAYS = 7   # ceyreklik Z.1/FRED net-arz: ~10-11hf yayin-gecikmeli -> gunluk ag-cekimi ISRAF.
#                     parquet bu kadar gunden taze ise ATLA (cache-TTL yeni ceyregi kacirmadan israfi keser).


def _bb_pull():
    """S&P 500 buyback bultenini OTOMATIK cek (manuel son-adimi oldurur). IDEMPOTENT: yeni ceyrek yoksa
    'guncel' der, dokunmaz; yeni ceyrek parse+12mo-self-check gecerse CSV'ye yazar + parquet'i kurar.
    BELIRSIZ-parse (verify_failed) / erisim-hatasi = loud-log AMA yanlis-sayi YAZMAZ (manuel-fallback)."""
    import importlib
    F = importlib.import_module("screen.fetch_supply_components")
    res = F.auto_pull_buyback(write=True)
    st = res.get("status")
    if st == "written":
        F._rebuild_parquet()
        _log(f"  buyback: YENI ceyrek {res.get('quarter')} auto-yazildi + parquet kuruldu (bb={res.get('bb_bn')})")
    elif st == "verify_failed":
        _log(f"  ⚠ buyback: {res.get('quarter')} AUTO-PARSE BELIRSIZ (12mo-self-check tutmadi) — "
             f"YAZILMADI, elle-dogrulama bekliyor: {res.get('detail')}")
    elif st in ("parse_failed", "no_listing"):
        _log(f"  ⚠ buyback: bulten cekimi/listesi alinamadi (status={st}) — atlandi, sonraki kosu dener")
    else:
        _log(f"  buyback: {st} (son {res.get('quarter')}) — yeni ceyrek yok, dokunulmadi")


def _ipo_pull():
    """EDGAR S-1/F-1 boru-hatti gozcusu — GUNLUK MESRU taze (yeni dev-IPO dosyalamasini ayni gun yakalar).
    Kapanmis ceyrekler kalici-cache; yalniz acik ceyregin form.idx'i >20saat eskiyse yeniden iner -> ag-nazik."""
    import importlib
    M = importlib.import_module("screen.fetch_ipo_pipeline")
    if M.main() != 0:
        raise RuntimeError("fetch_ipo_pipeline nonzero")


def _net_supply_pull():
    """Z.1/FRED net-arz — CACHE-GATED (ceyreklik veri, gunluk ag-israfi yok). parquet FRED_TTL_DAYS'ten
    taze ise ATLA; eksik/eski ise FRED'den tazele (yeni ceyrek ancak ayda-bir gelir)."""
    import importlib
    p = ROOT / "data" / "cache" / "net_equity_supply.parquet"
    if p.exists():
        age_d = (time.time() - p.stat().st_mtime) / 86400.0
        if age_d < FRED_TTL_DAYS:
            _log(f"  net-arz: cache taze ({age_d:.1f}g < {FRED_TTL_DAYS}g TTL) -> FRED'e gidilmedi (ag-israfi yok)")
            return
    M = importlib.import_module("screen.fetch_net_equity_supply")
    if M.main() != 0:
        raise RuntimeError("fetch_net_equity_supply nonzero")


def _balance_derive():
    """K1 arz-talep denge paneli — diger parquet'lerden TURETIR (ucuz, ag-yok). En son calismali ki
    yukaridaki tazelemeler dengeye yansisin."""
    import importlib
    M = importlib.import_module("screen.fetch_supply_demand_balance")
    if M.main() != 0:
        raise RuntimeError("fetch_supply_demand_balance nonzero")


def _refresh_constan():
    """Constan ARZ/TALEP bantlarini brief'TEN ONCE tazele — bantlar her gun ELLE kosmadan canli kalsin.
    BEST-EFFORT / NON-FATAL: her alt-fetch _substep ile sarili (fail loud-log ama exception SIZMAZ);
    bir kaynak dususe (FRED/EDGAR/press down) bant graceful-stale olur, run_daily ASLA bu yuzden
    olmez (kritik adimlar collect+brief+ledger ayri _step). Sira: IPO (gunluk taze) -> buyback
    (auto-pull) -> net-arz (cache-gated) -> denge (turetir, en son)."""
    _log("constan-refresh BASLADI (best-effort, non-fatal):")
    n_ok = 0
    n_ok += _substep("IPO boru-hatti (EDGAR S-1/F-1, gunluk taze)", _ipo_pull)
    n_ok += _substep("buyback auto-pull (S&P DJI bulteni)", _bb_pull)
    n_ok += _substep("net-arz (Z.1/FRED, cache-gated)", _net_supply_pull)
    n_ok += _substep("arz-talep denge (K1, turetir)", _balance_derive)
    _log(f"constan-refresh BITTI — {n_ok}/4 alt-adim OK "
         f"(dususler graceful-stale; daily-run akisini etkilemez)")


def _brief_and_ledger():
    import copy
    from config import load_config
    import run
    from engine import brief as B
    from validation import ledger as L
    for tic in ("SPY", "QQQ"):
        B.main(["--ticker", tic, "--json", "--quiet"])             # canlı tide tazele + JSON
    # forward-ledger: current çağrıyı kaydet (STALE eklenmez) + mark
    cfg = copy.deepcopy(load_config()); cfg.setdefault("spine", {})["source"] = "live"
    d = run.build_decision(cfg)
    run.write_latest(d)   # EQ-2: latest.json otomasyonda da her koşuda güncellenir (bayat 'current' artefakt biter)
    _log(f"latest.json güncellendi (as_of={d.get('as_of')}, status={d.get('call_status')})")
    # F8 (denetim 2026-07-05): kapı TEK kaynak run.ledger_eligible — market_open dahil (tatil koşusunun
    # kapalı-gün çağrısı deftere girip aynı ileri-getiriyi İKİ kez saydırmasın; 2026-06-19 Juneteenth dersi).
    ok, why = run.ledger_eligible(d)
    if ok:
        # EQ-3 (denetim 2026-07-04): ölü 'scores = overlays' satırı kaldırıldı; kayıt run.ledger_record'dan
        # (tek şema kaynağı) → m9/m5/m2 attribution kolonları artık GERÇEKTEN dolar (şemada boş kolon biter).
        rec = run.ledger_record(d)
        L.append_call(rec); L.mark_to_market()
        _log(f"ledger: current çağrı kaydedildi as_of={d['as_of']} dir={d['direction']} pos={d['position_target']}")
    else:
        _log(f"ledger: çağrı EKLENMEDİ ({why})")


def _forward_watch():
    """Aylık-hafif: m9 21g-corr alarmı + COR1M episode durumu (her gün ucuz; tek satır özet)."""
    from validation import attribution as A
    att = A.evaluate()
    alarms = [m for m, dd in att["modules"].items() if dd.get("alarm")]
    _log(f"forward-watch: attribution alarm={alarms or 'yok'} | "
         f"m9 skor {att['modules'].get('m9', {}).get('live_score')} "
         f"(tarihsel %{att['modules'].get('m9', {}).get('score_pctile')})")


def _lock_guard():
    """EQ-D6-01 (denetim 2026-07-04): kilit bekçisi artık otomasyonda — donmuş spine byte-parite
    (backtest/reproduce_baseline) her günlük koşuda doğrulanır; drift → adım FAIL → exit 1 (fail-loud)."""
    r = subprocess.run([PY, str(ROOT / "backtest" / "reproduce_baseline.py")],
                       cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=600)
    if r.returncode != 0:
        tail = (r.stdout or r.stderr or "").strip().splitlines()[-3:]
        raise RuntimeError("kilit drift (reproduce_baseline exit %s): %s" % (r.returncode, " | ".join(tail)[:240]))
    _log("lock-guard: donmuş spine byte-parite PASS")


def main() -> int:
    _log("=" * 60)
    _log("run_daily BAŞLADI")
    results = []
    results.append(_step("collect_daily (gamma+surface+levels)", _collect))
    # Constan bant-tazeleme: brief'TEN ONCE (bantlar canli olsun) AMA NON-FATAL (_refresh_constan
    # kendi-icinde her fetch'i _substep ile sarar; bir kaynak dususe exception SIZMAZ, _step yesil
    # kalir, brief+ledger yine calisir). Constan dususe bile kritik trade-borusu etkilenmez.
    results.append(_step("constan-refresh (IPO+buyback+net-arz+denge, best-effort)", _refresh_constan))
    results.append(_step("brief SPY+QQQ + forward-ledger", _brief_and_ledger))
    results.append(_step("forward-watch (attribution)", _forward_watch))
    results.append(_step("lock drift-guard (reproduce_baseline)", _lock_guard))
    n_fail = sum(1 for ok, _ in results if not ok)
    _alert_if_degraded(n_fail, results)                      # P0-A: bayatlık/degradasyon → ANINDA push
    if n_fail:
        _log(f"⛔ run_daily BİTTİ — {n_fail} ADIM PATLADI (fail-loud, exit 1). Üstteki FAIL satırlarına bak.")
        return 1
    _log("✓ run_daily BİTTİ — tüm adımlar OK")
    return 0


def _alert_if_degraded(n_fail: int, results: list) -> None:
    """P0-A anında-alarm (denetim 2026-07-07): koşu-sonu latest.json'ı denetle; bayat/degrade/adım-patlağı varsa
    notify.alert → Emir'e push (webhook varsa) + yerel STALE_ALERT.json. Temizse önceki alarmı sil. BEST-EFFORT."""
    try:
        import json as _json
        import notify
        reasons = []
        if n_fail:
            failed = [n for (ok, tb), n in zip(results, ("collect", "constan", "brief+ledger", "forward-watch", "lock-guard")) if not ok]
            reasons.append(f"{n_fail} adım PATLADI ({', '.join(failed)})")
        p = ROOT / "output" / "kader_equity_latest.json"   # run.write_latest bu dosyayı yazar (2026-07-07 fix: eski 'latest.json' HİÇ yoktu → her koşu sahte 'latest.json YOK' alarmı + clear_alert asla çağrılmıyordu)
        if p.exists():
            d = _json.loads(p.read_text(encoding="utf-8"))
            if d.get("call_status") != "current":
                fr = d.get("freshness", {}) or {}
                reasons.append(f"call_status={d.get('call_status')} (as_of {d.get('as_of')}, {fr.get('age_days')}g)")
            if d.get("data_source_stale"):
                reasons.append(f"BAYAT GİRDİ: {d['data_source_stale']}")
            if d.get("overlay_block"):
                reasons.append(f"overlay_block: {d.get('overlay_block_reason')}")
            if (d.get("spine") or {}).get("tide_degraded"):
                # Denetim 07-11 KOK C: degraded tide artik push-alarmda da (stale-damga run.py'de)
                reasons.append(f"TIDE DEGRADED: eksik modul, kayip agirlik "
                               f"{(d.get('spine') or {}).get('missing_weight_frac', '?')}")
        else:
            reasons.append("latest.json YOK (brief/ledger üretmedi)")
        if reasons:
            notify.alert("VERİ BAYAT / DEGRADE", " | ".join(str(r) for r in reasons))
            _log(f"🔔 ALARM gönderildi: {' | '.join(str(r) for r in reasons)}")
        else:
            notify.clear_alert()
            _log("alarm: temiz (call current + tüm adımlar OK)")
    except Exception as e:
        _log(f"⚠ alarm-adımı hata verdi (best-effort, run'ı öldürmez): {type(e).__name__}: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
