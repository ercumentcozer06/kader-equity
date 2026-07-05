"""
run — kader-equity karar motoru. MODEL = tide × COR1M-froth × GEX-shield (SPX 1.64 / NDX 1.77 @2019+).

Akış: spine (frozen 8-modül sweep vektörü → TIDE_SCORE → TIDE_DIR) → overlay'ler (trim-only) → tazelik
kapısı → §çıktı kontratı. Aktif overlay'ler (strict-FDR/stack-doğrulanmış): COR1M-froth (İLK alfa, düşük
implied-corr=froth) + GEX-shield (dealer short-gamma drawdown kalkanı). Yeni overlay = screen'i geçince modules/'a.

Kullanım:
  python run.py                 # terminal kararı (en güncel frozen tarih)
  python run.py --json          # + output/kader_equity_YYYYMMDD.json
  python run.py --validate      # son JSON'u doğrula (fetch yok)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from config import load_config                            # noqa: E402
from spine import contract as C, tide as T                # noqa: E402

MODEL_TAG = "kader-equity-stack-2026-06-09"  # tide × COR1M-froth × GEX-shield (SPX 1.64/NDX 1.77 @2019+; honest forward ~1.0-1.3)

# NYSE tatil takvimi (Audit 2026-06-19; F5 denetim 2026-07-05) — kapalı-piyasa gününde "current/age 0"
# çağrı damgalanmasın. Eski sabit 2025-27 frozenset SESSİZ SON-KULLANMA tarihliydi: 2027-01-01 sonrası her
# NYSE tatilinde 'açık' der, market_open bayrağı yanlışlanır ve run_preopen.missed_weekday 2027+ tatillerde
# (MLK 2027-01-18 vb.) SAHTE 'KAÇIRILMIŞ İŞGÜNÜ' bayrağı yazardı. Tek tatil kaynağı artık
# modules/opex_calendar (pandas AbstractHolidayCalendar NYSE varyantı; 2035'e deterministik, ağ yok).
def market_closed_reason(today) -> str | None:
    """NYSE bugün kapalı mı + neden. 'hafta sonu' | 'NYSE tatili' | None(açık). Saf/test-edilebilir."""
    if today.weekday() >= 5:
        return "hafta sonu"
    from modules.opex_calendar import is_market_holiday
    if is_market_holiday(today):
        return "NYSE tatili"
    return None


def ledger_eligible(d: dict) -> tuple[bool, str]:
    """F8 (denetim 2026-07-05): forward-ledger'a girme kapısı — TEK kaynak (main + run_daily._brief_and_ledger).
    Eski kapı market_open'ı YOK SAYIYORDU: tatil koşusunda kapalı günün çağrısı deftere giriyor ve
    mark_to_market (searchsorted → önceki kapanış) AYNI ileri-getiriyi farklı pozisyonla İKİ kez sayıyordu
    (2026-06-19 Juneteenth satırı = 06-18 satırıyla aynı Jun-18→Jun-22 getirisi). current + overlay_block
    yok + piyasa AÇIK ⇒ eklenebilir. Fail-closed etkilenmez (hiçbir şey canlı diye servis edilmiyor)."""
    if d.get("call_status") != "current":
        return False, f"status={d.get('call_status')}"
    if d.get("overlay_block"):
        return False, "overlay_block=True"
    if not d.get("market_open", True):
        return False, f"piyasa kapalı ({d.get('market_closed_reason')}) — deftere eklenmedi"
    return True, "current + piyasa açık"


def build_decision(cfg: dict) -> dict:
    import pandas as pd                                   # EQ-3 (denetim 2026-07-04): frozen yol da pd kullanır (aşağıdan taşındı)
    scfg = cfg.get("spine", {}) or {}
    source = scfg.get("source", "frozen")
    live_error = None                                     # canlı denendi+patladı → frozen-fallback notu (STALE damgalanır)

    input_stale = []                                      # bayat-FRED-girdi listesi (canlı yol doldurur)
    if source == "live":
        try:
            from spine.reconstruct import reconstruct_live    # Faz 0 task 3 (canlı skorlar + raw-m2)
            scores_row, vector, as_of, data_source, input_stale = reconstruct_live(cfg)
            td = T.decide(scores_row, vector)
        except Exception as e:
            # CANLI başarısız (FRED/ağ/kader-macro repo) → CRASH YOK; frozen'a düş. AMA fallback ASLA "current"
            # sunulmaz (Bible: bayat/fallback veri güncel çağrı değil) → aşağıda stale zorlanır + ledger'a girmez.
            live_error = f"{type(e).__name__}: {e}"
            source = "frozen-fallback"

    if source != "live":                                  # frozen (config) VEYA frozen-fallback (canlı-patladı)
        scores, _prices, vector, _prov = C.read_frozen()
        as_of = scores.index[-1]
        sc = T.tide_score_series(scores, vector)
        td = {"tide_score": round(float(sc.iloc[-1]), 4), "tide_dir": int(sc.iloc[-1] > 0),
              "modules_used": [k for k in vector if abs(float(vector[k])) > 1e-9]}
        data_source = "frozen"
        # EQ-3 (denetim 2026-07-04): frozen yolda da modül-skor satırı çıkar (canlı yol scores_row'u
        # reconstruct_live'dan alır) → defter attribution kolonları (m9/m5/m2) her iki yoldan da dolar.
        scores_row = {m: (None if pd.isna(v) else float(v)) for m, v in scores.iloc[-1].items()}

    fresh = C.snapshot_freshness(as_of, int(scfg.get("max_staleness_days", 5)))
    cap = float((cfg.get("sizing", {}) or {}).get("net_exposure_cap", 1.0))

    # ── OVERLAY'LER (trim-only, rebound-safe; nihai pozisyon = tide_dir × Π faktör, kaldıraçsız). ──
    #    frozen → sinyali data/cache parquet'inden tide as-of'ta hesapla (ağsız); live → canlı fetch.
    from modules import cor1m_froth, gex_shield            # noqa: E402
    overlays_cfg = cfg.get("overlays", {}) or {}
    factor, overlays_out, active_overlays = 1.0, {}, []

    # OVERLAY 1: COR1M-froth — düşük COR1M (call-froth/complacency) → de-risk. İLK strict-FDR alfa.
    ov = overlays_cfg.get("cor1m_froth", {}) or {}
    if bool(ov.get("enabled")):
        lo, hi, fl = float(ov.get("lo", 8.0)), float(ov.get("hi", 11.0)), float(ov.get("floor", 0.0))
        if data_source == "frozen":
            cpp = ROOT / "data" / "cache" / "corr_pc.parquet"
            cval = None
            if cpp.exists():
                cser = pd.read_parquet(cpp)["COR1M"].dropna()
                cval = float(cser.asof(as_of)) if len(cser) else None
            f1 = cor1m_froth.froth_factor(cval, lo, hi, fl)
            info1 = {"cor1m": round(cval, 2) if cval is not None and not pd.isna(cval) else None,
                     "factor": round(f1, 3), "cor1m_as_of": "frozen(@tide as_of)"}
        else:
            info1 = cor1m_froth.evaluate(cfg)
            f1 = float(info1.get("factor", 1.0))
        factor *= f1
        overlays_out["cor1m_froth"] = info1
        active_overlays.append("cor1m_froth")

    # OVERLAY 2: GEX-shield — z(GEX,252g) derin-düşük (dealer short-gamma) → drawdown kalkanı (alfa DEĞİL).
    ov2 = overlays_cfg.get("gex_shield", {}) or {}
    if bool(ov2.get("enabled")):
        k, thr = float(ov2.get("k", 0.5)), float(ov2.get("thr", 1.0))
        fl2, win = float(ov2.get("floor", 0.4)), int(ov2.get("win", 252))
        if data_source == "frozen":
            gpp = ROOT / "data" / "cache" / "squeeze_dix_gex.parquet"
            zval = None
            if gpp.exists():
                zs = gex_shield.gex_zscore(pd.read_parquet(gpp)["gex"].dropna(), win)
                zval = float(zs.asof(as_of)) if len(zs) else None
            f2 = gex_shield.shield_factor(zval, k, thr, fl2)
            info2 = {"gex_z": round(zval, 2) if zval is not None and not pd.isna(zval) else None,
                     "factor": round(f2, 3), "gex_as_of": "frozen(@tide as_of)"}
        else:
            info2 = gex_shield.evaluate(cfg)
            f2 = float(info2.get("factor", 1.0))
        factor *= f2
        overlays_out["gex_shield"] = info2
        active_overlays.append("gex_shield")

    # H3 fail-safe: GEX-z taşınamaz-bayat (>5g) → sessiz koruma-kapalı YERİNE no-trade bloğu (brief honor eder)
    overlay_block = bool((overlays_out.get("gex_shield", {}) or {}).get("fail_safe_block"))
    overlay_block_reason = (overlays_out.get("gex_shield", {}) or {}).get("error") if overlay_block else None

    pos = float(td["tide_dir"]) * factor                  # tide_dir × Π overlay-faktör (kaldıraçsız, trim-only)
    deploy = min(cap, pos)
    # Audit 2026-06-19 (KRİTİK): altta yatan bir FRED bacağı DONMUŞSA (backtest harness 'always fresh'
    # varsayımını delen denetim) çağrı GÜNCEL değildir → fail-LOUD: call_status STALE + log.error.
    if input_stale:
        import logging as _logging
        _logging.getLogger("kader_equity").error(
            "CANLI TIDE BAYAT FRED GİRDİSİ: %s — çağrı STALE damgalanıyor (current DEĞİL)",
            ", ".join(f"{s.get('series')}({s.get('age_bd', s.get('error'))})" for s in input_stale))
    stale = bool(fresh["stale"]) or (live_error is not None) or bool(input_stale)   # canlı-patlak/bayat-girdi ASLA "current" değil

    # ── OpEx TAKTİK KAPISI (frozen stack DIŞI; per-asset ifade override + takvim uyarısı). ──
    #    position_target DEĞİŞMEZ; yalnız NDX sleeve OpEx günü deploy→0 (FINDING 23, p=0.001 anomali).
    #    Referans = BUGÜN (ileriye-dönük bilinen takvim; bayat snapshot tarihine bağlı DEĞİL → "3 gün önce uyar" çalışır).
    from modules import opex_calendar
    ocfg = cfg.get("opex_gate", {}) or {}
    today = datetime.now(timezone.utc).date()
    opex = opex_calendar.evaluate(today, ocfg) if bool(ocfg.get("enabled")) else None

    # ── CONSTAN BAĞLAM BANTLARI (2026-06-13; pozisyon etkisi SIFIR — OpEx-etiket emsali). ──
    #    santa_window: koşullu Noel penceresi (standalone 1928+ GERÇEK, stack'e ABSORBED → etiket).
    #    net_supply_context: hisse net-arzı paneli (yön-sinyali FAIL → betimsel + uç-kuyruk bayrağı).
    #    Her ikisi graceful: veri yoksa None; flag OFF → anahtar None (çıktı şekli sabit).
    santa = None
    scfg = cfg.get("santa_context", {}) or {}
    if bool(scfg.get("enabled")):
        try:
            from modules import santa_window
            santa = santa_window.evaluate(today, scfg)
        except Exception as _e:
            print(f"  [!] santa_window hata (band atlandi): {_e}", file=sys.stderr)
    nsup = None
    ncfg = cfg.get("net_supply_context", {}) or {}
    if bool(ncfg.get("enabled")):
        try:
            from modules import net_supply_context
            nsup = net_supply_context.evaluate(ncfg, today=today)
        except Exception as _e:
            print(f"  [!] net_supply_context hata (panel atlandi): {_e}", file=sys.stderr)
    ipo = None
    icfg = cfg.get("ipo_pipeline_context", {}) or {}
    if bool(icfg.get("enabled")):
        try:
            from modules import ipo_pipeline_context
            ipo = ipo_pipeline_context.evaluate(icfg, today=today)
        except Exception as _e:
            print(f"  [!] ipo_pipeline_context hata (band atlandi): {_e}", file=sys.stderr)
    # dev-IPO GECİKMELİ arz dalgaları (lock-up bitişi + endeks-dahil; betimsel, pozisyon etkisi YOK)
    waves = None
    wcfg = cfg.get("ipo_supply_waves", {}) or {}
    if bool(wcfg.get("enabled")):
        try:
            from modules import ipo_supply_waves
            waves = ipo_supply_waves.evaluate(wcfg, today=today)
        except Exception as _e:
            print(f"  [!] ipo_supply_waves hata (band atlandi): {_e}", file=sys.stderr)
    # K1 arz-talep denge kadranı (betimsel; pozisyon etkisi YOK)
    sdbal = None
    bcfg = cfg.get("supply_demand_balance", {}) or {}
    if bool(bcfg.get("enabled")):
        try:
            from modules import supply_demand_balance
            sdbal = supply_demand_balance.evaluate(bcfg, today=today)
        except Exception as _e:
            print(f"  [!] supply_demand_balance hata (panel atlandi): {_e}", file=sys.stderr)
    # K2 koşullu de-risk (arz-aşırı VE talep-zayıf → asset_deploy trim; OpEx emsali, frozen stack DIŞI)
    sdrisk = None
    dcfg = cfg.get("supply_demand_derisk", {}) or {}
    if bool(dcfg.get("enabled")):
        try:
            from modules import supply_demand_derisk
            # #4 (2026-06-19): CANLI tide'ı geçir → K2 talep-kapısı 28g-bayat frozen-son-değeri kullanmasın
            _lts = td["tide_score"] if data_source == "live" else None
            _ltd = td["tide_dir"] if data_source == "live" else None
            sdrisk = supply_demand_derisk.evaluate(dcfg, live_tide_score=_lts, live_tide_dir=_ltd)
        except Exception as _e:
            print(f"  [!] supply_demand_derisk hata (trim atlandi): {_e}", file=sys.stderr)

    asset_deploy = {}
    # K2 trim faktörü: ateşlediyse SPX+NDX deploy'una OpEx'le ÇARPIMSAL uygulanır (rejim-değişim
    # sigortası; frozen position_target DEĞİŞMEZ — yalnız bu asset_deploy katmanı). position_effect
    # flag'i OFF ise trim hesaplanır+raporlanır ama deploy'a UYGULANMAZ (gözcü modu).
    _k2_factor = (float(sdrisk["trim_factor"]) if (sdrisk and sdrisk.get("fired")
                  and bool(dcfg.get("position_effect", True))) else 1.0)
    if opex or _k2_factor != 1.0:
        for a in (cfg.get("assets", []) or []):
            ov = (opex.get("asset_overrides", {}) or {}).get(a) if opex else None
            base_a = float(ov["deploy"]) if ov else deploy        # OpEx NDX günü→0
            asset_deploy[a] = round(base_a * _k2_factor, 4)        # ×K2-trim (OpEx-0 ise 0 kalır)

    # Audit 2026-06-19: takvim tatil-farkındalığı yoktu → kapalı-piyasa gününde (örn. Juneteenth)
    # "current/age 0" çağrı damgalanıp ledger'a yazılıyordu. POZİSYONU DEĞİŞTİRMEZ (veri = son işgünü,
    # çağrı bir sonraki açılışa uygulanır); yalnız market_open bayrağı + pusula notu ekler (görünürlük).
    _mkt_closed_reason = market_closed_reason(today)
    market_open = _mkt_closed_reason is None

    return {
        "model_tag": MODEL_TAG,
        "as_of": str(as_of.date() if hasattr(as_of, "date") else as_of),
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,                       # frozen | live
        "call_status": "STALE" if stale else "current",   # bayatsa GÜNCEL çağrı değil, yalnız as_of durumu
        "data_source_stale": input_stale or None,         # DONMUŞ alt-FRED bacakları (boş=temiz); doluysa call_status STALE
        "market_open": market_open,                       # bugün NYSE açık mı (kapalıysa çağrı sonraki açılışa)
        "market_closed_reason": _mkt_closed_reason,       # hafta sonu | NYSE tatili | None
        "_live_error": live_error,                         # canlı başarısız→frozen fallback nedeni (None=normal)
        "freshness": fresh,
        "tide_score": td["tide_score"],
        "tide_dir": td["tide_dir"],
        # EQ-3 (denetim 2026-07-04): modül günlük skorları (defter m9/m5/m2 kolonları buradan beslenir; None-safe)
        "module_scores": {m: (None if v is None else round(float(v), 4)) for m, v in scores_row.items()},
        "direction": "LONG" if td["tide_dir"] else "FLAT",
        "position_target": round(pos, 4),
        "deploy_fraction": round(deploy, 4),              # TEK deploy sayısı = min(cap, |pos|), kaldıraçsız sermaye oranı
        "leverage": 1.0,
        "active_overlays": active_overlays,               # aktif overlay'ler (cor1m_froth, gex_shield, ...)
        "overlay_block": overlay_block,                   # H3: True → brief no-trade (fail-loud, sessiz koruma-kapalı değil)
        "overlay_block_reason": overlay_block_reason,
        "overlays": overlays_out,
        "opex_gate": opex,                                # OpEx takvim uyarısı + NDX-flat (FINDING 23; stack DIŞI)
        "asset_deploy": asset_deploy or None,             # per-asset deploy (NDX OpEx günü 0, yoksa = deploy_fraction)
        "santa_context": santa,                           # Constan koşullu-Noel bandı (etiket; pozisyon etkisi yok)
        "net_supply_context": nsup,                       # Constan net-arz paneli (betimsel + uç-kuyruk bayrağı)
        "ipo_pipeline_context": ipo,                      # EDGAR S-1 boru-hattı + dev-arz izleme (FINDING 26)
        "ipo_supply_waves": waves,                        # dev-IPO gecikmeli arz dalgaları (lock-up + endeks; betimsel)
        "supply_demand_balance": sdbal,                   # K1 arz-talep denge kadranı (betimsel; FINDING 27)
        "supply_demand_derisk": sdrisk,                   # K2 koşullu de-risk (asset_deploy trim; FINDING 27)
        "_sd_derisk_position_effect": bool(dcfg.get("position_effect", True)),
        "spine": {
            "recipe": "8-modül sweep winner (RAW m2, +1g lag); m9 .563/m5 .214/m2 .118/m0 .061/m3 .025/m6 .01/m8 .006/m4 .002",
            "vector": {k: round(float(v), 4) for k, v in vector.items() if abs(float(v)) > 1e-9},
            "modules_used": td["modules_used"],
            "modules_missing": td.get("modules_missing", []),      # eksik (sessiz 0-oy) modüller — görünür
            "tide_degraded": bool(td.get("degraded", False)),      # eksik modül VARSA True (m9=ağırlığın %56'sı)
            "missing_weight_frac": td.get("missing_weight_frac", 0.0),
        },
        "_pass": (f"spine + {' × '.join(active_overlays)}" if active_overlays
                  else "spine-only (overlay yok → çıktı == tide_dir)"),
    }


def _render(d: dict) -> None:
    stale = d.get("call_status") == "STALE"
    fr = d.get("freshness", {})
    print("=" * 72)
    if stale:
        print("  ⚠ KADER EQUITY — VERİ BAYAT: GÜNCEL ÇAĞRI YOK")
        if d.get("_live_error"):
            print(f"  • CANLI BAŞARISIZ → frozen'a düşüldü: {d['_live_error']}")
            print(f"    (FRED/ağ/kader-macro erişimini kontrol et; aşağısı {fr.get('as_of')} frozen snapshot'tan)")
        else:
            print(f"  • spine snapshot {fr.get('as_of')} ({fr.get('age_days')}g > {fr.get('max_staleness_days')}g eşik).")
            print("    canlı skor için: spine.source=live (rekonstrüktör) ya da gen_snapshot.py tazele.")
        print("  ↓ aşağısı TEMKİNLİ oku — bugünü tam temsil etmeyebilir ↓")
    else:
        print(f"  KADER EQUITY PUSULA — {d['direction']}  (spine + {len(d.get('active_overlays', []))} overlay)")
    print("=" * 72)
    print(f"  veri kaynağı   : {d['data_source']}")
    tag = "tarih (BAYAT)  " if stale else "tarih          "
    print(f"  {tag}: {d['as_of']}" + (f"  ({fr.get('age_days')}g eski)" if stale else ""))
    if not d.get("market_open", True):
        print(f"  ⚠ piyasa KAPALI : {d.get('market_closed_reason')} — çağrı bir sonraki açılışa uygulanır (bugün exec yok)")
    if d.get("data_source_stale"):
        _ds = d["data_source_stale"]
        print(f"  ⛔ BAYAT FRED GİRDİSİ: " + ", ".join(
            f"{s.get('series')} {s.get('age_bd', s.get('error'))}{'g' if s.get('age_bd') is not None else ''}" for s in _ds)
            + " — çağrı GÜNCEL DEĞİL (donmuş bacak)")
    print(f"  tide           : score {d['tide_score']:+.2f} → {d['direction']} (yön {d['tide_dir']})")
    if d.get("spine", {}).get("tide_degraded"):
        _sp = d["spine"]
        print(f"  ⚠ TIDE DEGRADED: eksik modül {_sp.get('modules_missing')} "
              f"(kayıp ağırlık %{100*float(_sp.get('missing_weight_frac', 0)):.0f}) — skor eksik-0 ile hesaplandı")
    print(f"  spine          : {d['spine']['recipe']}")
    for name in d.get("active_overlays", []):
        inf = d.get("overlays", {}).get(name, {})
        # #3 (2026-06-19): bir overlay fetch-fail/stale yüzünden DISARMED (factor 1.0) olduysa SESSİZ değil
        # GÖRÜNÜR olsun — 'factor 1.0' tek başına 'normal' mi yoksa 'körleşmiş mi' ayırt edilemiyordu.
        _disarmed = (inf.get("available") is False) or bool(inf.get("stale"))
        _why = inf.get("error") or inf.get("reason") if _disarmed else None
        _mark = f"  ⚠ DISARMED ({_why})" if _disarmed else ""
        print(f"  overlay        : {name} → factor {inf.get('factor')}{_mark}")
    if not d.get("active_overlays"):
        print("  overlay        : — (yok)")
    print(f"  → POZİSYON     : {d['position_target']:+.2f}  "
          f"(deploy %{100*d['deploy_fraction']:.0f} sermaye, kaldıraçsız)")
    ox = d.get("opex_gate")
    if ox:
        if ox.get("is_opex_today"):
            print(f"  ⚠ OpEx GÜNÜ    : BUGÜN monthly OpEx → NDX sleeve FLAT (anomali −14bps p=0.001)")
        elif ox.get("warn"):
            qw = " [QUAD-WITCH]" if ox.get("is_quad_witch_next") else ""
            sh = " (3.-Cuma tatil → Perşembe'ye kaydı)" if ox.get("holiday_shifted") else ""
            wd = ox.get("next_opex_weekday", "")
            print(f"  ⚠ OpEx UYARI   : fiili monthly OpEx {ox['trading_days_until']} işgünü sonra "
                  f"({ox['next_opex']} {wd}{qw}{sh}) → o gün NDX FLAT planı")
        ad = d.get("asset_deploy")
        if ad:
            print(f"  per-asset      : " + " / ".join(f"{k} %{100*v:.0f}" for k, v in ad.items()))
    sc = d.get("santa_context")
    if sc:
        st = sc.get("state")
        if st == "QUALIFYING_ACTIVE":
            print(f"  ★ Noel bandı   : AKTİF — 1 Kas YTD %{sc.get('ytd_at_nov1_pct'):+.1f} ≥ %+10 "
                  f"(1928+: ort +4.5%, isabet %88) [etiket; pozisyon etkisi yok]")
        elif st == "NON_QUALIFYING":
            print(f"  ⚠ Noel bandı   : NİTELİKSİZ — 1 Kas YTD %{sc.get('ytd_at_nov1_pct'):+.1f} < %+10 "
                  f"→ Kas-Ara mevsimsel rüzgar YOK (1928+: ort ~0, min −22.7%)")
        elif st == "INACTIVE" and sc.get("preview_note"):
            print(f"  Noel ön-izleme : {sc['preview_note']}")
    ns = d.get("net_supply_context")
    if ns:
        stale_tag = " [BAYAT]" if ns.get("stale") else ""
        flip = f" | {ns['flip_note']}" if ns.get("flip_note") else ""
        print(f"  net-arz paneli : {ns['quarter']} NFC {ns.get('ratio4q_nfc_pct_ngdp'):+.2f}% NGDP "
              f"(z10y {ns.get('z10y_nfc'):+.2f}){flip}{stale_tag} [betimsel]")
        if ns.get("tail_flag"):
            print(f"  ⚠ İHRAÇ-ÇILGINLIĞI bayrağı: z10y ≥ +{ns['tail_threshold']} — {ns['tail_note']}")
        comp = ns.get("components")
        if comp:
            bits = []
            oi = comp.get("opco_ipo")
            if oi:
                bits.append(f"opco-IPO {oi['quarter']} 4Ç {oi.get('n_4q')} adet (z {oi['z10y']:+.2f})")
            sp = comp.get("spac")
            if sp:
                bits.append(f"SPAC {sp['year']}: {sp['n']} (2021: 613)")
            bb = comp.get("buyback")
            if bb:
                pre = " ön-değer" if bb.get("prelim") else ""
                bits.append(f"SPX-buyback {bb['quarter']} ${bb['bn']:.0f}B"
                            f" ({bb.get('ratio4q_pct_ngdp')}% NGDP, z {bb.get('z10y'):+.2f}){pre}")
            if bits:
                print(f"    bileşenler   : " + " | ".join(bits) + "  [betimsel]")
    ip = d.get("ipo_pipeline_context")
    if ip:
        st = " [BAYAT]" if ip.get("stale") else ""
        print(f"  S-1 boru-hattı : {ip['last_closed_quarter']} 4Ç {ip.get('roll4q_filings')} dosyalama "
              f"(z {ip.get('z10y'):+.2f}) | son-90g {ip.get('last90d_new')} yeni{st}  [niyet≠gerçekleşme]")
        for m in (ip.get("mega_hits_120d") or []):
            ceil = m.get("max_ceiling_usd")
            ceil_s = f", kayıt-tavanı ${ceil/1e9:.1f}B" if ceil else ""
            print(f"  ⚠ DEV-ARZ İZLEME: {m['watch_name']} S-1 dosyaladı "
                  f"({m['first_filed']}, {m['n_filings']} dosyalama{ceil_s}) — Constan-tezi CANLI")
    wv = d.get("ipo_supply_waves")
    if wv and wv.get("waves"):
        print(f"  ─ DEV-IPO GECİKMELİ ARZ DALGALARI ({wv['n_active']} aktif; TAHMİN, pozisyon etkisi YOK) ─")
        for w in wv["waves"]:
            sh = w.get("lockup_shares_est")
            sh_s = (f"~{sh/1e9:.1f}B hisse" if sh and sh >= 1e9 else
                    (f"~{sh/1e6:.0f}M hisse" if sh else "hisse adedi belirsiz"))
            pctn = f" ({w['lockup_pct_note']})" if w.get("lockup_pct_note") else ""
            dtl = w.get("days_to_lockup")
            par = "" if w.get("lockup_days_parsed") else " [VARSAYILAN 180g — prospektüsten okunamadı]"
            print(f"  ⚠ LOCK-UP UYARISI: {w['company']} {sh_s}{pctn} {dtl} gün sonra serbest "
                  f"({w['lockup_expiry_date']}, {w['lockup_days']}-gün kilit{par})")
            iw = w.get("index_incl_window")
            if iw:
                lo = w.get("forced_passive_demand_usd_lo")
                hi = w.get("forced_passive_demand_usd_hi")
                fd = (f", forced pasif talep ~${lo/1e9:.0f}-{hi/1e9:.0f}B"
                      if lo and hi else "")
                print(f"     endeks-dahil penceresi: {iw['earliest']}..{iw['latest']} "
                      f"(en erken {iw['days_to_earliest']} gün sonra){fd}")
        print(f"     [{wv['label']}]")
    sb = d.get("supply_demand_balance")
    if sb and sb.get("net_supply_pressure") is not None:
        print(f"  arz-talep denge: {sb['quarter']} baskı {sb['net_supply_pressure']:+.2f}z "
              f"(arz {sb.get('supply_z'):+.2f} / talep {sb.get('demand_z'):+.2f}) → "
              f"{sb.get('direction')}  [betimsel, ayraç değil]")
    dr = d.get("supply_demand_derisk")
    if dr:
        if dr.get("fired"):
            pe = bool((d.get("_sd_derisk_position_effect", True)))
            tag = f"→ SPX/NDX deploy ×{dr['trim_factor']}" if pe else "(gözcü modu; deploy'a uygulanmadı)"
            arm = " [MEGA-IPO ANLIK arz kolu]" if dr.get("supply_arm") == "mega-IPO" else ""
            print(f"  ⚠ ARZ-ŞOKU DE-RİSK: ATEŞLEDİ{arm} — {dr.get('reason')} {tag}")
            print(f"     ({dr.get('label')})")
        else:
            arm = ""
            if dr.get("mega_hi"):
                # mega kol AKTİF ama talep güçlü → SUSAR (2020-tipi koruma görünür kalsın)
                arm = (f" [MEGA-IPO {dr.get('mega_label')} ${dr['mega_ceiling_usd']/1e9:.0f}B kayıtlı "
                       f"ama talep güçlü → SUSAR]")
            print(f"  arz-şoku de-risk: sus ({dr.get('reason')}){arm} [rejim-sigortası, alfa değil]")
    print("-" * 72)


def _validate(_cfg: dict) -> int:
    out_dir = ROOT / "output"
    files = sorted(out_dir.glob("kader_equity_*.json")) if out_dir.exists() else []
    if not files:
        print("  [!] doğrulanacak çıktı yok (önce --json ile üret).")
        return 2
    d = json.loads(files[-1].read_text(encoding="utf-8"))
    schema_ok = (d.get("direction") in ("LONG", "FLAT")
                 and 0.0 <= d.get("position_target", 99) <= 1.0)
    stale = d.get("call_status") == "STALE"
    print(f"  {files[-1].name}: dir={d.get('direction')} pos={d.get('position_target')} "
          f"call={d.get('call_status')} → {'[OK]' if schema_ok else '[!] şema dışı'}"
          + ("  (BAYAT)" if stale else ""))
    return 0 if schema_ok else 1


def ledger_record(d: dict) -> dict:
    """EQ-3 (denetim 2026-07-04): karar dict'i → forward-ledger kaydı (TEK şema kaynağı; run.py main +
    run_daily._brief_and_ledger aynı fonksiyonu kullanır → iki kayıt yolu ıraksamaz). GÖREV 6a attribution
    kolonları (m9/m5/m2) module_scores'tan None-safe doldurulur — şemada olup HİÇ yazılmayan kolon biter."""
    _msc = (d.get("module_scores") or {})
    return {"as_of": d["as_of"], "computed_at": d["computed_at"], "model_tag": d["model_tag"],
            "call_status": d["call_status"], "position_target": d["position_target"],
            "direction": d["direction"], "size": d["deploy_fraction"], "tide_dir": d["tide_dir"],
            "tide_score": d["tide_score"], "active_overlays": ",".join(d["active_overlays"]),
            "data_source": d["data_source"],
            "m9_score": _msc.get("m9"), "m5_score": _msc.get("m5"), "m2_score": _msc.get("m2")}


def write_latest(d: dict) -> None:
    """EQ-2 (denetim 2026-07-04): insan-okunur latest artefaktı — #20 garantisi artık CLI'ye özel değil;
    otomasyon yolu (run_daily) da her koşuda bunu çağırır. STALE çağrıda da yazılır (durum dürüst görünür)."""
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "kader_equity_latest.json").write_text(
        json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="kader-equity karar motoru (tide × COR1M-froth × GEX-shield)")
    ap.add_argument("--json", action="store_true", help="output/kader_equity_YYYYMMDD.json yaz")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--validate", action="store_true", help="son JSON'u doğrula (fetch yok)")
    args = ap.parse_args(argv)

    cfg = load_config()
    if args.validate:
        return _validate(cfg)

    d = build_decision(cfg)

    _elig, _why = ledger_eligible(d)                      # F8: STALE / overlay_block / piyasa-KAPALI çağrı deftere EKLENMEZ
    if _elig:                                             # (EQ-2 write_latest aşağıda her durumda)
        try:
            from validation import ledger as _ledger
            _ledger.append_call(ledger_record(d))         # EQ-3 (denetim 2026-07-04): tek şema + m9/m5/m2 dolu
        except Exception as e:
            if not args.quiet:
                print(f"  (forward-ledger atlandı: {e})")
    elif not args.quiet and d.get("call_status") == "current":
        print(f"  (forward-ledger'a eklenmedi: {_why})")

    if not args.quiet:
        _render(d)

    # #20 (2026-06-19): --json'suz koşuda dated snapshot yazılmıyordu → diskteki eski brief CANLI çağrıyla
    # ÇELİŞİYORDU (pos=1.0 vs gerçek 0.007). Artık HER koşu `kader_equity_latest.json`'u günceller
    # (insan-okunur artefakt asla bayat kalmaz). Dated snapshot (--json) tarihçe için opt-in kalır.
    out_dir = ROOT / "output"
    try:
        write_latest(d)
    except Exception as e:
        if not args.quiet:
            print(f"  (latest.json yazılamadı: {e})")

    if args.json:
        f = out_dir / f"kader_equity_{datetime.now(timezone.utc):%Y%m%d}.json"
        f.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        if not args.quiet:
            print(f"  JSON → {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
