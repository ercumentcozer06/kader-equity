"""
backtest/remeasure/RC0_gapcheck.py — RC0.2 rolling-window kayıp-kontrolü + WAVE-0 kapı raporu (TEŞHİS-ONLY).

(a) data/raw_chains/<SYM>/ dosya-tarihleri vs beklenen liste (md_spy tarihleri + son-md-tarihinden
    bugün-1'e iş günleri; R0_backfill.py:72-77 kurulumuyla BİREBİR) → sembol-başına eksik/fazla listesi.
(b) Pencere-kaybı: sembol-başına diskteki en-eski gün; md_spy'ın ilk gününün diskte olup olmadığı
    (+ dosya içi _fetch_ts teyidi).
(c) TEK 1-kredi probe: md-ilk-gününün BİR ÖNCEKİ iş günü istenir (binary search YOK) → 402/no_data
    beklenir = API rolling-window'unun en-eskisi ≥ md-ilk-günü. Kayıp tarih = beklenen ∧ diskte-yok ∧
    API-penceresi-dışı (kalıcı kayıp); diskte-yok ∧ pencere-içi olanlar AYRI listelenir (hâlâ çekilebilir).

Hiçbir cache dosyası yazılmaz/silinmez (append-only'e dokunmaz); tek API isteği = 1 kredi.
Token .env'den okunur, ASLA basılmaz. Çıktı: RC0_gapcheck.json (config_sha dahil).
  & <venv python> backtest/remeasure/RC0_gapcheck.py
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402  (TEK-GERÇEK-KAYNAK)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# --- orkestrasyon sabitleri (model/ölçüm sabiti DEĞİL; R0_backfill.py:34 ve wave0_credit_probe.py ile aynı) ---
BASE = "https://api.marketdata.app/v1/options/chain"
OUT_PATH = config.REMEASURE_DIR / "RC0_gapcheck.json"
# WAVE-0 fetcher sonuçları (görev girdisi; çapraz-doğrulama için kayda geçer)
FETCHER_RESULTS = [
    {"sym": "SPY", "fetched": 87, "done": True, "missing": []},
    {"sym": "QQQ", "fetched": 88, "done": True, "missing": []},
    {"sym": "SPX", "fetched": 89, "done": True, "missing": ["2025-12-12"]},
    {"sym": "NDX", "fetched": 87, "done": True, "missing": ["2025-12-12"]},
]


def expected_dates() -> list[str]:
    """R0_backfill.py:72-77 ile BİREBİR: md_spy tarihleri + son-md-tarihinden bugün-1'e iş günleri."""
    md = pd.read_parquet(config.ROOT / "data" / "historical_chains" / "md_spy.parquet")
    dates = [d.date().isoformat() for d in sorted(pd.to_datetime(md["date"].unique()))]
    ext = [d.date().isoformat() for d in pd.bdate_range(pd.Timestamp(dates[-1]) + timedelta(days=1),
                                                        pd.Timestamp(date.today()) - timedelta(days=1))]
    return dates + ext


def _fetch_ts(p: Path) -> str | None:
    try:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f).get("_fetch_ts")
    except Exception:
        return None


def _sanitize(msg: str, token: str) -> str:
    return msg.replace(token, "***TOKEN***") if token else msg


def main() -> int:
    exp = expected_dates()
    exp_set = set(exp)
    md_first = exp[0]

    out: dict = {
        "config_sha": config.config_sha(),
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "RC0.2 rolling-window gap-check + WAVE-0 kapı",
        "expected": {"n": len(exp), "first": exp[0], "last": exp[-1],
                     "source": "md_spy.parquet + bdate-ext bugün-1'e (R0_backfill.py:72-77 birebir)"},
        "fetcher_results_input": FETCHER_RESULTS,
    }

    # ---- (a) sembol-başına kapsama + (b) pencere-kaybı ----
    per_sym: dict = {}
    all_missing: dict[str, list[str]] = {}
    for sym in config.SYMS:
        raw = config.RAW_DIR / sym
        have = sorted(p.name.removesuffix(".json.gz") for p in raw.glob("*.json.gz")) if raw.exists() else []
        have_set = set(have)
        missing = [d for d in exp if d not in have_set]
        extra = [d for d in have if d not in exp_set]
        first_file = raw / f"{md_first}.json.gz"
        per_sym[sym] = {
            "have_n": len(have),
            "oldest_on_disk": have[0] if have else None,
            "newest_on_disk": have[-1] if have else None,
            "missing_n": len(missing),
            "missing": missing,
            "extra_n": len(extra),
            "extra": extra,
            "md_first_on_disk": first_file.exists(),
            "md_first_fetch_ts": _fetch_ts(first_file) if first_file.exists() else None,
        }
        all_missing[sym] = missing
    out["per_symbol"] = per_sym
    out["window"] = {
        "md_first": md_first,
        "md_first_on_disk_all_syms": all(per_sym[s]["md_first_on_disk"] for s in config.SYMS),
    }

    # ---- (c) TEK 1-kredi probe: md-ilk-günün bir önceki iş günü (binary search YOK) ----
    # idempotent kredi-guard: bugünkü koşumdan probe sonucu varsa REUSE (re-run yeni kredi yakmaz)
    probe_date = (pd.Timestamp(md_first) - pd.offsets.BDay(1)).date().isoformat()
    probe_sym = config.SYMS[0]   # SPY; expiration param YOK → 1 kredi (wave0_credit_probe ile aynı)
    today_utc = datetime.now(timezone.utc).date().isoformat()
    if OUT_PATH.exists():
        try:
            prev = json.loads(OUT_PATH.read_text(encoding="utf-8"))
            pp = prev.get("probe", {})
            if pp.get("date") == probe_date and pp.get("http_status") is not None \
                    and prev.get("ts_utc", "").startswith(today_utc):
                pp["reused_from_prior_run"] = prev.get("ts_utc")
                out["probe"] = pp
        except Exception:
            pass
    load_dotenv(config.ROOT / ".env")
    token = os.environ.get("MARKETDATA_TOKEN", "")
    probe: dict = out.get("probe") or {"date": probe_date, "symbol": probe_sym,
                                       "expectation": "402/no_data → API en-eski ≥ md-ilk-günü"}
    if "reused_from_prior_run" in probe:
        pass   # bugünkü probe zaten atıldı; tekrar kredi yakma
    elif not token:
        probe["error"] = "MARKETDATA_TOKEN .env'de bulunamadı"
    else:
        try:
            r = requests.get(f"{BASE}/{probe_sym}/", params={"date": probe_date, "token": token}, timeout=60)
            h = {k.lower(): v for k, v in r.headers.items()}
            probe["http_status"] = r.status_code
            for name, key in [("x-api-ratelimit-remaining", "ratelimit_remaining"),
                              ("x-api-ratelimit-consumed", "consumed_this_call")]:
                try:
                    probe[key] = int(h.get(name))
                except (TypeError, ValueError):
                    probe[key] = None
            try:
                body = r.json()
                probe["body_s"] = body.get("s")
                probe["body_errmsg"] = body.get("errmsg")
                probe["n_contracts_returned"] = len(body.get("optionSymbol") or [])
            except Exception:
                probe["body_s"] = None
        except Exception as e:  # noqa: BLE001
            probe["error"] = _sanitize(f"{type(e).__name__}: {e}", token)

    served = probe.get("http_status") in (200, 203) and probe.get("body_s") == "ok" \
        and (probe.get("n_contracts_returned") or 0) > 0
    probe["probe_date_served"] = bool(served)
    # tek probe'dan çıkan PENCERE-TABANI (kanıtlanmış en-eski servis edilen sınır):
    #  - probe SERVİS EDİLDİ → pencere ≥ probe_date'e kadar geri gidiyor (api_oldest ≤ probe_date; tam değeri
    #    tek-kredi bütçesiyle ÖLÇÜLEMEDİ — binary search gerekirdi, görev yasaklıyor)
    #  - servis EDİLMEDİ (402/no_data) → kanıtlanmış taban = md-ilk-günü (bugün fetch edildi, diskte)
    window_floor_proven = probe_date if served else md_first
    probe["window_floor_proven"] = window_floor_proven
    probe["interpretation"] = (
        f"probe {probe_date} SERVİS EDİLDİ → API penceresi md-ilk-gününden ({md_first}) en az 1 iş günü daha "
        "eskiyi kapsıyor; beklenen 402/no_data GERÇEKLEŞMEDİ; pencere-kaybı riski bugün YOK"
        if served else
        f"probe {probe_date} verilmedi (beklendiği gibi) → API en-eskisi ≥ {md_first}; "
        "md-ilk-günü diskte olduğundan pencere-kaybı YOK"
    )
    out["probe"] = probe

    # ---- kayıp-tarih ayrımı: pencere-DIŞI (kalıcı) vs pencere-İÇİ (hâlâ çekilebilir / sağlayıcı-deliği) ----
    lost = {s: [d for d in m if d < window_floor_proven] for s, m in all_missing.items()}
    in_window_missing = {s: [d for d in m if d >= window_floor_proven] for s, m in all_missing.items()}
    out["lost_dates_window"] = lost                      # rolling-window'a düşmüş = kalıcı kayıp
    out["missing_in_window"] = in_window_missing         # pencere-içi eksik (fetcher 'missing' ile çapraz)
    out["fetcher_crosscheck"] = {
        fr["sym"]: {"fetcher_missing": fr["missing"],
                    "disk_missing": all_missing.get(fr["sym"], []),
                    "match": fr["missing"] == all_missing.get(fr["sym"], [])}
        for fr in FETCHER_RESULTS
    }

    # ---- WAVE-0 kapı: pencere-kaybı YOK + md-ilk-günü 4 sembolde diskte → PASS ----
    # (probe'un servis edilmesi kapıyı DÜŞÜRMEZ — pencere beklenenden GENİŞ demektir, kayıp tersi yönde olur)
    window_ok = out["window"]["md_first_on_disk_all_syms"] and all(len(v) == 0 for v in lost.values())
    coverage_full = all(len(v) == 0 for v in all_missing.values())
    out["gate"] = {
        "window_ok": bool(window_ok),
        "coverage_full": bool(coverage_full),
        "wave0_pass": bool(window_ok),   # kapı = pencere-kaybı YOK; pencere-içi sağlayıcı-deliği kapıyı düşürmez
        "note": "pencere-içi eksikler (varsa) kalıcı kayıp DEĞİL; sağlayıcı no-data ya da yeniden-denenebilir",
    }

    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
