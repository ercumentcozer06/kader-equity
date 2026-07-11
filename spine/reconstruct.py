"""
spine/reconstruct — CANLI tide rekonstrüktörü (FAZ-0 task 3). Donmuş snapshot (2026-05-22'de biten) yerine
kader-macro'nun GÜNCEL modül skorlarını okur → her zaman canlı tide. STALE damgası kalkar.

NEDEN gerekli: build_module_matrix (sweep4000) BACKTEST harness'ı → `_pit_signals_brakes_full.csv`'yi okur
ve `END="2026-05-22"` hard-cap'ler. Yani o yol ASLA bugüne gelmez. Bunun yerine PIT-replay makinesini
(pit_replay.run_pit_signals) DOĞRUDAN, bugüne uzatılmış kısa-grid üzerinde koşarız: seriler tam-tarih çekilip
as_of(t) ile dilimlenir → rolling z-skorları tüm geçmişi kullanır → kısa grid yeterli, SON satır = bugünün
makro okuması. 05-22 örtüşmesinde frozen CSV ile BYTE-AYNI doğrulandı (max diff 0.0) → vektör normalizasyonu
KAYMAZ. m2 RAW net-liq (SMART_M2 pop). m3 (auction) + m6 (Moody's Baa-Aaa) build_module_matrix ile AYNI enjekte.

MALİYET: fetch_all_series ~150s (cache'li sonra) + replay ~1-2dk → bu yüzden GÜN-BAZLI cache
(data/cache/live_tide_latest.json); aynı UTC gün içinde tekrar çağrı anında döner.

Yalnız spine.source=live iken çağrılır (frozen path kader-macro'yu hiç import etmez → reproducible/ağsız kalır).
run.py akışı: scores_row, vector, as_of, data_source = reconstruct_live(cfg); td = tide.decide(scores_row, vector).
kader-macro KISITI: sadece READ — dl/sg/pit_replay fonksiyonlarını import edip çağırırız; run.py/modules/analysis'e
DOKUNMAYIZ; paylaşılan _pit_signals_brakes_full.csv'yi de YAZMAYIZ (frozen-spine reproducibility korunur).
"""
from __future__ import annotations

import functools
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FROZEN = Path(__file__).resolve().parent / "frozen"
CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
LIVE_CACHE = CACHE / "live_tide_latest.json"

# build_module_matrix (sweep4000) modül evreni — vektörün beklediği sıra/küme.
MODS = ["m0", "m1", "m2", "m3", "m4", "m5", "m6", "m8", "m9", "m10", "m11"]
GRID_BUFFER_DAYS = 60          # bugüne kadar kısa grid (rolling z tüm seriyi kullanır → buffer yeterli)


def _inject_m3_m6(cfg: dict, dl, sg, index: pd.DatetimeIndex) -> tuple[dict, object]:
    """m3 (auction-demand pctl) + m6 (Moody's Baa-Aaa kredi-stres) — build_module_matrix ile BİREBİR aynı.

    Döndürür (out, m3_asof): m3_asof = son geçerli auction tarihi (None=boş/erişilemez) → P1-C tazelik-denetimi
    (denetim 2026-07-07): M3 SPECS-dışıydı, fetch_auctions boş dönerse m3 sessizce 0'a düşüp çağrı 'current'
    kalıyordu (KÖR-NOKTA). Artık m3_asof _build_live_panel'de denetlenip bayatsa input_stale'e girer."""
    import requests
    out: dict = {}
    # m3 — canlı auction talebi (cache-gap düzeltmesi, build_module_matrix ile aynı)
    auctions = dl.fetch_auctions(cfg)
    pctl = sg.auction_demand_pctl(auctions)
    pctl = pctl[~pctl.index.duplicated(keep="last")].sort_index()
    m3_asof = pctl.index[-1] if len(pctl) else None       # M3 tazelik-imzası (boş→None→STALE)
    out["m3"] = ((pctl.reindex(index, method="ffill") - 0.5) * 40).clip(-20, 20)
    # m6 — Moody's Baa-Aaa spread (FRED DBAA/DAAA, ICE-kısıtsız tam tarih); genişleme = stres = negatif
    key = os.environ.get("FRED_API_KEY")

    def _fred_direct(sid):
        r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                         params={"series_id": sid, "api_key": key, "file_type": "json",
                                 "observation_start": "2005-01-01"}, timeout=30)
        o = r.json().get("observations", [])
        return pd.Series({pd.Timestamp(x["date"]): float(x["value"])
                          for x in o if x["value"] != "."}).sort_index()

    spr = (_fred_direct("DBAA") - _fred_direct("DAAA")).dropna()
    rm = spr.rolling(504, min_periods=126).mean()
    rs = spr.rolling(504, min_periods=126).std()
    z = (spr - rm) / rs
    out["m6"] = (-z * 8.0).clip(-20, 20).reindex(index, method="ffill")
    return out, m3_asof


M3_MAX_BD = 10   # auctions >=haftalık (bill'ler Pzt/Sal); 10 işgünü yeni-auction yoksa FiscalData feed donmuş/kırık


def _audit_m3_freshness(m3_asof, today: date) -> dict | None:
    """P1-C (denetim 2026-07-07): M3 auction (FiscalData) SPECS-dışı ayrı kaynak → ayrı tazelik-denetimi.
    Boş (API timeout/erişilemez) VEYA donuk (>M3_MAX_BD) → input_stale satırı (dict); temiz → None."""
    if m3_asof is None:
        return {"series": "FISCALDATA_AUCTIONS", "module": "m3",
                "error": "auction verisi BOŞ (API timeout/erişilemez) → m3 kör"}
    last = m3_asof.date() if hasattr(m3_asof, "date") else m3_asof
    age = _bd_age(last, today)
    if age > M3_MAX_BD:
        return {"series": "FISCALDATA_AUCTIONS", "module": "m3", "age_bd": int(age), "max_bd": M3_MAX_BD,
                "error": f"auction feed DONMUŞ (son {last} {int(age)}bd > {M3_MAX_BD}bd)"}
    return None


# ── Tazelik-denetimi çekirdeği (F1/F2/F4, denetim 2026-07-05) — SAF, test-edilebilir, ağ/IO yok ──────
# (series_id, modül, max_business_days) — günlük=4, haftalık=8/9 (WDTGAL F2: HAFTALIK H.4.1 Çarşamba
# seviyesi + Perşembe yayını; eski 'günlük' 5bd tavanı HER Perşembe 09:31 ET koşusunda yanlış-STALE
# atacaktı → kardeşi WALCL gibi 8). MTS satırları max_bd=None → RELEASE-AWARE kural (_check_spec).
SPECS = [
    ("WALCL", "m2", 8), ("WDTGAL", "m2", 8), ("RRPONTSYD", "m2", 4),
    ("VIXCLS", "m5", 4), ("DTWEXBGS", "m10", 9),
    ("MTSO133FMS", "m9", None), ("MTSR133FMS", "m9", None),
    ("DBAA", "m6", 4), ("DAAA", "m6", 4),
]
_MTS_SERIES = ("MTSO133FMS", "MTSR133FMS")
_MTS_GRACE_BD = 2               # beklenen yayın günü + 2 işgünü tolerans (yayın ~14:00 ET; sabah koşusu görmez)
_MTS_META_MAX_CAL_DAYS = 45     # FRED last_updated bundan eskiyse kaynak GERÇEKTEN donmuş → STALE (fail-closed)


@functools.lru_cache(maxsize=1)
def _fed_holidays_np() -> np.ndarray:
    """ABD federal tatilleri (pandas USFederalHolidayCalendar; FRED/Treasury yayın takvimi federal takvimi
    izler) — np.busday_* için datetime64[D]. Yeni bağımlılık YOK, deterministik 2000-2035."""
    from pandas.tseries.holiday import USFederalHolidayCalendar
    return USFederalHolidayCalendar().holidays(
        pd.Timestamp("2000-01-01"), pd.Timestamp("2035-12-31")).values.astype("datetime64[D]")


def _bd_age(last, today) -> int:
    """F4: TATİL-FARKINDALIKLI işgünü yaşı. kader-macro _freshness.business_days_between tatil-KÖR
    (docstring: 'Holidays are not modelled') → 2026-07-03 gibi federal tatiller işgünü sayılıp her tatil
    haftasında SPECS yaşlarını +1 şişiriyordu (MTS 07-04'te '46bd' diye ateşledi; VIXCLS 07-05'te 3bd okudu,
    gerçek 2). Paylaşılan kader-macro modülüne DOKUNMADAN yerel sayaç. Konvansiyon aynı: last dahil, today hariç."""
    if last is None or today is None or today <= last:
        return 0
    return int(np.busday_count(np.datetime64(last, "D"), np.datetime64(today, "D"),
                               holidays=_fed_holidays_np()))


def _mts_next_release(obs_date: date) -> date:
    """F1: MTS yayın takvimi. FRED, MTS gözlemini AY-BAŞINA damgalar (Mayıs 2026 print = obs 2026-05-01) ve
    TAKİP EDEN ayın ~8. işgününde yayımlar (Mayıs MTS → 2026-06-10, tam takviminde). Elimizdeki gözlemden
    SONRAKİ yayın (obs+1 ayının print'i) = obs+2 ayının 8. federal-işgünü (Haziran MTS → 2026-07-13;
    4-Tem-Cumartesi'nin 3-Tem-Cuma'ya kayan tatili dahil doğru)."""
    t = obs_date.month + 2
    y = obs_date.year + (t - 1) // 12
    m = (t - 1) % 12 + 1
    first = np.datetime64(f"{y:04d}-{m:02d}-01", "D")
    return np.busday_offset(first, 7, roll="forward", holidays=_fed_holidays_np()).astype(object)


def _check_spec(sid: str, mod: str, max_bd: int | None, last: date, today: date,
                last_updated=None) -> dict | None:
    """SAF karar çekirdeği: bir SPECS satırı bayat mı? (dict=bayat, None=temiz.)

    • MTS satırları (m9): RELEASE-AWARE — düz işgünü tavanı DEĞİL. Ay-başı damga + ertesi-ay-8.-işgünü
      yayın takvimi yüzünden bir sonraki yayının HEMEN öncesinde gözlem yaşı KAÇINILMAZ ~49-51 işgünüdür;
      eski düz 45bd tavanı HER AY ~5 işgünü yanlış-STALE atıyordu (ilk ateşleme 2026-07-04/05 → çağrı
      STALE + aylık sistematik forward-defter deliği). Yeni kural: bayat ⇔ bugün > beklenen-sonraki-yayın
      + 2 işgünü. FAIL-CLOSED KORUNUR: gerçek donma en geç yayın+2bd'de YÜKSEK SESLE yakalanır; ayrıca
      FRED meta last_updated dedektörü (>45 takvim günü = takvimde asla olmayan yaş) donmayı yayın
      beklemeden yakalar. GERÇEK donma ASLA sessiz geçmez.
    • diğer satırlar: tatil-farkındalıklı işgünü yaşı (_bd_age) > max_bd ⇒ bayat.
    """
    if sid in _MTS_SERIES:
        expected = _mts_next_release(last)
        deadline = np.busday_offset(np.datetime64(expected, "D"), _MTS_GRACE_BD,
                                    roll="forward", holidays=_fed_holidays_np()).astype(object)
        if today > deadline:
            return {"series": sid, "module": mod, "last_date": str(last),
                    "age_bd": int(_bd_age(last, today)), "expected_release": str(expected),
                    "grace_bd": _MTS_GRACE_BD,
                    "reason": f"beklenen MTS yayını {expected}+{_MTS_GRACE_BD}bd geçti (release-aware)"}
        if last_updated is not None:                        # donma-dedektörü (cache probe _meta.last_updated)
            try:
                lu = pd.Timestamp(str(last_updated)[:10]).date()
                meta_age = (today - lu).days
                if meta_age > _MTS_META_MAX_CAL_DAYS:
                    return {"series": sid, "module": mod, "last_date": str(last),
                            "age_bd": int(_bd_age(last, today)), "meta_last_updated": str(last_updated),
                            "meta_age_cal": int(meta_age),
                            "reason": (f"FRED last_updated {meta_age} takvim günü eski "
                                       f"(>{_MTS_META_MAX_CAL_DAYS}) → kaynak DONMUŞ")}
            except Exception:
                pass                                        # meta parse edilemedi → takvim kapısı zaten fail-closed
        return None
    age = _bd_age(last, today)
    if age > max_bd:
        return {"series": sid, "module": mod, "last_date": str(last),
                "age_bd": int(age), "max_bd": max_bd}
    return None


def _audit_input_freshness(cfg_km: dict) -> list:
    """Canlı tide'ın altındaki GERÇEK FRED girdilerinin son-tarihini cadence'ine göre denetle.
    (kader-macro sys.path penceresi İÇİNDE çağrılır — _fred oradan çözülür.)

    Backtest harness 'always fresh' varsayar; bu denetim onu deler: yüksek-ağırlıklı modüllerin
    (m9 .563 / m5 .214 / m2 .118 + m6/m10) FRED bacağı DONMUŞSA listeler → run.py call_status=STALE.
    F1/F2/F4 (denetim 2026-07-05): karar çekirdeği saf _check_spec'e taşındı — yaş sayacı federal-tatil
    farkındalıklı, MTS satırları release-aware + last_updated donma-dedektörlü, WDTGAL haftalık(8bd).
    Eşikler yayın takvimini modeller (normal gecikme YANLIŞ-bayat tetiklemez); gerçek donma YÜKSEK SESLE."""
    from modules import _fred
    today = datetime.now(timezone.utc).date()
    cache_dir = Path((cfg_km.get("fred", {}) or {}).get("cache_dir", "data/fred_cache"))
    stale: list = []
    for sid, mod, max_bd in SPECS:
        try:
            df = _fred.fetch_series(sid, cfg_km)
            if df is None or len(df) == 0:
                stale.append({"series": sid, "module": mod, "error": "empty/none"})
                continue
            last = df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1]
            last_updated = None
            if sid in _MTS_SERIES:                          # donma-dedektörü hammaddesi (_fred._save_cache yazar)
                try:
                    meta = _fred._load_cache_meta(_fred._cache_path(cache_dir, sid)) or {}
                    last_updated = meta.get("last_updated")
                except Exception:
                    last_updated = None                     # meta yoksa takvim kapısı tek başına fail-closed
            v = _check_spec(sid, mod, max_bd, last, today, last_updated=last_updated)
            if v:
                stale.append(v)
        except Exception as e:                              # fetch RAISE = girdi erişilemez → bayat say (fail-closed)
            stale.append({"series": sid, "module": mod, "error": f"{type(e).__name__}: {str(e)[:60]}"})
    return stale


def _evict_macro_modules(macro_repo: Path) -> None:
    """kader-macro'nun KENDİ kaynak paketlerini (modules / backtest / signals / integrity / validation /
    analysis / datafeed …) sys.modules'tan at → kader-equity'nin aynı-isimli paketleri sonraki import'ta
    TAZE çözülür. Neden gerekli: kader-macro import edilince sys.modules'a kendi 'modules' paketini yazar;
    temizlenmezse kader-equity'ninkini gölgeler → run.py'deki `from modules import cor1m_froth` patlar.

    KRİTİK: kader-macro venv'i macro_repo ALTINDA (…/kader-macro/.venv/Lib/site-packages) → numpy/pandas/
    pyarrow gibi KURULU paketleri ASLA atma (yoksa 'numpy import edilemiyor' → parquet/okuma patlar). Bu yüzden
    site-packages ve .venv alt-ağacı HARİÇ; yalnız doğrudan-repo kaynak paketleri atılır."""
    try:
        mr = os.path.normcase(str(macro_repo.resolve()))
    except Exception:
        mr = os.path.normcase(str(macro_repo))
    venv_marker = os.sep + ".venv" + os.sep
    for name in list(sys.modules):
        mod = sys.modules.get(name)
        if mod is None:
            continue
        locs = []
        f = getattr(mod, "__file__", None)
        if f:
            locs.append(f)
        pth = getattr(mod, "__path__", None)
        if pth:
            try:
                locs.extend(list(pth))
            except TypeError:
                pass
        for loc in locs:
            if not loc:
                continue
            nloc = os.path.normcase(str(loc))
            if nloc.startswith(mr) and "site-packages" not in nloc and venv_marker not in nloc:
                del sys.modules[name]                       # kader-macro KAYNAK paketi → at (kurulu paket DEĞİL)
                break


# kader-equity ile kader-macro AYNI isimli kaynak paketlere sahip → macro işine girmeden önce equity'ninkini at.
_MACRO_EQUITY_COLLISIONS = ("modules", "backtest", "signals", "integrity", "validation", "analysis", "datafeed")


def _evict_equity_shadows() -> None:
    """macro_repo sys.path[0]'dayken macro'nun paketleri TAZE çözülsün diye, equity-konumlu ÇAKIŞAN paketleri
    (modules/backtest/…) sys.modules'tan at. NEDEN gerekli (2026-07-07 kök-neden): run.py, reconstruct_live'dan
    ÖNCE equity'nin `modules`'ünü import ediyor (market_closed_reason → `from modules.opex_calendar import
    is_market_holiday`), böylece `modules` sys.modules'a girer. sys.path.insert TEK BAŞINA gölgeyi kaldırmaz —
    zaten-import edilmiş paket path'e BAKILMADAN döner → macro data_loader.py:26 `from modules import _fred`
    equity'nin (_fred'siz) modules'üne düşer → ImportError → canlı tide çöker, run.py STALE damgalar. Simetrik:
    çıkışta _evict_macro_modules macro'nunkileri atar → equity taze döner. SADECE equity-repo konumlu paketleri
    atar (site-packages/kurulu paketlere ve macro'nunkine DOKUNMAZ) → fail-safe."""
    eq_root = os.path.normcase(str(Path(__file__).resolve().parents[1]))
    for name in list(sys.modules):
        if name.split(".", 1)[0] not in _MACRO_EQUITY_COLLISIONS:
            continue
        mod = sys.modules.get(name)
        if mod is None:
            continue
        locs = []
        f = getattr(mod, "__file__", None)
        if f:
            locs.append(f)
        pth = getattr(mod, "__path__", None)
        if pth:
            try:
                locs.extend(list(pth))
            except TypeError:
                pass
        if any(os.path.normcase(str(loc)).startswith(eq_root) for loc in locs if loc):
            del sys.modules[name]                           # YALNIZ equity-repo konumlu → at (macro/kurulu DOKUNULMAZ)


def _build_live_panel(macro_repo: Path) -> pd.DataFrame:
    """PIT-replay'i bugüne uzatılmış kısa grid üzerinde koş → m0..m11 canlı panel (m2 RAW, m3/m6 enjekte).

    İZOLASYON (kalıcı sertleştirme): kader-macro import'u sys.path[0]'a macro_repo'yu sokar VE kader-equity
    ile çakışan paket isimlerini (modules/backtest/signals/integrity/validation) sys.modules'a cache'ler →
    sonradan kader-equity'nin kendi paketlerini gölgeler. Bu yüzden tüm macro işini try/finally ile İZOLE
    ediyoruz: çıkışta sys.path + SMART_M2 env eski haline döner ve macro_repo altından yüklenen her modül
    sys.modules'tan atılır. Böylece `engine.brief` SOĞUK başlatılsa bile (ön-import gerekmeden) çalışır."""
    saved_path = list(sys.path)
    saved_smart_m2 = os.environ.get("SMART_M2")
    sys.path.insert(0, str(macro_repo))
    os.environ.pop("SMART_M2", None)                       # RAW m2 zorla (smart-RRP DEĞİL)
    try:
        _evict_equity_shadows()                            # equity'nin çakışan paketlerini at → macro'nunki (modules._fred dahil) taze çözülür
        from dotenv import load_dotenv                      # noqa: E402
        import yaml                                         # noqa: E402
        from backtest.revalidation import oos_judge as J    # noqa: E402
        from backtest import data_loader as dl              # noqa: E402
        from backtest import signal_generator as sg         # noqa: E402
        from backtest.revalidation import pit_replay as P   # noqa: E402

        load_dotenv(J.ROOT / ".env")                        # FRED key
        cfg_km = yaml.safe_load((J.ROOT / "config.yaml").read_text(encoding="utf-8"))
        # Audit 2026-06-19 (#1 kök-neden): cache_min_seconds=1e9 cache_max'ı (günlük tazeleme tavanı) DEFETTİ
        # → equity FRED cache'i (VIXCLS/WDTGAL/RRP) DONUYORDU (~9g bayat veriyle 'current' tide). 30dk'ya indir:
        # tek-build-içi (~150s) tutarlılık KORUNUR (age<1800→cache), AMA günlük rebuild cache_max+probe ile TAZELER.
        cfg_km.setdefault("fred", {})["cache_min_seconds"] = 1800   # 30dk (eski 1e9 = asla-tazeleme bug'ı)

        today = datetime.now(timezone.utc).date()
        start = (today - timedelta(days=GRID_BUFFER_DAYS)).isoformat()
        dates = dl.business_day_grid(start, today.isoformat())
        df = P.run_pit_signals(cfg_km, dates)               # m0..m11 PIT-clean, son satır = bugün
        df = df.dropna(how="all")
        if df.empty:
            raise RuntimeError("run_pit_signals boş panel döndü (FRED/veri sorunu).")

        inj, m3_asof = _inject_m3_m6(cfg_km, dl, sg, df.index)   # m3/m6 enjeksiyon + M3 tazelik-imzası
        for m, s in inj.items():
            df[m] = s
        # Audit 2026-06-19 (Emir, KRİTİK): backtest harness 'always fresh' varsayar (freshness=1.0).
        # Canlı çağrıda bir FRED bacağı DONARSA (güncellemeyi durdurup eski tarihçeyi döndürür) tide
        # sessizce bayat son-değeri ffill'leyip 'current' damgalıyordu. Burada GERÇEK son-tarihleri
        # cadence'ine göre denetle → bayatsa fail-LOUD (run.py call_status=STALE + log.error).
        input_stale = _audit_input_freshness(cfg_km)
        # P1-C (denetim 2026-07-07): M3 auction SPECS-dışı → ayrıca denetle (boş/donuk → STALE, kör-nokta kapandı).
        m3_stale = _audit_m3_freshness(m3_asof, today)
        if m3_stale:
            input_stale.append(m3_stale)
        return df[MODS], input_stale
    finally:
        sys.path[:] = saved_path                            # macro_repo'yu yoldan çıkar (gölgeleme bitsin)
        if saved_smart_m2 is not None:                      # SMART_M2 env'ini olduğu gibi geri koy
            os.environ["SMART_M2"] = saved_smart_m2
        _evict_macro_modules(macro_repo)                    # macro paketlerini sys.modules'tan at → equity taze resolve


def _build_live_panel_subprocess(macro_repo: Path) -> tuple[pd.DataFrame, list]:
    """CANLI paneli AYRI SUBPROCESS'te üret — KALICI `modules`-gölge izolasyonu (gold/silver deseni).

    spine/build_live_panel.py, TEMİZ bir child interpreter'da _build_live_panel'i çağırır: child'ın sys.modules'ü
    boş başlar → macro_repo sys.path[0]'a sokulunca `from modules import _fred` DOĞRUDAN macro'ya çözülür →
    equity'nin kendi `modules`'ü asla gölgeleyemez (in-process yolun 46-gün STALE yapan kök-nedeni yapısal biter).
    Aynı _build_live_panel çağrıldığı için panel BYTE-AYNI (Sharpe değişmez). Fail-closed: child exit!=0 → raise →
    reconstruct_live'ın çağıranı (run.py) frozen-STALE'e düşer, ASLA sessiz-frozen 'current'."""
    import subprocess
    import tempfile
    script = Path(__file__).resolve().parent / "build_live_panel.py"
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "panel.json"
        # Denetim 07-11 P2 ([10]): timeout'suz subprocess run_daily'yi SURESIZ asabiliyordu
        # (satir yok, exit-code yok, sebep izsiz) -> 10dk tavan; asim = gorunur RuntimeError.
        r = subprocess.run([sys.executable, str(script), str(macro_repo), str(out)],
                           cwd=str(root), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=600,
                           env={"PYTHONIOENCODING": "utf-8", **os.environ})
        if r.returncode != 0 or not out.exists():
            tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
            raise RuntimeError(f"build_live_panel subprocess başarısız (exit {r.returncode}): "
                               f"{' | '.join(tail)[:220]}")
        payload = json.loads(out.read_text(encoding="utf-8"))
    df = pd.DataFrame(payload["data"], index=pd.to_datetime(payload["index"]), columns=payload["columns"])
    return df, payload.get("input_stale", [])


def _read_cache_today(max_age_h: float = 4.0) -> dict | None:
    if not LIVE_CACHE.exists():
        return None
    try:
        c = json.loads(LIVE_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return None
    if c.get("computed_date") == datetime.now(timezone.utc).date().isoformat():
        # Denetim 07-11 P2 ([14]): gun-cache 09:31 panelini TUM UTC-gunu pinliyordu — aksam
        # deira kosusu sabah panelini yiyordu. 4 saatten yasli ayni-gun cache = yeniden hesap
        # (mtime-tabanli; hesap zaten ~dk, gunde 2-3 tazeleme ucuz).
        try:
            age_h = (datetime.now(timezone.utc).timestamp() - LIVE_CACHE.stat().st_mtime) / 3600
            if age_h > max_age_h:
                return None
        except Exception:
            pass
        return c
    return None


def reconstruct_live(cfg: dict, force: bool = False) -> tuple[dict, dict, pd.Timestamp, str]:
    """Döndürür (scores_row{modül:skor}, vector{modül:w}, as_of, 'live'). Canlı kader-macro modül skorları.

    Gün-bazlı cache: aynı UTC günde ikinci çağrı replay'i atlar (force=True → yeniden hesapla)."""
    macro_repo = Path((cfg.get("macro", {}) or {}).get("repo_path", r"C:\Users\admin\Downloads\kader-macro"))
    if not macro_repo.exists():
        raise FileNotFoundError(f"kader-macro repo yok: {macro_repo} — canlı tide için gerekli "
                                f"(config.macro.repo_path ya da spine.source=frozen kullan).")

    vec_p = FROZEN / "vector.json"
    if not vec_p.exists():
        raise FileNotFoundError(f"donmuş vektör yok: {vec_p} (gen_snapshot.py ile üret).")
    vector = json.loads(vec_p.read_text(encoding="utf-8"))
    # EQ-D6-02 (denetim 2026-07-04): canlı yol kilit çapraz-doğrulaması — vector.json,
    # provenance['vector'] kopyasıyla birebir olmalı; sessiz vektör drift'i canlı çağrıya sızamaz.
    prov_p = FROZEN / "provenance.json"
    if prov_p.exists():
        pv = (json.loads(prov_p.read_text(encoding="utf-8")) or {}).get("vector")
        if pv is not None and {k: round(float(v), 9) for k, v in pv.items()} != \
                              {k: round(float(v), 9) for k, v in vector.items()}:
            raise RuntimeError("KİLİT: vector.json ≠ provenance['vector'] — donmuş tide vektörü drift etti. "
                               "Kasıtlıysa gen_snapshot + provenance'ı birlikte güncelle.")

    cached = None if force else _read_cache_today()
    if cached is not None:
        as_of = pd.Timestamp(cached["as_of"])
        # Gün-cache yalnız aynı UTC günde geçerli DEĞİL, as_of de tazelik penceresinde olmalı: FRED gecikmesi/
        # disruption'da aynı gün üretilmiş ama as_of'u eski cache'i servis etme → yeniden hesaplamayı dene.
        max_stale = int((cfg.get("spine", {}) or {}).get("max_staleness_days", 5))
        age = (datetime.now(timezone.utc).date() - as_of.date()).days
        if age <= max_stale:
            return cached["scores_row"], vector, as_of, "live", cached.get("input_stale", [])
        # cache as_of çok eski → düş, _build_live_panel ile tazele (yine eski gelirse run.py STALE damgalar)

    # KALICI (2026-07-07): panel'i AYRI SUBPROCESS'te üret → `modules`-gölge yapısal olarak imkansız (varsayılan).
    # Rollback gerekirse config spine.panel_isolation: inprocess (eski eviction'lı in-process yol). Byte-aynı.
    iso = str((cfg.get("spine", {}) or {}).get("panel_isolation", "subprocess")).lower()
    if iso == "inprocess":
        df, input_stale = _build_live_panel(macro_repo)
    else:
        df, input_stale = _build_live_panel_subprocess(macro_repo)
    as_of = df.index[-1]
    row = df.loc[as_of]
    scores_row = {m: (None if pd.isna(v) else float(v)) for m, v in row.items()}

    CACHE.mkdir(parents=True, exist_ok=True)
    LIVE_CACHE.write_text(json.dumps({
        "computed_date": datetime.now(timezone.utc).date().isoformat(),
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "as_of": str(as_of.date() if hasattr(as_of, "date") else as_of),
        "scores_row": scores_row,
        "input_stale": input_stale,                         # bayat-FRED-girdi listesi (boş=temiz); cache'te de korunur
        "panel_tail": {str(ix.date()): {m: (None if pd.isna(df.loc[ix, m]) else round(float(df.loc[ix, m]), 4))
                                        for m in MODS} for ix in df.index[-6:]},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return scores_row, vector, as_of, "live", input_stale
