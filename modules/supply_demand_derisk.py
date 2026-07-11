"""
modules/supply_demand_derisk — K2: KOŞULLU forward de-risk (arz-aşırı VE talep-zayıf AYNI ANDA).

GEREKÇE + DÜRÜST ETİKET (2026-06-13, FINDING 27; Emir-onaylı, candidate_conditional_derisk):
  Tek-arz yön-sinyali backtest'te FAIL'di (2020 H2 arz-yüksekti AMA +%8..%27 rali vardı → arz-tek
  trim raliyi keserdi). ÇÖZÜM (Emir'in formülü): arz-yüksekliğini TALEP-zayıflığıyla KOŞULLA.
  • 2020 SESSİZLİĞİ KANITLANDI (2 adversarial doğrulayıcı, ham-veriden): 2020 H2 tide +5.9..+11.7
    (düzey-kapısı +2 ÜSTÜ → SUSAR → +%27 rali KESİLMEZ); 2021 H2 tide +1.1..−0.5'e çöker → ATEŞLER.
  • Üç ön-kayıtlı kriter GEÇTİ: 2020-sessiz / 2021-ateşler / 2019+ zararsız (dSharpe +0.005/+0.003).
  DÜRÜST SINIR (verify-2): incremental etki ≈ SIFIR (hatta kümülatif hafif negatif SPX −0.13%/
  NDX −0.54%; Sharpe-kazancı kozmetik, lag-2'de işaret döner) — çünkü donmuş tide 2021-22'yi ZATEN
  kendi FLAT'ına geçerek de-risk ediyor; trim'e az maruziyet kalıyor. Bu bir ALFA DEĞİL, REJİM-DEĞİŞİM
  SİGORTASIDIR: $800B-capex-kayması / dev-IPO-şoku gibi arz GERÇEKLEŞİP tide HENÜZ dönmeden vurursa
  (backtest'in göremediği rejim — Constan'ın Q3-Q4 tezi) trim koruma sağlar; sigorta primi ≈ 0.
  DİSİPLİN: frozen position_target (tide×froth×shield) DEĞİŞMEZ; trim YALNIZ asset_deploy katmanında
  (OpEx emsali) → reproduce 1.64/1.77 byte-exact. trim-only, rebound-safe (faktör≤1).

Ön-kayıtlı sabitler (candidate ile birebir; oynanmadı): SUPPLY_Z_THR=1.0, DEMAND_WEAK_LEVEL=2.0,
DECLINE_LB=63, TRIM=0.85. Saf değerlendirme test edilebilir (tide_score_series + arz-z parametreli).

MEGA-IPO ANLIK ARZ KOLU (2026-06-13, Emir-talebi "5.5 ay kabul edilemez"):
  Çeyreklik net-arz z'si (z10y_nfc) PIT-yayınla ~5.5 ay gecikir → SpaceX gibi tek-dev-IPO ANLIK
  arz şokunu o kadar geç görür. ÇÖZÜM: arz-aşırı KOLUNU genişlet (OR), TALEP-ZAYIF kapısı SABİT kalsın:
    supply_hi = (z10y_nfc >= SUPPLY_Z_THR) OR (mega_ceiling_usd >= MEGA_CEILING_THR)
    fired     = supply_hi AND demand_weak           # demand_weak DEĞİŞMEZ → 2020-sessizlik OTOMATİK korunur
  MEGA_CEILING_THR=50e9 GEREKÇE: tarihsel rekor tek-arz ~$26B (Aramco 2019, Alibaba $25B 2014).
  $50B eşiği AÇIKÇA-REJİM-DIŞI tek-arz demektir (SpaceX 631.5M hisse × $135 = ~$86.25B ARZ; bu likidite
  emişinin doğru sayısıdır, ~$1.67T değerleme DEĞİL). Mega-IPO TEK BAŞINA ASLA tetiklemez: 2020-tipi
  güçlü-talepte (tide +8..+11) dev IPO gelse bile demand_weak FALSE → SUSAR (rali kesilmez); 2021-tipi
  (tide çöker) + dev arz → ATEŞLER. mega_ceiling=0 (hit yok) → eski z-only davranışı bire bir.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("kader_equity.sd_derisk")

ROOT = Path(__file__).resolve().parents[1]
NET_SUPPLY_PARQUET = ROOT / "data" / "cache" / "net_equity_supply.parquet"
MEGA_HITS_JSON = ROOT / "data" / "cache" / "mega_ipo_hits.json"
MEGA_ACTIVE_JSON = ROOT / "data" / "cache" / "mega_active_ipos.json"   # KALICI dev-IPO defteri

SUPPLY_Z_THR = 1.0          # arz-aşırı eşiği (z10y_nfc >= +1.0)
DEMAND_WEAK_LEVEL = 2.0     # talep-zayıf düzey-kapısı (tide_score <= +2.0)
DECLINE_LB = 63             # talep-düşüş geriye-bakış (işlem-günü)
TRIM = 0.85                 # nitelikli günde deploy çarpanı (trim-only)
MEGA_CEILING_THR = 50e9     # ANLIK dev-IPO arz eşiği ($50B; tarihsel rekor ~$26B Aramco → açıkça-rejim-dışı)
# ARM SÜRESİ (Emir 2026-06-13 "körleşmesin"): dev-IPO defterde dosyalamadan itibaren bu kadar gün
# ARMED kalır — bekleyen S-1 + fiyatlama + ilk-listeleme emişi süresini kapsar. 540g (~18ay) hiçbir
# gerçekçi bekleme-süresinin dışına düşmez. (424B/RW ile fiyatlama/geri-çekme tespiti eklenince bu
# 'listelenene kadar'a döner; o zamana kadar 540g cap. ESKİ 120g KALDIRILDI — kaynak+derisk körleşiyordu.)
MEGA_ARM_MAX_DAYS = 540

HONEST_LABEL = ("rejim-degisim SIGORTASI (alfa DEGIL); incremental ~0, donmus tide 2021-22'yi "
                "zaten de-risk ediyor; deger = backtest'in goremedigi arz-soku rejimi (Constan Q3-Q4)")


def evaluate_pure(supply_z_pit: pd.Series, tide_score: pd.Series, tide_dir: pd.Series,
                  *, as_of=None, mega_ceiling_usd: float = 0.0, mega_label: str = "",
                  mega_ceiling_thr: float = MEGA_CEILING_THR) -> dict:
    """Saf fire-değerlendirmesi (test edilebilir). Girdiler GÜNLÜK hizalı seriler (idx ortak).
    supply_z_pit = z10y_nfc PIT-ffill; tide_score/tide_dir = donmuş tide serileri.
    mega_ceiling_usd = KALICI defterdeki aktif dev-IPO MAX kayıt-tavanı (ANLIK arz kolu; 0 → z-only).
    mega_label = tetikleyen şirket adı (reason'a yazılır). as_of None → son ortak tarih.
    Döndürür: bugünkü fire-durumu + faktör + bileşenler.
    """
    idx = tide_score.index
    if as_of is None:
        as_of = idx[-1]
    as_of = pd.Timestamp(as_of)
    ts_at = float(tide_score.loc[:as_of].iloc[-1]) if len(tide_score.loc[:as_of]) else float("nan")
    dir_at = int(tide_dir.loc[:as_of].iloc[-1]) if len(tide_dir.loc[:as_of]) else 0
    z_at = float(supply_z_pit.loc[:as_of].iloc[-1]) if len(supply_z_pit.loc[:as_of]) else float("nan")
    # 63g-düşüş: as_of tide_score vs 63 işlem-günü öncesi
    prior = tide_score.loc[:as_of]
    declining = bool(len(prior) > DECLINE_LB and (prior.iloc[-1] - prior.iloc[-1 - DECLINE_LB]) < 0)

    mega_ceiling_usd = float(mega_ceiling_usd or 0.0)
    z_hi = (z_at == z_at) and z_at >= SUPPLY_Z_THR            # NaN-güvenli (çeyreklik z, ~5.5ay gecikme)
    mega_hi = mega_ceiling_usd >= float(mega_ceiling_thr)     # ANLIK dev-IPO arz kolu (gecikmesiz)
    supply_hi = bool(z_hi or mega_hi)                         # arz-aşırı = z-koldan VEYA mega-koldan
    demand_weak = (dir_at == 0) or (ts_at <= DEMAND_WEAK_LEVEL and declining)
    fired = bool(supply_hi and demand_weak)
    why = (("dir=0 (likidite FLAT)" if dir_at == 0
            else f"tide {ts_at:+.1f}<=+2 ve 63g-dususte") if demand_weak else
           f"talep guclu (tide {ts_at:+.1f})")
    # arz kolu etiketi (hangi koldan): mega öncelikli gösterilir (anlık+somut), yoksa z
    if mega_hi:
        _co = (mega_label or "dev-IPO").strip()
        supply_arm = f"mega-IPO {_co} ${mega_ceiling_usd/1e9:.0f}B"
    else:
        supply_arm = f"arz-z {z_at:+.1f}>=+1"
    return {
        "fired": fired,
        "trim_factor": TRIM if fired else 1.0,
        "supply_z": round(z_at, 2) if z_at == z_at else None,
        "supply_hi": bool(supply_hi),
        "supply_arm": ("mega-IPO" if mega_hi else ("z-arz" if z_hi else None)),
        "mega_ceiling_usd": mega_ceiling_usd if mega_ceiling_usd > 0 else None,
        "mega_label": mega_label or None,
        "mega_hi": bool(mega_hi),
        "tide_score": round(ts_at, 2) if ts_at == ts_at else None,
        "tide_dir": dir_at,
        "tide_declining_63d": declining,
        "demand_weak": bool(demand_weak),
        "as_of": str(as_of.date()),
        "reason": (f"ATEŞLE: {supply_arm} VE {why}" if fired else
                   f"sus: " + ("arz-z dusuk (mega yok)" if not supply_hi else why)),
        "label": HONEST_LABEL,
        "thresholds": {"supply_z": SUPPLY_Z_THR, "demand_weak_level": DEMAND_WEAK_LEVEL,
                       "decline_lb": DECLINE_LB, "trim": TRIM, "mega_ceiling_usd": float(mega_ceiling_thr)},
    }


def _supply_z_pit(idx: pd.DatetimeIndex) -> pd.Series:
    """net_equity_supply.parquet z10y_nfc → PIT-ffill günlük seri (pub-lag parquet'te)."""
    sup = pd.read_parquet(NET_SUPPLY_PARQUET).dropna(subset=["z10y_nfc"])
    z = pd.Series(sup["z10y_nfc"].values, index=pd.DatetimeIndex(sup["pit_date"])).sort_index()
    z = z[~z.index.duplicated(keep="last")]
    return z.reindex(z.index.union(idx)).ffill().reindex(idx)


def _refresh_mega_active_ledger() -> dict:
    """KALICI dev-IPO defteri (mega_active_ipos.json). mega_ipo_hits.json'ın recent_120d listesi
    SADECE son-120g tutar → SpaceX 120g sonra KAYNAKTAN düşer (Emir'in 'körleşme' yakaladığı kök).
    Çözüm: strict-tier (gerçek watchlist) dev-IPO'ları şirket-bazlı KALICI defterde biriktir; her
    şirket için en-erken dosyalama + EN-YÜKSEK tavan (placeholder→gerçek revizyonu: SpaceX $1B→$86B)
    izlenir. Defter 120g penceresiyle SİLİNMEZ — yalnız okuma anında MEGA_ARM_MAX_DAYS ile elenir.
    Idempotent upsert. Döner: güncel defter {company: {filed, max_ceiling, first_seen}}."""
    ledger: dict = {}
    if MEGA_ACTIVE_JSON.exists():
        try:
            ledger = (json.loads(MEGA_ACTIVE_JSON.read_text(encoding="utf-8")) or {}).get("entries", {})
        except Exception as e:
            log.warning("sd_derisk mega-defter okunamadi: %s", e)
            ledger = {}
    if not MEGA_HITS_JSON.exists():
        return ledger
    try:
        hits = json.loads(MEGA_HITS_JSON.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("sd_derisk mega-hits okunamadi: %s", e)
        return ledger
    today_s = _dt.date.today().isoformat()
    changed = False
    for h in hits.get("recent_120d", []):
        if (h.get("tier") or "").lower() != "strict":     # yalnız gerçek watchlist (loose=yanlış-pozitif)
            continue
        comp = (h.get("watch_name") or h.get("company") or "").strip()
        filed = str(h.get("date_filed", ""))[:10]
        if not comp or len(filed) != 10:
            continue
        amt = h.get("proposed_max_aggregate_usd")
        amt = float(amt) if amt is not None else 0.0
        # PIT-DÜRÜST: her (dosyalama-tarihi, tavan) AYRI saklanır — okuma anında yalnız as_of'a kadar
        # görünen dosyalamaların max'ı alınır. Tavanı en-erken tarihe bağlamak GELECEK-SIZINTISI olurdu
        # (SpaceX $86B 06-03'te bilindi; 05-20'de görünmemeli).
        e = ledger.setdefault(comp, {"filings": [], "first_seen": today_s})
        if "filings" not in e:                            # eski-şema göçü (varsa)
            e["filings"] = [{"date": e.get("filed", filed), "ceiling": float(e.get("max_ceiling", 0.0))}]
        if not any(f["date"] == filed and float(f["ceiling"]) == amt for f in e["filings"]):
            e["filings"].append({"date": filed, "ceiling": amt})
            changed = True
    if changed:
        try:
            MEGA_ACTIVE_JSON.write_text(
                json.dumps({"entries": ledger, "updated": today_s}, ensure_ascii=False, indent=1),
                encoding="utf-8")
        except OSError as e:
            log.warning("sd_derisk mega-defter yazilamadi: %s", e)
    return ledger


def _mega_ceiling_pit(as_of=None, *, max_days: int = MEGA_ARM_MAX_DAYS,
                      ledger: dict | None = None) -> tuple[float, str]:
    """ANLIK arz kolu girdisi: KALICI defterden MAX tavan + şirket adı. PIT: dosyalama as_of'tan
    önce VE as_of'tan en fazla max_days gün önce (bekleyen IPO armed kalır, körleşmez; truly-abandoned
    >540g filing elenir). Hit yoksa (0.0, ''). max_days=120 ESKİSİ KALDIRILDI."""
    if ledger is None:
        ledger = _refresh_mega_active_ledger()
    ref = _dt.date.today() if as_of is None else pd.Timestamp(as_of).date()
    best_amt, best_label = 0.0, ""
    for comp, e in (ledger or {}).items():
        # PIT: yalnız as_of'a kadar dosyalanmış (date <= ref) bacakların tavanları görünür
        visible = []
        for f in e.get("filings", []):
            try:
                fd = _dt.date.fromisoformat(str(f.get("date", ""))[:10])
            except (ValueError, TypeError):
                continue
            if fd <= ref:
                visible.append((fd, float(f.get("ceiling", 0.0) or 0.0)))
        if not visible:
            continue
        big_date, big_ceiling = max(visible, key=lambda x: x[1])   # PIT-max tavan + o dosyalamanın tarihi
        # ARM: süre BÜYÜK dosyalamanın tarihinden işler (bekleyen+fiyatlama+listeleme; körleşmez)
        if (ref - big_date).days <= max_days and big_ceiling > best_amt:
            best_amt, best_label = big_ceiling, comp
    return best_amt, best_label


def evaluate(cfg: dict | None = None, *, tide_score=None, tide_dir=None, as_of=None,
             live_tide_score=None, live_tide_dir=None) -> dict | None:
    """Canlı giriş: tide serileri verilirse onları kullan (run.py geçirir), yoksa donmuş spine'den
    kur. arz-z parquet'ten PIT-ffill; mega-IPO kayıt-tavanı mega_ipo_hits.json'dan PIT-okunur.
    Graceful — veri yoksa None (mega yoksa eski z-only davranış).

    Audit 2026-06-19 (#4): K2 talep-kapısı donmuş tide'ın SON değerini (28g bayat olabilir) 'güncel
    talep' diye kullanıyordu. run.py canlı tide'ı (live_tide_score/dir) geçerse: donmuş seri yalnız
    z-hizalama + 63g-tarihçe için kalır, GÜNCEL nokta CANLI tide'dan gelir. tide_source çıktıda görünür."""
    cfg = cfg or {}
    try:
        tide_source = "live_series" if tide_score is not None else "frozen"
        frozen_last = None
        if tide_score is None or tide_dir is None:
            from spine import contract as C, tide as T
            scores, prices, vector, prov = C.read_frozen()
            tide_score = T.tide_score_series(scores, vector)
            tide_dir = T.tide_dir_series(tide_score)
            frozen_last = tide_score.index[-1]
        # CANLI tide noktası enjekte (run.py canlı yolda geçer) → bayat-frozen-son-değer kapısını deler
        if live_tide_score is not None and live_tide_dir is not None:
            ts_idx = pd.Timestamp(as_of) if as_of is not None else (
                frozen_last + pd.Timedelta(days=1) if frozen_last is not None else tide_score.index[-1])
            tide_score = tide_score.copy(); tide_score.loc[ts_idx] = float(live_tide_score)
            tide_dir = tide_dir.copy(); tide_dir.loc[ts_idx] = int(live_tide_dir)
            tide_score = tide_score.sort_index(); tide_dir = tide_dir.sort_index()
            as_of = ts_idx
            tide_source = "live"
        if not NET_SUPPLY_PARQUET.exists():
            return None
        z = _supply_z_pit(tide_score.index)
        # mega-IPO ANLIK arz kolu: eşik config'ten (yoksa modül default 50e9), tavan hits.json'dan PIT
        thr = float(cfg.get("mega_ceiling_thr_usd", MEGA_CEILING_THR))
        mega_amt, mega_label = _mega_ceiling_pit(as_of)
        res = evaluate_pure(z, tide_score, tide_dir, as_of=as_of,
                            mega_ceiling_usd=mega_amt, mega_label=mega_label,
                            mega_ceiling_thr=thr)
        if res is not None:
            res["tide_source"] = tide_source     # live | live_series | frozen — talep-girdisinin kaynağı (görünür)
        return res
    except Exception as e:
        # Denetim 07-11 P3 ([28]): pozisyon-etkili trim'in sessizce devre-disi kalmasi log.warning'de
        # kayboluyordu -> ERROR + tip; None doner (fail-soft ama BAGIRARAK; overlay_block katmani
        # available=False'i zaten yakalar).
        log.error("sd_derisk evaluate FAILED (trim devre-disi bu kosuda): %s: %s", type(e).__name__, e)
        return None
