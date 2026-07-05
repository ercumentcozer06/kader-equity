"""
modules/ipo_supply_waves — dev-IPO'nun GECİKMELİ arz dalgalarını izleyen BETİMSEL band.

GEREKÇE (2026-06-13, Emir-talebi): İlk-arz ($86B SpaceX) yalnızca BAŞLANGIÇ. Likidite-emişi
açısından asıl GECİKMELİ şoklar:
  (1) LOCK-UP BİTİŞİ — IPO sonrası kurucu+erken-yatırımcı hisseleri kilitli kalır; bitişte
      MASİF gecikmeli arz piyasaya iner. SpaceX prospektüsü (S-1/A 2026-06-03, EDGAR'dan
      doğrulandı): iki kademeli kilit — geniş havuz 180-gün, KURUCU (Musk) 366-gün. Founder +
      erken-yatırımcı kilidi ~7.8 MİLYAR hisse = "greater than 63% of our shares outstanding
      immediately prior to this offering" (prospektüs birebir alıntı). 366-gün bitişi = en uzun
      = bağlayıcı dev-overhang tarihi.
  (2) ENDEKS-DAHİL — $50B+ şirket IPO sonrası ~3-12 ayda S&P 500 / Nasdaq-100'e girer; pasif
      fonlar float'ın ~%15-20'sini almak ZORUNDA = FORCED talep (arzı KISMEN dengeler ama
      zamanlama şoku getirir).

DÜRÜST ETİKETLER (her çıktıda):
  • Tüm rakamlar TAHMİN / prospektüs-okuması; kesin değil.
  • lock-up = arz-YUKARI (overhang), endeks-dahil = talep-YUKARI (forced passive); NET etki
    zamanlama-bağımlı.
  • Bu band BETİMSELDİR — pozisyon/deploy etkisi SIFIR (OpEx/ipo_pipeline emsali bağlam-bandı).
    Anlık-arz POZİSYON kolu AYRI yaşar (modules/supply_demand_derisk mega-IPO kolu); bu modül
    yalnız GELECEK takvimi resmeder.

PIT / VERİ:
  • mega_ipo_hits.json recent_120d → her aktif dev-IPO hit'i (proposed_max_aggregate_usd ile).
  • Lock-up süresi: prospektüs gövdesinden (EX yok, ana S-1 .htm) '(\\d+)-day lock-up' regex'leri
    → MAX gün (kurucu kilidi geniş havuzdan uzundur). Gövde data/cache/raw/edgar/'da cache'lenir;
    yoksa ve ağ açıksa indirilir (graceful — indirilemezse varsayılan 180g + parse_status not).
  • Toplam-hisse / kilit-hisse / >%63: prospektüsten regex (best-effort; bulunamazsa None +
    dürüst not). SpaceX için doğrulanmış: ~7.8B / >63%.
  • Tarihler: lock_up_expiry = (IPO-tarihi VEYA dosyalama-tarihi) + lockup_days. IPO henüz
    fiyatlanmadıysa dosyalama-tarihi taban alınır (TAHMİN etiketi); fiyatlanınca gerçek tarih.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import re
from pathlib import Path

log = logging.getLogger("kader_equity.ipo_supply_waves")

ROOT = Path(__file__).resolve().parents[1]
MEGA_HITS_JSON = ROOT / "data" / "cache" / "mega_ipo_hits.json"
RAW_EDGAR = ROOT / "data" / "cache" / "raw" / "edgar"

DEFAULT_LOCKUP_DAYS = 180        # prospektüsten parse edilemezse standart varsayım
MEGA_USD_THR = 50e9              # endeks-dahil ZORUNLULUĞU eşiği ($50B+; supply_demand_derisk ile uyumlu)
WINDOW_DAYS = 120               # son window_days içindeki dev-IPO hit'leri aktif say (PIT)
PASSIVE_TAKE_LO = 0.15          # endeks-dahilde pasif fonların alacağı float oranı (alt)
PASSIVE_TAKE_HI = 0.20          # (üst) — kaba forced-talep bandı
# Endeks-dahil tipik pencere (IPO sonrası gün): S&P 500/Nasdaq-100 hızlı-yol ~3-12 ay
INDEX_INCL_LO_DAYS = 90
INDEX_INCL_HI_DAYS = 365

SEC_UA = {"User-Agent": "kader-research emirsancar2003@gmail.com",
          "Accept-Encoding": "gzip, deflate"}

HONEST_LABEL = ("TAHMIN (prospektus-okumasi, kesin degil); lock-up = arz-YUKARI overhang, "
                "endeks-dahil = talep-YUKARI forced-passive, NET zamanlama-bagimli; "
                "BETIMSEL band — pozisyon etkisi SIFIR")


# ─────────────────────────── prospektüs gövdesi: cache + (gerekirse) indir ───────────────────────────

def _accession_from_file(file_path: str) -> str | None:
    """edgar/data/.../0001628280-26-040364.txt → '0001628280-26-040364'."""
    m = re.search(r"(\d{10}-\d{2}-\d{6})", file_path or "")
    return m.group(1) if m else None


def _strip_html(html: str) -> str:
    t = re.sub(r"<[^>]+>", " ", html)
    t = (t.replace("&nbsp;", " ").replace("&#160;", " ").replace("&amp;", "&")
         .replace("&#8217;", "'").replace("&#8220;", '"').replace("&#8221;", '"')
         .replace("&#8212;", "-"))
    return re.sub(r"\s+", " ", t)


def _fetch_index_htm(cik: int, acc: str, *, allow_network: bool) -> str | None:
    """Dosyalama index sayfası (S-1 gövde linkini bulmak için). Cache'li fee-index varsa onu kullan."""
    acc_nd = acc.replace("-", "")
    cached = RAW_EDGAR / "fees" / f"{acc_nd}_index.htm"
    if cached.exists():
        return cached.read_text(encoding="utf-8", errors="replace")
    if not allow_network:
        return None
    try:
        import requests
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nd}/{acc}-index.htm"
        r = requests.get(url, headers=SEC_UA, timeout=60)
        return r.text if r.status_code == 200 else None
    except Exception as e:
        log.warning("ipo_waves: index indirilemedi: %s", e)
        return None


def _find_body_doc(index_html: str) -> str | None:
    """Index HTML'den ana S-1/F-1 gövde dokümanının dosya adını çıkar (Type sütunu S-1/S-1/A/F-1)."""
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", index_html, flags=re.S | re.I):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.S | re.I)
        if len(cells) < 4:
            continue
        types = " ".join(re.sub(r"<[^>]+>", " ", c) for c in cells)
        # Type sütununda S-1/F-1 (ama EX-... değil) ve .htm dokümanı olan satır = gövde
        if re.search(r"\b(S-1|S-1/A|F-1|F-1/A)\b", types) and "EX-" not in types.upper():
            m = re.search(r'href="([^"]+\.htm)"', row, flags=re.I)
            if m:
                return m.group(1).replace("/ix?doc=", "")
    return None


def load_prospectus_text(cik: int, acc: str, *, allow_network: bool = True) -> tuple[str | None, str]:
    """Ana prospektüs gövdesinin DÜZ METNİ. Cache (raw/edgar/_body_<acc>.htm) öncelikli; yoksa
    ve allow_network ise index'ten gövde-link bulup indirir (idempotent cache). -> (text|None, status)."""
    acc_nd = acc.replace("-", "")
    body_cache = RAW_EDGAR / f"_body_{acc_nd}.htm"
    if body_cache.exists() and body_cache.stat().st_size > 5000:
        return _strip_html(body_cache.read_text(encoding="utf-8", errors="replace")), "cache"
    idx = _fetch_index_htm(cik, acc, allow_network=allow_network)
    if idx is None:
        return None, "index-yok"
    href = _find_body_doc(idx)
    if href is None:
        return None, "govde-link-bulunamadi"
    if not allow_network:
        return None, "govde-cache-yok-ag-kapali"
    try:
        import requests
        url = href if href.startswith("http") else "https://www.sec.gov" + href
        r = requests.get(url, headers=SEC_UA, timeout=180)
        if r.status_code != 200 or len(r.content) < 5000:
            return None, f"govde-indirilemedi(http {r.status_code})"
        body_cache.parent.mkdir(parents=True, exist_ok=True)
        body_cache.write_bytes(r.content)
        return _strip_html(r.text), "indirildi"
    except Exception as e:
        log.warning("ipo_waves: govde indirilemedi: %s", e)
        return None, "govde-indirme-hatasi"


# ─────────────────────────── prospektüs parse (saf, test edilebilir) ───────────────────────────

def parse_lockup_days(text: str) -> tuple[int | None, list[int]]:
    """'(\\d+)-day lock-up' tüm eşleşmeleri → (MAX gün, bütün-günler-sıralı). MAX = bağlayıcı
    overhang (kurucu kilidi geniş havuzdan uzundur, SpaceX'te 366 vs 180). Bulunmazsa (None, [])."""
    days = sorted({int(m.group(1)) for m in re.finditer(r"(\d{2,4})[ -]day lock-?up", text, flags=re.I)})
    return (max(days) if days else None), days


def parse_lockup_shares(text: str) -> tuple[float | None, str | None]:
    """Kilit altındaki TOPLAM hisse tahmini (kurucu+erken-yatırımcı) + 'şu kadar %' notu.
    SpaceX birebir: 'represent approximately 7.8 billion shares ... or greater than 63% of our
    shares outstanding'. best-effort; bulunamazsa (None, None)."""
    shares = None
    pct_note = None
    # '... approximately N.N billion shares ... or (greater than)? P% of our shares outstanding'
    m = re.search(r"approximately\s+([\d.]+)\s*(billion|million)\s+shares[^.]{0,160}?"
                  r"(greater than\s*)?(\d{1,2})\s*%\s*of our shares outstanding", text, flags=re.I)
    if m:
        n = float(m.group(1)) * (1e9 if m.group(2).lower() == "billion" else 1e6)
        shares = n
        gt = "greater than " if m.group(3) else ""
        pct_note = f"{gt}{m.group(4)}% of shares outstanding (pre-offering)"
        return shares, pct_note
    # fallback: yalnız hisse adedi ('aggregate of N.N billion shares owned')
    m2 = re.search(r"aggregate of\s+([\d.]+)\s*(billion|million)\s+shares\s+owned", text, flags=re.I)
    if m2:
        shares = float(m2.group(1)) * (1e9 if m2.group(2).lower() == "billion" else 1e6)
        pct_note = "kilit havuzu (kurucu dahil)"
    return shares, pct_note


# ─────────────────────────── dalga hesabı (saf) ───────────────────────────

def compute_wave(hit: dict, *, today: _dt.date, prospectus_text: str | None = None,
                 default_lockup_days: int = DEFAULT_LOCKUP_DAYS) -> dict:
    """Tek dev-IPO hit'i için GELECEK arz dalgası. hit = mega_ipo_hits.json kaydı (recent_120d).
    prospectus_text verilmezse lock-up parse atlanır → varsayılan gün + parse_status='govde-yok'.
    Tüm sayılar TAHMİN; tarihler IPO-yoksa dosyalama-tabanlı."""
    company = (hit.get("watch_name") or hit.get("company") or "dev-IPO").strip()
    ceiling = hit.get("proposed_max_aggregate_usd")
    try:
        filed = _dt.date.fromisoformat(str(hit.get("date_filed", ""))[:10])
    except ValueError:
        filed = today
    # IPO tarihi: prospektüste fiyatlama yoksa dosyalama-tarihi taban (TAHMİN). hit'te varsa kullan.
    ipo_date = filed
    ipo_basis = "dosyalama-tarihi (IPO henuz fiyatlanmadi → TAHMIN tabani)"
    if hit.get("ipo_date"):
        try:
            ipo_date = _dt.date.fromisoformat(str(hit["ipo_date"])[:10])
            ipo_basis = "gerçek IPO tarihi"
        except ValueError:
            pass

    lockup_days, all_days = (None, [])
    lockup_status = "govde-yok (varsayilan)"
    lockup_shares, pct_note = None, None
    if prospectus_text:
        lockup_days, all_days = parse_lockup_days(prospectus_text)
        lockup_shares, pct_note = parse_lockup_shares(prospectus_text)
        lockup_status = ("parse-ok" if lockup_days else "ifade-bulunamadi (varsayilan)")
    eff_days = int(lockup_days or default_lockup_days)
    expiry = ipo_date + _dt.timedelta(days=eff_days)
    days_to = (expiry - today).days

    # endeks-dahil: $50B+ → S&P 500/Nasdaq-100 hızlı-yol pencere + forced pasif talep tahmini
    index_eligible = bool(ceiling and float(ceiling) >= MEGA_USD_THR)
    forced_lo = forced_hi = None
    index_window = None
    if index_eligible:
        # arz büyüklüğü ~ kayıt-tavanı (offering float proxy'si); pasif fonlar float'ın %15-20'sini alır
        forced_lo = float(ceiling) * PASSIVE_TAKE_LO
        forced_hi = float(ceiling) * PASSIVE_TAKE_HI
        win_lo = ipo_date + _dt.timedelta(days=INDEX_INCL_LO_DAYS)
        win_hi = ipo_date + _dt.timedelta(days=INDEX_INCL_HI_DAYS)
        index_window = {"earliest": str(win_lo), "latest": str(win_hi),
                        "days_to_earliest": (win_lo - today).days}

    return {
        "company": company,
        "ipo_date": str(ipo_date),
        "ipo_date_basis": ipo_basis,
        "offering_ceiling_usd": float(ceiling) if ceiling else None,
        # (1) LOCK-UP dalgası
        "lockup_days": eff_days,
        "lockup_days_parsed": lockup_days,             # None → varsayılan kullanıldı
        "lockup_days_all": all_days,                   # tüm bulunan kilit-süreleri (180+366 vb)
        "lockup_status": lockup_status,
        "lockup_expiry_date": str(expiry),
        "days_to_lockup": days_to,
        "lockup_shares_est": lockup_shares,            # ~7.8B (SpaceX)
        "lockup_pct_note": pct_note,                   # '>63% of shares outstanding'
        # (2) ENDEKS-DAHİL dalgası
        "index_incl_eligible": index_eligible,
        "index_incl_window": index_window,             # None → $50B altı
        "forced_passive_demand_usd_lo": forced_lo,     # float×%15
        "forced_passive_demand_usd_hi": forced_hi,     # float×%20
        "net_note": ("lock-up = arz-YUKARI (overhang), endeks-dahil = talep-YUKARI (forced) — "
                     "NET zamanlama-bagimli"),
        "label": HONEST_LABEL,
    }


# ─────────────────────────── canlı giriş ───────────────────────────

def evaluate(cfg: dict | None = None, *, today: _dt.date | None = None,
             allow_network: bool | None = None) -> dict | None:
    """mega_ipo_hits.json'daki son-120g dev-IPO hit'leri için GELECEK arz dalgaları.
    Aynı şirketin birden çok dosyalaması varsa kayıt-tavanı EN BÜYÜK olanı (en güncel arz)
    seçilir. Graceful — hit yoksa None. allow_network None → cfg'den (varsayılan True;
    lock-up gövdesi cache'te yoksa indirir). Pozisyon alanı ÜRETMEZ (betimsel sözleşme)."""
    cfg = cfg or {}
    today = today or _dt.date.today()
    if allow_network is None:
        allow_network = bool(cfg.get("allow_network", True))
    default_lockup = int(cfg.get("default_lockup_days", DEFAULT_LOCKUP_DAYS))
    if not MEGA_HITS_JSON.exists():
        return None
    try:
        hits = json.loads(MEGA_HITS_JSON.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("ipo_waves: mega-hits okunamadi: %s", e)
        return None

    # şirket-başına EN BÜYÜK kayıt-tavanlı hit (PIT: son window_days, gelecek dosyalama dışlanır)
    by_co: dict = {}
    for h in hits.get("recent_120d", []):
        amt = h.get("proposed_max_aggregate_usd")
        if amt is None:
            continue
        try:
            filed = _dt.date.fromisoformat(str(h.get("date_filed", ""))[:10])
        except ValueError:
            continue
        if not (filed <= today and (today - filed).days <= WINDOW_DAYS):
            continue
        key = h.get("cik") or h.get("company")
        cur = by_co.get(key)
        if cur is None or float(amt) > float(cur.get("proposed_max_aggregate_usd") or 0):
            by_co[key] = h

    waves = []
    for h in by_co.values():
        ptext, pstatus = (None, "atlandi")
        cik, acc = h.get("cik"), _accession_from_file(h.get("file", ""))
        if cik and acc:
            ptext, pstatus = load_prospectus_text(int(cik), acc, allow_network=allow_network)
        w = compute_wave(h, today=today, prospectus_text=ptext, default_lockup_days=default_lockup)
        w["prospectus_fetch_status"] = pstatus
        waves.append(w)

    if not waves:
        return None
    # en büyük arzdan başla (likidite-emişi sırası)
    waves.sort(key=lambda w: -(w.get("offering_ceiling_usd") or 0))
    return {
        "as_of": str(today),
        "n_active": len(waves),
        "waves": waves,
        "label": HONEST_LABEL,
    }
