"""
screen/fetch_ipo_pipeline — SEC EDGAR S-1/F-1 BORU-HATTI GOZCUSU.
Constan tezi: 2026'da dev halka-arzlar (SpaceX/OpenAI/Anthropic sinifi) piyasayi bogacak.
S-1/F-1 dosyalamasi fiyatlamadan AYLAR once gelir -> ILERIYE-BAKAN oncu gosterge.

TASARIM / KAYNAK KARARLARI (2026-06-13):
  1) TARIHCE: EDGAR ceyreklik form-indeksleri
     https://www.sec.gov/Archives/edgar/full-index/{YYYY}/QTR{N}/form.idx
     (duz metin, bot-acik; SEC fair-access: aciklayici UA zorunlu + <10 istek/sn -> throttle).
     2001Q1'den bugune S-1, S-1/A, F-1, F-1/A TAM SAYIMI (EXACT form eslesmesi; S-1MEF /
     POS AM / S-11 vb DAHIL DEGIL). ILK-dosyalama (S-1, F-1) = oncu sinyal; /A degisiklik
     trafigi AYRI kolon (aktif-boru-hatti yogunlugu proxy'si ama cifte-sayim icermesin diye
     toplam'a girmez).
     YAPISAL KIRILMA (kosuda kesfedildi, veriyle dogrulandi): SEC 33-8876 SB-2 kucuk-sirket
     formunu 2008-02'de kaldirdi -> kucuk kayitcilar S-1'e gocttu (SB-2 2007'de ~250-300/c,
     2008Q2'de SIFIR; S-1 2008Q2 571'e sicradi). Ham S-1 bu yuzden 2008-09 cokusunu GOSTERMEZ.
     Cozum: sb2_new_n tani kolonu + total_new_adj_n/roll4q_adj/z10y_adj kirilma-duzeltilmis seri
     (pre/post-2008 karsilastirma YALNIZ adj seriyle). Headline kolonlar spec geregi S-1/F-1.
  2) CANLI UC: efts full-text-search YERINE guncel ceyregin form.idx'i secildi — ayni parser,
     SEC gece gunceller, bot-acik (efts JSON semasi degisken + ayri kota; form.idx daha saglam).
     Kural: ceyrek HENUZ BITMEMISSE (ya da bitis+5g icindeyse) cache >20 saat eskiyse yeniden
     indirilir; KAPANMIS ceyrekler kalici idempotent cache (data/cache/raw/edgar/).
  3) MEGA-IZLEME: data/manual/mega_ipo_watchlist.txt (satir-basina isim; yoksa varsayilanla
     olusturulur). Iki kademe eslesme:
       strict = kelime-siniri + noktalama-normalize (X.AI -> XAI varyanti dahil)
       loose  = tum-bosluk-sikistirilmis substring (FP riskli — OPEN AIR CINEMAS 'OPENAI'
                yakalar; yalniz raporda/json'da etiketli, sayima girmez)
     parquet mega_hits_n kolonu = strict + yalniz ILK-dosyalama (S-1/F-1, /A haric).
  4) $ BUYUKLUK (best-effort): son-120g strict eslesmelerde dosyalamanin EX-FILING FEES
     (exhibit 107, 2022+ zorunlu) eki cekilir; 'proposed maximum aggregate offering price'
     ifadesinden sonraki pencerede MAX sayi alinir. Placeholder ($ kucuk) / 'indeterminate'
     yaygin -> parse_status ile durust etiketlenir, parse edilemezse null.

PIT DISIPLINI: dosyalama tarihi = kamuya-gorunme ani (EDGAR aninda yayinlar, SIFIR gecikme).
  Ceyreklik satirda pit_date = ceyrek-sonu (sayim o gun kesinlesir); KISMI guncel ceyrek
  partial=1 + pit_date = veri-cekim gunu. Aylik seri ayni mantik (ay-sonu).

DURUSTLUK ETIKETI: S-1/F-1 dosyalamasi NIYET olcer, TAMAMLANAN IPO DEGIL (cogu dosyalama
  fiyatlanmaz / geri cekilir / SPAC'tir). Ritter tamamlanan-IPO sayimiyla yalniz YON
  karsilastirilir (Spearman, dosyalama->fiyatlama gecikme taramasiyla), birebir DEGIL.

CIKTI:
  data/cache/ipo_pipeline.parquet          ceyreklik panel (sayimlar + roll4q + z10y + mega + pit)
  data/cache/ipo_pipeline_monthly.parquet  aylik sayimlar
  data/cache/mega_ipo_hits.json            mega-izleme isabetleri (son-120g detay + tarihce)
  data/cache/ipo_pipeline_live.json        canli blok (son-90g sayim + z-karsiligi + mega ozet)
  data/cache/raw/edgar/                    ham form.idx + fee-eki landing'leri (idempotent)
  output/ipo_pipeline_report.txt           kaynak raporu + saglamalar + son ceyrekler + canli blok

KULLANIM:
  python -X utf8 screen/fetch_ipo_pipeline.py            # tam kosu
  python -X utf8 screen/fetch_ipo_pipeline.py --quick    # $-buyukluk parse'ini atla (hizli)
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "data" / "cache"
RAW_EDGAR = CACHE / "raw" / "edgar"
OUT = ROOT / "output"
WATCHLIST_TXT = ROOT / "data" / "manual" / "mega_ipo_watchlist.txt"

# SEC fair-access kurali: tanimlayici UA (kurum + iletisim) zorunlu, yoksa 403.
UA = {"User-Agent": "kader-research emirsancar2003@gmail.com",
      "Accept-Encoding": "gzip, deflate"}
THROTTLE_S = 0.15            # <10 istek/sn SEC limiti; nazik kalmak icin ~6-7/sn tavan
START_YEAR = 2001
# SB-2 = YAPISAL-KIRILMA TANI KOLONU (sayima/headline'a girmez, asagiya bak):
# SEC Release 33-8876 (yururluk 2008-02-04) SB-1/SB-2 kucuk-sirket formlarini kaldirdi ->
# kucuk kayitcilar 2008Q2'den itibaren S-1'e gocttu (veride dogrulandi: SB-2 2007'de ~250-300/c,
# 2008Q2'de SIFIR; S-1 2008Q2'de 571'e sicradi). Ham S-1 seviyesi pre/post-2008 KARSILASTIRILAMAZ;
# kirilma-duzeltilmis seri = s1+f1+sb2 (total_new_adj_n).
FORMS = ("S-1", "S-1/A", "F-1", "F-1/A", "SB-2", "SB-2/A")
HEADLINE_FORMS = ("S-1", "S-1/A", "F-1", "F-1/A")
MAX_FEE_PARSE = 15           # son-120g eslesmede en cok bu kadar dosyalamanin $'i denenir

DEFAULT_WATCHLIST = [
    "SpaceX", "Space Exploration Technologies", "OpenAI", "Anthropic", "Stripe",
    "Databricks", "xAI", "Anduril", "Epic Games", "Canva", "SHEIN", "Revolut",
]

LANDINGS: list[str] = []


def land(msg: str) -> None:
    print("  " + msg)
    LANDINGS.append(msg)


# ─────────────────────────── EDGAR http katmani (throttle + retry + cache) ───────────────────────────

_SESS = requests.Session()
_SESS.headers.update(UA)
_LAST_REQ = [0.0]


def edgar_get(url: str, timeout: int = 120, retries: int = 3) -> requests.Response | None:
    for i in range(retries):
        wait = THROTTLE_S - (time.time() - _LAST_REQ[0])
        if wait > 0:
            time.sleep(wait)
        try:
            r = _SESS.get(url, timeout=timeout)
            _LAST_REQ[0] = time.time()
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None                      # yok = yok; tekrar denemek anlamsiz
            if r.status_code in (403, 429, 503):
                time.sleep(3.0 * (i + 1))        # rate-limit/WAF -> geri cekil
        except requests.RequestException:
            _LAST_REQ[0] = time.time()
            if i < retries - 1:
                time.sleep(3.0 * (i + 1))
    return None


def edgar_cached(fname: str, url: str, min_bytes: int = 200, force: bool = False) -> Path | None:
    """Idempotent landing: raw/edgar/fname varsa dokunma; force=True ise tazele
    (tazeleme BASARISIZSA eski dosya korunur — canli uc icin kritik)."""
    p = RAW_EDGAR / fname
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.stat().st_size >= min_bytes and not force:
        return p
    r = edgar_get(url)
    if r is not None and len(r.content) >= min_bytes:
        p.write_bytes(r.content)
        return p
    return p if (p.exists() and p.stat().st_size >= min_bytes) else None


# ─────────────────────────── form.idx indirme + parse ───────────────────────────

def qstart_of(year: int, qn: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=3 * qn - 2, day=1)


def qend_of(year: int, qn: int) -> pd.Timestamp:
    return qstart_of(year, qn) + pd.offsets.QuarterEnd(0)


def qlab(q: pd.Timestamp) -> str:
    return f"{q.year}Q{(q.month - 1) // 3 + 1}"


def fetch_form_idx(year: int, qn: int, today: pd.Timestamp) -> Path | None:
    """Kapanmis ceyrek = kalici cache. Acik/yeni-kapanmis ceyrek (bitis+5g) = >20 saat
    eskiyse yeniden indir (SEC guncel ceyregin form.idx'ini gece-gece buyutur)."""
    cutoff = qend_of(year, qn) + pd.Timedelta(days=5)
    final = today > cutoff
    fname = f"form_{year}_QTR{qn}.idx"
    p = RAW_EDGAR / fname
    force = False
    if p.exists():
        if final:
            # Denetim 2026-06-13: kalıcı-cache ancak dosya çeyrek-bitişi+5g SONRASI indirildiyse
            # geçerli — yoksa çeyrek-ortası indirilen idx 'final' sanılıp son günlerin dosyalamaları
            # kalıcı eksik kalır (sessiz undercount).
            force = pd.Timestamp(p.stat().st_mtime, unit="s") <= cutoff
        else:
            force = (time.time() - p.stat().st_mtime) / 3600.0 > 20.0
    url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{qn}/form.idx"
    return edgar_cached(fname, url, min_bytes=2000, force=force)


def parse_form_idx(p: Path) -> pd.DataFrame:
    """form.idx parse. DIKKAT: baslik-satiri offset'leri VERI satirlariyla hizali DEGIL
    (2021Q1'de header 'Company Name'=12 ama veri 17'de basliyor) -> sabit-genislik kirik.
    Saglam yontem: 2+ bosluk ayraciyla bol + SAGDAN sabitle (file=son, date=sondan-2,
    cik=sondan-3; sirket adindaki ic cift-bosluklar company'yi bolse de sag taraf sabit).
    Yalniz FORMS exact-eslesme satirlari doner."""
    text = p.read_text(encoding="latin-1", errors="replace")
    lines = text.splitlines()
    hdr_i = None
    for i, ln in enumerate(lines[:30]):
        if ln.startswith("Form Type") and "Company Name" in ln and "Date Filed" in ln:
            hdr_i = i
            break
    if hdr_i is None:
        return pd.DataFrame()
    rows = []
    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for ln in lines[hdr_i + 1:]:
        if not ln.strip() or ln.startswith("---"):
            continue
        # hizli on-filtre: satir bizim formlardan biriyle baslamali (bosluk = form sonu)
        head = ln.split(None, 1)[0] if ln.strip() else ""
        if head not in FORMS:
            continue
        toks = re.split(r"\s{2,}", ln.strip())
        if len(toks) < 5 or toks[0] != head or not date_re.match(toks[-2]):
            continue
        rows.append({
            "form": head,
            "company": " ".join(toks[1:-3]).strip(),
            "cik": pd.to_numeric(toks[-3], errors="coerce"),
            "date": toks[-2],
            "file": toks[-1],
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).reset_index(drop=True)


def collect_filings(today: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    """2001Q1 -> guncel ceyrek tum S-1/F-1 satirlari + indirilemeyen ceyrek listesi."""
    parts, failed = [], []
    per_year: dict[int, int] = {}
    for year in range(START_YEAR, today.year + 1):
        for qn in range(1, 5):
            if qstart_of(year, qn) > today:
                break
            p = fetch_form_idx(year, qn, today)
            if p is None:
                failed.append(f"{year}QTR{qn}")
                continue
            df = parse_form_idx(p)
            if len(df):
                parts.append(df)
                per_year[year] = per_year.get(year, 0) + len(df)
        if year in per_year and year % 5 == 0:
            print(f"    .. {year} tamam (yil-ici S-1/F-1 satiri: {per_year[year]})")
    allf = pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)
    land(f"[edgar] OK ceyreklik form.idx 2001Q1 -> {qlab(pd.Timestamp(today))} kapsami; "
         f"S-1/F-1 toplam satir {len(allf)}; indirilemeyen ceyrek: "
         f"{', '.join(failed) if failed else 'YOK'}; UA+throttle {THROTTLE_S}s (SEC fair-access)")
    return allf, failed


# ─────────────────────────── mega-izleme listesi ───────────────────────────

def _norm_spaced(s: str) -> str:
    """noktalama -> bosluk, cok-bosluk tek, upper. 'X.AI Corp' -> 'X AI CORP'"""
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", s.upper())).strip()


def _norm_glued(s: str) -> str:
    """noktalama SILINIR (bosluk korunur). 'X.AI Corp' -> 'XAI CORP'"""
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]+", "", s.upper())).strip()


def _norm_squash(s: str) -> str:
    """her sey silinir. 'Open AI Inc' -> 'OPENAIINC' (loose kademe, FP riskli)"""
    return re.sub(r"[^A-Z0-9]+", "", s.upper())


def load_watchlist() -> list[str]:
    if not WATCHLIST_TXT.exists():
        WATCHLIST_TXT.parent.mkdir(parents=True, exist_ok=True)
        WATCHLIST_TXT.write_text(
            "# mega-IPO izleme listesi — satir basina bir isim (# = yorum)\n"
            + "\n".join(DEFAULT_WATCHLIST) + "\n", encoding="utf-8")
        land(f"[watch] mega_ipo_watchlist.txt YOKTU -> varsayilan {len(DEFAULT_WATCHLIST)} "
             f"isimle olusturuldu ({WATCHLIST_TXT})")
    names = [ln.strip() for ln in WATCHLIST_TXT.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    return names


def build_matchers(names: list[str]):
    """isim -> (strict regex'ler [spaced+glued formda], squash str). strict = kelime-siniri."""
    out = []
    for nm in names:
        pats = []
        for v in {_norm_spaced(nm), _norm_glued(nm)}:
            if v:
                pats.append(re.compile(r"(?<![A-Z0-9])" + re.escape(v) + r"(?![A-Z0-9])"))
        out.append((nm, pats, _norm_squash(nm)))
    return out


def match_company(company: str, matchers) -> tuple[str | None, str | None]:
    """-> (eslesen izleme adi, kademe 'strict'|'loose') ya da (None, None)."""
    sp, gl, sq = _norm_spaced(company), _norm_glued(company), _norm_squash(company)
    for nm, pats, _ in matchers:
        for pat in pats:
            if pat.search(sp) or pat.search(gl):
                return nm, "strict"
    for nm, _, nsq in matchers:
        if len(nsq) >= 5 and nsq in sq:          # kisa adlar (XAI) loose'ta FP bombasi -> atla
            return nm, "loose"
    return None, None


# ─────────────────────────── $-buyukluk: EX-FILING FEES parse (best-effort) ───────────────────────────

FEE_PHRASES = ["proposed maximum aggregate offering price", "maximum aggregate offering price"]


def _strip_html(html: str) -> str:
    t = re.sub(r"<[^>]+>", " ", html)
    t = t.replace("&nbsp;", " ").replace("&#160;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", t)


def fee_exhibit_usd(cik: int, acc: str) -> tuple[float | None, str, str | None]:
    """Dosyalamanin index sayfasindan EX-FILING FEES ekini bul, 'proposed maximum aggregate
    offering price' sonrasi pencereden MAX sayiyi al. -> (usd|None, parse_status, exhibit_url).
    DURUSTLUK: placeholder tutarlar ($10-100M roundlot) ve 'indeterminate' yaygindir;
    deal-buyuklugu DEGIL kayit-tavani okunur."""
    acc_nd = acc.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nd}"
    pidx = edgar_cached(f"fees/{acc_nd}_index.htm", f"{base}/{acc}-index.htm", min_bytes=500)
    if pidx is None:
        return None, "index-erisilemedi", None
    html = pidx.read_text(encoding="utf-8", errors="replace")
    href = None
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S | re.I):
        if "EX-FILING FEES" in row.upper():
            m = re.search(r'href="([^"]+)"', row)
            if m:
                href = m.group(1)
                break
    if href is None:
        return None, "fee-eki-yok(107 oncesi ya da eksiz)", None
    href = href.replace("/ix?doc=", "")          # inline-XBRL viewer sarmasini soy
    url = href if href.startswith("http") else "https://www.sec.gov" + href
    fname = "fees/" + acc_nd + "_" + url.rsplit("/", 1)[-1].split("?")[0]
    pdoc = edgar_cached(fname, url, min_bytes=200)
    if pdoc is None:
        return None, "fee-eki-indirilemedi", url
    text = _strip_html(pdoc.read_text(encoding="utf-8", errors="replace"))
    low = text.lower()
    pos = -1
    for ph in FEE_PHRASES:
        pos = low.find(ph)
        if pos >= 0:
            break
    if pos < 0:
        return None, "ifade-bulunamadi", url
    win = text[pos:pos + 900]
    if "indeterminate" in win.lower():
        return None, "indeterminate(kayit-tavani belirsiz)", url
    nums = [float(x.replace(",", ""))
            for x in re.findall(r"\$?\s*([0-9][\d,]{4,}(?:\.\d{1,2})?)", win)]
    nums = [n for n in nums if n >= 1e5]         # kucuk sayilar = fee orani/adet gurultusu
    if not nums:
        return None, "ifade-var-sayi-yok", url
    return max(nums), "parse-ok(kayit-tavani, deal-buyuklugu degil)", url


# ─────────────────────────── panel insasi ───────────────────────────

COUNT_COLS = ["s1_new_n", "s1_amend_n", "f1_new_n", "f1_amend_n"]


def _agg_counts(f: pd.DataFrame, keycol: str) -> pd.DataFrame:
    g = f.groupby([keycol, "form"]).size().unstack("form").fillna(0.0)
    out = pd.DataFrame(index=g.index)
    out["s1_new_n"] = g.get("S-1", 0.0)
    out["s1_amend_n"] = g.get("S-1/A", 0.0)
    out["f1_new_n"] = g.get("F-1", 0.0)
    out["f1_amend_n"] = g.get("F-1/A", 0.0)
    out["total_new_n"] = out["s1_new_n"] + out["f1_new_n"]
    out["total_all_n"] = out[COUNT_COLS].sum(axis=1)
    # yapisal-kirilma tanisi: SB-2 (2008-02'de kalkti, kucuk kayitcilar S-1'e gocttu)
    out["sb2_new_n"] = g.get("SB-2", 0.0)
    out["sb2_amend_n"] = g.get("SB-2/A", 0.0)
    out["total_new_adj_n"] = out["total_new_n"] + out["sb2_new_n"]
    return out


def build_panels(filings: pd.DataFrame, failed_q: list[str], matchers,
                 today: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    f = filings.copy()
    f["qdate"] = pd.to_datetime(f["date"]).dt.to_period("Q").dt.start_time
    f["mdate"] = pd.to_datetime(f["date"]).dt.to_period("M").dt.start_time

    # mega eslesme (tum tarihce; strict sayilir, loose yalniz json/rapor; SB-2 tani-kolonu haric)
    hits_hist: list[dict] = []
    f["mega_strict"] = ""
    uniq = f.loc[f["form"].isin(HEADLINE_FORMS), "company"].unique()
    match_map = {c: match_company(c, matchers) for c in uniq}
    for i, r in f.iterrows():
        if r["form"] not in HEADLINE_FORMS:
            continue
        nm, tier = match_map[r["company"]]
        if nm is None:
            continue
        if tier == "strict":
            f.at[i, "mega_strict"] = nm
        hits_hist.append({"company": r["company"], "watch_name": nm, "tier": tier,
                          "form": r["form"], "date_filed": str(r["date"].date()),
                          "cik": int(r["cik"]) if np.isfinite(r["cik"]) else None,
                          "file": r["file"]})

    # ceyreklik
    cur_q = pd.Timestamp(today).to_period("Q").start_time
    qidx = pd.date_range(qstart_of(START_YEAR, 1), cur_q, freq="QS")
    panel = pd.DataFrame(index=qidx)
    panel.index.name = "qdate"
    panel = panel.join(_agg_counts(f, "qdate"), how="left").fillna(0.0)
    # indirilemeyen ceyrek = NaN (sahte-0 degil)
    for fq in failed_q:
        y, qn = int(fq[:4]), int(fq[-1])
        qd = qstart_of(y, qn)
        if qd in panel.index:
            panel.loc[qd, :] = np.nan
    mega_q = (f[(f["mega_strict"] != "") & (~f["form"].str.endswith("/A"))]
              .groupby("qdate").size())
    panel["mega_hits_n"] = mega_q.reindex(panel.index).fillna(0.0)

    panel["roll4q"] = panel["total_new_n"].rolling(4, min_periods=4).sum()
    s = panel["roll4q"]
    panel["z10y"] = (s - s.rolling(40, min_periods=20).mean()) / s.rolling(40, min_periods=20).std()
    # kirilma-duzeltilmis seri (SB-2 dahil; pre/post-2008 karsilastirilabilir tek seri)
    panel["roll4q_adj"] = panel["total_new_adj_n"].rolling(4, min_periods=4).sum()
    sa = panel["roll4q_adj"]
    panel["z10y_adj"] = (sa - sa.rolling(40, min_periods=20).mean()) / sa.rolling(40, min_periods=20).std()

    panel["partial"] = 0.0
    panel.loc[panel.index == cur_q, "partial"] = 1.0
    # kismi ceyrekte roll4q/z10y SISTEMATIK asagi-yanli (eksik toplam) -> NaN; canli sinyal
    # ipo_pipeline_live.json'daki z90 (son-90g pencere, yanli degil)
    panel.loc[panel["partial"] == 1.0, ["roll4q", "z10y", "roll4q_adj", "z10y_adj"]] = np.nan
    qe = pd.Series([q + pd.offsets.QuarterEnd(0) for q in panel.index], index=panel.index)
    panel["pit_date"] = qe                       # sayim ceyrek-sonunda kesin (dosyalama=aninda kamu)
    panel.loc[panel["partial"] == 1.0, "pit_date"] = today

    # aylik
    mon = _agg_counts(f, "mdate")
    mon.index.name = "mdate"
    me = mon.index + pd.offsets.MonthEnd(0)
    mon["pit_date"] = me
    mon["partial"] = 0.0
    cur_m = pd.Timestamp(today).to_period("M").start_time
    if cur_m in mon.index:
        mon.loc[cur_m, "partial"] = 1.0
        mon.loc[cur_m, "pit_date"] = today
    return panel, mon, hits_hist


# ─────────────────────────── canli blok + saglamalar ───────────────────────────

def live_block(filings: pd.DataFrame, panel: pd.DataFrame, hits_hist: list[dict],
               today: pd.Timestamp, quick: bool) -> dict:
    w90 = filings[filings["date"] >= today - pd.Timedelta(days=90)]
    w120 = filings[filings["date"] >= today - pd.Timedelta(days=120)]
    n90 = {"s1_new": int((w90["form"] == "S-1").sum()),
           "s1_amend": int((w90["form"] == "S-1/A").sum()),
           "f1_new": int((w90["form"] == "F-1").sum()),
           "f1_amend": int((w90["form"] == "F-1/A").sum())}
    n90["total_new"] = n90["s1_new"] + n90["f1_new"]

    # z-karsiligi: son-90g yeni-toplam, KAPANMIS son 40 ceyregin dagilimina vurulur (ceyrek~91g)
    qq = panel.loc[panel["partial"] == 0.0, "total_new_n"].dropna().tail(40)
    z90 = float((n90["total_new"] - qq.mean()) / qq.std()) if len(qq) >= 20 and qq.std() > 0 else np.nan

    # mega isabetler (son-120g) + $-buyukluk denemesi
    recent_hits = []
    cutoff = (today - pd.Timedelta(days=120)).date()
    for h in hits_hist:
        if pd.Timestamp(h["date_filed"]).date() >= cutoff:
            recent_hits.append(dict(h))
    parsed = 0
    for h in recent_hits:
        h["proposed_max_aggregate_usd"] = None
        h["fee_parse_status"] = "atlandi(--quick)" if quick else "denenmedi"
        h["fee_exhibit_url"] = None
        if quick or h["tier"] != "strict" or h["cik"] is None or parsed >= MAX_FEE_PARSE:
            continue
        m = re.search(r"(\d{10}-\d{2}-\d{6})", h["file"])
        if not m:
            h["fee_parse_status"] = "accession-cikartilamadi"
            continue
        usd, status, url = fee_exhibit_usd(h["cik"], m.group(1))
        h["proposed_max_aggregate_usd"] = usd
        h["fee_parse_status"] = status
        h["fee_exhibit_url"] = url
        parsed += 1
    return {"as_of": str(today.date()), "window_90d_counts": n90, "z90_vs_40q": z90,
            "mega_recent_120d": recent_hits,
            "honest_label": "S-1/F-1 dosyalamasi NIYET olcer, tamamlanan IPO degil; "
                            "$ = kayit-tavani (deal-buyuklugu degil), best-effort"}


def ritter_crosscheck(panel: pd.DataFrame) -> list[str]:
    """YON kontrolu: roll4q(S-1+F-1 yeni) vs Ritter tamamlanan-IPO 4Q adedi (varsa).
    Dosyalama fiyatlamayi ONCULER -> pozitif Spearman, en guclusu 0-2 ceyrek gecikmede beklenir."""
    lines = []
    sc = CACHE / "supply_components.parquet"
    if not sc.exists():
        return ["    supply_components.parquet yok -> Ritter capraz-kontrolu atlandi"]
    sp = pd.read_parquet(sc)
    col = "ritter_ipo_n_gross_4q"
    if col not in sp.columns:
        return [f"    {col} kolonu yok -> atlandi"]
    rit = sp[col].dropna()
    base = panel["roll4q"].dropna()
    for lag in (0, 1, 2, 3, 4):
        a = base.shift(lag)                      # ritter[t] vs dosyalama[t-lag]
        j = pd.concat([a, rit], axis=1, join="inner").dropna()
        if len(j) < 20:
            continue
        rho = float(j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman"))
        lines.append(f"    lag={lag}c  spearman {rho:+.3f}  (n={len(j)})")
    lines.append("    NOT: dosyalama!=tamamlanan-IPO (niyet vs gerceklesme); yalniz YON uyumu aranir")
    return lines


def sanity_checks(panel: pd.DataFrame) -> list[str]:
    L = []
    p = panel.dropna(subset=["total_new_n"])
    y2021 = p.loc["2021-01-01":"2021-12-31", "s1_new_n"]
    med10 = p.loc["2010-01-01":"2019-12-31", "s1_new_n"].median()
    L.append(f"    2021 SPAC-cagi S-1 patlamasi: ceyrek-basina {y2021.min():.0f}-{y2021.max():.0f} "
             f"(2010'lar medyani {med10:.0f}) -> {'GORUNUYOR' if y2021.max() > 2 * med10 else 'GORUNMUYOR!'}")
    # YAPISAL KIRILMA (veride dogrulandi): SEC 33-8876 ile SB-2 2008-02'de kalkti; kucuk
    # kayitcilar S-1'e gocttu (SB-2 2007 ~250-300/c -> 2008Q2 SIFIR; S-1 2008Q2 571'e sicradi).
    # HAM S-1 serisi 2008-09 cokusunu BU YUZDEN gostermez; durust test = duzeltilmis seri + F-1.
    y0809a = p.loc["2008-07-01":"2009-06-30", "total_new_adj_n"]
    prea = p.loc["2006-01-01":"2007-12-31", "total_new_adj_n"].mean()
    L.append(f"    2008-09 cokusu (KIRILMA-DUZELTILMIS s1+f1+sb2): 2008H2-2009H1 ceyrek-ort "
             f"{y0809a.mean():.0f} vs 2006-07 ort {prea:.0f} "
             f"-> {'GORUNUYOR' if y0809a.mean() < 0.75 * prea else 'GORUNMUYOR!'}")
    f1c = p.loc["2008-10-01":"2009-06-30", "f1_new_n"].mean()
    f1pre = p.loc["2006-01-01":"2007-12-31", "f1_new_n"].mean()
    L.append(f"    2008-09 cokusu (F-1, goc-kirlenmesi YOK): 2008Q4-2009H1 ort {f1c:.1f}/c vs "
             f"2006-07 ort {f1pre:.1f}/c -> {'GORUNUYOR' if f1c < 0.5 * f1pre else 'GORUNMUYOR!'}")
    L.append("    NOT: ham S-1 2008'de DUSMEZ (SB-2 gocu maskeler) — pre/post-2008 seviye "
             "karsilastirmasi icin total_new_adj_n / z10y_adj kullan")
    zmax = panel["z10y"].dropna()
    if len(zmax):
        L.append(f"    z10y tepe: {zmax.max():+.2f} ({qlab(zmax.idxmax())}), dip: {zmax.min():+.2f} "
                 f"({qlab(zmax.idxmin())})")
    return L


# ─────────────────────────── rapor ───────────────────────────

def write_report(panel: pd.DataFrame, mon: pd.DataFrame, live: dict,
                 hits_hist: list[dict], watch: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    L: list[str] = []
    L.append("=" * 112)
    L.append("  SEC EDGAR S-1/F-1 BORU-HATTI GOZCUSU — Constan dev-arz tezi ileriye-bakan dedektoru")
    L.append("  DURUSTLUK: dosyalama = NIYET (cogu fiyatlanmaz/geri cekilir); tamamlanan IPO DEGIL.")
    L.append("=" * 112)
    L.append("")
    L.append("  KAYNAK LANDING RAPORU:")
    for ln in LANDINGS:
        L.append("  " + ln)
    L.append("")
    L.append("  SAGLAMALAR:")
    L.extend(sanity_checks(panel))
    L.append("")
    L.append("  RITTER CAPRAZ-KONTROL (tamamlanan-IPO 4Q adedi vs dosyalama roll4q; YON uyumu):")
    L.extend(ritter_crosscheck(panel))
    L.append("")
    L.append("  SON 12 CEYREK:")
    show = ["s1_new_n", "s1_amend_n", "f1_new_n", "f1_amend_n", "total_new_n", "roll4q", "z10y",
            "mega_hits_n"]
    L.append(f"  {'ceyrek':<8}" + "".join(f"{c:>13}" for c in show) + f"{'partial':>9}")
    for q, row in panel.tail(12).iterrows():
        cells = "".join(f"{row[c]:>13.2f}" if np.isfinite(row[c]) else f"{'-':>13}" for c in show)
        L.append(f"  {qlab(q):<8}" + cells + f"{row['partial']:>9.0f}")
    L.append("")
    L.append("  CANLI BLOK (son-90g, PIT=dosyalama-tarihi sifir-gecikme):")
    n = live["window_90d_counts"]
    L.append(f"    S-1 yeni {n['s1_new']} | S-1/A {n['s1_amend']} | F-1 yeni {n['f1_new']} | "
             f"F-1/A {n['f1_amend']} | YENI-TOPLAM {n['total_new']}")
    z = live["z90_vs_40q"]
    L.append(f"    z-karsiligi (son-40-kapanmis-ceyrek dagilimina gore): "
             f"{z:+.2f}" if np.isfinite(z) else "    z-karsiligi: hesaplanamadi")
    L.append("")
    L.append(f"  MEGA-IZLEME ({len(watch)} isim: {', '.join(watch)}):")
    rec = live["mega_recent_120d"]
    if rec:
        for h in rec:
            usd = h.get("proposed_max_aggregate_usd")
            usd_s = f"${usd:,.0f}" if usd else "null"
            L.append(f"    [{h['tier']}] {h['date_filed']} {h['form']:<6} {h['company'][:45]:<45} "
                     f"-> {h['watch_name']} | kayit-tavani {usd_s} ({h.get('fee_parse_status')})")
    else:
        L.append("    son-120g eslesme YOK (SpaceX/OpenAI/Anthropic sinifi henuz dosyalamadi)")
    hist_strict = [h for h in hits_hist if h["tier"] == "strict"]
    L.append(f"    tarihce: 2001'den beri strict-eslesme {len(hist_strict)} dosyalama "
             f"(loose {len(hits_hist) - len(hist_strict)}; detay mega_ipo_hits.json)")
    L.append("")
    L.append("  PIT kurallari: ceyreklik pit_date = ceyrek-sonu (dosyalama EDGAR'da ANINDA kamu;")
    L.append("  sayim ceyrek kapaninca kesin); kismi guncel ceyrek partial=1 + pit=cekme-gunu.")
    L.append("  Aylik seri ayni mantik. Z-taban: roll4q uzerinde 40c pencere (min 20).")
    L.append("  YAPISAL KIRILMA: SB-2 formu 2008-02'de kalkti (SEC 33-8876) -> ham S-1 seviyesi")
    L.append("  pre/post-2008 KARSILASTIRILAMAZ; uzun-tarih karsilastirmasi total_new_adj_n /")
    L.append("  roll4q_adj / z10y_adj (s1+f1+sb2) ile yapilir. 2008-sonrasi pencerede fark yok.")
    L.append("=" * 112)
    (OUT / "ipo_pipeline_report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  rapor -> {OUT / 'ipo_pipeline_report.txt'}")


# ─────────────────────────── main ───────────────────────────

def main() -> int:
    quick = "--quick" in sys.argv[1:]
    print("=" * 100)
    print("  FETCH: SEC EDGAR S-1/F-1 boru-hatti gozcusu (dosyalama = oncu IPO-arz sinyali)")
    print("=" * 100)
    RAW_EDGAR.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.today().normalize()

    print("  [1/5] ceyreklik form.idx indeksleri (2001Q1 -> bugun; ilk kosu ~100 dosya, sabir)...")
    filings, failed_q = collect_filings(today)

    print("  [2/5] mega-izleme listesi + eslesme...")
    watch = load_watchlist()
    matchers = build_matchers(watch)
    land(f"[watch] izleme listesi {len(watch)} isim ({WATCHLIST_TXT.name}); strict=kelime-siniri"
         f"+noktalama-normalize, loose=bosluk-sikistirilmis (FP-riskli, sayima girmez)")

    print("  [3/5] ceyreklik/aylik panel + roll4q + z10y...")
    panel, mon, hits_hist = build_panels(filings, failed_q, matchers, today)

    print("  [4/5] canli blok (son-90g) + mega $-buyukluk denemesi...")
    live = live_block(filings, panel, hits_hist, today, quick)

    print("  [5/5] yaz + rapor...")
    CACHE.mkdir(parents=True, exist_ok=True)
    pq = CACHE / "ipo_pipeline.parquet"
    panel.to_parquet(pq)
    print(f"  -> {pq}  ({len(panel)} ceyrek, {panel.shape[1]} kolon)")
    pm = CACHE / "ipo_pipeline_monthly.parquet"
    mon.to_parquet(pm)
    print(f"  -> {pm}  ({len(mon)} ay)")
    mj = CACHE / "mega_ipo_hits.json"
    mj.write_text(json.dumps({
        "as_of": str(today.date()), "watchlist": watch,
        "recent_120d": live["mega_recent_120d"],
        "historical_all": hits_hist,
        "honest_label": live["honest_label"],
    }, indent=2, default=str), encoding="utf-8")
    print(f"  -> {mj}  (son-120g {len(live['mega_recent_120d'])} + tarihce {len(hits_hist)} isabet)")
    lj = CACHE / "ipo_pipeline_live.json"
    lj.write_text(json.dumps(live, indent=2, default=str), encoding="utf-8")
    print(f"  -> {lj}")
    write_report(panel, mon, live, hits_hist, watch)

    print()
    print("  OZET:")
    n = live["window_90d_counts"]
    z = live["z90_vs_40q"]
    last = panel.dropna(subset=["z10y"])
    print(f"    kapsam: {qlab(panel.index.min())} -> {qlab(panel.index.max())} "
          f"({int(panel['partial'].sum())} kismi ceyrek)")
    print(f"    son-90g: S-1 yeni {n['s1_new']} + F-1 yeni {n['f1_new']} = {n['total_new']} "
          f"(z90 {z:+.2f})" if np.isfinite(z) else f"    son-90g: {n['total_new']} (z yok)")
    if len(last):
        lq = last.index[-1]
        print(f"    son-kapanmis-z10y: {last.loc[lq, 'z10y']:+.2f} ({qlab(lq)}, roll4q "
              f"{last.loc[lq, 'roll4q']:.0f})")
    for ln in sanity_checks(panel):
        print(ln)
    mr = live["mega_recent_120d"]
    print(f"    mega-izleme: son-120g {len(mr)} isabet"
          + (" -> " + "; ".join(f"{h['watch_name']}({h['form']},{h['date_filed']})" for h in mr[:6])
             if mr else " (dev isimler henuz dosyalamadi)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
