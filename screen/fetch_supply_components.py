"""
screen/fetch_supply_components — Constan hisse-arzi grafiginin BILESEN borulari (IPO / ikincil /
geri-alim / SPAC). Onceki faz NET'i kurdu (fetch_net_equity_supply.py -> net_equity_supply.parquet);
bu faz NET'in altindaki AYRISTIRMAYI indirir, hizalar, turetir. BETIMSEL boru — icerik testi ayri faz.

KAYNAK KESIF SONUCLARI (2026-06-13, hepsi denendi, dusenler durustce raporlanir):
  1) Z.1 BULK (federalreserve.gov/releases/z1/current/z1_csv_files.zip, 8MB, raw/'a indirildi):
     2025 SNA yeniden-tasarimi sonrasi hisse enstrumani F.223 degil F.51 tablolari. SOZLUK TARAMASI
     (574 dosya, issuance/IPO/retire/buyback/seasoned anahtarlari): BRUT ihrac / geri-alim / itfa
     BILESEN serisi YOK — ne bulk'ta ne FRED'de (search endpoint 7 sorguyla tarandi). Fed net seriyi
     lisansli kaynaklardan (LSEG/S&P) ICERIDE derleyip yalniz NET yayinliyor -> Z.1'den ayristirma
     ALINAMAZ. Z.1 katkisi = NET referans kolonu (mevcut net_equity_supply.parquet'ten okunur).
  2) SIFMA aylik ihrac: modern xlsx HubSpot e-posta-formu arkasinda (share.hsforms.com) -> bot-fetch
     YOK. COZUM: Wayback'teki legacy dosya cm-us-equity-sifma.xls (snapshot 20140521): YILLIK
     1990-2013 (common/preferred/total/all-IPO/"true"-IPO/secondary, $mlr) + AYLIK yalniz 2013-2014.
     VERI-KAZASI: 2012 yillik satiri $mn biriminde (digerleri $mlr) -> total>5000 ise /1000 normalize.
     Convertible legacy dosyada YOK (modern SIFMA'da var ama form-kapili) -> KAPSAM BOSLUGU.
  3) Jay Ritter (site.warrington.ufl.edu/ritter): IPOALL.xlsx = AYLIK IPO sayisi 1960-2025 (brut/net
     adet + ort ilk-gun getiri + midpoint-ustu fiyatlama %'si='hotness'). IPOs-SPACs.pdf Table 15b =
     YILLIK SPAC IPO adedi + proceeds $mlr 1990-2025 (pypdf ile parse) -> SPAC AYRISTIRILABILIR (yillik).
     NOT: brut adet SPAC+kapali-uclu+penny+unit vb icerir; net adet operasyonel-sirket IPO'su.
  4) S&P 500 buybacks: spglobal dogrudan 403 (WAF). COZUM: Wayback snapshot DIKISI — ayni xlsx'in
     2014-2025 arasi ~20 snapshot'i (us.spindices.com + spglobal.com) indirilir, TABLE/table/Sheet*
     icindeki PERIOD/BUYBACKS bloklari genel tarayiciyla bulunur, ceyrekler SON-snapshot-kazanir
     kuraliyla dikilir -> ceyreklik buyback $mlr ~2008Q3-2024Q4 (+prelim bayragi).
  5) S&P 500 buybacks MANUEL KATMAN (2026-06-13): Wayback'te 2025-04-01 sonrasi snapshot YOK (CDX
     dogrulandi) -> 2025Q1+ ceyrekler S&P DJI ceyreklik BASIN BULTENLERINDEN (prnewswire, bot-acik)
     elle girilir: data/manual/spx_buybacks_manual.csv (quarter,bb_bn,div_bn,source_url,entered_at,
     prelim,pit_date,xcheck_url,note). MERGE ONCELIGI: Wayback-xlsx KAZANIR (resmi revizyonlu seri);
     manuel yalniz xlsx'te OLMAYAN ceyrekleri doldurur -> Wayback yetisince manuel satir otomatik
     devre-disi kalir. Panel'de manuel satirlar spx_bb_src='manual_press' + spx_bb_prelim=1 (resmi
     xlsx'e girmedikce revize edilebilir kabul edilir); CSV'deki prelim kolonu daha ince capraz-
     teyit durumunu tasir (0=sonraki bultende restate edildi, 1=tek bulten) -> spx_bb_xchecked.
     PIT: manuel satirda pit_date = bultenin GERCEK yayin tarihi (genel q-sonu+90g kuralini ezer).
     CEYREKLIK PROSEDUR: python screen/fetch_supply_components.py --check-buyback
       -> Wayback CDX'te yeni snapshot + press.spglobal.com'da yeni bulten arar; yoksa
          'manuel giris bekliyor: <sonraki ceyrek>' uyarisi basar.

PIT / YAYIN GECIKMELERI (kaynak basina ayri, kolon olarak saklanir):
  z1     : ceyrek-baslangici +165g (onceki fazla ayni kural)
  sifma  : yil-sonu +30g (aylik yayim ~ay+30g; yillik toplam Ocak-sonunda kesinlesir varsayimi)
  ritter : ceyrek-sonu +7g (IPO adedi/ilk-gun getirisi KAMUYA ANINDA gorunur; Ritter DOSYASI yillik
           guncellenir ama PIT bilgi-bazlidir — varsayim durustce budur, dosya-bazli degil)
  spac   : yil-sonu +7g (ayni bilgi-bazli mantik, yillik seri)
  spx_bb : ceyrek-sonu +90g (S&P prelim buyback raporu ~3 ay sonra; snapshot tarihleriyle uyumlu)

ROLLING-4Q KURALI (kolon-basina dogru yontem; SAAR dersi #3):
  - Z.1 NET (SAAR) -> 4Q ORTALAMASI (onceki fazda yapildi, burada hazir ratio okunur)
  - SAAR-OLMAYAN ceyreklik akislar (spx_bb $mlr/ceyrek) -> 4Q TOPLAMI / (4Q-ort NGDP)
  - yillik akislar (sifma, spac) -> dogrudan yil / (o yilin 4Q-ort NGDP'si), ceyreklere yayinlanir (_ay)
  - adetler (ritter) -> 4Q TOPLAM adet (NGDP normalizasyonu yok; z10y yeterli)
z10y = 40-ceyrek pencere (min 20) rolling z. SAHTE-TREND dersi #1: her sinyal icin sinyal~zaman
Spearman'i CLI/panelde raporlanir; anlamlilik iddiasi BU fazda yapilmaz (betimsel boru).

CIKTI:
  data/cache/supply_components.parquet          ceyreklik hizali panel (+turevler +pit kolonlari)
  data/cache/supply_components_monthly.parquet  aylik-HAM (ritter 1960+ aylik, sifma 2013-14 aylik)
  data/cache/raw/                                ham landing'ler (idempotent cache)
  output/supply_components_panel.txt             kaynak-raporu + son 8 ceyrek + spearman tanisi
"""
from __future__ import annotations

import io
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
RAW = CACHE / "raw"
OUT = ROOT / "output"
MANUAL_BB_CSV = ROOT / "data" / "manual" / "spx_buybacks_manual.csv"
ENV = Path(r"C:\Users\admin\Downloads\kader-macro\.env")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

PUB_LAG = {"z1": 165, "sifma": 30, "ritter": 7, "spac": 7, "spx_bb": 90}

# S&P buyback wayback snapshotlari (CDX kesfi 2026-06-13; statuscode:200, >20KB olanlar).
# Dikis kurali: ASCENDING sirayla islenir, ayni ceyrekte SON snapshot kazanir (revize/final deger).
SPX_BB_SNAPSHOTS = [
    ("20141224033348", "http://spindices.com/documents/additional-material/sp-500-buyback.xlsx"),
    ("20150126125348", "http://spindices.com/documents/additional-material/sp-500-buyback.xlsx"),
    ("20150221034339", "http://us.spindices.com/documents/additional-material/sp-500-buyback.xlsx"),
    ("20150323135325", "http://us.spindices.com/documents/additional-material/sp-500-buyback.xlsx"),
    ("20150423045610", "http://us.spindices.com/documents/additional-material/sp-500-buyback.xlsx"),
    ("20150524013436", "http://us.spindices.com/documents/additional-material/sp-500-buyback.xlsx"),
    ("20150623142707", "http://us.spindices.com/documents/additional-material/sp-500-buyback.xlsx"),
    ("20150725023205", "http://us.spindices.com/documents/additional-material/sp-500-buyback.xlsx"),
    ("20150825072021", "http://us.spindices.com/documents/additional-material/sp-500-buyback.xlsx"),
    ("20151114091514", "http://us.spindices.com/documents/additional-material/sp-500-buyback.xlsx"),
    ("20160304175416", "http://us.spindices.com/documents/additional-material/sp-500-buyback.xlsx"),
    ("20160808171436", "http://us.spindices.com/documents/additional-material/sp-500-buyback.xlsx"),
    ("20161011070857", "http://us.spindices.com/documents/additional-material/sp-500-buyback.xlsx"),
    ("20170119224357", "http://us.spindices.com/documents/additional-material/sp-500-buyback.xlsx"),
    ("20170408220506", "http://us.spindices.com/documents/additional-material/sp-500-buyback.xlsx"),
    ("20170428185717", "http://us.spindices.com/documents/additional-material/sp-500-buyback.xlsx"),
    ("20181009160011", "http://us.spindices.com/documents/additional-material/sp-500-buyback.xlsx"),
    ("20190503212923", "http://us.spindices.com/documents/additional-material/sp-500-buyback.xlsx"),
    ("20210322035125", "https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-buyback.xlsx"),
    ("20220329193118", "https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-buyback.xlsx"),
    ("20230123062209", "https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-buyback.xlsx"),
    ("20240507050317", "https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-buyback.xlsx"),
    ("20250401130739", "https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-buyback.xlsx"),
]
SIFMA_SNAP = ("20140521040111",
              "http://www.sifma.org/uploadedfiles/research/statistics/statisticsfiles/cm-us-equity-sifma.xls")

LANDINGS: list[str] = []   # kaynak-basina landing raporu satirlari


def land(msg: str) -> None:
    print("  " + msg)
    LANDINGS.append(msg)


def fred_key() -> str:
    import os
    k = os.environ.get("FRED_API_KEY")
    if not k and ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            if line.startswith("FRED_API_KEY"):
                k = line.split("=", 1)[1].strip()
    if not k:
        raise RuntimeError("FRED_API_KEY bulunamadi (kader-macro/.env)")
    return k


def fred_obs(sid: str, key: str) -> pd.Series:
    r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                     params={"series_id": sid, "api_key": key, "file_type": "json", "limit": 100000},
                     timeout=60)
    r.raise_for_status()
    obs = r.json()["observations"]
    idx = pd.to_datetime([o["date"] for o in obs])
    val = pd.to_numeric([o["value"] for o in obs], errors="coerce")
    return pd.Series(val, index=idx, name=sid).dropna()


def cached_download(fname: str, url: str, min_bytes: int = 5000, retries: int = 3,
                    backoff: float = 20.0) -> Path | None:
    """Idempotent landing: raw/fname varsa dokunma; yoksa indir (wayback hiz-siniri icin backoff)."""
    p = RAW / fname
    if p.exists() and p.stat().st_size >= min_bytes:
        return p
    for i in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=300)
            if r.status_code == 200 and len(r.content) >= min_bytes:
                p.write_bytes(r.content)
                return p
            if r.status_code == 200:
                return None          # kucuk govde = arsiv stub'i; deneme tekrari anlamsiz
        except requests.RequestException:
            pass
        if i < retries - 1:
            time.sleep(backoff * (i + 1))
    return None


def qstart(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=3 * ((ts.month - 1) // 3) + 1, day=1)


def quarter_end(q: pd.Timestamp) -> pd.Timestamp:
    return q + pd.offsets.QuarterEnd(0)


# ─────────────────────────── 3) RITTER: aylik IPO sayilari (1960+) ───────────────────────────

def fetch_ritter_monthly() -> pd.DataFrame:
    p = cached_download("ritter_IPOALL.xlsx", "https://site.warrington.ufl.edu/ritter/files/IPOALL.xlsx",
                        min_bytes=10000)
    if p is None:
        land("[ritter] FAIL: IPOALL.xlsx indirilemedi")
        return pd.DataFrame()
    df = pd.ExcelFile(p).parse("IPOALL", header=None)
    rows = []
    for _, r in df.iterrows():
        m, y = pd.to_numeric(r.iloc[0], errors="coerce"), pd.to_numeric(r.iloc[1], errors="coerce")
        if not (np.isfinite(m) and np.isfinite(y) and 1 <= m <= 12):
            continue
        yy = int(y)
        year = 1900 + yy if yy >= 30 else 2000 + yy
        rows.append({
            "date": pd.Timestamp(year=year, month=int(m), day=1),
            "ritter_ipo_fdret": pd.to_numeric(r.iloc[2], errors="coerce"),
            "ritter_ipo_n_gross": pd.to_numeric(r.iloc[3], errors="coerce"),
            "ritter_ipo_n_net": pd.to_numeric(r.iloc[4], errors="coerce"),
            "ritter_ipo_hot_pct": pd.to_numeric(r.iloc[5], errors="coerce"),
        })
    out = pd.DataFrame(rows).set_index("date").sort_index()
    out["ritter_ipo_nonop_n"] = out["ritter_ipo_n_gross"] - out["ritter_ipo_n_net"]
    land(f"[ritter] OK canli-indirme IPOALL.xlsx: AYLIK {out.index.min():%Y-%m} -> "
         f"{out.index.max():%Y-%m} (n={len(out)}); adet brut/net + ilk-gun getiri + hotness; "
         f"net-adet 1975+, hotness 1980+; pub-lag varsayimi ceyrek-sonu +{PUB_LAG['ritter']}g (bilgi-bazli)")
    return out


# ─────────────────────────── 3b) RITTER: yillik SPAC (Table 15b, pdf) ───────────────────────────

def fetch_ritter_spac() -> pd.DataFrame:
    p = cached_download("ritter_IPOs-SPACs.pdf", "https://site.warrington.ufl.edu/ritter/files/IPOs-SPACs.pdf",
                        min_bytes=20000)
    if p is None:
        land("[spac] FAIL: IPOs-SPACs.pdf indirilemedi")
        return pd.DataFrame()
    import pypdf
    rd = pypdf.PdfReader(p)
    text = ""
    for pg in rd.pages:
        t = pg.extract_text() or ""
        if "Table 15b" in t or "SPAC) IPOs, 19" in t:
            text += t + "\n"
    rows, fails = [], 0
    for line in text.splitlines():
        toks = line.split()
        if len(toks) < 6 or not re.fullmatch(r"(19|20)\d{2}", toks[0]):
            continue
        try:
            year = int(toks[0])
            opco_n = float(toks[1].replace(",", ""))
            # toks[2] = opco mean IR (%); 3,4,5 = nonunit, unit, total SPAC adet
            nonunit, unit, total = (float(toks[3]), float(toks[4]), float(toks[5]))
            proceeds = np.nan
            for t in toks[6:]:
                if t.startswith("$"):
                    proceeds = float(t.replace("$", "").replace(",", ""))
                    break
            rows.append({"year": year, "ritter_opco_n_ay": opco_n, "spac_n_ay": total,
                         "spac_unit_n_ay": unit, "spac_nonunit_n_ay": nonunit,
                         "spac_proceeds_bn_ay": proceeds})
        except (ValueError, IndexError):
            fails += 1
    if not rows:
        land("[spac] FAIL: Table 15b satirlari parse edilemedi")
        return pd.DataFrame()
    out = pd.DataFrame(rows).drop_duplicates("year").set_index("year").sort_index()
    land(f"[spac] OK canli-indirme IPOs-SPACs.pdf Table 15b (pypdf): YILLIK {out.index.min()} -> "
         f"{out.index.max()} (n={len(out)}, parse-fail {fails} satir); SPAC adet (unit/non-unit) + "
         f"proceeds $mlr + operasyonel-IPO adedi; pub-lag varsayimi yil-sonu +{PUB_LAG['spac']}g")
    return out


# ─────────────────────────── 2) SIFMA: legacy wayback (yillik 1990-2013 + aylik 2013-14) ─────────

SIFMA_COLS = ["sifma_common_bn", "sifma_preferred_bn", "sifma_total_equity_bn",
              "sifma_ipo_all_bn", "sifma_ipo_true_bn", "sifma_secondary_bn"]
_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "june": 6, "jul": 7,
           "july": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12}


def fetch_sifma() -> tuple[pd.DataFrame, pd.DataFrame]:
    ts, orig = SIFMA_SNAP
    p = cached_download("sifma_cm_us_equity_wayback_20140521.xls",
                        f"https://web.archive.org/web/{ts}if_/{orig}", min_bytes=50000)
    if p is None:
        land("[sifma] FAIL: wayback cm-us-equity-sifma.xls indirilemedi")
        return pd.DataFrame(), pd.DataFrame()
    df = pd.ExcelFile(p).parse("Equity", header=None)
    ann_rows, mon_rows, cur_year, unit_fixed = [], [], None, 0
    for i in range(len(df)):
        c0 = df.iloc[i, 0]
        vals = [pd.to_numeric(df.iloc[i, j], errors="coerce") for j in range(1, 7)]
        if isinstance(c0, (int, float)) and np.isfinite(pd.to_numeric(c0, errors="coerce")) \
                and 1980 < float(c0) < 2030:
            if np.isfinite(vals[2]):                     # yillik veri satiri (total dolu)
                v = vals
                if v[2] > 5000:                          # 2012 $mn birim-kazasi -> $mlr'a cevir
                    v = [x / 1000.0 if np.isfinite(x) else x for x in v]
                    unit_fixed += 1
                ann_rows.append({"year": int(c0), **dict(zip(SIFMA_COLS, v))})
            else:                                        # aylik blok basligi (yalniz yil)
                cur_year = int(c0)
        elif isinstance(c0, str) and c0.strip().lower()[:4].rstrip(".") in _MONTHS or \
                (isinstance(c0, str) and c0.strip().lower() in _MONTHS):
            if cur_year is None:
                continue
            key = c0.strip().lower()
            mnum = _MONTHS.get(key, _MONTHS.get(key[:3]))
            if mnum is None or not np.isfinite(vals[2]):
                continue
            mon_rows.append({"date": pd.Timestamp(year=cur_year, month=mnum, day=1),
                             **dict(zip(SIFMA_COLS, vals))})
    ann = pd.DataFrame(ann_rows).drop_duplicates("year").set_index("year").sort_index()
    mon = (pd.DataFrame(mon_rows).set_index("date").sort_index()
           if mon_rows else pd.DataFrame())
    land(f"[sifma] OK wayback-arsiv (snapshot {ts[:8]}, modern xlsx HubSpot-form-kapili): YILLIK "
         f"{ann.index.min()}-{ann.index.max()} (n={len(ann)}, birim-fix {unit_fixed} satir) "
         f"common/preferred/total/IPO-all/IPO-true/secondary $mlr; AYLIK yalniz "
         f"{mon.index.min():%Y-%m}->{mon.index.max():%Y-%m} (n={len(mon)}); 2014+ KAPALI (form-gate); "
         f"pub-lag yil-sonu +{PUB_LAG['sifma']}g")
    return ann, mon


# ─────────────────────────── 4) S&P 500 buybacks: wayback dikisi ───────────────────────────

def _parse_period(v):
    """PERIOD hucresi -> ('Q', qstart_ts, prelim) | ('A', yil, prelim) | None.
    DIKKAT: Excel datetime hucreleri pandas object-kolonda datetime.datetime olarak gelir
    (pd.Timestamp DEGIL) — ilk surumdeki isinstance(pd.Timestamp) bug'i ceyreklerin %70'ini
    dusuruyordu; datetime/date/np.datetime64 de kabul edilir."""
    import datetime as _dt
    if isinstance(v, (pd.Timestamp, _dt.datetime, _dt.date, np.datetime64)):
        return ("Q", qstart(pd.Timestamp(v)), False)
    if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) \
            and 1990 <= float(v) <= 2035:
        return ("A", int(v), False)
    if isinstance(v, str):
        s = v.strip()
        if not s or "Mo" in s:                       # '12 Mo Sep,20' satirlarini atla
            return None
        prelim = "prelim" in s.lower()
        s2 = re.sub(r"prelim\.?", "", s, flags=re.I).strip().rstrip(".").strip()
        if re.fullmatch(r"(19|20)\d{2}", s2):
            return ("A", int(s2), prelim)
        try:
            ts = pd.Timestamp(s2)
            return ("Q", qstart(ts), prelim)
        except Exception:
            return None
    return None


def _scan_buyback_sheet(df: pd.DataFrame):
    """Bir sheet'te PERIOD basligini + BUYBACKS/DIVIDENDS $-kolonlarini bul; (satirlar) dondur."""
    hdr = None
    for i in range(min(25, len(df))):
        for j in range(min(6, df.shape[1])):
            v = df.iloc[i, j]
            if isinstance(v, str) and v.strip().upper() == "PERIOD":
                hdr = (i, j)
                break
        if hdr:
            break
    if hdr is None:
        return None
    hi, hj = hdr
    # ÖLÇÜ-KARIŞIMI FIX (adversarial denetim 2026-06-13): eski döngü 'BUYBACK içeren SON kolonu'
    # alıyordu → 2014-2019 snapshot'larında birleşik 'DIVIDENDS & BUYBACKS' kolonu kazanıp
    # 2008Q3-2016Q2 dikişini %55-120 şişiriyordu (B1/B2/H3 o haliyle geçersizdi). Yeni kural:
    # SAF buyback kolonu öncelikli; saf yoksa (birleşik − temettü) kimliği — denetçi bu kimliği
    # 6/6 çeyrekte bilinen gerçekle birebir doğruladı (örn. 2014Q1 241.2−82.0=159.3).
    bb_pure = bb_combo = div_col = None
    for j in range(df.shape[1]):
        blob = " ".join(str(df.iloc[k, j]).upper() for k in range(hi, min(hi + 5, len(df)))
                        if isinstance(df.iloc[k, j], str))
        if "YIELD" in blob:
            continue
        has_bb, has_div = "BUYBACK" in blob, "DIVIDEND" in blob
        if has_bb and not has_div and bb_pure is None:
            bb_pure = j
        elif has_bb and has_div and bb_combo is None:
            bb_combo = j
        elif has_div and not has_bb and div_col is None:
            div_col = j
    if bb_pure is None and not (bb_combo is not None and div_col is not None):
        return None
    recs = []
    for i in range(hi + 1, len(df)):
        per = _parse_period(df.iloc[i, hj])
        if per is None:
            continue
        dv = pd.to_numeric(df.iloc[i, div_col], errors="coerce") if div_col is not None else np.nan
        if bb_pure is not None:
            bb = pd.to_numeric(df.iloc[i, bb_pure], errors="coerce")
        else:
            combo = pd.to_numeric(df.iloc[i, bb_combo], errors="coerce")
            bb = (combo - dv) if (np.isfinite(combo) and np.isfinite(dv)) else np.nan
        if np.isfinite(bb):
            recs.append((*per, float(bb), float(dv) if np.isfinite(dv) else np.nan))
    return recs or None


def load_manual_buybacks() -> pd.DataFrame:
    """data/manual/spx_buybacks_manual.csv -> qdate-indeksli DataFrame (bos olabilir).
    Kolon sozlesmesi: quarter(2025Q1),bb_bn,div_bn,source_url,entered_at,prelim,pit_date,
    xcheck_url,note. prelim: 0=sonraki resmi bultende restate-teyitli, 1=tek-bulten/teyitsiz."""
    if not MANUAL_BB_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(MANUAL_BB_CSV)
    rows = []
    for _, r in df.iterrows():
        m = re.fullmatch(r"((?:19|20)\d{2})Q([1-4])", str(r["quarter"]).strip())
        if not m:
            continue
        rows.append({
            "qdate": pd.Timestamp(year=int(m.group(1)), month=3 * int(m.group(2)) - 2, day=1),
            "bb_bn": pd.to_numeric(r["bb_bn"], errors="coerce"),
            "div_bn": pd.to_numeric(r.get("div_bn"), errors="coerce"),
            "csv_prelim": float(pd.to_numeric(r.get("prelim"), errors="coerce")),
            "pit_date": pd.to_datetime(r.get("pit_date"), errors="coerce"),
            "source_url": str(r.get("source_url", "")),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("qdate").sort_index()


# ─────────────────────────── 4b) S&P 500 buyback OTOMATIK CEKIM (manuel adimi oldurur) ───────────
#
# TASARIM (2026-06-13, ham-HTML kanit-tabanli, WebFetch-ozeti GUVENILMEZ -> ham regex):
#   S&P DJI ceyreklik buyback bulteni (press.spglobal.com VE/VEYA prnewswire, ikisi de bot-acik 200)
#   govdesinde SEKTOR-TOTAL satirini standart bir tablo formatinda basar:
#       'S&P 500  $249,004 $234,570 $226,557 $1,020,268 $918,398 $4,449,728 $7,575,704 $674,035'
#   KOLON DUZENI (3 bultende — Q1/Q2/Q3 2025 — birebir dogrulandi, $milyon):
#       col0 = MEVCUT ceyrek buyback        (parse hedefi -> /1000 = $mlr)
#       col1 = onceki ceyrek
#       col2 = yil-once ayni ceyrek
#       col3 = TRAILING-12-AY toplami        (SELF-CHECK capari)
#       col4 = onceki-12-ay | col5 = 5y | col6 = 10y | col7 = ...
#   Ayrica anlatida 'share repurchases were $249.0 billion' (col0 ile +-0.5 capraz-teyit, yuvarlanmis).
#   Temettu: index-toplami satirinda 'M/D/YYYY ... $<div> $<bb>' (ornek 9/30/2025 ... $168.08 $249.00).
#
# YANLIS-SATIR TUZAGI (workflow gecmisi): WebFetch ozetleyici Q1 bulteninde 'Top 20' satirini
#   sektor-TOTAL sanip yanlis rakam okumustu. Onlem: SADECE '\bS&P 500\b' total satiri (Top-20 / Top 20
#   / sektor-adi satirlari DEGIL), ham-HTML'den; ayrica anlati-rakami + 12mo-aritmetik cifte-kapi.


def _clean_html(raw: str) -> str:
    """HTML -> bosluk-normalize duz metin (etiket/entity temizligi; sayi-tablolari korunur)."""
    txt = re.sub(r"<[^>]+>", " ", raw)
    txt = txt.replace("\xa0", " ")
    txt = re.sub(r"&nbsp;|&#160;", " ", txt)
    txt = re.sub(r"&amp;", "&", txt)
    txt = re.sub(r"&#39;|&rsquo;|&apos;", "'", txt)
    return re.sub(r"[ \t\r\n]+", " ", txt)


def parse_buyback_bulletin(raw_html: str) -> dict | None:
    """Bir S&P DJI buyback bulteninin ham HTML'inden sektor-TOTAL rakamlarini cek.
    Donus: {'bb_bn', 'div_bn', 'bb_narr_bn', 'ttm_bn'} ($mlr) ya da None (total satiri yoksa).
    bb_bn = sektor-TOTAL 'S&P 500' satirinin col0'i (mevcut ceyrek, $mln -> $mlr).
    ttm_bn = ayni satirin col3'u (trailing-12mo, SELF-CHECK icin). div_bn opsiyonel (yoksa NaN)."""
    txt = _clean_html(raw_html)
    # SEKTOR-TOTAL: kelime-siniri 'S&P 500' + ardisik >=4 dolar-tutari. 'Top 20'/'Top 20 %' satiri
    # de '\bS&P 500\b' icermez (orneklerde 'Top 20 % of S&P 500' GECER ama ardindan $-dizisi YOK,
    # %-degeri var -> {4,} dolar-tutari kosulu onu eler).
    m = re.search(r"\bS&P 500\b((?:\s+\$[\d,]+){4,})", txt)
    if not m:
        return None
    nums = [int(x.replace(",", "")) for x in re.findall(r"\$([\d,]+)", m.group(1))]
    if len(nums) < 4:
        return None
    bb_bn = nums[0] / 1000.0          # col0 mevcut ceyrek ($mln -> $mlr)
    prior_q_bn = nums[1] / 1000.0     # col1 ONCEKI ceyrek ($mln -> $mlr) — BAGIMSIZ-capraz icin
    ttm_bn = nums[3] / 1000.0         # col3 trailing-12-ay ($mln -> $mlr)
    # anlati cifte-teyidi (yuvarlanmis $bln): 'share repurchases were $249.0 billion'
    nb = re.search(r"share repurchases were \$([\d.]+) billion", txt)
    bb_narr_bn = float(nb.group(1)) if nb else np.nan
    # temettu: index-toplam satiri 'M/D/YYYY Estimate? $<...> ... $<div> $<bb>' — bb_bn'e en yakin
    # $-cifti '... $div $bb_yuvarlanmis'. Guvenli yaklasim: col0-yuvarlanmasiyla biten ciftte div'i al.
    div_bn = np.nan
    cur_round = round(bb_bn, 0)
    for dm in re.finditer(r"\$([\d.]+)\s+\$([\d.]+)\b", txt):
        try:
            d_cand, bb_cand = float(dm.group(1)), float(dm.group(2))
        except ValueError:
            continue
        if abs(bb_cand - bb_bn) <= 0.6 and 50.0 <= d_cand <= 400.0:   # bb ~ col0; div makul aralik
            div_bn = d_cand
            break
    return {"bb_bn": bb_bn, "div_bn": div_bn, "bb_narr_bn": bb_narr_bn,
            "ttm_bn": ttm_bn, "prior_q_bn": prior_q_bn}


# Self-check toleranslari (saf-fonksiyon; testlerde kilitli):
_SC_TTM_TOL = 1.0       # 4Q-toplam vs bültenin 12mo'su ($mlr)
_SC_PRIOR_TOL = 5.0     # bültenin col1 (önceki-Q) vs bizim CSV önceki-Q (prelim->final revizyon payı)


def buyback_selfcheck(bb_bn: float, ttm_bn: float, prior_q_bn: float,
                      priors: list[float]) -> tuple[bool, str]:
    """SESSİZ-BOZULMA önleyici BAĞIMSIZ self-check (saf, test edilebilir).

    Hesaplanabilen HER bağımsız kontrol GEÇMELİ, ve EN AZ BİRİ hesaplanabilmeli:
      • BAĞIMSIZ-ÇAPA (≥1 prior): bültenin bastığı önceki-çeyrek (col1) ≈ bizim CSV'deki en
        yeni önceki-çeyrek (±5$mlr; iki AYRI bülten → gerçekten bağımsız). Top-20 tuzağı/birim
        hatası burada onlarca-yüzlerce milyar saparak DÜŞER.
      • 12mo-ARİTMETİK (≥3 prior): parse + önceki-3 = bültenin kendi 12mo'su (±1$mlr).
    0 prior (taze CSV) → REFUSE (aynı-kaynak anlatı-yedeğine ASLA düşülmez — eski deliğin kapanışı).
    Döner (ok, detay)."""
    checks, fails = [], []
    # bağımsız çapa
    if priors:
        d = abs(prior_q_bn - priors[-1]) if (prior_q_bn == prior_q_bn) else float("inf")
        ok = d <= _SC_PRIOR_TOL
        checks.append(ok)
        (fails.append if not ok else (lambda *_: None))(
            f"col1-capra: bulten-onceki-Q {prior_q_bn:.1f} vs CSV {priors[-1]:.1f} fark {d:.1f}>5")
        detail_anchor = f"col1 {prior_q_bn:.1f}~CSV {priors[-1]:.1f} (fark {d:.1f})"
    else:
        detail_anchor = "BAGIMSIZ-CAPA YOK (0 prior)"
    # 12mo aritmetik
    if len(priors) >= 3:
        recon = bb_bn + sum(priors[-3:])
        d = abs(recon - ttm_bn)
        ok = d <= _SC_TTM_TOL
        checks.append(ok)
        if not ok:
            fails.append(f"12mo-aritmetik: 4Q {recon:.1f} vs bulten-12mo {ttm_bn:.1f} fark {d:.1f}>1")
        detail_arith = f"12mo 4Q {recon:.1f}~bulten {ttm_bn:.1f}"
    else:
        detail_arith = "12mo atlandi (<3 prior)"
    if not checks:
        return False, f"hicbir BAGIMSIZ kontrol hesaplanamadi ({detail_anchor}) -> REFUSE"
    if fails:
        return False, "; ".join(fails)
    return True, f"{detail_anchor} | {detail_arith}"


def _bulletin_listing() -> list[dict]:
    """press.spglobal.com buyback bulten listesini tara -> [{quarter(ts), title, url, pub_date}].
    En yeni bulteni + URL'sini saglar; bos liste = sayfa-formati degismis/erisim yok."""
    out: list[dict] = []
    try:
        r = requests.get("https://press.spglobal.com/index.php?s=2429&l=100", headers=UA, timeout=60)
        if r.status_code != 200:
            return out
    except requests.RequestException:
        return out
    # anchor: href + baslik (baslikta 'S&P 500 Q<n> <yyyy> ... Buyback')
    for href, title in re.findall(r'<a[^>]+href="([^"]+)"[^>]*>([^<]*[Bb]uyback[^<]*)</a>', r.text):
        tm = re.search(r"Q([1-4])[\s\-]+(20\d{2})", title)
        if not tm:
            continue
        qn, yy = int(tm.group(1)), int(tm.group(2))
        q = pd.Timestamp(year=yy, month=3 * qn - 2, day=1)
        # bulten yayin tarihi URL on-ekinden (press.spglobal.com/YYYY-MM-DD-...)
        dm = re.search(r"/(20\d{2}-\d{2}-\d{2})-", href)
        pub = pd.to_datetime(dm.group(1)) if dm else pd.NaT
        out.append({"quarter": q, "title": title.strip(), "url": href, "pub_date": pub})
    # ceyrek-bazli dedup: ayni ceyrekte ilk (en ust = en yeni) kalir
    seen, dedup = set(), []
    for d in out:
        if d["quarter"] in seen:
            continue
        seen.add(d["quarter"])
        dedup.append(d)
    return dedup


def _existing_quarters() -> set[pd.Timestamp]:
    """Auto-parser'in ATLAMASI gereken ceyrekler: manuel CSV'de OLAN + Wayback-xlsx dikisinde OLAN.
    (Wayback xlsx = resmi revizyonlu seri; orada varsa otomatik-cekime gerek yok.)"""
    have: set[pd.Timestamp] = set()
    man = load_manual_buybacks()
    if len(man):
        have |= set(man.index)
    # wayback-xlsx ceyrekleri: parquet'ten oku (varsa) — wayback satirlari spx_bb_manual==0.
    pq = CACHE / "supply_components.parquet"
    if pq.exists():
        try:
            panel = pd.read_parquet(pq)
            if "spx_bb_bn" in panel and "spx_bb_manual" in panel:
                wb = panel[(panel["spx_bb_bn"].notna()) & (panel["spx_bb_manual"] == 0.0)]
                have |= set(wb.index)
        except Exception:
            pass
    return have


def auto_pull_buyback(write: bool = True, force_url: str | None = None,
                      force_quarter: str | None = None) -> dict:
    """S&P 500 buyback bultenini OTOMATIK cek + dogrula + (write ise) manuel-CSV'ye yaz.
    Manuel son-adimi oldurur. ASLA dogrulanmamis sayi yazmaz: 12mo-aritmetik self-check tutmazsa
    loud-log + return (manuel fallback). Donus dict = {status, quarter, ...} (test/CLI icin).

    status: 'guncel'        — yeni ceyrek yok (en yeni zaten kapsamda)
            'written'       — yeni ceyrek parse+dogrulandi -> CSV'ye yazildi
            'verify_failed' — parse oldu ama self-check tutmadi -> YAZILMADI (manuel bekliyor)
            'parse_failed'  — bulten cekildi ama total satiri parse edilemedi
            'no_listing'    — press.spglobal listesi bos/erisilemez (manuel kontrole dus)

    force_url/force_quarter: TEST/yeniden-parse icin belirli bir bulteni hedefle (kesif atla)."""
    print("=" * 100)
    print("  AUTO-BUYBACK: S&P 500 buyback bulteni OTOMATIK cekim + 12mo-aritmetik self-check")
    print("=" * 100)
    have = _existing_quarters()

    # 1) hedef bulteni belirle: force varsa onu, yoksa listeden kapsamdisi en-yeni ceyregi sec
    if force_url:
        m = re.fullmatch(r"((?:19|20)\d{2})Q([1-4])", (force_quarter or "").strip()) if force_quarter else None
        tgt_q = (pd.Timestamp(year=int(m.group(1)), month=3 * int(m.group(2)) - 2, day=1)
                 if m else pd.NaT)
        target = {"quarter": tgt_q, "url": force_url, "title": "(force)", "pub_date": pd.NaT}
        print(f"  [force] hedef bulten: {force_url[:80]}")
    else:
        listing = _bulletin_listing()
        if not listing:
            print("  [liste] press.spglobal.com bulten listesi bos/erisilemez -> manuel kontrole dusuldu")
            return {"status": "no_listing", "quarter": None}
        newest = max(d["quarter"] for d in listing)
        missing = [d for d in listing if d["quarter"] not in have]
        if not missing:
            qlab = f"{newest.year}Q{(newest.month - 1)//3 + 1}"
            print(f"  guncel: son {qlab} (manuel CSV + wayback-xlsx kapsami tam, yeni ceyrek yok)")
            return {"status": "guncel", "quarter": qlab}
        target = max(missing, key=lambda d: d["quarter"])    # en yeni kapsamdisi ceyrek
        ql = f"{target['quarter'].year}Q{(target['quarter'].month - 1)//3 + 1}"
        print(f"  [liste] KAPSAMDISI en yeni ceyrek: {ql} -> {target['url'][:70]}")

    # 2) bulteni cek (press.spglobal VEYA prnewswire — ikisi de ayni total satirini tasir)
    try:
        rr = requests.get(target["url"], headers=UA, timeout=120)
        if rr.status_code != 200 or len(rr.text) < 5000:
            print(f"  HATA: bulten cekilemedi (status {rr.status_code}, len {len(rr.text)}) -> manuel bekliyor")
            return {"status": "parse_failed", "quarter": None, "url": target["url"]}
    except requests.RequestException as e:
        print(f"  HATA: bulten istegi basarisiz ({type(e).__name__}) -> manuel bekliyor")
        return {"status": "parse_failed", "quarter": None, "url": target["url"]}

    parsed = parse_buyback_bulletin(rr.text)
    if parsed is None:
        print("  HATA: sektor-TOTAL 'S&P 500' satiri parse edilemedi (bulten formati degismis?) -> manuel bekliyor")
        return {"status": "parse_failed", "quarter": None, "url": target["url"]}

    qts = target["quarter"]
    qlab = (f"{qts.year}Q{(qts.month - 1)//3 + 1}" if qts is not pd.NaT and pd.notna(qts) else "?")
    bb, ttm, narr, div = parsed["bb_bn"], parsed["ttm_bn"], parsed["bb_narr_bn"], parsed["div_bn"]
    prior_q = parsed.get("prior_q_bn", float("nan"))
    print(f"  parse: {qlab}  buyback={bb:.3f} $mlr  (anlati {narr if np.isfinite(narr) else '-'}, "
          f"bulten-onceki-Q {prior_q if np.isfinite(prior_q) else '-'}), "
          f"temettu={div if np.isfinite(div) else '-'}, bulten-12mo={ttm:.3f}")

    # 3) ZORUNLU BAĞIMSIZ SELF-CHECK (saf-fonksiyon, sessiz-bozulma önleyici; aynı-kaynak anlatı-yedeği
    #    KALDIRILDI). col1-çapa (≥1 prior, BAĞIMSIZ iki-bülten) + 12mo-aritmetik (≥3 prior); 0-prior REFUSE.
    man = load_manual_buybacks()
    priors_all = (man[man.index < qts]["bb_bn"].dropna().sort_index().tolist()
                  if len(man) and qts is not pd.NaT and pd.notna(qts) else [])
    sc_ok, sc_detail = buyback_selfcheck(bb, ttm, prior_q, priors_all)
    print(f"  self-check [BAGIMSIZ]: {sc_detail} -> {'GECTI' if sc_ok else 'KALDI'}")

    if not sc_ok:
        print(f"  !! AUTO-PARSE BELIRSIZ: {qlab}, bagimsiz capraz tutmadi ({sc_detail}). "
              f"CSV'ye YAZILMADI -> elle-dogrulama bekliyor")
        return {"status": "verify_failed", "quarter": qlab, "bb_bn": bb, "ttm_bn": ttm,
                "detail": sc_detail}

    print(f"  OK self-check GECTI ({qlab})")
    if not write:
        return {"status": "verified_nowrite", "quarter": qlab, "bb_bn": bb, "div_bn": div, "ttm_bn": ttm}

    # 4) zaten kapsamda mi? (force ile mevcut ceyregi reparse ediyorsak yazma — sadece parse-dogrulugu goster)
    if qts in have:
        print(f"  not: {qlab} zaten kapsamda (manuel/wayback) -> WRITE-SKIP (parse-dogrulugu yukarida gosterildi)")
        return {"status": "verified_nowrite", "quarter": qlab, "bb_bn": bb, "div_bn": div, "ttm_bn": ttm}

    # 5) CSV'ye yaz (append, prelim=1: resmi xlsx'e girene kadar revize-edilebilir)
    today = pd.Timestamp.today().normalize()
    pit = target.get("pub_date")
    pit_s = f"{pd.to_datetime(pit):%Y-%m-%d}" if pd.notna(pit) else f"{today:%Y-%m-%d}"
    new_row = {
        "quarter": qlab,
        "bb_bn": f"{bb:.3f}",
        "div_bn": f"{div:.2f}" if np.isfinite(div) else "",
        "source_url": target["url"],
        "entered_at": f"{today:%Y-%m-%d}",
        "prelim": 1,
        "pit_date": pit_s,
        "xcheck_url": target["url"],
        "note": "auto-pulled + bagimsiz-capraz-verified (col1-capa + 12mo-aritmetik)",
    }
    if MANUAL_BB_CSV.exists():
        df = pd.read_csv(MANUAL_BB_CSV)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    df.to_csv(MANUAL_BB_CSV, index=False)
    print(f"  -> CSV'ye yazildi: {MANUAL_BB_CSV.name} satiri {qlab} bb={bb:.3f} div={div if np.isfinite(div) else '-'} "
          f"prelim=1 pit={pit_s} note=auto-pulled+12mo-verified")
    return {"status": "written", "quarter": qlab, "bb_bn": bb, "div_bn": div, "ttm_bn": ttm,
            "pit_date": pit_s}


def fetch_spx_buybacks() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    q_map: dict[pd.Timestamp, tuple] = {}
    a_map: dict[int, tuple] = {}
    used, failed = [], []
    for ts, orig in SPX_BB_SNAPSHOTS:               # ascending: son snapshot ayni ceyregi ezer
        p = cached_download(f"sp500_buyback_wayback_{ts}.xlsx",
                            f"https://web.archive.org/web/{ts}if_/{orig}", min_bytes=20000)
        if p is None:
            failed.append(ts)
            continue
        got = False
        try:
            xl = pd.ExcelFile(p)
            for sh in xl.sheet_names:
                df = xl.parse(sh, header=None)
                recs = _scan_buyback_sheet(df)
                if not recs:
                    continue
                got = True
                for kind, key, prelim, bb, dv in recs:
                    if kind == "Q":
                        q_map[key] = (bb, dv, prelim, ts)
                    else:
                        a_map[key] = (bb, dv, prelim, ts)
                break                                # ilk PERIOD/BUYBACKS sheet'i yeter
        except Exception:
            pass
        (used if got else failed).append(ts)

    # ── MANUEL KATMAN: xlsx KAZANIR, manuel yalniz xlsx'te olmayan ceyrekleri doldurur ──
    man = load_manual_buybacks()
    man_used, man_skip = [], []
    man_meta: dict[pd.Timestamp, tuple] = {}        # qdate -> (pit, xchecked)
    for q, r in (man.iterrows() if len(man) else []):
        if not np.isfinite(r["bb_bn"]):
            continue
        if q in q_map:                               # resmi xlsx mevcut -> manuel devre-disi
            man_skip.append(q)
            continue
        # panel-prelim=1: resmi revizyonlu xlsx'e girene kadar revize-edilebilir kabul edilir
        q_map[q] = (float(r["bb_bn"]), float(r["div_bn"]), True, "manual_press")
        man_meta[q] = (r["pit_date"], 1.0 - (r["csv_prelim"] if np.isfinite(r["csv_prelim"]) else 1.0))
        man_used.append(q)

    qdf = (pd.DataFrame([{"qdate": k, "spx_bb_bn": v[0], "spx_div_bn": v[1],
                          "spx_bb_prelim": float(v[2]),
                          "spx_bb_manual": 1.0 if v[3] == "manual_press" else 0.0,
                          "spx_bb_xchecked": man_meta[k][1] if k in man_meta else np.nan,
                          "spx_bb_pit_manual": man_meta[k][0] if k in man_meta else pd.NaT,
                          "spx_bb_src_snap": v[3]}
                         for k, v in q_map.items()])
           .set_index("qdate").sort_index()) if q_map else pd.DataFrame()
    adf = (pd.DataFrame([{"year": k, "spx_bb_ann_bn": v[0]} for k, v in a_map.items()])
           .set_index("year").sort_index()) if a_map else pd.DataFrame()
    if len(qdf):
        land(f"[spx_bb] OK wayback-DIKIS (dogrudan spglobal 403/WAF): {len(used)} snapshot kullanildi, "
             f"{len(failed)} dusttu ({','.join(t[:8] for t in failed) if failed else '-'}); CEYREKLIK "
             f"buyback $mlr {qdf.index.min():%Y-%m} -> {qdf.index.max():%Y-%m} (n={len(qdf)}, "
             f"son-snapshot-kazanir, prelim bayrakli) + dividend $mlr; yillik referans n={len(adf)}; "
             f"pub-lag ceyrek-sonu +{PUB_LAG['spx_bb']}g")
        if man_used:
            qlab = lambda q: f"{q.year}Q{(q.month - 1)//3 + 1}"  # noqa: E731
            land(f"[spx_bb] OK MANUEL basin-bulten katmani ({MANUAL_BB_CSV.name}): "
                 f"{len(man_used)} ceyrek eklendi ({', '.join(qlab(q) for q in man_used)}; "
                 f"src=manual_press, panel-prelim=1, pit=bulten-tarihi), "
                 f"{len(man_skip)} satir xlsx-tarafindan-ezildi "
                 f"({', '.join(qlab(q) for q in man_skip) if man_skip else '-'}); "
                 f"xlsx KAZANIR kurali: wayback yetisince manuel otomatik devre-disi")
        elif len(man):
            land(f"[spx_bb] MANUEL katman: {len(man)} satir okundu ama hepsi xlsx-tarafindan-ezildi")
    else:
        land(f"[spx_bb] FAIL: hicbir snapshot parse edilemedi ({len(failed)} deneme)")
    return qdf, adf, failed


# ─────────────────────────── panel insasi ───────────────────────────

def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    key = fred_key()
    gdp = fred_obs("GDP", key)                       # NGDP SAAR $mlr, ceyrek-baslangic indeksli
    land(f"[fred] OK GDP (NGDP SAAR $mlr) {gdp.index.min():%Y-%m} -> {gdp.index.max():%Y-%m} "
         f"(%NGDP normalizasyonu icin)")

    net_p = CACHE / "net_equity_supply.parquet"
    if not net_p.exists():
        raise RuntimeError("net_equity_supply.parquet yok — once screen/fetch_net_equity_supply.py kos")
    net = pd.read_parquet(net_p)
    land(f"[z1] OK NET referans net_equity_supply.parquet'ten (NFC transactions SAAR): "
         f"{net.index.min():%Y-%m} -> {net.index.max():%Y-%m}; BILESEN AYRISTIRMASI Z.1/FRED'DE YOK "
         f"(bulk-zip 574 dosya + FRED search tarandi; Fed yalniz NET yayinliyor) -> kapsam boslugu")

    ritter_m = fetch_ritter_monthly()
    spac_a = fetch_ritter_spac()
    sifma_a, sifma_m = fetch_sifma()
    spx_q, spx_a, _ = fetch_spx_buybacks()

    # ── ceyreklik iskelet ──
    pieces = []
    if len(ritter_m):
        g = ritter_m.copy()
        g["qdate"] = [qstart(d) for d in g.index]
        w = g["ritter_ipo_n_net"].fillna(0.0)
        # min_count=1: 1975-oncesi net-adet NaN'lari sahte-0'a donmesin
        _s = lambda s: s.sum(min_count=1)  # noqa: E731
        agg = g.groupby("qdate").agg(
            ritter_ipo_n_gross=("ritter_ipo_n_gross", _s),
            ritter_ipo_n_net=("ritter_ipo_n_net", _s),
            ritter_ipo_nonop_n=("ritter_ipo_nonop_n", _s),
        )
        # adet-agirlikli ort ilk-gun getiri + hotness (agirlik = net adet)
        g["_w"] = w
        for src, dst in (("ritter_ipo_fdret", "ritter_ipo_fdret_w"),
                         ("ritter_ipo_hot_pct", "ritter_ipo_hot_w")):
            num = (g[src] * g["_w"]).groupby(g["qdate"]).sum()
            den = g["_w"].where(g[src].notna(), 0.0).groupby(g["qdate"]).sum()
            agg[dst] = num / den.replace(0.0, np.nan)
        pieces.append(agg)
    if len(spx_q):
        pieces.append(spx_q.drop(columns=["spx_bb_src_snap", "spx_bb_pit_manual"]))

    idx_min = min(p.index.min() for p in pieces)
    idx_max = max(max(p.index.max() for p in pieces), net.index.max())
    qidx = pd.date_range(idx_min, idx_max, freq="QS")
    panel = pd.DataFrame(index=qidx)
    panel.index.name = "qdate"
    for p in pieces:
        panel = panel.join(p, how="left")

    # yillik serileri ceyreklere yay (_ay = annual-yayilmis; deger o YILIN TOPLAMI, 4 ceyrekte ayni)
    years = pd.Series(panel.index.year, index=panel.index)
    for adf in (sifma_a.add_suffix("_ay") if len(sifma_a) else pd.DataFrame(),
                spac_a if len(spac_a) else pd.DataFrame(),
                spx_a.add_suffix("_ay") if len(spx_a) else pd.DataFrame()):
        for c in adf.columns:
            panel[c] = years.map(adf[c]).astype(float)

    # Z.1 NET referans + NGDP
    panel = panel.join(net[["nfc_saar_mn", "ratio4q_nfc_pct", "z10y_nfc"]]
                       .rename(columns={"nfc_saar_mn": "z1_net_nfc_saar_mn",
                                        "ratio4q_nfc_pct": "z1_ratio4q_nfc_pct",
                                        "z10y_nfc": "z1_z10y_nfc"}), how="left")
    panel["ngdp_saar_bn"] = gdp.reindex(panel.index)
    gdp4 = panel["ngdp_saar_bn"].rolling(4, min_periods=4).mean()

    # ── turevler ──
    # spx buyback: SAAR DEGIL -> 4Q TOPLAM / yillik NGDP
    if "spx_bb_bn" in panel:
        panel["spx_bb_4q_bn"] = panel["spx_bb_bn"].rolling(4, min_periods=4).sum()
        panel["ratio4q_spx_bb_pct"] = 100.0 * panel["spx_bb_4q_bn"] / gdp4
    # sifma yillik akislar: yil / yil-NGDP (gdp4 ~ yil ortalamasi)
    for c, rc in (("sifma_total_equity_bn_ay", "ratio_ann_sifma_total_pct"),
                  ("sifma_ipo_true_bn_ay", "ratio_ann_sifma_ipo_true_pct"),
                  ("sifma_secondary_bn_ay", "ratio_ann_sifma_secondary_pct")):
        if c in panel:
            panel[rc] = 100.0 * panel[c] / gdp4
    # ritter adetler: 4Q toplam
    if "ritter_ipo_n_net" in panel:
        panel["ritter_ipo_n_net_4q"] = panel["ritter_ipo_n_net"].rolling(4, min_periods=4).sum()
        panel["ritter_ipo_n_gross_4q"] = panel["ritter_ipo_n_gross"].rolling(4, min_periods=4).sum()
    # spac yillik adet zaten _ay; proceeds %NGDP
    if "spac_proceeds_bn_ay" in panel:
        panel["ratio_ann_spac_proceeds_pct"] = 100.0 * panel["spac_proceeds_bn_ay"] / gdp4

    # z10y'ler (40 ceyrek, min 20)
    zsrc = ["ratio4q_spx_bb_pct", "ratio_ann_sifma_total_pct", "ratio_ann_sifma_ipo_true_pct",
            "ratio_ann_sifma_secondary_pct", "ritter_ipo_n_net_4q", "ritter_ipo_n_gross_4q",
            "spac_n_ay", "ratio_ann_spac_proceeds_pct"]
    for c in zsrc:
        if c in panel:
            s = panel[c]
            panel[f"z10y_{c}"] = (s - s.rolling(40, min_periods=20).mean()) / s.rolling(40, min_periods=20).std()

    # ── PIT kolonlari (kaynak basina) ──
    qe = pd.Series([quarter_end(q) for q in panel.index], index=panel.index)
    ye = pd.Series([pd.Timestamp(year=y, month=12, day=31) for y in panel.index.year], index=panel.index)
    panel["pit_date_z1"] = pd.Series(panel.index, index=panel.index) + pd.Timedelta(days=PUB_LAG["z1"])
    panel["pit_date_ritter"] = qe + pd.Timedelta(days=PUB_LAG["ritter"])
    panel["pit_date_spx_bb"] = qe + pd.Timedelta(days=PUB_LAG["spx_bb"])
    # manuel basin-bulten satirlari: pit = bultenin GERCEK yayin tarihi (genel +90g kuralini ezer)
    if len(spx_q) and "spx_bb_pit_manual" in spx_q:
        mp = spx_q["spx_bb_pit_manual"].dropna()
        common = mp.index.intersection(panel.index)
        panel.loc[common, "pit_date_spx_bb"] = mp.loc[common]
    panel["pit_date_sifma"] = ye + pd.Timedelta(days=PUB_LAG["sifma"])
    panel["pit_date_spac"] = ye + pd.Timedelta(days=PUB_LAG["spac"])

    # ── aylik-HAM panel ──
    mon = ritter_m.copy() if len(ritter_m) else pd.DataFrame()
    if len(sifma_m):
        mon = mon.join(sifma_m, how="outer") if len(mon) else sifma_m
    if len(mon):
        me = mon.index + pd.offsets.MonthEnd(0)
        mon["pit_date_ritter"] = me + pd.Timedelta(days=PUB_LAG["ritter"])
        mon["pit_date_sifma"] = me + pd.Timedelta(days=PUB_LAG["sifma"])
        mon.index.name = "mdate"
    return panel, mon


# ─────────────────────────── rapor ───────────────────────────

def signal_time_spearman(panel: pd.DataFrame) -> list[str]:
    """Sahte-trend dersi: her turetilmis sinyal icin sinyal~takvim-zamani Spearman."""
    lines = []
    cols = [c for c in panel.columns
            if c.startswith(("ratio", "z10y_", "ritter_ipo_n_", "spac_n")) and panel[c].notna().sum() > 20]
    for c in sorted(cols):
        s = panel[c].dropna()
        t = pd.Series(np.arange(len(s), dtype=float), index=s.index)
        rho = float(s.corr(t, method="spearman"))
        warn = "  <-- SEKULER-TREND riski (|rho|>0.5): anlamliligi yalniz z/detrend formda say" \
            if abs(rho) > 0.5 else ""
        lines.append(f"    {c:<38} rho_zaman {rho:+.3f}  (n={len(s)}){warn}")
    return lines


def write_panel_report(panel: pd.DataFrame, mon: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    L: list[str] = []
    L.append("=" * 112)
    L.append("  HISSE-ARZI BILESEN PANELI — Constan ayristirmasi (IPO / ikincil / SPAC / geri-alim)")
    L.append("  BETIMSEL boru: icerik testi (candidate fazi) YAPILMADI; NET'in on-kayitli testi zaten")
    L.append("  KALICILIK-FAIL vermisti (net_supply_report.txt) — bilesenler ayri test ister.")
    L.append("=" * 112)
    L.append("")
    L.append("  KAYNAK LANDING RAPORU:")
    for ln in LANDINGS:
        L.append("  " + ln)
    L.append("")
    L.append("  KOLON -> CONSTAN-BILESENI ESLEMESI:")
    L.append("    IPO            : ritter_ipo_n_net/gross (AYLIK adet 1960+), sifma_ipo_all/true_bn_ay")
    L.append("                     (YILLIK $mlr 1990-2013), spx yok")
    L.append("    ikincil/ek     : sifma_secondary_bn_ay (YILLIK $mlr 1990-2013) — 2014+ BOSLUK")
    L.append("    donusturulebilir: ALINAMADI (legacy SIFMA dosyasinda yok; modern form-kapili) — BOSLUK")
    L.append("    SPAC           : spac_n_ay + spac_proceeds_bn_ay (YILLIK 1990-2025, Ritter 15b);")
    L.append("                     ritter_ipo_nonop_n (AYLIK brut-net farki, SPAC+CEF+penny KARISIK proxy)")
    L.append("    geri-alim      : spx_bb_bn (CEYREKLIK $mlr ~2008Q3+, S&P 500 evreni; SP500/NGDP ~%85-")
    L.append("                     100 oldugu icin makro-proxy kabul edilebilir, TUM-piyasa DEGIL).")
    L.append("                     2024Q4-sonrasi = MANUEL basin-bulten katmani (spx_bb_manual=1,")
    L.append("                     prelim=1, spx_bb_xchecked=capraz-teyit; xlsx yetisince otomatik ezer);")
    L.append("                     ceyreklik kontrol: python screen/fetch_supply_components.py --check-buyback")
    L.append("    NET referans   : z1_net_nfc_saar_mn + z1_ratio4q_nfc_pct (onceki faz, Z.1)")
    L.append("")
    L.append("  SAHTE-TREND TANISI (sinyal~zaman Spearman; lesson #1):")
    L.extend(signal_time_spearman(panel))
    L.append("")
    L.append("  SON 8 CEYREK (ana bilesenler):")
    show = [c for c in ("ritter_ipo_n_net", "ritter_ipo_nonop_n", "spac_n_ay", "spx_bb_bn",
                        "ratio4q_spx_bb_pct", "sifma_secondary_bn_ay", "z1_ratio4q_nfc_pct") if c in panel]
    hdr = f"  {'ceyrek':<9}" + "".join(f"{c[:16]:>17}" for c in show)
    L.append(hdr)
    tail = panel.dropna(subset=["ritter_ipo_n_net"], how="all").tail(8)
    for q, row in tail.iterrows():
        lab = f"{q.year}Q{(q.month - 1)//3 + 1}"
        cells = "".join(f"{row[c]:>17.2f}" if np.isfinite(row[c]) else f"{'-':>17}" for c in show)
        L.append(f"  {lab:<9}" + cells)
    L.append("")
    L.append("  PIT kurallari: z1 q-bas+165g | sifma yil-sonu+30g | ritter q-sonu+7g (bilgi-bazli; dosya")
    L.append("  yillik guncellenir) | spac yil-sonu+7g | spx_bb q-sonu+90g. Kaynak-basina pit_date_* kolonu.")
    L.append("  ISTISNA: manuel spx_bb satirlarinda pit_date_spx_bb = basin-bulteninin GERCEK yayin tarihi.")
    L.append("=" * 112)
    (OUT / "supply_components_panel.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  panel-raporu -> {OUT / 'supply_components_panel.txt'}")


# ─────────────────────────── --check-buyback: ceyreklik tek-komut kontrol ───────────────────────────

def check_buyback() -> int:
    """Yeni buyback ceyregi var mi? 3 kaynak sirayla: (1) Wayback CDX yeni xlsx snapshot,
    (2) press.spglobal.com bulten listesi, (3) yoksa 'manuel giris bekliyor' uyarisi.
    Hicbir sey YAZMAZ — yalniz rapor eder (idempotent kontrol)."""
    print("=" * 100)
    print("  CHECK-BUYBACK: S&P 500 buyback serisinde yeni ceyrek var mi?")
    print("=" * 100)

    # mevcut kapsam: xlsx-dikis ceyrekleri (raw cache'ten yeniden taranmaz; parquet referans) + manuel
    pq = CACHE / "supply_components.parquet"
    last_q, last_src = None, "?"
    if pq.exists():
        panel = pd.read_parquet(pq)
        s = panel["spx_bb_bn"].dropna()
        if len(s):
            last_q = s.index.max()
            if "spx_bb_manual" in panel:
                last_src = "manual_press" if panel.loc[last_q, "spx_bb_manual"] == 1.0 else "wayback_xlsx"
    man = load_manual_buybacks()
    if last_q is None and len(man):
        last_q, last_src = man.index.max(), "manual_press"
    if last_q is None:
        print("  HATA: ne parquet ne manuel CSV var — once fetch'i kos")
        return 1
    next_q = last_q + pd.offsets.QuarterBegin(startingMonth=1)
    nlab = f"{next_q.year}Q{(next_q.month - 1)//3 + 1}"
    llab = f"{last_q.year}Q{(last_q.month - 1)//3 + 1}"
    print(f"  mevcut son ceyrek : {llab} (kaynak: {last_src})")
    print(f"  aranan ceyrek     : {nlab}")
    found = False

    # 1) Wayback CDX: hardcode listesindeki son snapshot'tan sonra yeni snapshot olustu mu?
    last_snap = max(ts for ts, _ in SPX_BB_SNAPSHOTS)
    try:
        r = requests.get("https://web.archive.org/cdx/search/cdx",
                         params={"url": "spglobal.com/spdji/en/documents/additional-material/"
                                        "sp-500-buyback.xlsx",
                                 "from": str(int(last_snap[:8]) + 1), "output": "json",
                                 "filter": "statuscode:200"},
                         headers=UA, timeout=60)
        rows = r.json()[1:] if r.status_code == 200 and r.text.strip() else []
        new_ts = sorted({row[1] for row in rows if row[1] > last_snap})
        if new_ts:
            found = True
            print(f"  [wayback] YENI snapshot var: {', '.join(t[:8] for t in new_ts)}")
            print(f"            -> SPX_BB_SNAPSHOTS listesine ekle (ts, spglobal-url) + fetch'i yeniden kos;")
            print(f"               resmi xlsx manuel satirlari otomatik ezer (en temiz yol)")
        else:
            print(f"  [wayback] yeni snapshot yok (son: {last_snap[:8]})")
    except Exception as e:
        print(f"  [wayback] CDX sorgusu basarisiz ({type(e).__name__}) — atlandi")

    # 2) press.spglobal.com bulten listesi (bot-acik; 2026-06-13 dogrulandi)
    try:
        r = requests.get("https://press.spglobal.com/index.php?s=2429&l=100", headers=UA, timeout=60)
        titles = set(re.findall(r"Q([1-4])[\s\-+](20\d{2})[\s\-+]Buybacks", r.text, flags=re.I)) \
            if r.status_code == 200 else set()
        qs = sorted(pd.Timestamp(year=int(y), month=3 * int(qn) - 2, day=1)
                    for qn, y in titles)
        new_qs = [q for q in qs if q > last_q]
        if new_qs:
            found = True
            for q in new_qs:
                print(f"  [bulten ] YENI bulten basligi: {q.year}Q{(q.month-1)//3+1} Buybacks "
                      f"(press.spglobal.com / prnewswire'da ara)")
            print(f"            -> rakami {MANUAL_BB_CSV} dosyasina kaynak-URL'siyle ekle + fetch'i kos")
        elif qs:
            print(f"  [bulten ] yeni bulten yok (listede en yeni: "
                  f"{max(qs).year}Q{(max(qs).month-1)//3+1})")
        else:
            print("  [bulten ] listede buyback basligi bulunamadi (sayfa formati degismis olabilir)")
    except Exception as e:
        print(f"  [bulten ] press.spglobal.com sorgusu basarisiz ({type(e).__name__}) — atlandi")

    # 3) sonuc
    print()
    if not found:
        # bulten tipik gecikmesi: ceyrek-sonu +80-95g (Q1->Haziran-sonu, Q3->Aralik-ortasi)
        due = quarter_end(next_q) + pd.Timedelta(days=80)
        today = pd.Timestamp.today().normalize()
        if today >= due:
            print(f"  SONUC: manuel giris bekliyor: {nlab} (bulten vadesi ~{due:%Y-%m-%d} gecti;")
            print(f"         prnewswire'da 'S&P 500 {nlab[-2:]} {nlab[:4]} Buybacks' ara, bulunca")
            print(f"         {MANUAL_BB_CSV.name} dosyasina yaz + fetch'i yeniden kos)")
        else:
            print(f"  SONUC: yeni ceyrek yok; {nlab} bulteni ~{due:%Y-%m-%d} civari beklenir")
    else:
        print("  SONUC: yeni veri kaynagi bulundu — yukaridaki adimi uygula")
    return 0


def _rebuild_parquet() -> None:
    """Panel'i yeniden kur + yaz (auto-buyback yeni satir yazdiktan sonra cagrilir)."""
    RAW.mkdir(parents=True, exist_ok=True)
    panel, mon = build()
    CACHE.mkdir(parents=True, exist_ok=True)
    pq = CACHE / "supply_components.parquet"
    panel.to_parquet(pq)
    print(f"  -> parquet yeniden kuruldu: {pq}  ({len(panel)} ceyrek, {panel.shape[1]} kolon)")
    if len(mon):
        pm = CACHE / "supply_components_monthly.parquet"
        mon.to_parquet(pm)
    write_panel_report(panel, mon)


def main() -> int:
    if "--check-buyback" in sys.argv[1:]:
        return check_buyback()
    if "--auto-buyback" in sys.argv[1:]:
        res = auto_pull_buyback(write=True)
        # yalniz GERCEKTEN yeni satir yazildiysa parquet'i yeniden kur (idempotent: guncel/skip ise dokunma)
        if res.get("status") == "written":
            print()
            _rebuild_parquet()
        # cikis kodu: belirsiz-parse (verify_failed) loud-fail; guncel/written/skip = temiz exit 0
        return 1 if res.get("status") == "verify_failed" else 0
    print("=" * 100)
    print("  FETCH: hisse-arzi BILESENLERI (Z.1-net ref + SIFMA + Ritter IPO/SPAC + S&P buyback)")
    print("=" * 100)
    RAW.mkdir(parents=True, exist_ok=True)
    panel, mon = build()
    CACHE.mkdir(parents=True, exist_ok=True)
    pq = CACHE / "supply_components.parquet"
    panel.to_parquet(pq)
    print(f"  -> {pq}  ({len(panel)} ceyrek, {panel.shape[1]} kolon)")
    if len(mon):
        pm = CACHE / "supply_components_monthly.parquet"
        mon.to_parquet(pm)
        print(f"  -> {pm}  ({len(mon)} ay, {mon.shape[1]} kolon)")
    write_panel_report(panel, mon)
    print()
    print("  SAHTE-TREND TANISI (sinyal~zaman Spearman):")
    for ln in signal_time_spearman(panel):
        print(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
