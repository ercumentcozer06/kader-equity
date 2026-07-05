"""
modules/santa_window — Constan KOŞULLU Noel penceresi: takvim/bağlam BANDI (pozisyon etkisi SIFIR).

GEREKÇE (Constan 101 Canon + candidate_santa_conditional + candidate_santa_incremental, 2026-06-13):
  • STANDALONE GERÇEK: 1 Kas'ta SPX YTD ≥ +10% → Kas-Ara getirisi 1928-2025'te n=43, ort +4.54%,
    isabet %88, min −2.10%; ÜÇ dönemde stabil (1928-79 %90 / 1980-2004 %92 / 2005+ %80), perm-p<0.001.
    Koşul GERÇEK bilgi taşır: koşulsuz yıllarda Kas-Ara ort SIFIR, isabet %49, min −22.7% (2008).
  • STACK'E KARŞI ABSORBED: nitelikli pencerelerde stack zaten ~tam-long (ort poz 0.816, tide long %86);
    boost'un dokunduğu 55 gün tam da overlay'lerin de-risk dediği günler (2024: froth-trim Aralık
    zayıflığını atlatıp B&H'a +4.2pp kazandırdı — boost o alfayı SİLERDİ). V1/V2/V3 hiçbiri FDR geçmedi;
    ters-yüz trim (V3) güvenilir ZARARLI (2020 rallisini yarıyor).
  → BİÇİM: OpEx-kapısı emsali ETİKET — pozisyon/deploy'a dokunmaz; brief çıktısına pencere-durumu bandı.
    Nitelikli pencere = "mevsimsel rüzgar ARKADA" bağlamı; nitelik tutmayan Kas-Ara = "Noel'e güvenme,
    kuyruk açık (min −22.7%)" uyarısı. DÜRÜST ETİKET: alfa-değil, veri-doğrulanmış takvim-bağlamı.

Veri: data/cache/spx_gspc_long.csv (^GSPC günlük 1927+); pencere aylarında bayatsa yfinance tazeler
(graceful — tazelenemezse durum UNKNOWN + sebep). Saf değerlendirme test edilebilir (closes parametreli).
"""
from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("kader_equity.santa_window")

ROOT = Path(__file__).resolve().parents[1]
CACHE_CSV = ROOT / "data" / "cache" / "spx_gspc_long.csv"

STATS_NOTE = "1928-2025: n=43, ort +4.54%, isabet %88, min -2.10% (perm-p<0.001; 3 donem stabil)"
NONQ_NOTE = "kosul tutmayan yillarda Kas-Ara ort ~0, isabet %49, min -22.7% (2008) - mevsimsel ruzgar YOK"


def _to_date(as_of) -> _dt.date:
    if isinstance(as_of, _dt.date) and not isinstance(as_of, _dt.datetime):
        return as_of
    return pd.Timestamp(as_of).date()


def load_spx_closes(*, refresh_if_stale_days: int = 7) -> pd.Series | None:
    """Uzun-tarih ^GSPC kapanışları; bayatsa yfinance ile tazele (graceful)."""
    s = None
    if CACHE_CSV.exists():
        try:
            df = pd.read_csv(CACHE_CSV, parse_dates=["Date"]).set_index("Date")
            s = df["Close"].dropna().sort_index()
        except Exception as e:
            log.warning("santa: cache okunamadi: %s", e)
    today = _dt.date.today()
    age = (today - s.index.max().date()).days if s is not None and len(s) else 999
    if age > refresh_if_stale_days:
        try:
            import yfinance as yf
            df = yf.Ticker("^GSPC").history(period="max", auto_adjust=False)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.index.name = "Date"
                CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
                df[["Close"]].to_csv(CACHE_CSV)
                s = df["Close"].dropna().sort_index()
        except Exception as e:
            log.warning("santa: yfinance tazeleme basarisiz (%s) — eldeki cache ile devam", e)
    return s


def evaluate_pure(as_of, closes: pd.Series, cfg: dict | None = None) -> dict:
    """Saf durum makinesi (test edilebilir). closes = ^GSPC günlük kapanış (yıl-içi tarihçe yeter).

    Durumlar: INACTIVE (Oca-Eki) | QUALIFYING_ACTIVE | NON_QUALIFYING | UNKNOWN (veri yetersiz/bayat).
    Nitelik: yılın İLK kapanışı → 1 Kas (veya sonraki ilk işgünü) kapanışı YTD ≥ eşik (varsayılan +10%).
    PIT-temiz: yalnız as_of'a kadarki kapanışlar kullanılır.
    """
    cfg = cfg or {}
    thr = float(cfg.get("ytd_threshold", 0.10))
    a = _to_date(as_of)
    out: dict = {"as_of": str(a), "window_months": [11, 12], "ytd_threshold": thr,
                 "stats_note": STATS_NOTE}

    if a.month not in (11, 12):
        # Ekim ortasından itibaren ön-izleme: 1 Kas'a giderken mevcut YTD (heads-up; nitelik DEĞİL)
        out["state"] = "INACTIVE"
        if a.month == 10 and a.day >= 15 and closes is not None:
            yr = closes[(closes.index >= f"{a.year}-01-01") & (closes.index.date <= a)]
            if len(yr) > 10:
                out["preview_ytd_pct"] = round(float(yr.iloc[-1] / yr.iloc[0] - 1) * 100, 1)
                out["preview_note"] = f"1 Kas yaklasiyor; su anki YTD %{out['preview_ytd_pct']:+.1f} (esik %+10)"
        return out

    if closes is None or len(closes) == 0:
        out["state"] = "UNKNOWN"
        out["reason"] = "SPX kapanis verisi yok"
        return out

    pit = closes[closes.index.date <= a]
    yr = pit[pit.index >= f"{a.year}-01-01"]
    nov = yr[yr.index >= f"{a.year}-11-01"]
    if len(yr) < 100 or nov.empty:
        out["state"] = "UNKNOWN"
        out["reason"] = f"yil-ici veri yetersiz (n={len(yr)}, kas-gunu={len(nov)})"
        return out
    # bayatlık: pencere içindeyiz ama son kapanış 7 günden eskiyse durum şüpheli → yine hesapla, etiketle
    staleness = (a - pit.index.max().date()).days
    ytd_nov1 = float(nov.iloc[0] / yr.iloc[0] - 1)
    qualifying = ytd_nov1 >= thr
    out.update({
        "state": "QUALIFYING_ACTIVE" if qualifying else "NON_QUALIFYING",
        "ytd_at_nov1_pct": round(ytd_nov1 * 100, 1),
        "window_end": f"{a.year}-12-31",
        "data_staleness_days": staleness,
        "note": (f"Constan kosullu Noel penceresi AKTIF (1 Kas YTD %{ytd_nov1*100:+.1f} >= %+10). "
                 + STATS_NOTE) if qualifying else
                (f"Kas-Ara penceresi NITELIKSIZ (1 Kas YTD %{ytd_nov1*100:+.1f} < %+10). " + NONQ_NOTE),
        "label": "alfa-degil; veri-dogrulanmis takvim-baglami (stack'e ABSORBED — pozisyon etkisi SIFIR)",
    })
    return out


def evaluate(as_of, cfg: dict | None = None) -> dict:
    """Canlı giriş noktası: veriyi yükle (gerekirse tazele) + saf değerlendirme."""
    cfg = cfg or {}
    a = _to_date(as_of)
    # veri yalnız Eki-Ara'da gerekir; diğer aylarda yükleme maliyetine girme
    closes = load_spx_closes() if a.month in (10, 11, 12) else (
        pd.read_csv(CACHE_CSV, parse_dates=["Date"]).set_index("Date")["Close"].dropna().sort_index()
        if CACHE_CSV.exists() else None)
    return evaluate_pure(a, closes, cfg)
