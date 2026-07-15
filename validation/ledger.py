"""
validation/ledger — forward-validation defteri (CANLI OOS kaydı). run.py her CURRENT çağrıyı ekler
(as_of dedup → re-run günceller, çift-satır yok). Yalnız pandas+parquet; ağ opsiyonel. STALE çağrı EKLENMEZ
(bayat-veri dersi: yalnız güncel-call kaydı). output/forward_ledger.parquet.

GÖREV 1 — SİNYAL-PnL vs İFADE-PnL AYRIMI. Backtest Sharpe'ı (1.64/1.77) ENDEKS pozisyonuna ait; gerçek para
OPSİYON ifadesine giriyor → iki farklı nesne. Forward bozulduğunda "sinyal mi çürüdü, ifade mi kanıyor"
ayrışsın diye iki ayrı seri:
  • signal_pnl     = position_target × endeks(SPX) ertesi-gün getirisi   (modelin VAADİ, NAV-oranı)
  • expression_pnl = gerçekleşen opsiyon PnL / NAV (komisyon+slippage dahil)  (GERÇEKLEŞEN, NAV-oranı)
  • expression_drag = signal_pnl − expression_pnl  (ifadenin sinyalden kaçırdığı; izleme serisi)
Her ikisi de NAV-oranı (aynı taban) → fark anlamlı. İfade-tarafı GERÇEK trade kapanınca record_expression ile
girilir; trade yoksa NaN (model 06-10 canlı → geçmiş forku yok, backfill edilemez = dürüst boşluk).
"""
from __future__ import annotations

import json                                # M1 (ops-fix 2026-07-06): KALICI-GAP sidecar okuyucu
from datetime import datetime, timezone   # EQ-1 (denetim 2026-07-04): H4 bekçisi import'suz NameError'la ölüydü
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# eski sinyal alanları + GÖREV 1 PnL ayrımı + GÖREV 6a modül attribution (geriye-uyumlu: eksik kolon → NaN)
_COLS = ["as_of", "computed_at", "model_tag", "call_status", "position_target", "direction",
         "size", "tide_dir", "tide_score", "active_overlays", "data_source",
         "signal_pnl", "expression_pnl", "expression_drag",
         "m9_score", "m5_score", "m2_score",                # GÖREV 6a: dominant modüllerin günlük skoru
         "price_stale"]                                     # H4: signal_pnl bayat-fiyat yüzünden işaretlenemedi

REF_ASSET = "SPX"                                    # sinyal-PnL referans endeksi (anchor asset)


def ledger_path() -> Path:
    return ROOT / "output" / "forward_ledger.parquet"


def gaps_path() -> Path:
    """M1 (ops-fix 2026-07-06): KALICI (kurtarılamaz) forward-ledger boşluk sidecar'ı."""
    return ROOT / "output" / "forward_ledger_gaps.json"


def load_permanent_gaps() -> set[str]:
    """KALICI-GAP tarihlerini sidecar'dan oku → skorlamada AÇIKÇA atlanacak gün kümesi.

    M1 (ops-fix 2026-07-06): 2026-06-22..07-04 makine-KAPALI penceresinde canlı opsiyon-zinciri
    snapshot'ı ALINMADI (gamma_spy/gamma_qqq cache'i 06-22 → 07-05 atlıyor) → o 8 işlem günü
    KURTARILAMAZ point-in-time. Bu satırlar ASLA uydurulmaz; forward-skorlama onları sessizce
    örneklemi küçültmek yerine GÖRÜNÜR biçimde atlar. Sidecar yoksa/bozuksa → boş küme (ama SÖYLE:
    bekçi körse bunu yut-ma). recoverable=true işaretli boşluklar atlanmaz (yalnız kalıcı olanlar)."""
    p = gaps_path()
    if not p.exists():
        return set()
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:   # bozuk sidecar'ı sessizce yutma — GÖRÜNÜR uyar, sonra güvenli-boş dön
        print(f"  ⚠ LEDGER: KALICI-GAP sidecar okunamadı ({type(e).__name__}: {e}) → boşluk atlaması BU KOŞUDA yok")
        return set()
    dates: set[str] = set()
    for g in (doc.get("gaps") or []):
        if g.get("recoverable") is False:                # yalnız KALICI (kurtarılamaz) boşluklar atlanır
            dates.update(str(d) for d in (g.get("dates") or []))
    return dates


def _atomic_to_parquet(df: pd.DataFrame, p: Path) -> None:
    """Denetim 07-11 P2: dei-ra'nin CANLI okudugu dosya yarim-yazimda bozulmasin — tmp+replace."""
    import os
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".", suffix=".tmp")
    os.close(fd)
    df.to_parquet(tmp)
    os.replace(tmp, str(p))


def _close_cache_path(asset: str) -> Path:
    """Son başarılı CANLI endeks kapanışları; tek ağ arızasında frozen'a geri düşmeyi önler."""
    return ROOT / "data" / "cache" / f"index_closes_{asset.lower()}.parquet"


def _clean_closes(raw, asset: str = REF_ASSET) -> pd.Series:
    """Ticker.history / yf.download çıktılarını aynı tz-naive Series biçimine getir."""
    if isinstance(raw, pd.DataFrame):
        if "Close" in raw.columns:
            raw = raw["Close"]
        elif isinstance(raw.columns, pd.MultiIndex) and "Close" in raw.columns.get_level_values(0):
            raw = raw.xs("Close", axis=1, level=0)
        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:, 0] if len(raw.columns) else pd.Series(dtype=float)
    s = pd.to_numeric(pd.Series(raw), errors="coerce").dropna()
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    return s[~s.index.duplicated(keep="last")].sort_index().rename(asset)


def append_call(record: dict) -> Path:
    p = ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {c: record.get(c) for c in _COLS}
    # Denetim 07-11 P1-B kuyrugu: price_stale mark_to_market tamamlanana dek FAIL-CLOSED (True) —
    # eskiden None kaliyordu; append sonrasi mark cokerse dei-ra None'i (falsy) temiz saniyordu.
    if rec.get("price_stale") is None:
        rec["price_stale"] = True
    df = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=_COLS)
    for c in _COLS:                                  # şema göçü (eski parquet'te eksik kolon → NaN)
        if c not in df.columns:
            df[c] = pd.NA
    if not df.empty:
        _old = df[df["as_of"].astype(str) == str(rec["as_of"])]
        if len(_old):
            # Denetim 07-11 P2 ([17]): ayni-gun re-run satiri SESSIZCE yeniden yaziyordu — dei-ra'nin
            # tukettigi deger degisir, iz kalmazdi. Davranis ayni (son kosu kazanir) ama BAGIRARAK.
            _op = _old.iloc[-1].get("position_target")
            if pd.notna(_op) and _op != rec.get("position_target"):
                print(f"  ⚠ LEDGER: {rec['as_of']} satiri YENIDEN yazildi — position_target "
                      f"{_op} -> {rec.get('position_target')} (gun-ici re-run; dei-ra onceki degeri okumus olabilir)")
        df = df[df["as_of"].astype(str) != str(rec["as_of"])]
    df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)[_COLS]
    df = df.sort_values("as_of").reset_index(drop=True)
    _atomic_to_parquet(df, p)
    return p


def load_ledger() -> pd.DataFrame:
    p = ledger_path()
    df = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=_COLS)
    for c in _COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[_COLS] if len(df.columns) else df


def _index_closes(asset: str = REF_ASSET) -> pd.Series:
    """Endeks günlük kapanışları. FORWARD ledger → CANLI gerekli (frozen prices 05-22'de biter, forward
    günleri işaretleyemez) → iki yfinance yolunu dene; ağ koparsa son başarılı CANLI cache,
    ancak o da yoksa frozen fallback. Tazelik kararı mark_to_market'te ayrıca fail-closed verilir."""
    sym = {"SPX": "^GSPC", "NDX": "^NDX"}.get(asset, "^GSPC")
    errors = []
    try:
        import yfinance as yf
        attempts = (
            ("Ticker.history", lambda: yf.Ticker(sym).history(period="2y")),
            ("download", lambda: yf.download(sym, period="2y", progress=False, auto_adjust=False)),
        )
        for label, fetch in attempts:
            try:
                h = _clean_closes(fetch(), asset)
                if not len(h):
                    raise RuntimeError("boş seri")
                cp = _close_cache_path(asset)
                cp.parent.mkdir(parents=True, exist_ok=True)
                _atomic_to_parquet(h.to_frame(), cp)
                return h
            except Exception as e:
                errors.append(f"{label}: {type(e).__name__}: {str(e)[:90]}")
    except Exception as e:
        errors.append(f"yfinance import: {type(e).__name__}: {str(e)[:90]}")

    cp = _close_cache_path(asset)
    if cp.exists():
        try:
            cached = pd.read_parquet(cp)
            h = _clean_closes(cached, asset)
            if len(h):
                print(f"  ⚠ LEDGER: canlı endeks fetch başarısız → son CANLI cache kullanılıyor "
                      f"(son {h.index.max().date()}; {' | '.join(errors)})")
                return h
        except Exception as e:
            errors.append(f"live-cache: {type(e).__name__}: {str(e)[:90]}")

    fp = ROOT / "spine" / "frozen" / "prices.parquet"     # fallback (ağ yok) — yalnız backtest-tarihi kapsar
    if fp.exists():
        px = pd.read_parquet(fp)
        if asset in px.columns:
            s = _clean_closes(px[asset], asset)
            print(f"  ⚠ LEDGER: canlı endeks + live-cache yok → FROZEN fallback "
                  f"(son {s.index.max().date()}; {' | '.join(errors)})")
            return s
    raise RuntimeError("endeks kapanışı alınamadı (yfinance + live-cache + frozen yok): "
                       + " | ".join(errors))


def mark_to_market(asset: str = REF_ASSET, closes: pd.Series | None = None) -> pd.DataFrame:
    """Her satırın signal_pnl'ini doldur: position_target × (endeks as_of→ertesi-gün getirisi).
    Ertesi kapanış henüz yoksa (bugünkü çağrı) NaN kalır (beklemede). expression_drag = signal − expression."""
    df = load_ledger()
    if df.empty:
        return df
    cl = closes if closes is not None else _index_closes(asset)
    src_stale_bd = 0
    if closes is None:
        # Denetim 07-11 P0: seans icinde yfinance gunluk tarihce BUGUNUN devam-eden barini icerir —
        # dunku satirin signal_pnl'i KISMI fiyatla FINAL gibi muhurlenip gun boyu kaliyordu (deira
        # paneli yanlis accrual; ertesi sabah sessiz degisim). Bugunun bari yalniz seans KAPANDIYSA
        # (>= ~17:00 ET, konservatif UTC-5) marka girer; oncesinde dusulur -> dun durustce BEKLEMEDE.
        try:
            _et = datetime.now(timezone.utc) - pd.Timedelta(hours=5)
            if _et.hour < 17:
                _today_et = pd.Timestamp(_et.date())
                _n_before = len(cl)
                cl = cl[cl.index.normalize() < _today_et]
                if len(cl) < _n_before:
                    print("  forward-score: bugunun DEVAM-EDEN bari marka alinmadi (seans acik) — "
                          "dunku satir kapanista skorlanacak (kismi-bar muhru yok)")
        except Exception as e:
            print(f"  ⚠ LEDGER: kismi-bar korumasi hesaplanamadi ({type(e).__name__}: {e})")
        # H4: fiyat-yaşı kontrolü — canlı kaynak >2 işlem günü bayatsa ALARM + STALE damga
        try:
            today = pd.Timestamp(datetime.now(timezone.utc).date())
            src_stale_bd = max(0, len(pd.bdate_range(cl.index.max().normalize(), today)) - 1)
            if src_stale_bd > 2:
                print(f"  ⚠ LEDGER ALARM: fiyat serisi {src_stale_bd} işlem günü bayat (son {cl.index.max().date()}) "
                      f"→ yeni signal_pnl GÜVENİLMEZ, satırlar STALE damgalı.")
                try:   # Denetim 07-11 P1: alarm print-only idi — push'a da cikar (best-effort)
                    import notify
                    notify.alert("EQUITY fiyat-kaynagi BAYAT",
                                 f"{src_stale_bd} isgunu (son {cl.index.max().date()}) — signal_pnl guvenilmez")
                except Exception:
                    pass
        except Exception as e:   # EQ-1: sessiz yutma kaldırıldı — bekçi körse bunu GÖRÜNÜR söyle
            print(f"  ⚠ LEDGER: fiyat-yaşı kontrolü hesaplanamadı ({type(e).__name__}: {e}) — H4 bekçisi bu koşuda KÖR")
    clmax = cl.index.max()
    fwd1 = cl.pct_change().shift(-1)                 # pozisyon[t] → close[t]→close[t+1] getirisi (look-ahead-free)
    # M1 (ops-fix 2026-07-06): KALICI-GAP (kurtarılamaz) günleri forward-skorlamada AÇIKÇA atla.
    # Bu 8 gün (2026-06-23..07-02, makine-KAPALI) için canlı opsiyon-zinciri snapshot'ı YOK → uydurma YASAK.
    # Örneklemi sessizce küçültmek yerine: satır varsa signal_pnl=None + GÖRÜNÜR not (sayıya katılmaz).
    gap_dates = load_permanent_gaps()
    n_gap_skipped = 0
    n_ghost = 0
    _trade_days = set(cl.index.normalize())
    _prev_sig = df["signal_pnl"] if "signal_pnl" in df.columns else pd.Series([None] * len(df))
    _keep_before = clmax - pd.Timedelta(days=21)
    sig, prc_stale = [], []
    for _i, r in df.iterrows():
        a = pd.Timestamp(str(r["as_of"]))
        if a.strftime("%Y-%m-%d") in gap_dates:      # KALICI-GAP → skorlanmaz, uydurulmaz, sayılmaz
            sig.append(None)
            prc_stale.append(False)                  # bayat-kaynak DEĞİL — ayrı, dürüst 'kalıcı-boşluk' kategorisi
            n_gap_skipped += 1
            continue
        # Denetim 07-11 P2 ([8]/[13]): işlem-günü-olmayan as_of (tatil hayaleti, ör. Juneteenth
        # 2026-06-19) searchsorted'la ÖNCEKİ günün getirisine yapışıp ÇİFT sayılıyordu → skorlanmaz.
        if a <= clmax and a.normalize() not in _trade_days:
            sig.append(None)
            prc_stale.append(False)
            n_ghost += 1
            continue
        # Denetim 07-11 P2 ([12]): yıkıcı yeniden-yazım koruması — eski (>21g) ve DOLU signal_pnl
        # korunur; tek bozuk closes-cekimi tum tarihi yeniden yazamaz (yakin satirlar tazelenir).
        _old_v = _prev_sig.iloc[_i] if _i < len(_prev_sig) else None
        if a < _keep_before and _old_v is not None and pd.notna(_old_v):
            sig.append(round(float(_old_v), 6))
            prc_stale.append(False)
            continue
        pos = r.get("position_target")
        idx = cl.index.searchsorted(a, side="right") - 1
        has_next = (0 <= idx < len(fwd1)) and pd.notna(fwd1.iloc[idx])
        rr = float(fwd1.iloc[idx]) if has_next else None
        sig.append(None if (rr is None or pd.isna(pos)) else round(float(pos) * rr, 6))
        # Denetim 07-11 P1 ([1]/[4]/[5]): eski kosul `a <= clmax+10g` tavani yuzunden frozen-fallback'te
        # (clmax donuk) 10 gunden DERIN kesintide yeni satirlar False aliyordu = tam en kotu anda
        # fail-open. Yeni kural: kaynak bayatsa (>2 isgunu) ve satir işaretlenemediyse -> STALE, nokta.
        prc_stale.append(bool((not has_next) and src_stale_bd > 2))
    if gap_dates:                                    # her koşuda GÖRÜNÜR not (defterde satır olsa da olmasa da)
        print(f"  forward-score: {len(gap_dates)} gün KALICI-GAP atlandı (2026-06-23..07-02, machine-off) "
              f"— defterde {n_gap_skipped} satır eşleşti, uydurulmadı/sayılmadı")
    df["signal_pnl"] = sig
    df["price_stale"] = prc_stale
    df["expression_drag"] = [
        (None if (pd.isna(s) or pd.isna(e)) else round(float(s) - float(e), 6))
        for s, e in zip(df["signal_pnl"], df["expression_pnl"])
    ]
    _atomic_to_parquet(df, ledger_path())
    return df


def record_expression(as_of: str, expression_pnl_nav: float) -> pd.DataFrame:
    """GERÇEK opsiyon trade'i kapanınca: gerçekleşen PnL'i NAV-oranı olarak gir (komisyon+slippage DAHİL)."""
    df = load_ledger()
    m = df["as_of"].astype(str) == str(as_of)
    if not m.any():
        raise KeyError(f"ledger'da as_of {as_of} yok")
    df.loc[m, "expression_pnl"] = float(expression_pnl_nav)
    s = df.loc[m, "signal_pnl"]
    df.loc[m, "expression_drag"] = (None if s.isna().all()
                                    else float(s.iloc[0]) - float(expression_pnl_nav))
    _atomic_to_parquet(df, ledger_path())
    return df


def drag_summary() -> dict:
    """Kümülatif sinyal-PnL, ifade-PnL ve drag (brief satırı için). NaN'lar atlanır.
    M1: n_permanent_gap = KALICI-GAP gün sayısı (2026-06-23..07-02) — aggregate SESSİZCE küçülmesin diye
    açıkça raporlanır (örneklem eksikliği görünür kalır)."""
    df = load_ledger()
    sp = pd.to_numeric(df["signal_pnl"], errors="coerce").dropna()
    ep = pd.to_numeric(df["expression_pnl"], errors="coerce").dropna()
    dg = pd.to_numeric(df["expression_drag"], errors="coerce").dropna()
    return {"n_calls": int(len(df)), "n_signal_marked": int(len(sp)), "n_expression": int(len(ep)),
            "n_permanent_gap": int(len(load_permanent_gaps())),
            "cum_signal_pnl": round(float(sp.sum()), 5) if len(sp) else None,
            "cum_expression_pnl": round(float(ep.sum()), 5) if len(ep) else None,
            "cum_drag": round(float(dg.sum()), 5) if len(dg) else None}
