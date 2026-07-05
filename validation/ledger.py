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


def append_call(record: dict) -> Path:
    p = ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {c: record.get(c) for c in _COLS}
    df = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=_COLS)
    for c in _COLS:                                  # şema göçü (eski parquet'te eksik kolon → NaN)
        if c not in df.columns:
            df[c] = pd.NA
    if not df.empty:
        df = df[df["as_of"].astype(str) != str(rec["as_of"])]
    df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)[_COLS]
    df = df.sort_values("as_of").reset_index(drop=True)
    df.to_parquet(p)
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
    günleri işaretleyemez) → önce canlı yfinance (^GSPC/^NDX, son ~2y), ağ koparsa frozen fallback."""
    sym = {"SPX": "^GSPC", "NDX": "^NDX"}.get(asset, "^GSPC")
    try:
        import yfinance as yf
        h = yf.Ticker(sym).history(period="2y")["Close"]
        h.index = pd.to_datetime(h.index).tz_localize(None)
        h = h.dropna().sort_index()
        if len(h):
            return h
    except Exception:
        pass
    fp = ROOT / "spine" / "frozen" / "prices.parquet"     # fallback (ağ yok) — yalnız backtest-tarihi kapsar
    if fp.exists():
        px = pd.read_parquet(fp)
        if asset in px.columns:
            s = px[asset].dropna()
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
    raise RuntimeError("endeks kapanışı alınamadı (yfinance + frozen ikisi de yok)")


def mark_to_market(asset: str = REF_ASSET, closes: pd.Series | None = None) -> pd.DataFrame:
    """Her satırın signal_pnl'ini doldur: position_target × (endeks as_of→ertesi-gün getirisi).
    Ertesi kapanış henüz yoksa (bugünkü çağrı) NaN kalır (beklemede). expression_drag = signal − expression."""
    df = load_ledger()
    if df.empty:
        return df
    cl = closes if closes is not None else _index_closes(asset)
    # H4: fiyat-yaşı kontrolü — closes CANLI çekildiyse ve son fiyat >2 işlem günü bayatsa ALARM + STALE damga
    # (yfinance düşüp frozen-fallback'e düşerse sessiz çöp signal_pnl birikimi YOK; satır STALE işaretlenir).
    src_stale_bd = 0
    if closes is None:
        try:
            today = pd.Timestamp(datetime.now(timezone.utc).date())
            src_stale_bd = max(0, len(pd.bdate_range(cl.index.max().normalize(), today)) - 1)
            if src_stale_bd > 2:
                print(f"  ⚠ LEDGER ALARM: fiyat serisi {src_stale_bd} işlem günü bayat (son {cl.index.max().date()}) "
                      f"→ yeni signal_pnl GÜVENİLMEZ, satırlar STALE damgalı.")
        except Exception as e:   # EQ-1: sessiz yutma kaldırıldı — bekçi körse bunu GÖRÜNÜR söyle
            print(f"  ⚠ LEDGER: fiyat-yaşı kontrolü hesaplanamadı ({type(e).__name__}: {e}) — H4 bekçisi bu koşuda KÖR")
    clmax = cl.index.max()
    fwd1 = cl.pct_change().shift(-1)                 # pozisyon[t] → close[t]→close[t+1] getirisi (look-ahead-free)
    # M1 (ops-fix 2026-07-06): KALICI-GAP (kurtarılamaz) günleri forward-skorlamada AÇIKÇA atla.
    # Bu 8 gün (2026-06-23..07-02, makine-KAPALI) için canlı opsiyon-zinciri snapshot'ı YOK → uydurma YASAK.
    # Örneklemi sessizce küçültmek yerine: satır varsa signal_pnl=None + GÖRÜNÜR not (sayıya katılmaz).
    gap_dates = load_permanent_gaps()
    n_gap_skipped = 0
    sig, prc_stale = [], []
    for _, r in df.iterrows():
        a = pd.Timestamp(str(r["as_of"]))
        if a.strftime("%Y-%m-%d") in gap_dates:      # KALICI-GAP → skorlanmaz, uydurulmaz, sayılmaz
            sig.append(None)
            prc_stale.append(False)                  # bayat-kaynak DEĞİL — ayrı, dürüst 'kalıcı-boşluk' kategorisi
            n_gap_skipped += 1
            continue
        pos = r.get("position_target")
        idx = cl.index.searchsorted(a, side="right") - 1
        has_next = (0 <= idx < len(fwd1)) and pd.notna(fwd1.iloc[idx])
        rr = float(fwd1.iloc[idx]) if has_next else None
        sig.append(None if (rr is None or pd.isna(pos)) else round(float(pos) * rr, 6))
        # bu satır işaretlenemedi VE kaynak bayat VE as_of yakın → bayat-kaynak yüzünden (normal bugün-beklemede DEĞİL)
        prc_stale.append(bool((not has_next) and src_stale_bd > 2 and a <= clmax + pd.Timedelta(days=10)))
    if gap_dates:                                    # her koşuda GÖRÜNÜR not (defterde satır olsa da olmasa da)
        print(f"  forward-score: {len(gap_dates)} gün KALICI-GAP atlandı (2026-06-23..07-02, machine-off) "
              f"— defterde {n_gap_skipped} satır eşleşti, uydurulmadı/sayılmadı")
    df["signal_pnl"] = sig
    df["price_stale"] = prc_stale
    df["expression_drag"] = [
        (None if (pd.isna(s) or pd.isna(e)) else round(float(s) - float(e), 6))
        for s, e in zip(df["signal_pnl"], df["expression_pnl"])
    ]
    df.to_parquet(ledger_path())
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
    df.to_parquet(ledger_path())
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
