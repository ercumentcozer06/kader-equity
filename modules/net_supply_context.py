"""
modules/net_supply_context — Constan HİSSE NET-ARZI paneli: BETİMSEL bağlam (pozisyon etkisi SIFIR).

GEREKÇE (candidate_net_supply + adversarial denetim, 2026-06-13 — DÜRÜST ETİKET):
  • Yön-sinyali olarak KALICILIK-FAIL: tam-örneklem anlamlılığı sahte-trend vakası (sinyal~zaman
    Spearman −0.726; arz 1984 öncesi yapısal-pozitif, sonrası negatif). Geri-alımın var olduğu
    dönemde (1984+) forward içerik ~SIFIR; 2005+ işaret TERS. 2019+ tide-üstü 8 kural hepsi FDR-FAIL.
  → BU PANEL SİNYAL DEĞİL: Constan-grafiğinin bedava ikizi olarak BETİMSEL akış-arşivi + iki bilgi:
    (1) son okuma (rolling-4Q NFC net-arz %NGDP + z10y + tek-çeyrek dönüm haberleri);
    (2) UÇ-KUYRUK İZLEME BAYRAĞI: z10y ≥ +2.5 = ihraç-çılgınlığı imzası (2000, 2021 — ikisi de
        fwd-252g derin negatif; AMA 2020 toparlanma-ihraçları z+1.4-2.3 → +27..+40% = ılımlı yükseklik
        karışık, n=2) → kural DEĞİL, izleme bayrağı; ateşlenirse Emir'e görünür uyarı.

Veri: data/cache/net_equity_supply.parquet (screen/fetch_net_equity_supply.py üretir; Z.1/FRED,
çeyreklik, pub-lag +165g PIT kaydırması). Çeyreklik seri → bayatlık eşiği geniş (200 gün).
"""
from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("kader_equity.net_supply")

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "cache" / "net_equity_supply.parquet"
COMP_PARQUET = ROOT / "data" / "cache" / "supply_components.parquet"

# Bileşen fazı (2026-06-13, FINDING 25): PANEL-ONLY — hiçbir bileşen-bayrağı üçlü sınavı
# (2000-yakala + 2021-yakala + 2020-ateşleme) geçemedi; sayım-z 2000-kör (90'lar tabanı yutuyor),
# dolar-z niyet-kör (2009 kurtarma-ihraçlarında da ateşledi). TEK gerçek ayrışma: operasyonel-IPO
# sayımı (S3) 2020-temiz/2021-yakala — toplam-z'nin yanlış-ateşlemesini düzeltiyor ama 2000-FAIL +
# tek-dönem → bayrak/pozisyon kablosu YOK, yalnız betimsel görünürlük.
COMP_LABEL = ("bilesen-paneli BETIMSEL - bayrak-yukseltme ucgen-sinavi FAIL "
              "(sayim 2000-kor, dolar niyet-kor); S3 opco-IPO 2020-temiz/2021-yakala notu gecerli")

HONEST_LABEL = ("BETIMSEL panel - yon-sinyali DEGIL (1984+ icerik ~0, 2005+ isaret ters, "
                "tam-orneklem anlamlilik sahte-trend; tide-ustu 8 kural FDR-FAIL)")
TAIL_NOTE = ("ihrac-cilginligi imzasi (z>=+2.5): 2000 ve 2021'de fwd-252g derin negatif; "
             "n=2 + 2020 karsi-ornek -> kural degil IZLEME bayragi")


def _components(today: _dt.date) -> dict | None:
    """Bileşen özetleri (FINDING 25, PANEL-ONLY): kaynak-bazlı PIT kapısıyla son okumalar.
    Graceful — parquet yoksa/kolon eksikse None/kısmi döner."""
    if not COMP_PARQUET.exists():
        return None
    try:
        df = pd.read_parquet(COMP_PARQUET)
    except Exception as e:
        log.warning("net-supply components: parquet okunamadi: %s", e)
        return None
    out: dict = {"label": COMP_LABEL}

    def _last(cols: list[str], pit_col: str):
        try:
            sub = df.dropna(subset=[cols[0]])
            sub = sub[pd.to_datetime(sub[pit_col]).dt.date <= today]
            if sub.empty:
                return None, None
            return sub.iloc[-1], pd.Timestamp(sub.index[-1])
        except (KeyError, TypeError):
            return None, None

    r, qd = _last(["z10y_ritter_ipo_n_net_4q"], "pit_date_ritter")
    if r is not None:
        out["opco_ipo"] = {
            "quarter": f"{qd.year}Q{qd.quarter}",
            "n_4q": int(r["ritter_ipo_n_net_4q"]) if pd.notna(r.get("ritter_ipo_n_net_4q")) else None,
            "z10y": round(float(r["z10y_ritter_ipo_n_net_4q"]), 2),
            "note": "S3: 2020-temiz/2021-yakala ayristirici (2021Q4 z +3.26); 2000-kor — betimsel",
        }
    r, qd = _last(["spac_n_ay"], "pit_date_spac")
    if r is not None:
        out["spac"] = {"year": int(qd.year), "n": int(r["spac_n_ay"]),
                       "ref": "2021 zirvesi 613; 2024: 57"}
    r, qd = _last(["spx_bb_bn"], "pit_date_spx_bb")
    if r is not None:
        out["buyback"] = {
            "quarter": f"{qd.year}Q{qd.quarter}",
            "bn": round(float(r["spx_bb_bn"]), 1),
            "ratio4q_pct_ngdp": (round(float(r["ratio4q_spx_bb_pct"]), 2)
                                 if pd.notna(r.get("ratio4q_spx_bb_pct")) else None),
            "z10y": (round(float(r["z10y_ratio4q_spx_bb_pct"]), 2)
                     if pd.notna(r.get("z10y_ratio4q_spx_bb_pct")) else None),
            "prelim": bool(r.get("spx_bb_prelim", 0)),
            "universe": "S&P 500 (tum-piyasa degil; olcu-karisimi dikisi 2026-06-13 ONARILDI)",
        }
    return out if len(out) > 1 else None


def evaluate(cfg: dict | None = None, *, today: _dt.date | None = None) -> dict | None:
    """Son PIT-geçerli okuma + uç-kuyruk bayrağı. Parquet yoksa/bozuksa None (graceful)."""
    cfg = cfg or {}
    tail_thr = float(cfg.get("tail_z_threshold", 2.5))
    stale_days = int(cfg.get("max_staleness_days", 200))
    today = today or _dt.date.today()
    if not PARQUET.exists():
        log.warning("net-supply: parquet yok (%s) — once: python -m screen.fetch_net_equity_supply", PARQUET)
        return None
    try:
        df = pd.read_parquet(PARQUET)
    except Exception as e:
        log.warning("net-supply: parquet okunamadi: %s", e)
        return None
    if df.empty or "pit_date" not in df.columns:
        return None
    df = df.copy()
    df["pit_date"] = pd.to_datetime(df["pit_date"])
    valid = df[df["pit_date"].dt.date <= today]                  # PIT: bugune kadar yayimlanmis ceyrekler
    if valid.empty:
        return None
    row = valid.iloc[-1]
    ratio = row.get("ratio4q_nfc_pct")
    z = row.get("z10y_nfc")
    q_flow = row.get("nfc_saar_mn")
    pit_age = (today - row["pit_date"].date()).days
    # tek-çeyrek işaret dönümü: önceki çeyrekle kıyas (betimsel haber-değeri)
    flip_note = None
    if len(valid) >= 2 and q_flow is not None:
        prev_flow = valid.iloc[-2].get("nfc_saar_mn")
        if prev_flow is not None and (q_flow > 0) != (prev_flow > 0):
            flip_note = (f"tek-ceyrek NFC ihraci isaret degistirdi: {prev_flow/1000:+,.0f} -> "
                         f"{q_flow/1000:+,.0f} $bn SAAR")
    tail = z is not None and pd.notna(z) and float(z) >= tail_thr
    qd = pd.Timestamp(valid.index[-1])
    return {
        "quarter": f"{qd.year}Q{qd.quarter}",
        "components": _components(today),
        "pit_date": str(row["pit_date"].date()),
        "pit_age_days": pit_age,
        "stale": pit_age > stale_days,
        "ratio4q_nfc_pct_ngdp": round(float(ratio), 2) if pd.notna(ratio) else None,
        "z10y_nfc": round(float(z), 2) if pd.notna(z) else None,
        "single_q_flow_bn_saar": round(float(q_flow) / 1000, 1) if pd.notna(q_flow) else None,
        "flip_note": flip_note,
        "tail_flag": bool(tail),
        "tail_threshold": tail_thr,
        "tail_note": TAIL_NOTE if tail else None,
        "label": HONEST_LABEL,
        "panel_file": "output/net_supply_panel.txt",
    }
