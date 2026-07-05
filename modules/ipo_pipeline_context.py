"""
modules/ipo_pipeline_context — EDGAR S-1/F-1 boru-hattı bandı: Constan dev-arz tezinin
İLERİYE-BAKAN dedektörü (bağlam; pozisyon etkisi SIFIR).

GEREKÇE (2026-06-13, FINDING 26): dosyalama, fiyatlamadan AYLAR önce gelir = arz-dalgasının
öncü göstergesi. Tarihçe 2001+ (SB-2 yapısal-kırılması düzeltilmiş seri); mega-izleme listesi
(SpaceX/OpenAI/Anthropic/...) iki-kademeli ad-eşleşmesiyle (strict sayılır, loose yalnız etiket —
tarihsel 41/41 loose-isabet yanlış-pozitifti). İLK KOŞUDA ATEŞLENDİ: SpaceX S-1 2026-05-20,
06-03 değişikliğinde kayıt-tavanı $86.25 mlr ($135/hisse). DÜRÜST ETİKETLER: dosyalama=NİYET
(gerçekleşme değil); kayıt-tavanı != arz büyüklüğü; sayım-z 2021-tarzı SELİ ölçer, tek-dev-arz
DOLAR şokunu mega-izleme yakalar.

Veri: data/cache/ipo_pipeline.parquet + ipo_pipeline_live.json + mega_ipo_hits.json
(screen/fetch_ipo_pipeline.py üretir; canlı koşuda parquet bayatsa fetch'i çağırmaz — graceful
bayat-etiket; tazeleme run_daily/elle: python -m screen.fetch_ipo_pipeline).
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("kader_equity.ipo_pipeline")

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "cache" / "ipo_pipeline.parquet"
LIVE_JSON = ROOT / "data" / "cache" / "ipo_pipeline_live.json"
HITS_JSON = ROOT / "data" / "cache" / "mega_ipo_hits.json"

HONEST_LABEL = ("dosyalama = NIYET, gerceklesme degil; kayit-tavani != arz-buyuklugu; "
                "baglam-bandi — pozisyon etkisi SIFIR")


def evaluate(cfg: dict | None = None, *, today: _dt.date | None = None) -> dict | None:
    """Boru-hattı özeti: kapanmış-çeyrek z + son-90g sayım + mega isabetler. Graceful None."""
    cfg = cfg or {}
    stale_days = int(cfg.get("max_staleness_days", 10))
    today = today or _dt.date.today()
    if not (PARQUET.exists() and LIVE_JSON.exists()):
        log.warning("ipo pipeline: cache yok — once: python -m screen.fetch_ipo_pipeline")
        return None
    try:
        df = pd.read_parquet(PARQUET)
        live = json.loads(LIVE_JSON.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("ipo pipeline: okuma hatasi: %s", e)
        return None

    closed = df[(df["partial"] == 0) & (pd.to_datetime(df["pit_date"]).dt.date <= today)]
    if closed.empty:
        return None
    r = closed.iloc[-1]
    qd = pd.Timestamp(closed.index[-1])

    as_of = str(live.get("as_of", ""))[:10]
    try:
        live_age = (today - _dt.date.fromisoformat(as_of)).days
    except ValueError:
        live_age = 999

    mega = []
    if HITS_JSON.exists():
        try:
            hits = json.loads(HITS_JSON.read_text(encoding="utf-8"))
            seen = {}
            for h in hits.get("recent_120d", []):
                k = h.get("cik")
                amt = h.get("proposed_max_aggregate_usd")
                cur = seen.setdefault(k, {"company": h.get("company"),
                                          "watch_name": h.get("watch_name"),
                                          "first_filed": h.get("date_filed"),
                                          "n_filings": 0, "max_ceiling_usd": None})
                cur["n_filings"] += 1
                cur["first_filed"] = min(cur["first_filed"], h.get("date_filed"))
                if amt is not None and (cur["max_ceiling_usd"] is None or amt > cur["max_ceiling_usd"]):
                    cur["max_ceiling_usd"] = amt
            mega = list(seen.values())
        except Exception as e:
            log.warning("ipo pipeline: mega-hits okunamadi: %s", e)

    return {
        "last_closed_quarter": f"{qd.year}Q{qd.quarter}",
        "roll4q_filings": int(r["roll4q_adj"]) if pd.notna(r.get("roll4q_adj")) else None,
        "z10y": round(float(r["z10y_adj"]), 2) if pd.notna(r.get("z10y_adj")) else None,
        "last90d_new": (live.get("window_90d_counts") or {}).get("total_new"),
        "z90": round(float(live.get("z90_vs_40q")), 2) if live.get("z90_vs_40q") is not None else None,
        "live_as_of": as_of,
        "stale": live_age > stale_days,
        "mega_hits_120d": mega,
        "ref_2021": "2021 seli: S-1 443-779/çeyrek, z tepe +3.49 (2021Q2)",
        "label": HONEST_LABEL,
    }
