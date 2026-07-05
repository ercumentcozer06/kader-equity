"""
modules/supply_demand_balance — K1: Constan ARZ-TALEP DENGE göstergesi (BETİMSEL kadran).

GEREKÇE + DÜRÜST SINIR (2026-06-13, FINDING 27; Emir: "model arzı VE talebi takip etsin"):
  Tam panel: ARZ-z (NFC net-ihraç + SIFMA + SPAC + IPO-boru + buyback-NEGATİF) vs TALEP-z
  (NGDP-büyüme + buyback-POZİTİF + tide-yardımcı), hepsi %NGDP normalize.
  KRİTİK DÜRÜSTLÜK (adversarial onarım): denge-SEVİYESİ yön-AYRACI DEĞİL. PIT-dürüst (genişleyen-z)
  onarımdan sonra 2020-rali ile 2021-tepe AYRILMIYOR (ayraç +0.59→−0.08, işaret döner; her ikisi de
  güçlü-bearish okunur). Arz-froth GERÇEK ama denge seviyesi tek başına yön-sinyali olarak okunamaz.
  → Bu panel BETİMSEL takip (Constan grafiğinin bedava ikizi). Gerçek 2020/2021 ayracı = K2'nin ham
  tide-DÜZEY kapısıdır ([[supply_demand_derisk]]), denge-z değil. Panelde 'AYRAÇ' iddiası YAPILMAZ.

Veri: data/cache/supply_demand_balance.parquet (screen/fetch_supply_demand_balance.py; PIT-onarımlı).
"""
from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("kader_equity.sd_balance")

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "cache" / "supply_demand_balance.parquet"

HONEST_LABEL = ("BETIMSEL denge-kadrani (Constan ikizi) — yon-AYRACI DEGIL: PIT-durust z'de "
                "2020-rali/2021-tepe ayrilmaz; gercek ayrac K2 tide-duzey kapisinda")


def evaluate(cfg: dict | None = None, *, today: _dt.date | None = None) -> dict | None:
    """Son denge okuması + arz/talep bacak ayrıştırması (betimsel). Graceful None."""
    cfg = cfg or {}
    today = today or _dt.date.today()
    if not PARQUET.exists():
        log.warning("sd_balance: parquet yok — once: python -m screen.fetch_supply_demand_balance")
        return None
    try:
        df = pd.read_parquet(PARQUET)
    except Exception as e:
        log.warning("sd_balance: okunamadi: %s", e)
        return None
    if df.empty:
        return None
    r = df.iloc[-1]
    qd = pd.Timestamp(df.index[-1])

    def _g(col):
        v = r.get(col)
        return round(float(v), 2) if v is not None and pd.notna(v) else None

    bal = _g("net_supply_pressure")
    return {
        "quarter": f"{qd.year}Q{qd.quarter}",
        "net_supply_pressure": bal,          # arz_z − talep_z; >0 = arz baskısı (bearish-eğilim)
        "supply_z": _g("supply_z"),
        "demand_z": _g("demand_z"),
        "direction": ("arz baskısı (bearish-eğilim)" if bal is not None and bal > 0.25
                      else "talep baskısı (bullish-eğilim)" if bal is not None and bal < -0.25
                      else "denge/nötr") if bal is not None else None,
        "label": HONEST_LABEL,
        "panel_file": "output/supply_demand_panel.txt",
    }
