"""
Denetim 2026-07-04 fix testleri:
  • EQ-3 — defter attribution kolonları (m9/m5/m2) artık GERÇEKTEN yazılıyor:
    build_decision → module_scores → run.ledger_record (tek şema) → append_call → parquet'te dolu.
  • EQ-5 — kaçırılmış işgünü artık sessizce yeşil değil: missed_weekday saf karar yardımcısı
    (pencere-sonrası + koşmamış + işgünü → True; hafta sonu / tatil / erken-tetik → False).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── EQ-3: defter skor kolonları ──────────────────────────────────────────────

def test_eq3_ledger_record_maps_module_scores():
    """Fake karar dict'i → ledger_record m9/m5/m2'yi module_scores'tan çeker (None-safe)."""
    import run
    d = {"as_of": "2026-07-03", "computed_at": "t", "model_tag": "test", "call_status": "current",
         "position_target": 0.8, "direction": "LONG", "deploy_fraction": 0.8, "tide_dir": 1,
         "tide_score": 5.0, "active_overlays": ["cor1m_froth"], "data_source": "live",
         "module_scores": {"m9": 12.34, "m5": -3.2, "m2": 0.0, "m0": 1.0}}
    rec = run.ledger_record(d)
    assert rec["m9_score"] == 12.34 and rec["m5_score"] == -3.2 and rec["m2_score"] == 0.0
    assert rec["size"] == 0.8 and rec["active_overlays"] == "cor1m_froth"


def test_eq3_ledger_record_none_safe():
    """module_scores yok/eksik → üç kolon None (crash yok; eski karar dict'leriyle geriye-uyumlu)."""
    import run
    d = {"as_of": "2026-07-03", "computed_at": "t", "model_tag": "test", "call_status": "current",
         "position_target": 1.0, "direction": "LONG", "deploy_fraction": 1.0, "tide_dir": 1,
         "tide_score": 5.0, "active_overlays": [], "data_source": "frozen"}
    rec = run.ledger_record(d)
    assert rec["m9_score"] is None and rec["m5_score"] is None and rec["m2_score"] is None


def test_eq3_append_call_writes_score_columns(tmp_path, monkeypatch):
    """append_call sonrası parquet'te üç attribution kolonu DOLU (şemada-var-ama-hiç-yazılmıyor bitti)."""
    import pandas as pd
    import run
    from validation import ledger as L
    monkeypatch.setattr(L, "ledger_path", lambda: tmp_path / "fl.parquet")
    d = {"as_of": "2026-07-03", "computed_at": "t", "model_tag": "test", "call_status": "current",
         "position_target": 0.9, "direction": "LONG", "deploy_fraction": 0.9, "tide_dir": 1,
         "tide_score": 7.0, "active_overlays": ["cor1m_froth", "gex_shield"], "data_source": "live",
         "module_scores": {"m9": 10.6, "m5": 15.46, "m2": 10.5373}}
    L.append_call(run.ledger_record(d))
    df = pd.read_parquet(tmp_path / "fl.parquet")
    row = df.set_index("as_of").loc["2026-07-03"]
    assert abs(float(row["m9_score"]) - 10.6) < 1e-9
    assert abs(float(row["m5_score"]) - 15.46) < 1e-9
    assert abs(float(row["m2_score"]) - 10.5373) < 1e-9


def test_eq3_build_decision_carries_module_scores():
    """Frozen yol da karar dict'ine module_scores koyar (m9/m5/m2 anahtarları mevcut + sayısal)."""
    import copy

    import run
    from config import load_config
    cfg = copy.deepcopy(load_config())
    cfg.setdefault("spine", {})["source"] = "frozen"
    d = run.build_decision(cfg)
    ms = d.get("module_scores")
    assert isinstance(ms, dict)
    for k in ("m9", "m5", "m2"):
        assert k in ms and ms[k] is not None and isinstance(ms[k], float)


# ── EQ-5: kaçırılmış işgünü karar yardımcısı ─────────────────────────────────

def _utc(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


def test_eq5_missed_weekday_true_after_window():
    """Çarşamba 16:00 UTC = 12:00 EDT (pencere geçti) + bugün koşmamış → KAÇIRILMIŞ (True)."""
    import run_preopen as P
    assert P.missed_weekday(_utc(2026, 7, 1, 16, 0), ran_today=False) is True


def test_eq5_missed_weekday_false_if_ran():
    """Aynı an ama run_daily bugün zaten koşmuş → False (bayrak yok)."""
    import run_preopen as P
    assert P.missed_weekday(_utc(2026, 7, 1, 16, 0), ran_today=True) is False


def test_eq5_missed_weekday_false_weekend():
    """Cumartesi → beklenen atlama, bayrak YOK."""
    import run_preopen as P
    assert P.missed_weekday(_utc(2026, 7, 4, 16, 0), ran_today=False) is False


def test_eq5_missed_weekday_false_nyse_holiday():
    """2026-07-03 Cuma = NYSE tatili (run.NYSE_HOLIDAYS) → beklenen atlama, bayrak YOK."""
    import run_preopen as P
    assert P.missed_weekday(_utc(2026, 7, 3, 16, 0), ran_today=False) is False


def test_eq5_missed_weekday_false_before_window():
    """Kış erken tetiği (08:35 EST, pencere HENÜZ gelmedi) → ikinci tetik gelecek, bayrak YOK."""
    import run_preopen as P
    assert P.missed_weekday(_utc(2026, 1, 7, 13, 35), ran_today=False) is False
