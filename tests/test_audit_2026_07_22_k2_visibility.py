"""
Denetim 2026-07-22 — K2 (supply_demand_derisk) sessiz-karanlık FIX:
  modules/supply_demand_derisk.py::evaluate() herhangi bir iç hatada None döner (207-252) AMA eskiden
  run.py bu None'ı overlays_out'a HİÇ kaydetmiyordu -> position_overlay_block (H3 fail-closed kapı)
  K2'yi göremiyordu, call_status "current" kalıyordu, _k2_factor sessizce 1.0'a düşüyordu (sigorta
  katmanı görünmez şekilde sönüyordu) VE run_daily._alert_if_degraded'de K2'ye özel bir kontrol yoktu.

  Bu dosya üç şeyi kilitler:
  (1) evaluate() gerçekten None döner iç hatada (önerme/premise doğrulaması).
  (2) run.build_decision artık K2 hatasını overlays_out'a available=False yazar, overlay_block/
      call_status=STALE'yi bunu görecek şekilde YENİDEN değerlendirir (diğer 3 overlayle aynı sözleşme).
  (3) run_daily._alert_if_degraded artık sd_derisk-özel bir nedeni notify.alert'e taşır.
"""
from __future__ import annotations

import copy
import json

import pytest


# ── (1) Önerme: evaluate() iç hatada gerçekten None döner (207-252) ────────────
class TestEvaluatePremise:
    def test_internal_exception_returns_none(self, monkeypatch):
        from modules import supply_demand_derisk as D
        from spine import contract as C, tide as T

        scores, _p, vector, _prov = C.read_frozen()
        ts = T.tide_score_series(scores, vector)
        td = T.tide_dir_series(ts)

        assert D.NET_SUPPLY_PARQUET.exists()   # ön-koşul: "veri yok" dalını değil, try-gövdesini vur

        def _boom(idx):
            raise RuntimeError("corrupt supply-z parquet (simulated)")
        monkeypatch.setattr(D, "_supply_z_pit", _boom)

        out = D.evaluate({}, tide_score=ts, tide_dir=td)
        assert out is None   # iç hata YUTULUR, call_status'a hiçbir iz bırakmadan (fix öncesi davranış)


# ── (2) run.build_decision: K2 hatası artık overlays_out + overlay_block + STALE'de görünür ────
class TestK2VisibilityInBuildDecision:
    def test_k2_failure_registers_unavailable_and_blocks(self, monkeypatch):
        from config import load_config
        from modules import supply_demand_derisk
        import run

        def _boom(*a, **kw):
            raise RuntimeError("sd_derisk boom (simulated)")
        monkeypatch.setattr(supply_demand_derisk, "evaluate", _boom)

        cfg = copy.deepcopy(load_config())
        cfg.setdefault("spine", {})["source"] = "frozen"
        assert bool((cfg.get("supply_demand_derisk", {}) or {}).get("enabled")) is True  # canlı-config varsayımı

        d = run.build_decision(cfg)

        ov = (d.get("overlays") or {}).get("supply_demand_derisk")
        assert ov is not None, "K2 hatası overlays_out'a hiç kaydedilmedi (eski bug geri geldi)"
        assert ov.get("available") is False
        assert ov.get("factor") == 1.0   # #4: fallback değeri DEĞİŞMEDİ (yalnız görünürlük)

        assert d["overlay_block"] is True
        assert "supply_demand_derisk" in (d.get("overlay_block_reason") or "")
        assert d["call_status"] == "STALE"   # eskiden "current" kalırdı — asimetri kapandı
        assert d.get("supply_demand_derisk") is None   # evaluate() gerçekten None döndü (K2 kendi bloğu)

    def test_k2_success_not_blocked(self):
        """Regresyon: K2 normal çalışırken (hata yok) overlay_block/STALE bundan ETKİLENMEMELİ."""
        from config import load_config
        import run

        cfg = copy.deepcopy(load_config())
        cfg.setdefault("spine", {})["source"] = "frozen"
        d = run.build_decision(cfg)

        ov = (d.get("overlays") or {}).get("supply_demand_derisk")
        assert ov is None   # başarılı yolda K2 overlays_out'a hiç yazılmaz (yalnız hata-yolunda yazılır)


# ── (3) run_daily._alert_if_degraded: K2-özel neden notify.alert'e taşınır ────
def _isolate_notify(tmp_path, monkeypatch):
    import notify
    (tmp_path / "output").mkdir(exist_ok=True)
    monkeypatch.setattr(notify, "ROOT", tmp_path)
    monkeypatch.setattr(notify, "_ALERT_STATE", tmp_path / "output" / ".alert_state.json")
    monkeypatch.setattr(notify, "_cfg_alert", lambda: {})
    calls = []
    monkeypatch.setattr(notify, "alert", lambda subj, body="": calls.append((subj, body)) or {"fired": []})
    monkeypatch.setattr(notify, "clear_alert", lambda: calls.append(("CLEAR", "")))
    return calls


class TestRunDailyAlertsOnK2Unavailable:
    def _write_latest(self, tmp_path, overlays):
        latest = {
            "call_status": "current", "data_source_stale": None,
            "overlay_block": False, "overlay_block_reason": None,
            "overlays": overlays, "spine": {}, "as_of": "2026-07-22",
            "freshness": {"age_days": 0},
        }
        (tmp_path / "output").mkdir(exist_ok=True)
        (tmp_path / "output" / "kader_equity_latest.json").write_text(
            json.dumps(latest), encoding="utf-8")

    def test_k2_unavailable_fires_alert(self, tmp_path, monkeypatch):
        import run_daily
        calls = _isolate_notify(tmp_path, monkeypatch)
        monkeypatch.setattr(run_daily, "ROOT", tmp_path)
        self._write_latest(tmp_path, {"supply_demand_derisk": {
            "available": False, "factor": 1.0,
            "error": "sd_derisk evaluate() None döndü (simulated)"}})

        run_daily._alert_if_degraded(0, [])

        assert calls, "K2 unavailable olduğu halde notify.alert hiç çağrılmadı"
        subj, body = calls[0]
        assert subj == "VERİ BAYAT / DEGRADE"
        assert "sd_derisk" in body and "K2" in body

    def test_clean_output_clears_alert_not_fires(self, tmp_path, monkeypatch):
        import run_daily
        calls = _isolate_notify(tmp_path, monkeypatch)
        monkeypatch.setattr(run_daily, "ROOT", tmp_path)
        self._write_latest(tmp_path, {})   # K2 yok/temiz (disabled ya da başarılı+overlays_out'ta yok)

        run_daily._alert_if_degraded(0, [])

        assert calls == [("CLEAR", "")]
