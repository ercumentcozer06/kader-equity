"""
test_freshness_failloud — 2026-06-19 audit fix'lerinin REGRESYON kilidi.

Bu testler "sessiz bayat→taze gibi davranma" garantilerini kilitler. Bir refactor onları kırarsa
(06-12'deki eksik-fix gibi) bu testler kırmızı yanar. Hepsi saf/mock — ağ yok.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from spine import tide as T


# ── NYSE tatil farkındalığı (#7) ──────────────────────────────────────
class TestMarketClosed:
    def test_juneteenth_closed(self):
        from run import market_closed_reason
        assert market_closed_reason(datetime(2026, 6, 19).date()) == "NYSE tatili"

    def test_weekend_closed(self):
        from run import market_closed_reason
        assert market_closed_reason(datetime(2026, 6, 20).date()) == "hafta sonu"   # Cumartesi

    def test_normal_trading_day_open(self):
        from run import market_closed_reason
        assert market_closed_reason(datetime(2026, 6, 18).date()) is None           # Perşembe


# ── tide.decide eksik-modül census (#8) — fail-OPEN yerine fail-VISIBLE ─
class TestTideDegraded:
    VEC = {"m9": 0.563, "m5": 0.214, "m2": 0.118, "m0": 0.061}

    def test_missing_high_weight_module_flags_degraded(self):
        r = T.decide({"m9": None, "m5": 2.0, "m2": 1.0, "m0": 1.0}, self.VEC)   # m9 (=%56) eksik
        assert r["degraded"] is True
        assert "m9" in r["modules_missing"]
        assert r["missing_weight_frac"] > 0.5

    def test_nan_module_also_flagged(self):
        r = T.decide({"m9": 5.0, "m5": float("nan"), "m2": 1.0, "m0": 1.0}, self.VEC)
        assert r["degraded"] is True
        assert "m5" in r["modules_missing"]

    def test_all_present_not_degraded(self):
        r = T.decide({"m9": 5.0, "m5": 2.0, "m2": 1.0, "m0": 1.0}, self.VEC)
        assert r["degraded"] is False
        assert r["modules_missing"] == []


# ── COR1M-froth staleness gate (#19) ──────────────────────────────────
class TestCor1mStaleGate:
    CFG = {"overlays": {"cor1m_froth": {"enabled": True, "lo": 8.0, "hi": 11.0, "floor": 0.0,
                                        "max_age_days": 4}}}

    def test_stale_cor1m_disarms_and_flags(self, monkeypatch):
        # 2026-07-06: birincil kaynak artık canlı quote → gate'i sınamak için ONU bayat yap.
        from modules import cor1m_froth
        monkeypatch.setattr(cor1m_froth, "fetch_cor1m_quote",
                            lambda timeout=15: (8.0, "2026-05-01"))          # ~7 hafta bayat
        r = cor1m_froth.evaluate(self.CFG)
        assert r["available"] is False
        assert r.get("stale") is True
        assert r["factor"] == 1.0                 # de-risk OFF (nötr, asla agresif)

    def test_fresh_cor1m_active(self, monkeypatch):
        from modules import cor1m_froth
        today = datetime.now(timezone.utc).date()
        monkeypatch.setattr(cor1m_froth, "fetch_cor1m_quote",
                            lambda timeout=15: (8.02, today.isoformat()))
        r = cor1m_froth.evaluate(self.CFG)
        assert r["available"] is True
        assert r.get("stale") is not True
        assert r["source"] == "cboe_quote_live"

    def test_quote_fails_falls_back_to_daily_csv(self, monkeypatch):
        # Canlı quote düşerse günlük CSV son kapanışına düşmeli (fail-open değil, taze-yedek).
        from modules import cor1m_froth
        def _boom(timeout=15):
            raise RuntimeError("quote endpoint down")
        today = datetime.now(timezone.utc).date()
        fresh = pd.Series([8.05], index=[pd.Timestamp(today)])
        monkeypatch.setattr(cor1m_froth, "fetch_cor1m_quote", _boom)
        monkeypatch.setattr(cor1m_froth, "fetch_cor1m_live", lambda timeout=20: fresh)
        r = cor1m_froth.evaluate(self.CFG)
        assert r["available"] is True
        assert r["source"] == "cboe_daily_csv_fallback"


# ── Fail-closed tepe-kapı: HERHANGİ pozisyon-overlay data-fail → no-trade (denetim 2026-07-06) ──
class TestPositionOverlayBlock:
    """cor1m/gex bayat→sessiz-flip asimetrisi kapatıldı: run.position_overlay_block jenerik fail-closed."""

    def test_stale_cor1m_blocks(self):
        from run import position_overlay_block
        ov = {"cor1m_froth": {"available": False, "stale": True, "factor": 1.0,
                              "reason": "COR1M STALE: ..."},
              "gex_shield": {"available": True, "factor": 1.0}}
        blk, reason = position_overlay_block(ov)
        assert blk is True and "cor1m_froth" in reason      # bayat cor1m artık bloke

    def test_gex_fail_safe_blocks(self):
        from run import position_overlay_block
        ov = {"gex_shield": {"available": False, "fail_safe_block": True, "error": "GEX DONMUŞ"}}
        assert position_overlay_block(ov)[0] is True

    def test_legit_neutral_does_not_block(self):
        # COR1M≥11 / GEX z≥−thr → available True, factor 1.0: de-risk-yok KARARI, veri-fail DEĞİL → bloke YOK.
        from run import position_overlay_block
        ov = {"cor1m_froth": {"available": True, "factor": 1.0, "froth": False},
              "gex_shield": {"available": True, "factor": 1.0}}
        assert position_overlay_block(ov)[0] is False

    def test_disabled_does_not_block(self):
        from run import position_overlay_block
        ov = {"cor1m_froth": {"available": False, "factor": 1.0, "reason": "disabled"}}
        assert position_overlay_block(ov)[0] is False

    def test_frozen_missing_keys_no_block(self):
        # Frozen yol: info dict'lerinde available/stale/fail_safe_block YOK → None → bloke YOK.
        from run import position_overlay_block
        ov = {"cor1m_froth": {"factor": 0.0, "as_of": "2026-05-22"},
              "gex_shield": {"factor": 1.0, "as_of": "2026-05-22"}}
        assert position_overlay_block(ov)[0] is False


# ── M3 auction tazelik kör-noktası kapatıldı (P1-C, denetim 2026-07-07) ──
class TestM3AuctionFreshness:
    """fetch_auctions boş/donuk dönerse m3 sessizce 0'a düşüp çağrı 'current' kalıyordu → artık input_stale."""

    def test_empty_auctions_flagged(self):
        import datetime as dt
        from spine.reconstruct import _audit_m3_freshness
        r = _audit_m3_freshness(None, dt.date(2026, 7, 7))
        assert r is not None and r["module"] == "m3" and "BOŞ" in r["error"]

    def test_frozen_feed_flagged(self):
        import datetime as dt
        import pandas as pd
        from spine.reconstruct import _audit_m3_freshness, M3_MAX_BD
        r = _audit_m3_freshness(pd.Timestamp("2026-05-20"), dt.date(2026, 7, 7))
        assert r is not None and r["age_bd"] > M3_MAX_BD

    def test_fresh_auctions_clean(self):
        import datetime as dt
        import pandas as pd
        from spine.reconstruct import _audit_m3_freshness
        assert _audit_m3_freshness(pd.Timestamp("2026-07-02"), dt.date(2026, 7, 7)) is None


# ── K2 supply_demand_derisk CANLI-tide enjeksiyonu (#4) ────────────────
class TestSdDeriskLiveTide:
    def test_live_tide_injection_tags_source(self):
        from modules import supply_demand_derisk as sd
        res = sd.evaluate({}, live_tide_score=7.6, live_tide_dir=1,
                          as_of=pd.Timestamp("2026-06-19"))
        if res is None:
            pytest.skip("net_supply parquet yok — ortam-bağımlı")
        assert res.get("tide_source") == "live"   # frozen-28g-bayat DEĞİL, canlı tide kullanıldı


# ── KRİTİK (#1): bayat FRED girdisi → call_status STALE (fail-LOUD) ─────
class TestBuildDecisionStaleInput:
    BASE_CFG = {"spine": {"source": "live", "max_staleness_days": 5},
                "sizing": {"net_exposure_cap": 1.0},
                "overlays": {}, "assets": ["SPX", "NDX"]}

    @staticmethod
    def _fake_reconstruct(input_stale):
        def _f(cfg, force=False):
            scores_row = {"m9": 5.0, "m5": 2.0, "m2": 1.0}
            vector = {"m9": 0.563, "m5": 0.214, "m2": 0.118}
            as_of = pd.Timestamp(datetime.now(timezone.utc).date())   # bugün → snapshot taze
            return scores_row, vector, as_of, "live", input_stale
        return _f

    def test_stale_fred_leg_forces_stale_call(self, monkeypatch):
        import spine.reconstruct as R
        import run
        stale = [{"series": "WALCL", "module": "m2", "age_bd": 20, "max_bd": 8}]
        monkeypatch.setattr(R, "reconstruct_live", self._fake_reconstruct(stale))
        d = run.build_decision(self.BASE_CFG)
        assert d["call_status"] == "STALE"               # donmuş bacak → GÜNCEL değil
        assert d["data_source_stale"] == stale

    def test_fresh_legs_stay_current(self, monkeypatch):
        import spine.reconstruct as R
        import run
        monkeypatch.setattr(R, "reconstruct_live", self._fake_reconstruct([]))
        d = run.build_decision(self.BASE_CFG)
        assert d["call_status"] == "current"             # tüm girdi taze → güncel çağrı
        assert d["data_source_stale"] is None


# ── #1 kök-fix: equity reconstruct FRED cache'i DONDURMAZ (cache_min 1e9 → günlük tazeleme açık) ──
class TestEquityFredRefreshes:
    def test_reconstruct_cache_min_not_frozen(self):
        import re
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "spine" / "reconstruct.py").read_text(encoding="utf-8")
        # ASSIGNMENT'ın RHS'ini al (yorumdaki '1e9' tarihçesini DEĞİL); düz-int + ≤1gün olmalı.
        m = re.search(r'cache_min_seconds"\]\s*=\s*([^\n#]+)', src)
        assert m, "cache_min_seconds ataması bulunamadı (refactor kırmış olabilir)"
        rhs = m.group(1).strip()
        assert rhs.replace("_", "").isdigit(), f"cache_min RHS '{rhs}' düz-int değil (10**9/1e9 bug'ı geri gelmiş)"
        assert int(rhs.replace("_", "")) <= 86400, f"cache_min={rhs} > 1gün → FRED günlük tazelenmez (cache_max defedilir)"
