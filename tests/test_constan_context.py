"""Constan bağlam bantları — santa_window durum makinesi + net_supply_context okuma.
Ağ YOK: santa saf-fonksiyon sentetik kapanışlarla; net-supply gerçek parquet'ten (repo içi) okur."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from modules import net_supply_context as NS
from modules import santa_window as SW


def _closes(year: int, ytd_at_nov1: float) -> pd.Series:
    """Sentetik yıl: 2 Oca = 100, 1 Kas'a lineer rampa → hedef YTD, sonra düz."""
    idx = pd.bdate_range(f"{year}-01-02", f"{year}-12-31")
    nov1 = next(d for d in idx if d >= pd.Timestamp(f"{year}-11-01"))
    n_pre = int((idx <= nov1).sum())
    target = 100.0 * (1 + ytd_at_nov1)
    pre = np.linspace(100.0, target, n_pre)
    post = np.full(len(idx) - n_pre, target)
    return pd.Series(np.concatenate([pre, post]), index=idx)


class TestSantaPure:
    def test_qualifying_active(self):
        out = SW.evaluate_pure(dt.date(2030, 11, 15), _closes(2030, 0.15))
        assert out["state"] == "QUALIFYING_ACTIVE"
        assert out["ytd_at_nov1_pct"] == pytest.approx(15.0, abs=0.3)

    def test_non_qualifying(self):
        out = SW.evaluate_pure(dt.date(2030, 12, 5), _closes(2030, 0.04))
        assert out["state"] == "NON_QUALIFYING"
        assert "min -22.7%" in out["note"]

    def test_threshold_edge_inclusive(self):
        out = SW.evaluate_pure(dt.date(2030, 11, 3), _closes(2030, 0.101))
        assert out["state"] == "QUALIFYING_ACTIVE"

    def test_inactive_months(self):
        out = SW.evaluate_pure(dt.date(2030, 6, 15), _closes(2030, 0.20))
        assert out["state"] == "INACTIVE"

    def test_october_preview(self):
        out = SW.evaluate_pure(dt.date(2030, 10, 20), _closes(2030, 0.15))
        assert out["state"] == "INACTIVE"
        assert "preview_ytd_pct" in out

    def test_pit_no_future_leak(self):
        """1 Kas'tan önceki as_of'ta pencere durumu hesaplanamaz (UNKNOWN değil INACTIVE-ay-dışı)."""
        c = _closes(2030, 0.15)
        out = SW.evaluate_pure(dt.date(2030, 11, 3), c[c.index <= "2030-11-03"])
        assert out["state"] == "QUALIFYING_ACTIVE"   # 1 Kas kapanışı mevcut → yeter

    def test_no_data_unknown(self):
        out = SW.evaluate_pure(dt.date(2030, 11, 15), pd.Series(dtype=float))
        assert out["state"] == "UNKNOWN"

    def test_zero_position_contract(self):
        """Bant pozisyon alanı ÜRETMEZ — etiket sözleşmesi (yanlışlıkla deploy sızmasın)."""
        out = SW.evaluate_pure(dt.date(2030, 11, 15), _closes(2030, 0.15))
        assert "deploy" not in out and "factor" not in out and "position" not in str(out.keys())


class TestNetSupply:
    def test_reads_latest_pit_valid(self):
        out = NS.evaluate({}, today=dt.date(2026, 6, 13))
        assert out is not None
        # 2026Q1 pit 2026-06-15 → 13 Haziran'da en taze PIT-geçerli çeyrek 2025Q4 olmalı
        assert out["quarter"] == "2025Q4"
        assert out["ratio4q_nfc_pct_ngdp"] is not None

    def test_pit_advances_after_release(self):
        out = NS.evaluate({}, today=dt.date(2026, 6, 16))
        assert out["quarter"] == "2026Q1"
        assert out["flip_note"] is not None   # 2026Q1 ilk pozitif tek-çeyrek → işaret-dönümü haberi

    def test_tail_flag_threshold(self):
        out_low = NS.evaluate({"tail_z_threshold": 99.0}, today=dt.date(2026, 6, 16))
        assert out_low["tail_flag"] is False
        out_hot = NS.evaluate({"tail_z_threshold": 1.0}, today=dt.date(2026, 6, 16))
        assert out_hot["tail_flag"] is True and out_hot["tail_note"] is not None

    def test_honest_label_present(self):
        out = NS.evaluate({}, today=dt.date(2026, 6, 13))
        assert "DEGIL" in out["label"]   # "yon-sinyali DEGIL" dürüst etiketi her çıktıda


class TestComponents:
    """FINDING 25: bileşen paneli — PANEL-ONLY (bayrak-yükseltme üçlü-sınavı FAIL)."""

    def test_components_present_and_pit_gated(self):
        out = NS.evaluate({}, today=dt.date(2026, 6, 13))
        comp = out.get("components")
        assert comp is not None
        # Ritter sayımı bilgi-bazlı PIT (+7g): 2026-06-13'te 2025Q4'e kadar görünür olmalı
        assert comp["opco_ipo"]["quarter"] in ("2025Q4", "2026Q1")
        # S&P buyback: Wayback-xlsx 2024Q4'te biter; MANUEL basın-bülteni katmanı (2026-06-13)
        # 2025Q1-Q3'ü gerçek bülten pit_date'iyle ekler → son görünür çeyrek 2025Q3 (pit 2025-12-18)
        assert comp["buyback"]["quarter"] == "2025Q3"
        assert comp["buyback"]["prelim"] is True   # manuel katman resmi xlsx'e girene dek prelim=1

    def test_buyback_splice_fix_values(self):
        """Ölçü-karışımı onarımı kalıcı: dikiş-öncesi çeyrek SAF buyback değerinde
        (birleşik buyback+temettü 151.1 DEĞİL, saf 89.7 — adversarial denetim 2026-06-13)."""
        import pandas as pd
        df = pd.read_parquet(NS.COMP_PARQUET)
        idx = df.index[df.index.astype(str).str.startswith("2008-07")]
        assert len(idx) == 1
        assert float(df.loc[idx[0], "spx_bb_bn"]) == pytest.approx(89.7, abs=0.1)

    def test_component_label_says_descriptive(self):
        out = NS.evaluate({}, today=dt.date(2026, 6, 13))
        assert "BETIMSEL" in out["components"]["label"]
        assert "FAIL" in out["components"]["label"]   # bayrak-yükseltme reddi görünür


class TestIpoPipeline:
    """FINDING 26: EDGAR S-1 boru-hattı bandı + dev-arz izleme (Constan Q3-Q4 dedektörü)."""

    def test_reads_pipeline_and_mega_hits(self):
        from modules import ipo_pipeline_context as IP
        out = IP.evaluate({}, today=dt.date(2026, 6, 13))
        assert out is not None
        assert out["last_closed_quarter"] == "2026Q1"
        assert out["roll4q_filings"] and out["z10y"] is not None
        # SpaceX 2026-05-20 S-1 → 120g penceresinde dev-arz isabeti olmalı (kayıt-tavanı $86.25B)
        hits = out["mega_hits_120d"]
        assert any("SPACE EXPLORATION" in (h.get("company") or "") for h in hits)
        spx = next(h for h in hits if "SPACE EXPLORATION" in h["company"])
        assert spx["max_ceiling_usd"] and spx["max_ceiling_usd"] > 8e10   # $86.25B tavan

    def test_honest_intent_label(self):
        from modules import ipo_pipeline_context as IP
        out = IP.evaluate({}, today=dt.date(2026, 6, 13))
        assert "NIYET" in out["label"]      # dosyalama=niyet etiketi her çıktıda
        assert "SIFIR" in out["label"]      # pozisyon-etkisi-yok sözleşmesi

    def test_zero_position_contract(self):
        from modules import ipo_pipeline_context as IP
        out = IP.evaluate({}, today=dt.date(2026, 6, 13))
        assert "deploy" not in out and "factor" not in out


class TestSupplyDemandDerisk:
    """FINDING 27 K2: koşullu de-risk — 2020-sessizlik (kritik) + 2021-ateşleme + trim-only."""

    @staticmethod
    def _series():
        from spine import contract as C, tide as T
        scores, _p, vector, _prov = C.read_frozen()
        ts = T.tide_score_series(scores, vector)
        td = T.tide_dir_series(ts)
        from modules import supply_demand_derisk as D
        z = D._supply_z_pit(ts.index)
        return z, ts, td

    def test_2020_silence_strong_demand(self):
        """KRİTİK: arz-yüksek AMA talep-güçlü (tide>+2) → ATEŞLEMEZ (2020 rali kesilmez)."""
        from modules import supply_demand_derisk as D
        z, ts, td = self._series()
        for d in ("2020-09-14", "2020-12-14"):
            o = D.evaluate_pure(z, ts, td, as_of=d)
            assert o["supply_hi"] is True          # arz gerçekten yüksekti
            assert o["fired"] is False             # ama talep güçlü → SUSAR
            assert o["trim_factor"] == 1.0

    def test_2021_fires_weak_demand(self):
        """arz-yüksek + talep-zayıf (tide<=+2 düşüşte / dir=0) → ATEŞLER."""
        from modules import supply_demand_derisk as D
        z, ts, td = self._series()
        for d in ("2021-09-13", "2021-12-13"):
            o = D.evaluate_pure(z, ts, td, as_of=d)
            assert o["fired"] is True
            assert o["trim_factor"] == 0.85

    def test_trim_only_rebound_safe(self):
        """Faktör daima ≤1 (trim-only); ateşlemezse tam 1.0."""
        from modules import supply_demand_derisk as D
        z, ts, td = self._series()
        o = D.evaluate_pure(z, ts, td, as_of="2026-05-22")
        assert o["trim_factor"] <= 1.0
        assert o["label"] and "alfa DEGIL" in o["label"]

    def test_thresholds_frozen(self):
        from modules import supply_demand_derisk as D
        assert (D.SUPPLY_Z_THR, D.DEMAND_WEAK_LEVEL, D.DECLINE_LB, D.TRIM) == (1.0, 2.0, 63, 0.85)
        assert D.MEGA_CEILING_THR == 50e9


class TestMegaIpoArm:
    """MEGA-IPO ANLIK arz kolu (2026-06-13, Emir "5.5 ay kabul edilemez"): çeyreklik z'nin
    ~5.5 ay gecikmesini kapatan OR kolu. KRİTİK kural: yalnız ARZ kolunu genişletir,
    TALEP-ZAYIF kapısını ASLA atlamaz → 2020-sessizlik mega-rağmen korunur."""

    SPACEX = 86_249_999_880.0   # SpaceX 2026-06-03 S-1/A kayıt-tavanı ($86.25B ARZ; değerleme değil)

    @staticmethod
    def _series():
        from spine import contract as C, tide as T
        scores, _p, vector, _prov = C.read_frozen()
        ts = T.tide_score_series(scores, vector)
        td = T.tide_dir_series(ts)
        from modules import supply_demand_derisk as D
        z = D._supply_z_pit(ts.index)
        return z, ts, td

    def test_mega_or_fires_weak_demand(self):
        """(a) SpaceX $86.25B + tide-ZAYIF (2021-12-13, tide -0.53) → mega kolundan ATEŞLER."""
        from modules import supply_demand_derisk as D
        z, ts, td = self._series()
        o = D.evaluate_pure(z, ts, td, as_of="2021-12-13",
                            mega_ceiling_usd=self.SPACEX, mega_label="SpaceX")
        assert o["fired"] is True
        assert o["trim_factor"] == 0.85
        assert o["supply_arm"] == "mega-IPO"      # z DEĞİL, mega kolundan
        assert o["mega_hi"] is True
        assert "SpaceX" in o["reason"] and "$86B" in o["reason"]

    def test_2020_silence_despite_mega(self):
        """(b) KRİTİK — SpaceX $86.25B + tide-GÜÇLÜ (2020-09-14 +8.8 / 2020-12-14 +5.9) → SUSAR.
        Mega kol AKTİF (supply_hi True) AMA demand_weak FALSE → fire FALSE. 2020-rali kesilmez."""
        from modules import supply_demand_derisk as D
        z, ts, td = self._series()
        for d in ("2020-09-14", "2020-12-14"):
            o = D.evaluate_pure(z, ts, td, as_of=d,
                                mega_ceiling_usd=self.SPACEX, mega_label="SpaceX")
            assert o["mega_hi"] is True            # mega tavan eşik üstü
            assert o["supply_hi"] is True          # arz-aşırı kol AKTİF
            assert o["demand_weak"] is False       # ama talep güçlü
            assert o["fired"] is False             # → SUSAR (2020-koruma mega-rağmen)
            assert o["trim_factor"] == 1.0

    def test_mega_zero_equals_z_only(self):
        """(c) mega_ceiling=0 (hit yok) → eski z-only davranış bire bir (default == açık-0)."""
        from modules import supply_demand_derisk as D
        z, ts, td = self._series()
        for d in ("2020-09-14", "2021-12-13", "2026-05-22"):
            o_def = D.evaluate_pure(z, ts, td, as_of=d)                       # mega arg yok
            o_zero = D.evaluate_pure(z, ts, td, as_of=d, mega_ceiling_usd=0.0)
            assert o_def["fired"] == o_zero["fired"]
            assert o_def["trim_factor"] == o_zero["trim_factor"]
            assert o_def["mega_hi"] is False and o_zero["mega_hi"] is False

    def test_mega_below_threshold_inert(self):
        """Eşik-altı tek-arz ($1B << $50B) → mega_hi False; tek başına ASLA tetiklemez."""
        from modules import supply_demand_derisk as D
        z, ts, td = self._series()
        # tide-güçlü günde küçük IPO → zaten susardı; tide-zayıf günde de mega kol inert kalmalı
        o = D.evaluate_pure(z, ts, td, as_of="2020-09-14",
                            mega_ceiling_usd=1e9, mega_label="SmallCo")
        assert o["mega_hi"] is False
        assert o["fired"] is False

    def test_mega_alone_never_fires_strong_tide(self):
        """Mega-IPO TEK BAŞINA tetiklemez: arz kolu mega'dan True olsa da talep güçlüyse SUSAR."""
        from modules import supply_demand_derisk as D
        z, ts, td = self._series()
        o = D.evaluate_pure(z, ts, td, as_of="2020-09-14",
                            mega_ceiling_usd=500e9, mega_label="GiantCo")  # absürt-büyük tavan
        assert o["mega_hi"] is True
        assert o["fired"] is False                 # talep güçlü → mega tek başına yetmez

    def test_custom_threshold_param(self):
        """evaluate_pure mega_ceiling_thr parametresi: config eşiği geçer (global mutasyon yok)."""
        from modules import supply_demand_derisk as D
        z, ts, td = self._series()
        # $86B tavan, eşik $100B → mega_hi False (eşik aşılmadı)
        o = D.evaluate_pure(z, ts, td, as_of="2021-12-13",
                            mega_ceiling_usd=self.SPACEX, mega_ceiling_thr=100e9)
        assert o["mega_hi"] is False
        assert o["thresholds"]["mega_ceiling_usd"] == 100e9
        # global default DEĞİŞMEDİ (mutasyon yok)
        assert D.MEGA_CEILING_THR == 50e9

    def test_evaluate_reads_spacex_from_hits_json(self):
        """evaluate() canlı: mega_ipo_hits.json'dan SpaceX $86.25B PIT-okur (date_filed 2026-06-03,
        as_of 2026-06-13 → 120g penceresinde). Mega kol AKTİF olmalı."""
        from modules import supply_demand_derisk as D
        amt, label = D._mega_ceiling_pit(as_of="2026-06-13")
        assert amt > 8e10                          # $86.25B kayıt-tavanı okundu
        assert "Space" in label or "SpaceX" in label

    def test_mega_pit_no_future_leak(self):
        """PIT: $86.25B tavan 06-03'te dosyalandı → 06-02'de GÖRÜNMEZ (gelecek-sızıntısı yok).
        06-04'te görünür. (Kalıcı defter tavanı erken-tarihe bağlamaz — sızıntı önlenir.)"""
        from modules import supply_demand_derisk as D
        led = D._refresh_mega_active_ledger()
        amt_in, _ = D._mega_ceiling_pit(as_of="2026-06-04", ledger=led)
        assert amt_in > 8e10                       # 06-04: $86.25B görünür
        amt_future, _ = D._mega_ceiling_pit(as_of="2026-06-02", ledger=led)
        assert amt_future < 8e10                   # 06-02: $86.25B HENÜZ yok (yalnız 05-20 $1B, eşik-altı)

    def test_mega_no_blindness_past_120d(self):
        """KÖRLEŞME YOK (Emir 2026-06-13): SpaceX $86B 120 GÜNDEN SONRA da armed kalır
        (eski 120g pencere körleştiriyordu). 2026-12-01 (~180g sonra) → hâlâ görünür."""
        from modules import supply_demand_derisk as D
        led = D._refresh_mega_active_ledger()
        amt_180, lbl = D._mega_ceiling_pit(as_of="2026-12-01", ledger=led)
        assert amt_180 > 8e10 and ("Space" in lbl or "SpaceX" in lbl)
        # ama 540g cap sonrası (truly-abandoned) elenir: 2028-01-01 (~570g)
        amt_old, _ = D._mega_ceiling_pit(as_of="2028-01-01", ledger=led)
        assert amt_old < 8e10

    def test_mega_arm_window_frozen(self):
        from modules import supply_demand_derisk as D
        assert D.MEGA_ARM_MAX_DAYS == 540


class TestIpoSupplyWaves:
    """dev-IPO GECİKMELİ arz dalgaları (lock-up bitişi + endeks-dahil). BETİMSEL — pozisyon etkisi YOK.
    SpaceX prospektüs gövdesi data/cache/raw/edgar/_body_<acc>.htm'de cache'li → ağsız test."""

    # SpaceX S-1/A 2026-06-03 prospektüsünden DOĞRULANMIŞ birebir alıntılar:
    SPACEX_TEXT = (
        "the shares subject to the 180-day lock-up period released from restrictions, and the "
        "shares held by our founder (which are subject to a 366-day lock-up period), represent "
        "approximately 7.8 billion shares of common stock, or greater than 63% of our shares "
        "outstanding immediately prior to this offering. These restrictions for over one year "
        "include an aggregate of 7.8 billion shares owned (including 100% of the shares owned by "
        "Mr. Musk).")

    def test_parse_lockup_days_picks_founder_366(self):
        from modules import ipo_supply_waves as W
        mx, alld = W.parse_lockup_days(self.SPACEX_TEXT)
        assert mx == 366                      # kurucu kilidi = bağlayıcı (180 değil)
        assert alld == [180, 366]             # her iki kademe de bulunur

    def test_parse_lockup_days_default_when_absent(self):
        from modules import ipo_supply_waves as W
        mx, alld = W.parse_lockup_days("no lock up info here")
        assert mx is None and alld == []      # bulunamaz → çağıran varsayılana düşer

    def test_parse_lockup_shares_7p8b_63pct(self):
        from modules import ipo_supply_waves as W
        sh, note = W.parse_lockup_shares(self.SPACEX_TEXT)
        assert sh == pytest.approx(7.8e9, rel=1e-9)
        assert "63%" in note and "greater than" in note

    def test_compute_wave_spacex_dates_and_index(self):
        from modules import ipo_supply_waves as W
        hit = {"watch_name": "SpaceX", "company": "SPACE EXPLORATION TECHNOLOGIES CORP",
               "date_filed": "2026-06-03", "proposed_max_aggregate_usd": 86_249_999_880.0}
        w = W.compute_wave(hit, today=dt.date(2026, 6, 13), prospectus_text=self.SPACEX_TEXT)
        # lock-up = dosyalama 2026-06-03 + 366g = 2027-06-04
        assert w["lockup_days"] == 366
        assert w["lockup_expiry_date"] == "2027-06-04"
        assert w["days_to_lockup"] == 356
        assert w["lockup_shares_est"] == pytest.approx(7.8e9, rel=1e-9)
        # endeks-dahil: $86B > $50B → eligible + forced pasif talep $86B×%15-20
        assert w["index_incl_eligible"] is True
        assert w["forced_passive_demand_usd_lo"] == pytest.approx(86_249_999_880.0 * 0.15)
        assert w["forced_passive_demand_usd_hi"] == pytest.approx(86_249_999_880.0 * 0.20)
        assert w["index_incl_window"]["earliest"] == "2026-09-01"

    def test_compute_wave_default_lockup_when_no_text(self):
        from modules import ipo_supply_waves as W
        hit = {"watch_name": "MidCo", "date_filed": "2026-06-03",
               "proposed_max_aggregate_usd": 60e9}
        w = W.compute_wave(hit, today=dt.date(2026, 6, 13), prospectus_text=None)
        assert w["lockup_days"] == 180        # varsayılan
        assert w["lockup_days_parsed"] is None
        assert "varsayilan" in w["lockup_status"].lower()

    def test_compute_wave_sub_50b_no_index(self):
        from modules import ipo_supply_waves as W
        hit = {"watch_name": "SmallCo", "date_filed": "2026-06-03",
               "proposed_max_aggregate_usd": 10e9}   # $10B < $50B
        w = W.compute_wave(hit, today=dt.date(2026, 6, 13), prospectus_text=self.SPACEX_TEXT)
        assert w["index_incl_eligible"] is False
        assert w["index_incl_window"] is None
        assert w["forced_passive_demand_usd_lo"] is None

    def test_evaluate_reads_spacex_offline(self):
        """evaluate() canlı: mega_ipo_hits.json'dan SpaceX'i okur, cache'li gövdeden 366g parse eder."""
        from modules import ipo_supply_waves as W
        out = W.evaluate({}, today=dt.date(2026, 6, 13), allow_network=False)
        assert out is not None and out["n_active"] >= 1
        sx = next(w for w in out["waves"] if "Space" in w["company"] or "SpaceX" in w["company"])
        assert sx["lockup_days"] == 366               # cache'li prospektüsten gerçek parse
        assert sx["lockup_expiry_date"] == "2027-06-04"
        assert sx["lockup_shares_est"] == pytest.approx(7.8e9, rel=1e-9)
        assert sx["prospectus_fetch_status"] == "cache"

    def test_zero_position_contract(self):
        """Band pozisyon/deploy alanı ÜRETMEZ (betimsel sözleşme; yanlışlıkla deploy sızmasın)."""
        from modules import ipo_supply_waves as W
        out = W.evaluate({}, today=dt.date(2026, 6, 13), allow_network=False)
        blob = str(out)
        assert "deploy" not in blob and "trim_factor" not in blob
        for w in out["waves"]:
            assert "factor" not in w and "position" not in w

    def test_honest_label_present(self):
        from modules import ipo_supply_waves as W
        out = W.evaluate({}, today=dt.date(2026, 6, 13), allow_network=False)
        assert "TAHMIN" in out["label"] and "SIFIR" in out["label"]

    def test_pit_excludes_future_ceiling(self):
        """PIT: gelecek dosyalama dışlanır. as_of 06-02'de 06-03'teki $86.25B S-1/A tavanı GÖRÜNMEZ
        (yalnız 05-20 S-1 $1B tavanı in-window) → SpaceX dalgası varsa $86B DEĞİL, $50B-altı/endeks-dışı."""
        from modules import ipo_supply_waves as W
        out_future = W.evaluate({}, today=dt.date(2026, 6, 2), allow_network=False)
        if out_future is not None:
            for w in out_future["waves"]:
                if "Space" in w["company"]:
                    assert (w["offering_ceiling_usd"] or 0) < 50e9   # $86B tavanı sızmadı
                    assert w["index_incl_eligible"] is False
        # as_of 06-13'te ise $86.25B tavan görünür ($50B üstü, endeks-eligible)
        out_now = W.evaluate({}, today=dt.date(2026, 6, 13), allow_network=False)
        sx = next(w for w in out_now["waves"] if "Space" in w["company"])
        assert sx["offering_ceiling_usd"] > 8e10 and sx["index_incl_eligible"] is True


class TestSupplyDemandBalance:
    """FINDING 27 K1: betimsel denge kadranı (yön-ayracı DEĞİL)."""

    def test_reads_balance(self):
        from modules import supply_demand_balance as B
        out = B.evaluate({}, today=dt.date(2026, 6, 13))
        assert out is not None
        assert out["net_supply_pressure"] is not None
        assert out["supply_z"] is not None and out["demand_z"] is not None

    def test_not_discriminator_label(self):
        from modules import supply_demand_balance as B
        out = B.evaluate({}, today=dt.date(2026, 6, 13))
        assert "AYRACI DEGIL" in out["label"]   # dürüst sınır her çıktıda


class TestBuybackSelfCheck:
    """FINDING 28: auto-buyback BAĞIMSIZ self-check — sessiz-bozulma önleyici (latent delik kapatıldı).
    Önemli: aynı-kaynak anlatı-yedeği KALDIRILDI; col1-çapa (bağımsız) + 12mo-aritmetik."""

    def test_correct_passes_with_3_priors(self):
        from screen.fetch_supply_components import buyback_selfcheck
        # Q4 sim: bb=249, bültenin col1 (Q3)=234.6, CSV son-prior=234.57; 12mo: 249+(293.45+234.57+...)
        priors = [293.45, 234.57, 234.6]   # son = bültenin col1 ile uyumlu
        ok, _ = buyback_selfcheck(bb_bn=250.0, ttm_bn=250.0 + sum(priors[-3:]),
                                  prior_q_bn=234.6, priors=priors)
        assert ok is True

    def test_top20_trap_blocked(self):
        """Top-20 satırı (~$123B) sektör-TOTAL (~$249B) yerine parse edilse → col1-çapa DÜŞER."""
        from screen.fetch_supply_components import buyback_selfcheck
        # yanlış parse: bb=123 + bültenin col1'i de yanlış-satırdan 100 gelir; CSV-prior 234.6
        ok, detail = buyback_selfcheck(bb_bn=123.0, ttm_bn=999.0, prior_q_bn=100.0,
                                       priors=[293.45, 234.57, 234.6])
        assert ok is False and "col1" in detail

    def test_units_error_blocked(self):
        """Birim hatası (×1000) → her iki bağımsız kontrol de gross sapar → KALDI."""
        from screen.fetch_supply_components import buyback_selfcheck
        ok, _ = buyback_selfcheck(bb_bn=249004.0, ttm_bn=1020268.0, prior_q_bn=234570.0,
                                  priors=[293.45, 234.57, 234.6])
        assert ok is False

    def test_zero_priors_refuses(self):
        """0 prior (taze CSV) → aynı-kaynak yedeğe DÜŞMEZ, REFUSE (eski deliğin kapanışı)."""
        from screen.fetch_supply_components import buyback_selfcheck
        ok, detail = buyback_selfcheck(bb_bn=249.0, ttm_bn=1020.0, prior_q_bn=234.6, priors=[])
        assert ok is False and "REFUSE" in detail

    def test_1_prior_independent_anchor_blocks_fake(self):
        """KRİTİK (doğrulayıcının kanıtladığı delik): tek prior + uyumlu-aynı-kaynak (999/999) AMA
        bültenin col1'i bizim CSV-prior'la UYUŞMAZSA → artık YAZMAZ (eskiden yazıyordu)."""
        from screen.fetch_supply_components import buyback_selfcheck
        # sahte: bb=999, bülten-col1=999 (saçma), CSV-prior=234.6 → col1-çapa 999 vs 234.6 = DÜŞER
        ok, detail = buyback_selfcheck(bb_bn=999.0, ttm_bn=999.0, prior_q_bn=999.0, priors=[234.6])
        assert ok is False and "col1" in detail

    def test_tolerances_frozen(self):
        from screen.fetch_supply_components import _SC_TTM_TOL, _SC_PRIOR_TOL
        assert (_SC_TTM_TOL, _SC_PRIOR_TOL) == (1.0, 5.0)
