"""
backtest/intraday_gex_v0 — Ş2b. INTRADAY GEX sleeve (=1 DSR-trial). VERİ GELİNCE çalışır (Ş2a: SPY+QQQ 1-dk
bar 2019+ & EOD chain strike-OI). ŞİMDİ: kural mantığı + friction PURE-FONKSİYON olarak kurulu + sentetik-bar
ile test edilir; GERÇEK VERİ YOK → load_data() net hata verir, SONUÇ İDDİA EDİLMEZ.

KURALLAR (a priori SABİT, grid YOK):
  R1 (mean-revert fade): pozitif-γ & spot HVL–call-wall bandında → UÇ FADE (short), hedef HVL, stop call-wall üstü.
  R2 (momentum):         negatif-γ & spot flip-altı KIRILIM → momentum devam (short dahil), hedef aşağı, stop flip üstü.
  F1 (no-trade filtre):  OPEX / CPI / FOMC günü → işlem YOK.
FRICTION: CFD spread 0.4–0.6pt + slippage (round-trip indeks-puanı → getiriye çevrilir).
KABUL (pre-registered): TIDE-only baz çizgisine göre prop_sim'de funded'a-SÜRE ↓ VE $/eval ↑;
  AYRICA sleeve günlük sub-budget: (TIDE-kitap + sleeve) KOMBİNE en-kötü gün < %5 (FTMO günlük limit).
ÇIKTI (veri gelince): sleeve'li vs sleeve'siz KARAR tablosu. Standalone Sharpe DEĞİL — prop_sim iyileşmesi ölçüt.

PIT: rejim/seviyeler GÜN-ÖNCESİ EOD chain'den (t-1 kapanış); intraday işlem gün-içi bar'da. Look-ahead yok.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── a priori SABİT parametreler (taranmaz) ──
SPREAD_PT = 0.5           # CFD spread (0.4–0.6pt aralığının ortası; round-trip'te ×2 bacak)
SLIPPAGE_PT = 0.25        # bacak başına slippage (a priori)
R1_STOP_BUF = 0.25        # call-wall üstü stop tamponu (HVL-callwall mesafesinin oranı)
R2_STOP_BUF = 0.25        # flip üstü stop tamponu
NEG_GAMMA_Z = 1.0         # gex_shield mevcut eşiği (YENİ DEĞİL): z < −1 → negatif-γ rejim


def r1_fade(spot, hvl, call_wall, regime_positive_gamma):
    """R1: pozitif-γ & spot ∈ (HVL, call_wall] → uç FADE short, hedef HVL. Aksi None."""
    if not regime_positive_gamma or hvl is None or call_wall is None:
        return None
    if hvl < spot <= call_wall and call_wall > hvl:
        stop = call_wall + R1_STOP_BUF * (call_wall - hvl)
        return {"rule": "R1", "side": "SHORT", "entry": spot, "target": hvl, "stop": stop}
    return None


def r2_momentum(spot, flip, regime_negative_gamma, broke_below_flip):
    """R2: negatif-γ & spot flip-altı kırılım → momentum devam SHORT. Aksi None."""
    if not regime_negative_gamma or flip is None or not broke_below_flip:
        return None
    if spot < flip:
        stop = flip + R2_STOP_BUF * abs(flip) * 0.0 + flip * 0.0 + flip  # stop = flip seviyesi (üstü)
        return {"rule": "R2", "side": "SHORT", "entry": spot, "target": None, "stop": flip}
    return None


def f1_no_trade(is_opex: bool, is_cpi: bool, is_fomc: bool) -> bool:
    """F1: OPEX/CPI/FOMC günü → True (işlem YOK)."""
    return bool(is_opex or is_cpi or is_fomc)


def friction_return(entry_px: float, spread_pt: float = SPREAD_PT, slippage_pt: float = SLIPPAGE_PT) -> float:
    """Round-trip friction (giriş+çıkış): (spread + 2·slippage) indeks-puanı / entry → getiri-drag (negatif)."""
    rt_pt = spread_pt + 2.0 * slippage_pt              # 1 spread (giriş+çıkış net ~1) + 2 bacak slippage
    return -(rt_pt / entry_px) if entry_px else 0.0


def signal(bar: dict, levels: dict, calendar: dict):
    """Tek-bar sinyal: F1 filtre → R1/R2. bar={spot,broke_below_flip,...}, levels={hvl,call_wall,flip,
    regime_z,net_gex}, calendar={is_opex,is_cpi,is_fomc}. PIT: levels t-1 EOD'den."""
    if f1_no_trade(calendar.get("is_opex"), calendar.get("is_cpi"), calendar.get("is_fomc")):
        return None
    pos_gamma = (levels.get("net_gex", 0) or 0) > 0 or (levels.get("regime_z") is not None and levels["regime_z"] >= -NEG_GAMMA_Z)
    neg_gamma = (levels.get("regime_z") is not None and levels["regime_z"] < -NEG_GAMMA_Z)
    s = r1_fade(bar.get("spot"), levels.get("hvl"), levels.get("call_wall"), pos_gamma)
    if s:
        return s
    return r2_momentum(bar.get("spot"), levels.get("flip"), neg_gamma, bar.get("broke_below_flip", False))


def load_data():
    """Ş2a verisi (1-dk bar + EOD chain). HENÜZ YOK → net hata. Veri gelince burası doldurulur."""
    raise NotImplementedError(
        "VERİ YOK — Ş2a bekleniyor (SPY+QQQ 1-dk bar 2019+ & EOD chain strike-OI). "
        "Veri tedarik edilince load_data() + run() doldurulacak; sonuç ŞİMDİ İDDİA EDİLEMEZ.")


def run():
    """Sleeve'i prop_sim TIDE-kitabına ekle, sleeve'li vs sleeve'siz prop_sim karşılaştır. VERİ GELİNCE."""
    load_data()                                        # raises — veri yok


if __name__ == "__main__":
    print("Ş2b intraday_gex_v0 — KURAL MANTIĞI kurulu + test edilir; GERÇEK VERİ YOK (Ş2a bekleniyor).")
    print("Kabul ölçütü: prop_sim funded-süre↓ & $/eval↑ & kombine-en-kötü-gün<%5. Standalone Sharpe DEĞİL.")
    run()
