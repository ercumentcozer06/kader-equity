"""
engine/state — MarketState'i kurar: kilitli model kararı (run.build_decision) + canlı gamma/vol snapshot'ları
(gamma_engine → data/cache/gamma_spy, surface_yf → surface_<tic>) → DECISION'ın tükettiği tek dict.

Snapshot'lar forward-collector tarafından (collect_daily / gamma_engine / surface_yf) üretilir; STATE onları
okur (ayırma: collector=veri, motor=karar). Tazelik kapısı snapshot yaşını işaretler (Bible: bayat≠canlı-çağrı).
"""
from __future__ import annotations
import json
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
TIC_MULT = {"SPY": ("SPX", 1), "QQQ": ("NDX", 41)}  # SPY arg -> CBOE _SPX (spot zaten index-native, mult=1); QQQ -> QQQ ETF (×41 NDX-eşdeğeri). gamma_engine/surface_yf CFG ile tek-kaynak (2026-06-17 CBOE geçişi).


def _latest(subdir: str) -> dict | None:
    d = CACHE / subdir
    files = sorted(d.glob("*.json")) if d.exists() else []
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


def _ts_ratio(surface: dict | None) -> float | None:
    """vol term: kısa-DTE ATM-IV / uzun-DTE ATM-IV (>1 backwardation/stres, <1 contango/sakin)."""
    if not surface:
        return None
    s = surface.get("surface", {}) or {}
    ivs = [(int(k.rstrip("d")), v.get("atm_iv")) for k, v in s.items() if v.get("atm_iv")]
    ivs = sorted(ivs)
    if len(ivs) < 2:
        return None
    return round(ivs[0][1] / ivs[-1][1], 3)


def _atm_iv_at(surface: dict | None, target_dte: int = 30) -> float | None:
    """Hedef-DTE'ye en yakın ATM-IV (%). VRP için ~30g (VIX/1-ay standardı)."""
    if not surface:
        return None
    s = surface.get("surface", {}) or {}
    ivs = [(int(k.rstrip("d")), v.get("atm_iv")) for k, v in s.items() if v.get("atm_iv")]
    return min(ivs, key=lambda x: abs(x[0] - target_dte))[1] if ivs else None


def _realized_vol_ewma(ticker: str, lam: float = 0.94) -> float | None:
    """EWMA realized vol (annualized %). λ=0.94 = RiskMetrics günlük standardı (a priori, optimize EDİLMEDİ;
    half-life ~11g → ~30g implied ile eşleşir). VRP = implied − realized'in realized bacağı."""
    try:
        import numpy as np
        import yfinance as yf
        h = yf.Ticker(ticker).history(period="3mo")["Close"].dropna()
        lr = np.log(h / h.shift(1)).dropna()
        if len(lr) < 10:
            return None
        ew = lr.ewm(alpha=1 - lam, adjust=False).std().iloc[-1]
        return round(float(ew) * np.sqrt(252) * 100, 2)
    except Exception:
        return None


def build_state(cfg: dict, ticker: str = "SPY") -> tuple[dict, dict, dict]:
    """Döndürür (model_decision, market_state, meta). meta = tazelik/kaynak.

    Motor CANLI ister (günlük saf-sinyal): spine.source=live zorlanır (config.yaml default=frozen kalır →
    pytest/backtest hızlı+reproducible). Canlı başarısızsa (FRED/ağ) frozen'a düşer, açık not bırakılır."""
    import copy
    import run
    cfg_live = copy.deepcopy(cfg)
    cfg_live.setdefault("spine", {})["source"] = "live"
    try:
        model = run.build_decision(cfg_live)
    except Exception as e:                                  # H1: fallback HER ZAMAN force-frozen (cfg=live/CLI olsa
        cfg_frozen = copy.deepcopy(cfg)                     #      bile crash YOK; STALE-damgalı muhafazakâr taşıma)
        cfg_frozen.setdefault("spine", {})["source"] = "frozen"
        model = run.build_decision(cfg_frozen)
        model["_live_error"] = f"{type(e).__name__}: {e}"
    # H7: QQQ gamma yoksa SPY'a düşüş SESSİZ değil — açık flag (brief'te uyarı). Ticker-doğru tercih edilir.
    gsub = "gamma_spy" if ticker == "SPY" else f"gamma_{ticker.lower()}"
    gamma = _latest(gsub)
    gamma_fallback = False
    if gamma is None and ticker != "SPY":
        gamma = _latest("gamma_spy"); gamma_fallback = gamma is not None
    surf = _latest(f"surface_{ticker.lower()}")
    surf_fallback = False
    if surf is None and ticker != "SPY":
        surf = _latest("surface_spy"); surf_fallback = surf is not None
    cor1m = ((model.get("overlays", {}) or {}).get("cor1m_froth", {}) or {}).get("cor1m")

    g = gamma or {}
    front = None
    if surf:
        s = surf.get("surface", {}) or {}
        if s:
            k0 = sorted(s, key=lambda k: int(k.rstrip("d")))[0]
            front = s[k0]

    state = {
        "ticker": ticker,
        "spot": g.get("spot"),
        "net_gex_bn": g.get("net_gex_bn"),
        "net_vanna_m": g.get("net_vanna_m"),
        "net_charm_m": g.get("net_charm_m"),
        "gex_flip": g.get("gex_flip"),
        "call_wall": g.get("call_wall"),                # #3: raw-OI yapısal direnç (Emir'in CALL WALL'u)
        "ghost": g.get("ghost"),                        # #3: gamma-peak yakın seviye (Emir'in GHOST'u)
        "put_wall": g.get("put_wall"),
        "max_pain": g.get("max_pain"),
        "exp_move_1d": g.get("exp_move_1d"),
        "rv_ratio": g.get("rv_ratio"),
        "gamma_regime": g.get("regime"),
        "cor1m": cor1m,
        "atm_iv": (front or {}).get("atm_iv"),
        "rr_skew": (front or {}).get("rr_skew"),
        "ts_ratio": _ts_ratio(surf),
    }
    # VRP (GÖREV 3): ~30g ATM implied − EWMA realized (vol-puanı). >0 implied zengin (prim-sat lehine),
    # düşük/negatif implied ucuz (prim-sat kanaması → konveksite-al lehine). Eşik decision'da (a priori).
    atm30 = _atm_iv_at(surf, 30)
    rvol = _realized_vol_ewma(ticker)
    state["atm_iv_30d"] = atm30
    state["realized_vol"] = rvol
    state["vrp"] = (round(float(atm30) - float(rvol), 2) if (atm30 is not None and rvol is not None) else None)
    # tazelik: snapshot yaşı
    today = datetime.now(timezone.utc).date()
    snap_as_of = g.get("as_of") or (surf or {}).get("as_of")
    age = None
    if snap_as_of:
        try:
            age = (today - date.fromisoformat(snap_as_of)).days
        except Exception:
            pass
    from engine import dataguard as DG                  # GÖREV 6b: veri kalite kapısı
    dq = DG.validate(gamma, surf)
    meta = {
        "snapshot_as_of": snap_as_of, "snapshot_age_days": age,
        "gamma_available": gamma is not None, "surface_available": surf is not None,
        "model_call_status": model.get("call_status"),
        "stale": (age is not None and age > 3) or (model.get("call_status") == "STALE"),
        "data_ok": dq["ok"], "data_fails": dq["fails"], "data_checks": dq["checks"],
        "data_junk": not dq["ok"],                      # True → trade üretme (VERİ ÇÖP)
        "gamma_fallback_to_spy": gamma_fallback,        # H7: QQQ→SPY gamma düşüşü (açık uyarı)
        "surface_fallback_to_spy": surf_fallback,
        "index": TIC_MULT.get(ticker, (ticker, 1))[0], "mult": TIC_MULT.get(ticker, (ticker, 1))[1],
    }
    return model, state, meta
