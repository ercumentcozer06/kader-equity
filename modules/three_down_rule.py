"""three_down_rule — UC-GUN KURALI long-boost (SPX-only; DEPLOY 2026-08-05, EMIR KARARI).

Tez (Suttmeier T2 tani labi -> three_down_rule_ablation E3 kolu): endeksin KENDI kapanis serisinde
3 ARDISIK asagi kapanis = kisa-vadeli kapitulasyon -> izleyen 3 is gununde pozitif drift. Aktif
pencerede final target > 0 ise target x1.25 (yalniz-long boost; target <= 0'a DOKUNULMAZ — stack
zaten long/flat). Sonuc clip: target <= 1.25.

Kanit sicili (backtest/research/three_down_rule_ablation.py + _result.json, 2026-08-05):
E1/E3 kolu SPX her pencerede (full/2014+/2020+/ex20H1) Sharpe 1.661->1.695, fark NW-t +3.27;
maxDD -13.2 -> -14.1 (house dd-no-worse kurali FAIL -> EMIR 2026-08-05 maxDD-taviziyle ACIK ONAY
verdi, deploy emri). NDX her pencerede FAIL (dSharpe -0.055, NW-t 0.41) -> NDX'te overlay KAPALI
(enable_ndx: false). ROLLBACK = config three_down_rule.enable_spx: false (tek anahtar).

PIT (ablation ile BIREBIR): bayrak t kapanisinda olusur (t-2,t-1,t uc kapanis da chg<0);
overlay pozisyon-tarihi t..t+W-1'de aktif -> exec lag=1 ile fiilen etkilenen getiriler t+1..t+W.
Canli serviste: bugunku servis-tarihi son W is gunu icinde bayrak varsa boost aktif.
Takvim ablation'la ayni: hafta-ici (Mon-Fri) takvime ffill-reindex; ffill'li duz gun (chg==0)
asagi SAYILMAZ ve seriyi KESER (ablation notu: "ffill flat gun asagi sayilmaz").

Veri: canli = yfinance ^GSPC gunluk kapanis (frozen prices.parquet SPX kolonunun canli ikizi;
S&P 500 endeks kapanisi). yfinance dusurse data/cache/spx_gspc_long.csv (santa_window cache'i)
taze ise fallback. FAIL-CLOSED NO-OP: seri eksik/bayat (son kapanis > max_age_bd is gunu eski)
-> factor 1.0 + available=False + reason. NOT: bu bir BOOST overlay'i — fail-safe yonu NOTR'dur
(boost uygulanmaz, taban pozisyon korunur); trim-overlay'lerin fail-closed tepe-kapisindan
(position_overlay_block -> model HALT) BILEREK ayri tutulur, veri-yoklugu tum modeli oldurmez.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FALLBACK_CSV = ROOT / "data" / "cache" / "spx_gspc_long.csv"   # santa_window ^GSPC cache (fallback)

WINDOW_BD, BOOST, CAP, MAX_AGE_BD = 3, 1.25, 1.25, 3
_DOWN_STREAK = 3   # bayrak tanimi: 3 ardisik asagi kapanis (SABIT kural; window_bd = aktivasyon penceresi)


def _flag_series(close: pd.Series) -> pd.Series:
    """Hafta-ici takvime ffill-reindex edilmis kapanislardan 3-asagi bayragi (ablation ile birebir)."""
    s = close.dropna().sort_index()
    cb = s.reindex(pd.bdate_range(s.index.min(), s.index.max()), method="ffill")
    chg = cb.diff()
    down = chg < 0
    flag = down
    for k in range(1, _DOWN_STREAK):
        flag = flag & down.shift(k).fillna(False)
    return flag


def boost_factor_series(close: pd.Series, window_bd: int = WINDOW_BD, boost: float = BOOST) -> pd.Series:
    """Tarihsel AKTIVASYON-faktor serisi (backtest/frozen-yol/test tek-kaynak): bayrak-gunu DAHIL
    son window_bd is gununde bayrak varsa boost, yoksa 1.0. Isaret kosulu (target>0) CAGIRANDA —
    bu seri yalniz pencere-aktivasyonunu kodlar (ablation act3 ile birebir)."""
    flag = _flag_series(close)
    act = flag.rolling(int(window_bd), min_periods=1).max().astype(bool)
    return pd.Series(np.where(act, float(boost), 1.0), index=flag.index)


def three_down_snapshot(close_hist: pd.Series | None, window_bd: int = WINDOW_BD,
                        max_age_bd: int | None = None, today=None) -> dict:
    """Son durum: {available, flag_active, days_left, last_flag_date, reason}. FAIL-CLOSED no-op:
    seri yok/kisa/bayat -> available=False + reason (cagiran factor 1.0 uygular).

    max_age_bd=None -> takvim-yas kapisi YOK (frozen yol: snapshot-tazeligi zaten ust katmanda
    damgalanir; as-of degerlendirme deterministik kalir). Canli yol max_age_bd verir."""
    out = {"available": False, "flag_active": False, "days_left": 0,
           "last_flag_date": None, "reason": None}
    if close_hist is None or len(close_hist.dropna()) < _DOWN_STREAK + 1:
        out["reason"] = (f"kapanis serisi eksik (n={0 if close_hist is None else len(close_hist.dropna())} "
                         f"< {_DOWN_STREAK + 1}) — overlay no-op")
        return out
    s = close_hist.dropna().sort_index()
    if max_age_bd is not None:
        today_d = pd.Timestamp(today).date() if today is not None else pd.Timestamp.now().date()
        age_bd = int(np.busday_count(s.index.max().date(), today_d))
        if age_bd > int(max_age_bd):
            out["reason"] = (f"son kapanis {s.index.max().date()} {age_bd} is gunu eski "
                             f"(>{int(max_age_bd)}) — overlay no-op (fail-closed)")
            return out
    flag = _flag_series(s)
    out["available"] = True
    if not bool(flag.any()):
        return out
    last_flag = flag[flag].index[-1]
    # bayraktan bu yana gecen POZISYON-gunu sayisi (bayrak gunu = 0); ablation act3 = rolling(W) dahil
    days_since = int(len(flag.loc[last_flag:]) - 1)
    out["last_flag_date"] = str(last_flag.date())
    if days_since < int(window_bd):
        out["flag_active"] = True
        out["days_left"] = int(window_bd) - days_since
    return out


def _live_closes():
    """Canli ^GSPC gunluk kapanislar (son ~1 ay yeter). yfinance -> fallback santa cache CSV.
    Basarisizlik: (None, kaynak-notu)."""
    try:
        import yfinance as yf
        px = yf.download("^GSPC", period="1mo", auto_adjust=False, progress=False)["Close"]
        if isinstance(px, pd.DataFrame):
            px = px.iloc[:, 0]
        px = px.dropna()
        if len(px):
            px.index = pd.to_datetime(px.index).tz_localize(None)
            return px, "yfinance ^GSPC"
    except Exception as e:
        err = f"yfinance ^GSPC fail ({type(e).__name__})"
    else:
        err = "yfinance ^GSPC bos donus"
    try:
        if FALLBACK_CSV.exists():
            s = (pd.read_csv(FALLBACK_CSV, parse_dates=["Date"]).set_index("Date")["Close"]
                 .dropna().sort_index())
            if len(s):
                return s, f"{err} -> fallback spx_gspc_long.csv"
    except Exception:
        pass
    return None, err


def evaluate(cfg: dict, asset: str = "SPX") -> dict:
    """Canli degerlendirme (SPX). Doner: {factor, available, flag_active, days_left, last_flag_date,
    as_of, age_bd, src, reason}. factor = aktivasyon-faktoru (boost|1.0); isaret kosulu (target>0)
    ve clip CAGIRANDA (run.py asset_deploy katmani). Fail-closed no-op: factor 1.0 + reason."""
    ov = ((cfg.get("three_down_rule", {}) or {}))
    window_bd = int(ov.get("window_bd", WINDOW_BD))
    boost = float(ov.get("boost", BOOST))
    max_age = int(ov.get("max_age_bd", MAX_AGE_BD))
    out = {"factor": 1.0, "available": False, "flag_active": False, "days_left": 0,
           "last_flag_date": None, "as_of": None, "age_bd": None, "src": None, "reason": None}
    if asset != "SPX":
        out["reason"] = f"{asset} icin canli kapanis ucu tanimsiz (yalniz SPX servis edilir)"
        return out
    closes, src = _live_closes()
    out["src"] = src
    if closes is None:
        out["reason"] = f"kapanis serisi alinamadi ({src}) — overlay no-op (fail-closed)"
        return out
    snap = three_down_snapshot(closes, window_bd=window_bd, max_age_bd=max_age)
    out.update({k: snap[k] for k in ("available", "flag_active", "days_left", "last_flag_date", "reason")})
    out["as_of"] = str(closes.index.max().date())
    out["age_bd"] = int(np.busday_count(closes.index.max().date(), pd.Timestamp.now().date()))
    if out["available"] and out["flag_active"]:
        out["factor"] = boost
    return out
