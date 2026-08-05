"""
backtest/research/breadth_regime_ablation — EQUITY BREADTH INTERNALS as regime filters,
ABLATION over the LIVE kader-equity stack replica (Suttmeier/BofA TA primer adaylari).

BASELINE (canli-replika): frozen tide spine (2019-01..2026-05-22) x COR1M-froth(8,11,0)
x GEX-shield(0.5,1.0,0.4) x es_basis unwind-gate (C1a: p75/dz-1.5/504/10/252, floor 0)
= config.yaml'daki DEPLOY stack'in birebiri (hedef canli Sharpe ~ SPX 1.63 / NDX 1.86).
Kaynaklar: spine/frozen, data/cache/{corr_pc,squeeze_dix_gex,es_basis_daily}.parquet;
modules.{cor1m_froth,gex_shield,es_basis_unwind} SADECE OKUNUR (import), degistirilmez.

VERI (breadth panel): S&P 500 CURRENT bileseninden (Wikipedia listesi + yfinance batch,
2014-06-01+) hesaplanir — fetch_breadth_internals.py house-emsali; SURVIVORSHIP caveat
(current-list; lookback derinlestikce iyimser-yanli, ozellikle NH/NL). NYSE genis-market
serileri (Barchart $S5TH/$NYHL/$ADRN) repo'da YOK ve ucretsiz kaynakta (stooq/yfinance
^ADD 404) bulunamadi -> B2/B3 SPX-500-bilesen PROXY'sidir, NYSE degil; deklarasyonda soyle.
McClellan RATIO-ADJUSTED (RANA = 1000*(adv-dec)/(adv+dec); osc = EMA19-EMA39) — issue-count
normalizasyonu, -70 esigi ancak boyle karsilastirilabilir. Panel cache:
backtest/research/_breadth_panel.parquet (data/cache'e YAZILMAZ).

PIT: breadth degerleri ayni-gun EOD; execution lag = +1 gun (engine ile ayni). Tum
donusumler rolling/EMA (trailing-only); full-sample istatistik YOK.

PRE-DECLARED GRID (kosulmadan ONCE deklare; cherry-pick yok, HEPSI raporlanir):
  BASE   : tide x froth x shield x esb  (canli-replika)
  G1     : pct>200dma < 30 VE 20g-dusuyor  -> x0.5 de-risk
  G1inv  : pct>200dma > 70 VE 20g-yukseliyor -> x0.5 (ayna ters-kontrol)
  G1sens : esik 35 (SENSITIVITY, tek varyant)
  G2     : NH/(NH+NL) 10dma > 25dma -> long(1), degilse FLAT(0)  [long/flat gate]
  G2inv  : ters cross (10dma < 25dma -> long)  (ters-kontrol)
  G2sens : 5dma x 20dma cross (SENSITIVITY, tek varyant)
  G3     : McClellan-osc < -70 (son 10 gun icinde) -> x1.25 tilt (entry-timing)
  G3inv  : osc > +70 (son 10 gun) -> x1.25 (ters-kontrol)
  G3sens : esik -50 (SENSITIVITY, tek varyant)
  G3b    : Summation Index 10g-dusuyor -> x0.75 (intermediate rejim; primer'de acikca var)
METRIK: Sharpe + maxDD; pencereler full(2019+ = frozen basi; 2014+ stack'te YOK, frozen
2019'da basliyor — acikca soyle), 2020+, 2020H1-haric. (cand-base) gunluk PnL farki
Newey-West t (lags=10). KARAR: aday TUM pencerelerde Sharpe VE maxDD'de base'i gececek
VE fark ekonomik-gorunur olacak; degilse RED/betimsel. ML YOK, grid-disi avlanma YOK.

  python backtest/research/breadth_regime_ablation.py [--refetch]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from spine import contract as C, tide as T                      # noqa: E402
from backtest import engine as E                                 # noqa: E402
from modules.cor1m_froth import froth_factor_series              # noqa: E402
from modules.gex_shield import gex_zscore, shield_factor_series  # noqa: E402
from modules.es_basis_unwind import unwind_factor_series         # noqa: E402

HERE = Path(__file__).resolve().parent
PANEL = HERE / "_breadth_panel.parquet"
OUT_JSON = HERE / "breadth_regime_ablation_result.json"
MIN_YEARS = 8.0
NW_LAGS = 10


# ── data: component-derived breadth panel ─────────────────────────────────────
def fetch_panel() -> pd.DataFrame:
    import io
    import requests
    import yfinance as yf
    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    tbl = pd.read_html(io.StringIO(html))[0]
    syms = [s.replace(".", "-") for s in tbl["Symbol"].astype(str).tolist()]
    print(f"  S&P 500 listesi (CURRENT -> survivorship caveat): {len(syms)} sembol")
    data = yf.download(syms, start="2014-06-01", auto_adjust=True, progress=False, threads=True)["Close"]
    data = data.dropna(axis=1, how="all")
    data.index = pd.to_datetime(data.index).tz_localize(None)
    print(f"  fiyat matrisi: {data.shape[0]} gun x {data.shape[1]} isim  "
          f"{data.index.min().date()}..{data.index.max().date()}")
    ma200 = data.rolling(200, min_periods=150).mean()
    n_ok = data.notna().sum(axis=1)
    pct200 = (data > ma200).sum(axis=1) / n_ok * 100
    hi252 = data.rolling(252, min_periods=200).max()
    lo252 = data.rolling(252, min_periods=200).min()
    nh = (data >= hi252).sum(axis=1)         # 52w new high (bugun >= trailing-252 max)
    nl = (data <= lo252).sum(axis=1)
    chg = data.pct_change()
    adv = (chg > 0).sum(axis=1)
    dec = (chg < 0).sum(axis=1)
    out = pd.DataFrame({"pct200": pct200, "nh": nh, "nl": nl, "adv": adv, "dec": dec,
                        "n_names": n_ok})
    out = out[out["n_names"] > 300]
    out.to_parquet(PANEL)
    return out


def load_panel(refetch: bool) -> pd.DataFrame:
    if PANEL.exists() and not refetch:
        return pd.read_parquet(PANEL)
    return fetch_panel()


# ── metrics ───────────────────────────────────────────────────────────────────
def _sh(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")


def _dd(r):
    eq = (1 + r.dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min())


def strat_ret(pos: pd.Series, close: pd.Series, lag: int = 1) -> pd.Series:
    idx = pos.index
    ret = E.fwd_ret(close, idx).values
    p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def win_slice(r: pd.Series, wname: str) -> pd.Series:
    if wname == "2020+":
        return r[r.index >= "2020-01-01"]
    if wname == "ex20H1":
        return r[(r.index < "2020-01-01") | (r.index > "2020-06-30")]
    return r


def nw_t(d: pd.Series, lags: int = NW_LAGS) -> float:
    """Newey-West t-stat of mean(d), Bartlett kernel, lags=10."""
    d = d.dropna().values
    n = len(d)
    if n < 50:
        return float("nan")
    mu = d.mean()
    e = d - mu
    s = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        s += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(s / n)
    return float(mu / se) if se > 0 else float("nan")


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    refetch = "--refetch" in sys.argv
    print("=" * 100)
    print("  BREADTH REGIME ABLATION — canli-stack replika uzerinde (kader-equity house harness)")
    print("=" * 100)

    # ---- 0) DATA CENSUS (sonuclardan ONCE) ----
    print("\n  [0] DATA CENSUS")
    pan = load_panel(refetch)
    census, halt = {}, {}
    series_map = {
        "B1 pct>200dma (SPX-bilesen)": pan["pct200"].dropna(),
        "B2 NH (52w, bilesen)":        pan["nh"].dropna(),
        "B2 NL (52w, bilesen)":        pan["nl"].dropna(),
        "B3 adv-dec (bilesen proxy)":  (pan["adv"] - pan["dec"]).dropna(),
    }
    for name, s in series_map.items():
        yrs = (s.index.max() - s.index.min()).days / 365.25
        ok = yrs >= MIN_YEARS
        leg = name.split()[0]
        halt[leg] = halt.get(leg, False) or (not ok)
        census[name] = {"n": int(len(s)), "start": str(s.index.min().date()),
                        "end": str(s.index.max().date()), "years": round(yrs, 1),
                        "status": "OK" if ok else "HALT(<8y)"}
        print(f"    {name:<30} N={len(s):>5}  {s.index.min().date()}..{s.index.max().date()}"
              f"  ({yrs:4.1f}y)  {'OK' if ok else 'HALT (<8y) — bu bacak KOSULMAZ'}")
    # cache tutarlilik capraz-kontrolu (repo'nun kendi breadth_internals'i ile)
    try:
        old = pd.read_parquet(ROOT / "data" / "cache" / "breadth_internals.parquet")["pct_above_200d"]
        both = pd.concat([pan["pct200"], old], axis=1).dropna()
        corr = float(both.corr().iloc[0, 1])
        print(f"    capraz-kontrol: yeni pct200 vs repo cache corr={corr:.3f} (n={len(both)})")
    except Exception as e:
        corr = None
        print(f"    capraz-kontrol atlandi: {type(e).__name__}")

    # ---- baseline live-stack replica ----
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    cor = pd.read_parquet(ROOT / "data" / "cache" / "corr_pc.parquet")["COR1M"].dropna()
    gex = pd.read_parquet(ROOT / "data" / "cache" / "squeeze_dix_gex.parquet")["gex"].dropna()
    esb = pd.read_parquet(ROOT / "data" / "cache" / "es_basis_daily.parquet")["spread_bps"].dropna()
    froth = froth_factor_series(cor.reindex(idx, method="ffill"), 8, 11, 0.0)
    shield = shield_factor_series(gex_zscore(gex).reindex(idx, method="ffill"), 0.5, 1.0, 0.4)
    esb_gate = unwind_factor_series(esb).reindex(idx, method="ffill").fillna(1.0)
    stack = (tdir * froth * shield * esb_gate).reindex(idx)
    win = prov.get("window", {})
    print(f"\n  [1] BASELINE = canli-replika stack (tide x froth x GEX-shield x esb-gate)")
    print(f"      frozen pencere {win.get('start')}..{win.get('end')} ({win.get('n_days')} gun)."
          f"  NOT: 2014+ penceresi frozen spine'da YOK -> 'full' = 2019+ (veri-basi).")

    # ---- breadth transforms (trailing-only) ----
    p200 = pan["pct200"].reindex(idx, method="ffill")
    fall20 = p200 < p200.shift(20)
    rise20 = p200 > p200.shift(20)
    nhnl = (pan["nh"] / (pan["nh"] + pan["nl"]).replace(0, np.nan)).fillna(0.5)
    r10, r25 = nhnl.rolling(10).mean(), nhnl.rolling(25).mean()
    r5, r20 = nhnl.rolling(5).mean(), nhnl.rolling(20).mean()
    rana = (1000.0 * (pan["adv"] - pan["dec"]) / (pan["adv"] + pan["dec"]).replace(0, np.nan))
    mco = ema(rana, 19) - ema(rana, 39)
    si = mco.cumsum()
    si_fall = si < si.shift(10)

    def rx(s):  # align to stack index, EOD ffill
        return s.reindex(idx, method="ffill")

    def recent(cond: pd.Series, k: int = 10) -> pd.Series:
        return cond.rolling(k, min_periods=1).max().astype(bool)

    gates = {
        "G1 p200<30 & fall -> x0.5":    pd.Series(np.where(rx(p200.lt(30)) & rx(fall20), 0.5, 1.0), index=idx),
        "G1inv p200>70 & rise -> x0.5": pd.Series(np.where(rx(p200.gt(70)) & rx(rise20), 0.5, 1.0), index=idx),
        "G1sens esik35 [SENS]":         pd.Series(np.where(rx(p200.lt(35)) & rx(fall20), 0.5, 1.0), index=idx),
        "G2 NHNL 10x25 long/flat":      rx((r10 > r25).astype(float)),
        "G2inv ters-cross":             rx((r10 < r25).astype(float)),
        "G2sens 5x20 [SENS]":           rx((r5 > r20).astype(float)),
        "G3 MCO<-70 -> x1.25 10g":      pd.Series(np.where(rx(recent(mco < -70)), 1.25, 1.0), index=idx),
        "G3inv MCO>+70 -> x1.25 10g":   pd.Series(np.where(rx(recent(mco > 70)), 1.25, 1.0), index=idx),
        "G3sens esik-50 [SENS]":        pd.Series(np.where(rx(recent(mco < -50)), 1.25, 1.0), index=idx),
        "G3b SI 10g-dusuyor -> x0.75":  pd.Series(np.where(rx(si_fall), 0.75, 1.0), index=idx),
    }
    # HALT enforcement: bacagi dusen gate kosulmaz
    for leg, bad in halt.items():
        if bad:
            gates = {k: v for k, v in gates.items() if not k.startswith(leg.replace("B", "G"))}

    windows = ["full", "2020+", "ex20H1"]
    results = {"census": census, "cross_check_corr": corr, "arms": {}}
    base_r = {a: strat_ret(stack, prices[a]) for a in ("SPX", "NDX")}

    print(f"\n  [2] ARM TABLOSU — Sharpe / maxDD per pencere + NW-t(lags={NW_LAGS}, full)  "
          f"(TUM kollar raporlanir)")
    for a in ("SPX", "NDX"):
        print(f"\n  [{a}]  base full Sh {_sh(base_r[a]):+.3f} dd {100*_dd(base_r[a]):+.0f}%")
        hdr = f"    {'kol':<30}"
        for w in windows:
            hdr += f"{w+' Sh':>10}{'dd':>7}"
        hdr += f"{'NW-t':>7}{'karar':>10}"
        print(hdr)
        # baseline row
        row = f"    {'BASE (canli-replika)':<30}"
        for w in windows:
            rr = win_slice(base_r[a], w)
            row += f"{_sh(rr):>+10.3f}{100*_dd(rr):>+6.0f}%"
        print(row + f"{'':>7}{'':>10}")
        results["arms"][a] = {"BASE": {w: {"sharpe": round(_sh(win_slice(base_r[a], w)), 3),
                                          "maxdd": round(_dd(win_slice(base_r[a], w)), 3)}
                                      for w in windows}}
        for label, fac in gates.items():
            vr = strat_ret((stack * fac).reindex(idx), prices[a])
            beats = True
            entry = {}
            row = f"    {label:<30}"
            for w in windows:
                rb, rv = win_slice(base_r[a], w), win_slice(vr, w)
                shb, shv, ddb, ddv = _sh(rb), _sh(rv), _dd(rb), _dd(rv)
                beats = beats and (shv > shb) and (ddv >= ddb)
                entry[w] = {"sharpe": round(shv, 3), "maxdd": round(ddv, 3),
                            "d_sharpe": round(shv - shb, 3)}
                row += f"{shv:>+10.3f}{100*ddv:>+6.0f}%"
            diff = (vr - base_r[a]).dropna()
            t = nw_t(diff)
            ann_diff = float(diff.mean() * 252)
            visible = abs(ann_diff) >= 0.005 and abs(t) >= 2.0   # >=50bp/yil ve |t|>=2
            verdict = "DEPLOY-aday" if (beats and visible and ann_diff > 0) else "RED"
            entry["nw_t"] = round(t, 2)
            entry["ann_pnl_diff"] = round(ann_diff, 4)
            entry["verdict"] = verdict
            results["arms"][a][label] = entry
            print(row + f"{t:>+7.2f}{verdict:>12}")

    # per-gate verdict (iki varlikta da gecmeli)
    print("\n  [3] KAPI-BAZLI NIHAI KARAR (SPX VE NDX ikisinde de tum-pencere gecis sarti)")
    finals = {}
    for label in gates:
        v = all(results["arms"][a][label]["verdict"] == "DEPLOY-aday" for a in ("SPX", "NDX"))
        finals[label] = "DEPLOY-aday" if v else "RED"
        print(f"    {label:<32} {finals[label]}")
    for leg, bad in halt.items():
        if bad:
            finals[leg] = "VERI-YOK-HALT"
            print(f"    {leg:<32} VERI-YOK-HALT")
    results["final"] = finals
    results["notes"] = ["survivorship: CURRENT S&P500 listesi (house-emsal fetch_breadth_internals)",
                        "B2/B3 = SPX-500 bilesen PROXY (NYSE genis-market degil)",
                        "McClellan ratio-adjusted (RANA), -70 esigi bu normalizasyonla",
                        "baseline = canli-replika stack; frozen 2019+; 2014+ penceresi mevcut degil"]
    OUT_JSON.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\n  JSON -> {OUT_JSON}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
