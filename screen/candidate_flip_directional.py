"""
screen/candidate_flip_directional — wave-2 ADAY: GAMMA-FLIP yön/rejim sinyali (front-month-proxy zincirden).

candidate_gex.py iki-aşamalı klonu. Sinyal = sign(spot − flip) (BİRİM-DEĞİŞMEZ işaret; SPY-spot vs SPY-flip
ile rejim, INDEX SPX/NDX getirisine uygulanır) + flip-distance z. data/cache/flip_history_{spy,qqq}.parquet okur.

AŞAMA 1 — STANDALONE directional (overlap penceresi):
  'spot >= flip GERÇEKTEN +fwd-getiri predict ediyor mu?' her asset {SPX via SPY, NDX via QQQ} için
  IC (fwd 1/5/21g INDEX getiri) + hit-rate + LONG/FLAT standalone Sharpe + alt-dönem. HEM naive HEM pw13.

AŞAMA 2 — INCREMENTAL over TIDE (4 form, 4 hipotez):
  base = strat_ret(tide_dir, INDEX_prices) overlap penceresinde; variant = strat_ret(tide_dir × overlay).
  (a) directional GATE  : spot>=flip → 1.0 else floor∈{0.0,0.4}
  (b) flip-distance z SHIELD : gex_shield FORMU AYNEN (k0.5/thr1.0/floor0.4 A-PRİORİ, yeni fit YOK)
  (c) cross-event 3g trim    : flip-sign-değişimi ±3g pencerede trim (rejim-geçişi kırılgan)
  (d) proximity trim         : |spot-flip| < exp_move iken trim (flip'e yakın = whipsaw riski)
  Metrikler: Sharpe, ΔSharpe, maxDD, CVaR5, exposure, P(v>b) paired-bootstrap (_util), BH-FDR.

KABUL:
  • directional PASS = STRICT both-{SPX,NDX} P(v>b) >= 95% (BH-FDR α=0.05) — TÜM 4 form üzerinde FDR (cherry-pick yok).
  • SHIELD PASS = maxDD/CVaR iyileşme @ Sharpe>=0 (gex_shield precedent).
  modules/flip_regime.py YARATILMAZ (Stage-2 PASS etmeden canlı modül yok).

CAVEAT (raporda tümü): front-month-proxy flip (canlı çok-vadeli DEĞİL), IV-from-mid, 243g TEK-REJİM
  (tide overlap'te kalıcı-LONG → base degenere), 0DTE-çağı, OOS-yok, in-sample.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from spine import contract as C, tide as T                          # noqa: E402
from backtest import engine as E                                    # noqa: E402
from screen._util import load_price_csv, paired_win_prob, fdr_bh    # noqa: E402
from modules.gex_shield import shield_factor_series                 # noqa: E402

DESK = Path(r"C:\Users\admin\Desktop\backtesting")
CACHE = ROOT / "data" / "cache"
PRICES = {"SPX": "SPX_daily.csv", "NDX": "NASDAQ_daily.csv"}
ASSET_TK = {"SPX": "spy", "NDX": "qqq"}   # INDEX getirisi <- ilgili ETF zincir-rejimi (birim-değişmez işaret)

# overlap içi alt-dönemler (243g tek-yıl; kabaca yarıya bölünmüş + son-çeyrek stres)
SUBPER = {"H1(25H2)": ("2025-06-13", "2025-12-31"), "H2(26)": ("2026-01-01", "2026-06-08")}


def _flip_df(tk: str) -> pd.DataFrame:
    p = CACHE / f"flip_history_{tk}.parquet"
    if not p.exists():
        raise SystemExit(f"HALT: {p} yok — önce screen/reconstruct_flip_history.py koş.")
    df = pd.read_parquet(p)
    df["as_of"] = pd.to_datetime(df["as_of"]).dt.normalize()
    return df.set_index("as_of").sort_index()


def z(s: pd.Series, win: int = 252, mp: int = 60) -> pd.Series:
    return (s - s.rolling(win, min_periods=mp).mean()) / s.rolling(win, min_periods=mp).std()


def strat_ret(pos: pd.Series, close: pd.Series, lag: int = 1) -> pd.Series:
    """signal[t] -> t+1 getiri, +1g exec lag (look-ahead-free). engine.fwd_ret kullanır."""
    idx = pos.index
    ret = E.fwd_ret(close, idx).values
    p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def _sh(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")


def _dd(r):
    eq = (1 + r.dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min()) if len(eq) else float("nan")


def _cv(r, q=0.05):
    r = r.dropna().values
    if len(r) == 0:
        return float("nan")
    k = max(1, int(q * len(r)))
    return float(np.sort(r)[:k].mean())


def _ep(r, s, e):
    w = r[(r.index >= pd.Timestamp(s)) & (r.index <= pd.Timestamp(e))]
    return _sh(w)


# ============================================================ STAGE 1
def stage1():
    print("\n" + "=" * 100)
    print("  AŞAMA 1 — STANDALONE DIRECTIONAL:  sign(spot−flip) GERÇEKTEN +fwd-getiri predict ediyor mu?")
    print("  sinyal_dir = +1 if spot>=flip else 0 (LONG/FLAT). dist_z = z(spot−flip). HEM naive HEM pw13.")
    print("=" * 100)
    print(f"  {'asset':<6}{'flip':<7}{'IC1d':>8}{'IC5d':>8}{'IC21d':>8}{'hit1d':>8}"
          f"{'Sh(L/F)':>9}{'Sh B&H':>9}{'expo':>7}   alt-dönem(Sh L/F)")
    rows = []
    for a, fn in PRICES.items():
        close = load_price_csv(DESK / fn)
        fdf = _flip_df(ASSET_TK[a])
        idx = fdf.index.intersection(close.index)
        if len(idx) < 30:
            print(f"  {a}: overlap {len(idx)} (<30) — atla")
            continue
        cb = close.reindex(idx, method="ffill")
        f1 = E.fwd_ret(close, idx)
        f5 = cb.shift(-5) / cb - 1
        f21 = cb.shift(-21) / cb - 1
        for kind, flipcol in (("naive", "gamma_flip_naive"), ("pw13", "gamma_flip_pw13")):
            flip = fdf.loc[idx, flipcol]
            spot = fdf.loc[idx, "spot"]
            dist = (spot - flip)                       # >0 = spot ÜSTÜ (long-gamma)
            distz = z(dist).reindex(idx)
            sig = (spot >= flip).astype(float)         # directional LONG/FLAT
            # IC = continuous flip-distance z vs fwd getiri (yön gücü)
            ic1 = distz.corr(f1, method="spearman")
            ic5 = distz.corr(f5, method="spearman")
            ic21 = distz.corr(f21, method="spearman")
            hit = float((sig.astype(int) == (f1 > 0).astype(int)).mean())
            rr = strat_ret(sig, close)
            bh = E.fwd_ret(close, idx).dropna()
            eps = "  ".join(f"{k}:{_ep(rr,s,e):+.1f}" for k, (s, e) in SUBPER.items())
            print(f"  {a:<6}{kind:<7}{ic1:>+8.3f}{ic5:>+8.3f}{ic21:>+8.3f}{hit:>8.2f}"
                  f"{_sh(rr):>+9.2f}{_sh(bh):>+9.2f}{float(sig.mean()):>7.2f}   {eps}")
            rows.append({"asset": a, "flip": kind, "ic1": ic1, "ic5": ic5, "ic21": ic21,
                         "hit": hit, "sh_lf": _sh(rr), "sh_bh": _sh(bh)})
    print("  (IC>0 = flip-ÜSTÜ→sonraki-getiri+ momentum; IC<0 = contrarian. Sh(L/F) ~ Sh B&H => sinyal değer katmıyor.)")
    return rows


# ============================================================ STAGE 2 overlays
def ov_gate(spot, flip, floor):
    """(a) directional GATE: spot>=flip → 1.0 else floor."""
    return pd.Series(np.where(spot.values >= flip.values, 1.0, floor), index=spot.index)


def ov_shield(dist, k=0.5, thr=1.0, floor=0.4):
    """(b) flip-distance z SHIELD: gex_shield FORMU AYNEN. dist DÜŞÜK (spot flip-altı/kırılgan) → trim.
    zg = z(dist); zg ≤ −thr → trim. shield_factor_series ile byte-aynı."""
    zg = z(dist)
    return shield_factor_series(zg, k=k, thr=thr, floor=floor)


def ov_cross_trim(flip_sign: pd.Series, window=3, floor=0.4):
    """(c) cross-event 3g trim: flip-işaret-değişimi (rejim-geçiş) ±window içinde trim (kırılgan)."""
    chg = (flip_sign.diff().abs() > 0).astype(int)
    near = chg.rolling(2 * window + 1, center=True, min_periods=1).max().fillna(0)
    return pd.Series(np.where(near.values > 0, floor, 1.0), index=flip_sign.index)


def ov_proximity_trim(dist_abs: pd.Series, exp_move: pd.Series, floor=0.4):
    """(d) proximity trim: |spot-flip| < exp_move (flip'e yakın → whipsaw riski) iken trim."""
    near = (dist_abs < exp_move).fillna(False)
    return pd.Series(np.where(near.values, floor, 1.0), index=dist_abs.index)


def stage2():
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    print("\n" + "=" * 100)
    print("  AŞAMA 2 — INCREMENTAL over TIDE (4 form = 4 hipotez):  base = tide_dir, variant = tide × overlay")
    print("  STRICT BH-FDR α=0.05 TÜM formlar üzerinde (cherry-pick yok). directional PASS = both P(v>b)>=95%.")
    print("=" * 100)
    # naive flip kullanılır (directional rejim için kanonik; pw13 Stage-1'de karşılaştırma olarak duruyor)
    flips = {a: _flip_df(ASSET_TK[a]) for a in PRICES}

    # overlap penceresi = tide ∩ flip ∩ index
    base_window = {}
    for a in PRICES:
        close = load_price_csv(DESK / PRICES[a])
        idx = tdir.index.intersection(flips[a].index).intersection(prices[a].index)
        base_window[a] = idx
        b = strat_ret(tdir.reindex(idx), prices[a])
        print(f"\n  [{a}]  overlap {len(idx)}g {idx.min().date()}..{idx.max().date()} | "
              f"base tide-LONG günleri %{100*tdir.reindex(idx).mean():.0f}")
        print(f"        base tide Sharpe {_sh(b):+.3f}  maxDD {100*_dd(b):+.0f}%  CVaR {100*_cv(b):+.2f}%  expo {b.ne(0).mean():.2f}")

    # form tanımları (label -> overlay üreten fn(a, idx))
    def forms_for(a, idx):
        fdf = flips[a]
        spot = fdf.loc[idx, "spot"]; flip = fdf.loc[idx, "gamma_flip_naive"]
        dist = (spot - flip); fsign = np.sign(dist)
        dabs = dist.abs(); em = fdf.loc[idx, "exp_move_1d"]
        return {
            "(a) GATE floor0.0":  ov_gate(spot, flip, 0.0),
            "(a) GATE floor0.4":  ov_gate(spot, flip, 0.4),
            "(b) dist-z SHIELD":  ov_shield(dist),                       # a-priori k.5/thr1/fl.4
            "(c) cross-3g trim":  ov_cross_trim(fsign, 3, 0.4),
            "(d) proximity trim": ov_proximity_trim(dabs, em, 0.4),
        }

    # her form için (label) -> {asset: P(v>b)} topla → FDR TÜM form×asset üzerinde
    all_pvals = {}      # "label::asset" -> 1 - P(v>b)
    rowstore = []       # tablo satırları
    metrics = {}        # (label,asset) -> dict
    for a in PRICES:
        idx = base_window[a]
        b = strat_ret(tdir.reindex(idx), prices[a])
        forms = forms_for(a, idx)
        for label, ov in forms.items():
            pos = (tdir.reindex(idx) * ov.reindex(idx)).clip(0, 1)
            v = strat_ret(pos, prices[a])
            wp = paired_win_prob(b, v)
            m = {"sh": _sh(v), "dsh": _sh(v) - _sh(b), "dd": _dd(v), "cv": _cv(v),
                 "expo": float(pos.ne(0).mean()), "wp": wp,
                 "dd_b": _dd(b), "cv_b": _cv(b)}
            metrics[(label, a)] = m
            if wp is not None:
                all_pvals[f"{label}::{a}"] = 1.0 - wp
            rowstore.append((a, label, m))

    # FDR tüm aile üzerinde
    passed = fdr_bh(all_pvals, alpha=0.05)

    # tablo bas
    for a in PRICES:
        print(f"\n  [{a}]  (base tide Sharpe {_sh(strat_ret(tdir.reindex(base_window[a]), prices[a])):+.3f})")
        print(f"    {'form':<20}{'Sharpe':>8}{'ΔSh':>7}{'maxDD':>8}{'CVaR':>8}{'expo':>7}{'P(v>b)':>8}{'FDR':>6}")
        for (aa, label, m) in [r for r in rowstore if r[0] == a]:
            fdrok = passed.get(f"{label}::{a}", False)
            wp = m["wp"]
            print(f"    {label:<20}{m['sh']:>+8.3f}{m['dsh']:>+7.2f}{100*m['dd']:>+7.0f}%{100*m['cv']:>+7.2f}%"
                  f"{m['expo']:>7.2f}{(f'{wp:.0%}' if wp is not None else 'n/a'):>8}{('YES' if fdrok else 'no'):>6}")

    # ---- KARARLAR ----
    print("\n" + "-" * 100)
    print("  KARAR — directional PASS / shield-only / dead")
    print("-" * 100)
    # directional formlar = GATE (a) ve cross-trim(c)/proximity(d) yön-amaçlı; SHIELD(b) ayrı bar
    dir_forms = ["(a) GATE floor0.0", "(a) GATE floor0.4", "(c) cross-3g trim", "(d) proximity trim"]
    directional_pass = False
    for label in dir_forms:
        both = all(passed.get(f"{label}::{a}", False) for a in PRICES)
        # ek: ΔSharpe pozitif mi (gate'in alfa kattığı yön)
        dpos = all(metrics[(label, a)]["dsh"] >= 0 for a in PRICES)
        if both and dpos:
            directional_pass = True
        wps = {a: metrics[(label, a)]["wp"] for a in PRICES}
        print(f"  {label:<20} both-FDR={both}  ΔSh≥0(her ikisi)={dpos}  P(v>b)={{{', '.join(f'{a}:{(wps[a] or 0):.0%}' for a in PRICES)}}}")

    # SHIELD PASS (gex_shield precedent'e SADIK): maxDD/CVaR iyileşme @ Sharpe>=0 VE Sharpe küçük-kayıp.
    # Precedent: gex_shield maxDD'yi 4-7pp KESERKEN Sharpe'ı ~düz tuttu (variance-reduction, alfa değil).
    # Burada tail-iyileşme 0.5-0.7 Sharpe pahasına geliyorsa o kalkan DEĞİL → ΔSharpe taban -0.20'den iyi olmalı.
    sl = "(b) dist-z SHIELD"
    DSH_FLOOR = -0.20          # a-priori: kalkan Sharpe'ı en fazla bu kadar düşürebilir (precedent ~düz)
    shield_pass = True
    for a in PRICES:
        m = metrics[(sl, a)]
        dd_better = m["dd"] >= m["dd_b"] + 0.01    # en az +1pp maxDD iyileşme (gürültü-üstü)
        cv_better = m["cv"] >= m["cv_b"]
        sh_ok = (not np.isnan(m["sh"])) and m["sh"] >= 0
        sh_cost_ok = m["dsh"] >= DSH_FLOOR          # Sharpe ağır kaybetmiyor
        ok = sh_ok and sh_cost_ok and (dd_better or cv_better)
        shield_pass = shield_pass and ok
        print(f"  SHIELD[{a}]  Sharpe {m['sh']:+.3f}(≥0={sh_ok})  ΔSh {m['dsh']:+.2f}(≥{DSH_FLOOR}={sh_cost_ok})  "
              f"maxDD {100*m['dd']:+.0f}% vs base {100*m['dd_b']:+.0f}% (iyi={dd_better})  "
              f"CVaR {100*m['cv']:+.2f}% vs {100*m['cv_b']:+.2f}% (iyi={cv_better}) → PASS={ok}")

    verdict = "directional-PASS" if directional_pass else ("shield-only" if shield_pass else "dead")
    print("\n" + "=" * 100)
    print(f"  VERDICT: {verdict}")
    print("  modules/flip_regime.py YARATILMADI (Stage-2 directional PASS yok → canlı modül yok).")
    print("=" * 100)
    print("\n  CAVEATS (hepsi):")
    print("   • front-month-proxy flip — canlı CBOE çok-vadeli gamma-flip DEĞİL (günde-tek-front-month zincir).")
    print("   • IV diskte %100 NULL → mid'den BS-implied (Brent); 0DTE/derin-ITM/no-arb satırları elendi.")
    print("   • 243g TEK-REJİM penceresi: tide overlap'te kalıcı-LONG (base ≈ her gün long) → directional bar yüksek.")
    print("   • 0DTE-çağı zinciri (front-month dte 0-25); sanity x-ref market-wide GEX ile DÜŞÜK uyum (yapısal fark).")
    print("   • OOS YOK (tek 12-aylık pencere); tüm sayılar IN-SAMPLE.")
    return {"directional_pass": directional_pass, "shield_pass": shield_pass, "verdict": verdict,
            "metrics": metrics, "passed": passed}


def main():
    stage1()
    stage2()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
