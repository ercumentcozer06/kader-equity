"""
backtest/research/three_down_rule_ablation — UC-GUN KURALI (3 ardisik asagi kapanis)
overlay ABLATION, canli kader-equity stack replikasi uzerinde.

BAGLAM: Tani labi (Desktop/backtesting/pa_diag_suttmeier_t2_results.json) SPX'te
3-asagi sonrasi fwd h1/h3/h5 fark-t 3.1-3.7 buldu (tam/2014+/2020+ tutarli, ters-kontrol
temiz); NDX zayif. Soru: bu bilgi CANLI stack'e Sharpe VE maxDD katkisi veriyor mu?

BASELINE (canli-replika, breadth_regime_ablation.py ile BIREBIR ayni insa):
frozen tide spine x COR1M-froth(8,11,0) x GEX-shield(0.5,1.0,0.4) x es_basis unwind-gate.
Replika dogrulama: SPX full Sharpe 1.661 / NDX 1.810 yeniden basilir; tutmazsa HALT.

BAYRAK (PIT): index kapanis serisi = stack'in kendi fiyat serisi (frozen prices, engine
ile ayni ffill-reindex). flag[t] = son 3 kapanis ardisik asagi (t-2,t-1,t hepsi chg<0).
Bayrak t kapanisinda olusur -> overlay pozisyon-tarihi t..t+W-1'de aktif; engine'in
lag=1'i ile ILK etkilenen getiri t+1 -> overlay fiilen t+1'den itibaren islenir
(gorev spec'i: "overlay t+1'den itibaren aktif"). Her endeksin bayragi KENDI serisinden.

PRE-DECLARED GRID (kosulmadan once ilan; yalniz bu 3 kol x 2 endeks, varyant avi YOK):
  E1: pencere = 3-asagi sonrasi 3 is gunu; target>0 -> x1.25, target<0 -> x0.75
  E2: E1 ama pencere 5 is gunu (tek duyarlilik kolu)
  E3: yalniz long-boost; target>0 -> x1.25, short'a dokunulmaz (3 is gunu)
CLIP: overlay sonrasi |target| <= base_max_abs x 1.25 (crack'teki +-1.25 mantigi);
tabanin target olcegi raporlanir.

METRIK: Sharpe + maxDD; pencereler full(2019+ = frozen basi; 2014+ frozen'da YOK ->
2014+ == full, acikca soyle), 2020+, ex20H1 (betimsel ek). Fark = (kol - base) gunluk
PnL, Newey-West t (lags=10) pencere-bazli. KARAR (once ilan): kol ancak SPX'te
tam/2014+/2020+ UCUNDE de Sharpe VE maxDD iyilestirirse DEPLOY-ADAY (house-kural:
Sh strict >, dd no-worse >=; dd tie ayrica raporlanir). NDX ayri raporlanir; NDX fail
SPX'i dusurmez ama SPX-only oneri acikca etiketlenir. |NW-t|<2 -> "kanit-demeti zayif".
HICBIR SEY entegre edilmez.

  python backtest/research/three_down_rule_ablation.py
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
OUT_JSON = HERE / "three_down_rule_ablation_result.json"
NW_LAGS = 10
EXPECT = {"SPX": 1.661, "NDX": 1.810}   # replika hedefi (breadth_regime_ablation ile ayni)
TOL = 0.0051


# ── metrics (breadth_regime_ablation ile birebir) ─────────────────────────────
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
    if wname == "2014+":
        return r[r.index >= "2014-01-01"]   # frozen 2019'da baslar -> full ile ozdes
    if wname == "2020+":
        return r[r.index >= "2020-01-01"]
    if wname == "ex20H1":
        return r[(r.index < "2020-01-01") | (r.index > "2020-06-30")]
    return r


def nw_t(d: pd.Series, lags: int = NW_LAGS) -> float:
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


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 100)
    print("  UC-GUN KURALI (3-asagi) OVERLAY ABLATION — canli-stack replika (kader-equity harness)")
    print("=" * 100)

    # ---- baseline live-stack replica (breadth_regime_ablation ile BIREBIR) ----
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

    # ---- 0) DATA CENSUS (sonuclardan ONCE) ----
    print("\n  [0] DATA CENSUS (her bacak N + aralik, sonuclardan once)")
    census = {}
    legs = {
        "SPX close (frozen prices)": prices["SPX"].dropna(),
        "NDX close (frozen prices)": prices["NDX"].dropna(),
        "tide dir (frozen spine)":   tdir.dropna(),
        "COR1M (froth)":             cor,
        "GEX (shield)":              gex,
        "ES basis (esb-gate)":       esb,
    }
    for name, s in legs.items():
        census[name] = {"n": int(len(s)), "start": str(s.index.min().date()),
                        "end": str(s.index.max().date())}
        print(f"    {name:<28} N={len(s):>5}  {s.index.min().date()}..{s.index.max().date()}")
    print(f"    frozen pencere: {win.get('start')}..{win.get('end')} ({win.get('n_days')} gun)."
          f"  NOT: frozen 2019'da baslar -> '2014+' penceresi 'full' ile OZDES (veri yok).")

    # ---- 1) replika dogrulama (tutmazsa HALT) ----
    base_r = {a: strat_ret(stack, prices[a]) for a in ("SPX", "NDX")}
    print("\n  [1] REPLIKA DOGRULAMA")
    for a in ("SPX", "NDX"):
        got = _sh(base_r[a])
        ok = abs(got - EXPECT[a]) <= TOL
        print(f"    {a}: base full Sharpe {got:+.3f}  (hedef {EXPECT[a]:.3f})  "
              f"{'OK' if ok else 'FAIL -> HALT'}")
        if not ok:
            print("    HALT: canli-replika taban tutmadi; ablation KOSULMAZ.")
            return 1
    base_maxabs = float(stack.abs().max())
    print(f"    taban target olcegi: |target| in [{float(stack.min()):+.3f}, "
          f"{float(stack.max()):+.3f}], max|target|={base_maxabs:.3f}"
          f"  -> overlay clip tavani = {base_maxabs:.3f} x 1.25 = {base_maxabs*1.25:.3f}")

    # ---- 2) bayrak (per-endeks, PIT) ----
    # engine ile ayni fiyat hizalamasi: close.reindex(idx, ffill); ffill dumduz gun (chg==0) asagi SAYILMAZ
    flags, factors = {}, {}
    for a in ("SPX", "NDX"):
        cb = prices[a].reindex(idx, method="ffill")
        chg = cb.diff()
        down = chg < 0
        flag = down & down.shift(1).fillna(False) & down.shift(2).fillna(False)
        flags[a] = flag
        # aktif pencere: bayrak gunu DAHIL son W pozisyon-gunu -> engine lag=1 ile
        # fiilen islenen gunler t+1..t+W (spec: overlay t+1'den itibaren aktif)
        act3 = flag.rolling(3, min_periods=1).max().astype(bool)
        act5 = flag.rolling(5, min_periods=1).max().astype(bool)
        pos = stack > 0
        neg = stack < 0
        factors[a] = {
            "E1 3g pencere +-1.25": pd.Series(
                np.where(act3 & pos, 1.25, np.where(act3 & neg, 0.75, 1.0)), index=idx),
            "E2 5g pencere +-1.25 [SENS]": pd.Series(
                np.where(act5 & pos, 1.25, np.where(act5 & neg, 0.75, 1.0)), index=idx),
            "E3 3g yalniz-long x1.25": pd.Series(
                np.where(act3 & pos, 1.25, 1.0), index=idx),
        }

    print("\n  [2] BAYRAK SAYIMI (pozisyon-gunu bazinda)")
    windows = ["full", "2014+", "2020+", "ex20H1"]
    flag_counts = {}
    for a in ("SPX", "NDX"):
        fc = {}
        for w in windows:
            f = win_slice(flags[a].astype(float), w)
            a3 = win_slice(flags[a].rolling(3, min_periods=1).max(), w)
            a5 = win_slice(flags[a].rolling(5, min_periods=1).max(), w)
            fc[w] = {"flag_days": int(f.sum()), "active3_days": int(a3.sum()),
                     "active5_days": int(a5.sum()), "n_days": int(len(f))}
        flag_counts[a] = fc
        print(f"    {a}: " + "  ".join(
            f"{w}: bayrak={fc[w]['flag_days']} aktif3={fc[w]['active3_days']} "
            f"aktif5={fc[w]['active5_days']}/{fc[w]['n_days']}" for w in windows))

    # ---- 3) kol tablosu ----
    results = {"census": census, "frozen_window": win, "base_target_scale":
               {"min": round(float(stack.min()), 3), "max": round(float(stack.max()), 3),
                "max_abs": round(base_maxabs, 3), "clip_cap": round(base_maxabs * 1.25, 3)},
               "flag_counts": flag_counts, "arms": {}, "notes": [
                   "frozen 2019+ -> '2014+' penceresi full ile OZDES (ayri veri yok)",
                   "ex20H1 = betimsel ek pencere (karar-disi; house-emsal)",
                   "bayrak per-endeks kendi frozen kapanis serisinden; ffill flat gun asagi sayilmaz",
                   "PIT: bayrak t kapanisinda; engine lag=1 -> ilk etkilenen getiri t+1",
                   "clip: factor<=1.25 oldugu icin |target|<=base_max_abs*1.25 otomatik saglanir"]}
    clip_cap = base_maxabs * 1.25
    print(f"\n  [3] KOL TABLOSU — Sharpe / maxDD per pencere + pencere-bazli NW-t(lags={NW_LAGS})")
    for a in ("SPX", "NDX"):
        print(f"\n  [{a}]")
        hdr = f"    {'kol':<30}"
        for w in windows:
            hdr += f"{w+' Sh':>10}{'dd':>7}{'NWt':>6}"
        hdr += f"{'karar':>14}"
        print(hdr)
        row = f"    {'BASE (canli-replika)':<30}"
        entry_b = {}
        for w in windows:
            rr = win_slice(base_r[a], w)
            entry_b[w] = {"sharpe": round(_sh(rr), 3), "maxdd": round(_dd(rr), 3)}
            row += f"{_sh(rr):>+10.3f}{100*_dd(rr):>+6.0f}%{'':>6}"
        print(row)
        results["arms"][a] = {"BASE": entry_b}
        for label, fac in factors[a].items():
            ovl = (stack * fac).clip(lower=-clip_cap, upper=clip_cap)
            vr = strat_ret(ovl, prices[a])
            diff_full = (vr - base_r[a]).dropna()
            entry = {}
            beats, dd_tie = True, []
            row = f"    {label:<30}"
            for w in windows:
                rb, rv = win_slice(base_r[a], w), win_slice(vr, w)
                dfw = win_slice(diff_full, w)
                shb, shv, ddb, ddv = _sh(rb), _sh(rv), _dd(rb), _dd(rv)
                t = nw_t(dfw)
                entry[w] = {"sharpe": round(shv, 3), "maxdd": round(ddv, 3),
                            "d_sharpe": round(shv - shb, 3), "d_maxdd": round(ddv - ddb, 4),
                            "nw_t": round(t, 2)}
                if w in ("full", "2014+", "2020+"):     # karar-pencereleri
                    beats = beats and (shv > shb) and (ddv >= ddb)
                    if abs(ddv - ddb) < 1e-9:
                        dd_tie.append(w)
                row += f"{shv:>+10.3f}{100*ddv:>+6.0f}%{t:>+6.2f}"
            ann_diff = float(diff_full.mean() * 252)
            t_full = nw_t(diff_full)
            weak = abs(t_full) < 2.0
            verdict = "DEPLOY-aday" if (beats and ann_diff > 0) else "RED"
            entry["ann_pnl_diff"] = round(ann_diff, 4)
            entry["nw_t_full"] = round(t_full, 2)
            entry["dd_tie_windows"] = dd_tie
            entry["evidence_weak"] = weak
            entry["verdict"] = verdict
            results["arms"][a][label] = entry
            tag = verdict + ("(zayif-t)" if (verdict == "DEPLOY-aday" and weak) else "")
            print(row + f"{tag:>16}")
            if dd_tie:
                print(f"      not: maxDD tie (iyilesme degil, no-worse) pencere(ler): {dd_tie}")

    # ---- 4) nihai karar ----
    print("\n  [4] NIHAI KARAR (karar-cetveli: SPX tam/2014+/2020+ ucunde Sh VE dd; NDX ayri rapor)")
    finals = {}
    for label in factors["SPX"]:
        s_ok = results["arms"]["SPX"][label]["verdict"] == "DEPLOY-aday"
        n_ok = results["arms"]["NDX"][label]["verdict"] == "DEPLOY-aday"
        if s_ok and n_ok:
            v = "DEPLOY-ADAY (SPX+NDX)"
        elif s_ok:
            v = "DEPLOY-ADAY (SPX-ONLY; NDX fail — tanida da zayifti)"
        else:
            v = "RED"
        finals[label] = v
        wk = results["arms"]["SPX"][label]["evidence_weak"]
        print(f"    {label:<32} {v}" + ("   [kanit-demeti zayif: |NW-t|<2]" if (s_ok and wk) else ""))
    results["final"] = finals
    OUT_JSON.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\n  JSON -> {OUT_JSON}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
