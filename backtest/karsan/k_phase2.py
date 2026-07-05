"""
backtest/karsan/k_phase2 — FAZ 2 (GATED, Emir onayı 2026-06-12): C8 survivor'ın TIDE üzerine INCREMENTAL edge'i.
C8 footprint = VIX9D/VIX ön-uç slope quad-witch'e yükseliyor (Karsan: vanna/charm support into expiry, sonra
normalize). İYİMSER kurgu: en güçlü directional + vol-regime okumalarını dene, ÇOK yönelim. Kriter = TIDE
üzerine incremental Sharpe (standalone Sharpe DEĞİL). t+1 lag, PIT-clean. DSR cumulative-N (Part1 28 + Part2 12
+ Faz2). Frozen TIDE/OVERLAYS okunur, DEĞİŞMEZ.
  & <venv> backtest/karsan/k_phase2.py → results/phase2_report.json + tablo.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import k_config as K
import k_stats as KS
import k_data as KD
from k_phase1 import third_fridays, bd_offset_to_events

RNG = K.boot_rng()
N_PRIOR_TRIALS = 28 + 12            # Part1 + Part2 (DSR cumulative)


def met(r):
    r = pd.Series(r).dropna()
    if len(r) < 30 or r.std() == 0:
        return dict(sharpe=0.0, maxdd=0.0, cum=0.0, n=len(r))
    eq = (1 + r).cumprod(); dd = float((eq / eq.cummax() - 1).min())
    return dict(sharpe=round(float(r.mean() / r.std() * np.sqrt(252)), 3), maxdd=round(dd, 4),
                cum=round(float(eq.iloc[-1] - 1), 3), n=len(r),
                sharpe_daily=float(r.mean() / r.std()) if r.std() > 0 else 0.0,
                skew=float(r.skew()), kurt=float(r.kurt() + 3))


def slope_series(S):
    """VIX9D/VIX ön-uç slope (C8 sinyali). >1 = backwardation (ön-uç stresi)."""
    vix = S["VIX"]; v9 = S["VIX9D"]["c"]
    sl = (v9.reindex(vix.index).ffill(limit=1) / vix).dropna()
    return sl


def main():
    S = KD.load_all()
    sl = slope_series(S)                                  # 2011+
    spx = S["SPX_ohlc"]["c"]
    ret = spx.pct_change()
    out = {"family": "PHASE-2 (incremental over TIDE)", "n_prior_trials": N_PRIOR_TRIALS, "trials": [], "tables": {}}
    trials = out["trials"]

    def regp(cid, desc, p, effect, note=""):
        trials.append({"id": cid, "desc": desc, "p": float(p), "effect": effect, "note": note})

    # ================= STANDALONE directional içerik (2011-26, max güç; İYİMSER: her yönelim) =================
    print("=" * 92); print("  FAZ 2 — C8 slope: STANDALONE directional içerik (2011-26, iyimser tarama)"); print("=" * 92)
    common = sl.index.intersection(ret.index)
    d = pd.DataFrame({"slope": sl.reindex(common), "ret": ret.reindex(common)}).dropna()
    d["fwd5"] = d["ret"].shift(-1).rolling(5).sum().shift(-4)
    d["fwd21"] = d["ret"].shift(-1).rolling(21).sum().shift(-20)
    # slope tercile → forward return (her iki yönelim doğal düşer)
    q = d["slope"].quantile([1/3, 2/3])
    hi = d["slope"] >= q.iloc[1]; lo = d["slope"] <= q.iloc[0]
    bucket = {}
    for hz in ("fwd5", "fwd21"):
        bucket[hz] = {"low_slope": round(1e2*float(d[lo][hz].mean()), 3), "high_slope": round(1e2*float(d[hi][hz].mean()), 3),
                      "n_hi": int(hi.sum())}
        # test: high-slope forward farkı (contrarian-bullish mı?)
        r = KS.mean_diff_boot(d[hz].values, hi.values, RNG)
        regp(f"P2.bucket.{hz}", f"high-slope vs rest forward {hz}", r["p"],
             f"{1e2*r['obs']:+.3f}% (t{r['t']})", note="iyimser: high-slope=capitulation→bullish?")
    out["tables"]["standalone_bucket"] = bucket
    # quad-witch into/post forward (Karsan support/weakness) — slope-elevated koşullu
    tf = third_fridays(d.index); quad = pd.DatetimeIndex([x for x in tf if x.month in K.QUAD_MONTHS])
    off = bd_offset_to_events(d.index, quad)
    into = off.between(-5, 0); post = off.between(1, 5)
    for nm, m in (("into_quad", into), ("post_quad", post)):
        r = KS.mean_diff_boot(d["ret"].values, m.values, RNG)
        regp(f"P2.{nm}", f"{nm} günlük ret vs baseline (support/weakness)", r["p"],
             f"{1e4*r['obs']:+.1f}bps (t{r['t']})", note="Karsan: into=support(+), post=weak(−)")
    # into-quad ∩ slope-elevated (vanna en güçlü)
    into_hi = into & hi
    r = KS.mean_diff_boot(d["ret"].values, into_hi.values, RNG)
    regp("P2.into_quad_hislope", "into-quad ∩ slope-elevated günlük ret", r["p"],
         f"{1e4*r['obs']:+.1f}bps (t{r['t']})", note="vanna-support en güçlü hücre")

    # ================= ABLATION: TIDE üzerine incremental (2019+, SPX+NDX) =================
    print("\n" + "=" * 92); print("  FAZ 2 — INCREMENTAL over TIDE (2019+ frozen; kriter = incremental Sharpe)"); print("=" * 92)
    from spine import contract as C, tide as T
    from backtest import engine as E
    from modules.cor1m_froth import froth_factor_series
    from modules.gex_shield import gex_zscore, shield_factor_series
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector)); idx = tdir.index
    cor = pd.read_parquet(ROOT/"data/cache/corr_pc.parquet")["COR1M"].dropna()
    gex = pd.read_parquet(ROOT/"data/cache/squeeze_dix_gex.parquet")["gex"].dropna()
    froth = froth_factor_series(cor.reindex(idx, method="ffill"), 8, 11, 0.0)
    shield = shield_factor_series(gex_zscore(gex).reindex(idx, method="ffill"), 0.5, 1.0, 0.4)
    slp = sl.reindex(idx, method="ffill")
    offX = bd_offset_to_events(idx, pd.DatetimeIndex([x for x in third_fridays(idx) if x.month in K.QUAD_MONTHS]))

    def strat(pos, close, lag=1):
        r = E.fwd_ret(close, pos.index).values; p = pos.astype(float).values
        p = np.concatenate([np.zeros(lag), p[:-lag]])
        return pd.Series(p * r, index=pos.index).dropna()

    # İYİMSER overlay seti (hepsi trim/tilt, t+1; en güçlü directional + vol-regime okumalar):
    # O1 slope-trim: backwardation(slope>1.0)→0.5 (hızlı-shield); O2 quad-support: into-quad'da shield-trim'i
    #   FULL'e geri al (vanna override); O3 post-quad-derisk: post-quad'da 0.5; O4 into-quad-long-boost:
    #   into-quad'da tide-FLAT'i bile LONG yap (support directional, en agresif iyimser)
    thr = 1.0
    slope_trim = pd.Series(np.where(slp > thr, 0.5, 1.0), index=idx)
    quad_support = pd.Series(1.0, index=idx)                                  # into-quad'da shield iptal
    post_derisk = pd.Series(np.where(offX.between(1, 5), 0.5, 1.0), index=idx)
    into_mask = offX.between(-5, 0)

    base_tide = tdir.copy()
    base_stack = tdir * froth * shield
    res_ab = {}
    cum_strats = 0
    for a in ("SPX", "NDX"):
        rows = {}
        rows["tide (base)"] = met(strat(base_tide, prices[a]))
        rows["stack (base: tide×froth×shield)"] = met(strat(base_stack, prices[a]))
        # incremental over TIDE
        rows["tide × slope-trim"] = met(strat(base_tide * slope_trim, prices[a]))
        rows["tide × post-quad-derisk"] = met(strat(base_tide * post_derisk, prices[a]))
        # into-quad long-boost: tide_dir FLAT→LONG into-quad (directional support, iyimser)
        boosted = base_tide.copy(); boosted[into_mask & (tdir == 0)] = 1.0
        rows["tide + into-quad-LONG"] = met(strat(boosted, prices[a]))
        # incremental over DEPLOYED stack (asıl soru: kitaba ekliyor mu)
        rows["stack × slope-trim"] = met(strat(base_stack * slope_trim, prices[a]))
        # quad-support: into-quad'da shield-trim'i full'e al (vanna override) → stack ama into-quad'da shield=1
        sh_override = shield.copy(); sh_override[into_mask] = 1.0
        rows["stack + quad-vanna-override"] = met(strat(tdir * froth * sh_override, prices[a]))
        res_ab[a] = rows
        # incremental Sharpe + paired-P + DSR (cumulative) for the key incremental candidates
        base_r = strat(base_stack, prices[a])
        for label in ("stack × slope-trim", "stack + quad-vanna-override"):
            pos = (base_stack * slope_trim) if "slope-trim" in label else (tdir * froth * sh_override)
            rr = strat(pos, prices[a])
            from screen._util import paired_win_prob
            wp = paired_win_prob(base_r, rr)
            cum_strats += 1
            dN = N_PRIOR_TRIALS + 8   # +8 phase2 ablation strateji (a-priori sabit aşağıda)
            m = met(rr)
            ds = KS.dsr(m["sharpe_daily"], m["n"], dN, m["skew"], m["kurt"])
            regp(f"P2.ablation.{a}.{label}", f"{a} {label} incremental over stack",
                 1 - (wp or 0.5), f"ΔSharpe {m['sharpe']-rows['stack (base: tide×froth×shield)']['sharpe']:+.3f}, DSR {ds}",
                 note=f"P(>stack)={wp:.2f}" if wp is not None else "")

    # BH-FDR (Phase-2 family)
    pv = [t["p"] for t in trials]; adj = KS.bh_fdr(pv)
    for t, ap in zip(trials, adj):
        t["p_bh"] = float(ap); t["pass_bh"] = bool(ap < K.FDR_ALPHA)
    out["tables"]["ablation"] = res_ab
    (K.KRESULTS/"phase2_report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

    # stdout
    print("\n  STANDALONE bucket (forward ret %, slope tercile):")
    for hz, b in bucket.items():
        print(f"    {hz}: low-slope {b['low_slope']:+.3f}% | high-slope {b['high_slope']:+.3f}% (n_hi={b['n_hi']})")
    for a in ("SPX", "NDX"):
        print(f"\n  [{a}] incremental ablation:")
        bs = res_ab[a]["stack (base: tide×froth×shield)"]["sharpe"]
        for label, m in res_ab[a].items():
            dlt = f"  (Δ vs stack {m['sharpe']-bs:+.3f})" if "stack" in label and label != "stack (base: tide×froth×shield)" else ""
            print(f"    {label:38} Sharpe {m['sharpe']:+.3f}  maxDD {100*m['maxdd']:+.0f}%  cum {100*m['cum']:+.0f}%{dlt}")
    print(f"\n  PHASE-2 TRIALS = {len(trials)} (ayrı BH-FDR); DSR cumulative-N = {N_PRIOR_TRIALS}+8")
    print(f"  {'trial':40}{'effect':34}{'rawp':>7}{'BHp':>7}{'geç':>4}")
    for t in trials:
        print(f"  {t['id'][:39]:40}{t['effect'][:33]:34}{t['p']:>7.3f}{t['p_bh']:>7.3f}{'✓' if t['pass_bh'] else '·':>4}")
    print("\n  KRİTER = incremental Sharpe over TIDE/stack. İYİMSER kuruldu (çok-yönelim). → phase2_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
