"""
T1 — FLIP-GATE / ASYMMETRIC SHIELD.

Emir's core rule: ABOVE flip (gex>0, long-gamma) play full; BELOW flip (gex<0, short-gamma) trim.
Two layers of evidence:
  STANDALONE (SPX, 2011-26, squeeze price): raw multi-regime edge of each trim rule. trim-only,
    PnL = position[D] × next-day SPX return (lag=1, look-ahead-free). Sub-period maxDD/cum_ret table.
  ABLATION (2019+, frozen tide harness, byte-faithful to finalize_stack): replace gex_shield with the
    flip-gate / asymmetric variant inside tide × froth × <shield>. Does it beat the current shield's
    maxDD/DSR without hurting Sharpe?

PIT discipline: signal[D] (EOD gex/z/price) earns return[D→D+1]. We compute position on signal[D] and
apply engine.fwd_ret (next-day) — identical to finalize_stack.strat_ret(pos, close, lag=1). The 252d
z-score is trailing (rolling, min_periods) so it never peeks. No same-day or future info in any position.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "backtest/gex_swing")
import gxs_config as G  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path("C:/Users/admin/Downloads/kader-equity")
RESULTS = ROOT / "backtest" / "gex_swing" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# ───────────────────────── STANDALONE (SPX, 2011-26) ─────────────────────────
# PIT helper: position is a signal series aligned to day D; PnL = pos[D] * (price[D+1]/price[D]-1).
# We implement this by computing next-day return r1[D] = price.pct_change().shift(-1)[D] and multiplying
# by pos[D]. This is EXACTLY lag=1: the position decided EOD-D earns the D→D+1 move. Last day NaN (no fut).


def standalone_pnl(pos: pd.Series, price: pd.Series) -> pd.Series:
    r1 = price.pct_change().shift(-1)              # r1[D] = D→D+1 return (next-day), look-ahead-free
    return (pos.astype(float) * r1).dropna()


def turnover(pos: pd.Series) -> float:
    return float(pos.astype(float).diff().abs().mean())


def subperiod_table(pos: pd.Series, price: pd.Series) -> dict:
    out = {}
    for name, (s, e) in G.SUBPERIODS.items():
        m = (price.index >= pd.Timestamp(s)) & (price.index <= pd.Timestamp(e))
        sub_price = price[m]
        sub_pos = pos[m]
        if len(sub_price) < 10:
            out[name] = {"maxdd": None, "cum_ret": None, "n": int(len(sub_price))}
            continue
        r = standalone_pnl(sub_pos, sub_price)
        if len(r) < 5:
            out[name] = {"maxdd": None, "cum_ret": None, "n": int(len(r))}
            continue
        eq = (1 + r).cumprod()
        dd = float((eq / eq.cummax() - 1).min())
        out[name] = {"maxdd": round(dd, 4), "cum_ret": round(float(eq.iloc[-1] - 1), 4), "n": int(len(r))}
    return out


def build_standalone():
    g = G.load_squeeze()
    price = g["price"]
    gex = g["gex"]
    z = G.gex_zscore(gex).reindex(g.index)        # trailing z, look-ahead-free

    variants = {}

    # BH baseline: always full long
    pos_bh = pd.Series(1.0, index=g.index)

    # V0_zshield = current shield: (1 − 0.5·clip(−z−1,0,3)).clip(0.4,1)
    pos_v0 = (1.0 - 0.5 * np.clip(-z - 1.0, 0.0, 3.0)).clip(0.4, 1.0)
    pos_v0 = pos_v0.fillna(1.0)                    # no z (warm-up) → 1.0 neutral (shield default)

    pos_map = {"BH": pos_bh, "V0_zshield": pos_v0}

    # V1_flipbin for each floor: floor if gex<0 else 1.0
    for fl in G.FLIP_FLOORS:
        p = pd.Series(np.where(gex.values < 0, fl, 1.0), index=g.index)
        pos_map[f"V1_flipbin_{fl}"] = p

    # V2_asym standalone: 200dma PRICE-TREND PROXY for direction (real tide unavailable pre-2019).
    #   bearish = close < 200dma ; bullish = close ≥ 200dma
    #   gex<0 & bearish → ASYM_FLOOR_LO (0.4, hard) ; gex<0 & bullish → ASYM_FLOOR_HI (0.7, light) ; else 1.0
    sma200 = price.rolling(200, min_periods=200).mean()
    bearish = price < sma200
    pos_v2 = pd.Series(1.0, index=g.index)
    neg = gex < 0
    pos_v2 = pos_v2.mask(neg & bearish, G.ASYM_FLOOR_LO)
    pos_v2 = pos_v2.mask(neg & (~bearish) & sma200.notna(), G.ASYM_FLOOR_HI)
    # before 200dma exists, bearish/bullish undefined → treat as bullish-light (HI) on neg so we still trim
    pos_v2 = pos_v2.mask(neg & sma200.isna(), G.ASYM_FLOOR_HI)
    pos_map["V2_asym_200dma"] = pos_v2

    results = {}
    for name, pos in pos_map.items():
        r = standalone_pnl(pos, price)
        full = G.metrics(r)
        results[name] = {
            "full": full,
            "subperiods": subperiod_table(pos, price),
            "turnover": round(turnover(pos), 5),
            "pct_trimmed": round(float((pos.astype(float) < 1.0).mean()), 4),
        }
    return results


# ───────────────────────── ABLATION (2019+, frozen tide) ─────────────────────────
def build_ablation():
    sys.path.insert(0, str(ROOT))
    from spine import contract as C, tide as T            # noqa: E402
    from backtest import engine as E                       # noqa: E402
    from screen._util import bootstrap_ci, paired_win_prob # noqa: E402
    from modules.cor1m_froth import froth_factor_series     # noqa: E402
    from modules.gex_shield import gex_zscore, shield_factor_series  # noqa: E402
    import math
    from scipy import stats

    GAMMA = 0.5772156649
    N_TRIALS = 60

    def _m(r):
        r = r.dropna()
        eq = (1 + r).cumprod(); dd = float((eq / eq.cummax() - 1).min())
        sh = float(r.mean() / r.std() * np.sqrt(252)); k = max(1, int(0.05 * len(r)))
        return sh, dd, float(np.sort(r.values)[:k].mean()), float((r != 0).mean())

    def strat_ret(pos, close, lag=1):
        idx = pos.index; ret = E.fwd_ret(close, idx).values; p = pos.astype(float).values
        if lag:
            p = np.concatenate([np.zeros(lag), p[:-lag]])
        return pd.Series(p * ret, index=idx).dropna()

    def dsr(r, n_trials=N_TRIALS):
        r = r.dropna().values
        if len(r) < 100 or r.std() == 0:
            return None
        sr = r.mean() / r.std(); n = len(r)
        sk = float(stats.skew(r)); ku = float(stats.kurtosis(r, fisher=False))
        var_sr = (1.0 / n)
        z1 = stats.norm.ppf(1 - 1.0 / n_trials); z2 = stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
        sr0 = math.sqrt(var_sr) * ((1 - GAMMA) * z1 + GAMMA * z2)
        denom = math.sqrt(max(1e-9, 1 - sk * sr + ((ku - 1) / 4) * sr * sr))
        return float(stats.norm.cdf((sr - sr0) * math.sqrt(n - 1) / denom))

    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    cor = pd.read_parquet(ROOT / "data" / "cache" / "corr_pc.parquet")["COR1M"].dropna()
    gex = pd.read_parquet(ROOT / "data" / "cache" / "squeeze_dix_gex.parquet")["gex"].dropna()
    froth = froth_factor_series(cor.reindex(idx, method="ffill"), 8, 11, 0.0)
    zg = gex_zscore(gex).reindex(idx, method="ffill")
    gex_shield = shield_factor_series(zg, 0.5, 1.0, 0.4)     # CURRENT shield (baseline)

    # V1 flipbin floor 0.5 as a shield-replacement factor: gex<0 → 0.5 else 1.0
    gex_sign = np.sign(gex).reindex(idx, method="ffill")    # +1 above flip, -1 below
    v1_factor = pd.Series(np.where(gex_sign.values < 0, 0.5, 1.0), index=idx)

    # V2 asym with REAL tide direction: gex<0 & tide<0 → 0.4 ; gex<0 & tide>0 → 0.7 ; else 1.0
    # tide_dir is 1 (LONG) / 0 (FLAT). "tide<0" interpreted as tide-bearish = FLAT (tdir==0);
    # "tide>0" = LONG (tdir==1). (tide_score sign drives tdir; FLAT == non-positive tide_score.)
    tide_long = tdir.reindex(idx) > 0
    neg = gex_sign < 0
    v2_factor = pd.Series(1.0, index=idx)
    v2_factor = v2_factor.mask(neg & (~tide_long), G.ASYM_FLOOR_LO)
    v2_factor = v2_factor.mask(neg & tide_long, G.ASYM_FLOOR_HI)

    out = {}
    for a in ("SPX", "NDX"):
        base_tide = strat_ret(tdir, prices[a])
        stacks = {
            "base_stack":  tdir * froth * gex_shield,   # tide × froth × current shield
            "V1_stack":    tdir * froth * v1_factor,    # tide × froth × flipbin(0.5)
            "V2_stack":    tdir * froth * v2_factor,    # tide × froth × asym(tide-dir)
        }
        a_out = {}
        for label, pos in stacks.items():
            r = strat_ret(pos.reindex(idx), prices[a])
            sh, dd, cv, ex = _m(r)
            ci = bootstrap_ci(r); wp = paired_win_prob(base_tide, r); ds = dsr(r)
            a_out[label] = {
                "sharpe": round(sh, 3), "maxdd": round(dd, 4), "cvar5": round(cv, 5),
                "expo": round(ex, 3), "n": int(len(r)),
                "cum_ret": round(float((1 + r).cumprod().iloc[-1] - 1), 4),
                "boot_p5": ci["p5"], "p_gt_tide": (round(wp, 4) if wp is not None else None),
                "dsr": (round(ds, 4) if ds is not None else None),
            }
        out[a] = a_out
    return out


def main():
    print("=" * 100)
    print("T1 — FLIP-GATE / ASYMMETRIC SHIELD")
    print("=" * 100)

    standalone = build_standalone()

    # ── standalone full-sample table ──
    print("\n[STANDALONE SPX 2011-26]  trim-only, PnL = pos[D] × next-day SPX ret (lag=1)")
    print(f"  {'variant':<20}{'Sharpe':>8}{'maxDD':>9}{'CVaR5':>9}{'cum_ret':>10}{'turnover':>10}{'%trim':>8}{'n':>7}")
    for name, d in standalone.items():
        f = d["full"]
        print(f"  {name:<20}{f['sharpe']:>+8.3f}{100*f['maxdd']:>+8.1f}%{100*f['cvar5']:>+8.2f}%"
              f"{100*f['cum_ret']:>+9.1f}%{d['turnover']:>10.4f}{100*d['pct_trimmed']:>+7.1f}%{f['n']:>7}")

    # ── standalone sub-period maxDD/cum_ret ──
    periods = list(G.SUBPERIODS.keys())
    print("\n[STANDALONE SUB-PERIOD maxDD]  (where multi-regime protection shows)")
    hdr = "  " + f"{'variant':<20}" + "".join(f"{p[:11]:>13}" for p in periods)
    print(hdr)
    for name, d in standalone.items():
        row = "  " + f"{name:<20}"
        for p in periods:
            sp = d["subperiods"][p]
            row += (f"{100*sp['maxdd']:>+12.1f}%" if sp["maxdd"] is not None else f"{'n/a':>13}")
        print(row)
    print("\n[STANDALONE SUB-PERIOD cum_ret]")
    print(hdr)
    for name, d in standalone.items():
        row = "  " + f"{name:<20}"
        for p in periods:
            sp = d["subperiods"][p]
            row += (f"{100*sp['cum_ret']:>+12.1f}%" if sp["cum_ret"] is not None else f"{'n/a':>13}")
        print(row)

    # ── ablation ──
    print("\n" + "=" * 100)
    print("[ABLATION 2019+ frozen tide]  tide × froth × <shield> ; replace shield with flip-gate / asym")
    print("=" * 100)
    ablation = build_ablation()
    for a in ("SPX", "NDX"):
        print(f"\n  [{a}]")
        print(f"    {'stack':<14}{'Sharpe':>8}{'maxDD':>9}{'CVaR5':>9}{'expo':>7}{'cum_ret':>10}"
              f"{'boot_p5':>9}{'P>tide':>8}{'DSR':>8}{'n':>6}")
        for label, d in ablation[a].items():
            p5 = (f"{d['boot_p5']:+.2f}" if d["boot_p5"] is not None else "n/a")
            pg = (f"{100*d['p_gt_tide']:.0f}%" if d["p_gt_tide"] is not None else "-")
            ds = (f"{d['dsr']:.3f}" if d["dsr"] is not None else "-")
            print(f"    {label:<14}{d['sharpe']:>+8.3f}{100*d['maxdd']:>+8.1f}%{100*d['cvar5']:>+8.2f}%"
                  f"{100*d['expo']:>+6.0f}%{100*d['cum_ret']:>+9.1f}%{p5:>9}{pg:>8}{ds:>8}{d['n']:>6}")

    # ── baseline reproduction check ──
    base_spx = ablation["SPX"]["base_stack"]["sharpe"]
    base_ndx = ablation["NDX"]["base_stack"]["sharpe"]
    repro_ok = (abs(base_spx - 1.64) < 0.03) and (abs(base_ndx - 1.77) < 0.03)
    print("\n" + "-" * 100)
    print(f"  REPRO CHECK: base_stack SPX {base_spx:.3f} (exp ~1.64) | NDX {base_ndx:.3f} (exp ~1.77) "
          f"→ {'MATCH ✓' if repro_ok else 'MISMATCH ✗'}")

    caveats = [
        "Standalone V2 uses a 200dma PRICE-TREND PROXY for direction (real tide is pre-2019 unavailable); "
        "the ablation V2 uses the REAL frozen tide direction.",
        "2022 bear was a SLOW grind (no sustained deep-negative GEX spike) — GEX-based trim barely fires; "
        "flip-gate/asym give little 2022 protection (check sub-period table).",
        "Ablation is 2019+ single-regime (m9 era) → forward Sharpes ~1.0-1.3, not the in-sample stack figure.",
        "Flip-bin (binary gex<0) is far more often-trimming than the z-shield (z≤−1 deep-neg only) → it "
        "trims in benign long-gamma-but-slightly-negative-gex days, costing upside; check turnover/%trim.",
        "DSR shown at N=60 (lower-bound, optimistic); honest selection universe N≈150-200 lowers it ~0.96/0.98.",
    ]

    payload = {
        "meta": {
            "task": "T1_flipgate", "date": "2026-06-12",
            "standalone_window": "2011-05-02..2026-06-08 (SPX squeeze price)",
            "ablation_window": "2019-01..2026-05 (frozen tide)",
            "pit": "signal[D] EOD → return[D+1], lag=1; z trailing-252d/min60",
            "flip_floors": list(G.FLIP_FLOORS), "asym_lo": G.ASYM_FLOOR_LO, "asym_hi": G.ASYM_FLOOR_HI,
            "n_t1_extra_forms": "V1×3 floors + V2 = 4 new a-priori forms (DSR accounting note)",
            "repro_base": {"SPX": base_spx, "NDX": base_ndx, "match": bool(repro_ok)},
        },
        "standalone": standalone,
        "ablation": ablation,
        "caveats": caveats,
    }
    (RESULTS / "T1.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n  wrote {RESULTS / 'T1.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
