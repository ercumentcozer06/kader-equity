"""
backtest/research/rsp_spy_breadth_placebo — RSP/SPY ablation'in KARAR-KESICI kontrol testi.

GOZLEM (rsp_spy_breadth_ablation ciktisi): her form icin "daralma->x0.5" kolunun NW-t'si ile
"daralma->x1.25" kolununki TAM simetrik (or. F2 SPX: -4.16 / +4.16; annPnL -7.72% / +3.86%).
Bu, farkin ZAMANLAMA'dan degil sadece MARUZIYET olceginden gelebilecegini dusundurur.

BU DOSYA IKI KONTROL KOSAR (ikisi de once ilan edildi, tek gecis, varyant avi YOK):

  E) PLASEBO-BAYRAK (canli stack uzerinde): gercek bayragi DONGUSEL KAYDIRARAK 1000 sahte
     bayrak uret. Kaydirma; gorev-donguSU (duty cycle), sure-dagilimi ve otokorelasyon
     yapisini AYNEN korur, yalniz "NE ZAMAN" bilgisini yok eder. Gercek bayragin dSharpe'i
     plasebo dagiliminin neresinde? Ortada (p~%50) ise -> sinyalde zamanlama bilgisi YOK,
     tum etki maruziyet-olcegi. Uc persentil (<%5 veya >%95) ise -> gercek bilgi var.
  F) MARUZIYET-ESLENIK SABIT KIYAS (uzun tarihce standalone): "genis-iken LONG" kuralinin
     maxDD kazanci, sadece piyasada daha az kalmaktan mi geliyor? Ayni ortalama maruziyete
     sahip SABIT pozisyon (ve ayni maruziyette 1000 rastgele-blok kural) ile kiyasla.

  python backtest/research/rsp_spy_breadth_placebo.py
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
from screen._util import load_price_csv                          # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "rsp_spy_breadth_placebo_result.json"
DESK = Path(r"C:\Users\admin\Desktop\backtesting")
LONG_PRICES = {"SPX": "SPX_daily.csv", "NDX": "NASDAQ_daily.csv"}
RATIO_COL = {"SPX": "RSP_SPY", "NDX": "QQEW_QQQ"}
EXPECT = {"SPX": 1.661, "NDX": 1.810}
TOL = 0.0051
NPLACEBO = 1000
SEED = 77


def _sh(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")


def _dd(r):
    eq = (1 + r.dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min())


def _z(s: pd.Series, win: int) -> pd.Series:
    return (s - s.rolling(win, min_periods=win // 4).mean()) / s.rolling(win, min_periods=win // 4).std()


def forms(b: pd.Series) -> dict[str, pd.Series]:
    return {"F1": _z(b / b.rolling(126, min_periods=40).mean(), 252),
            "F2": b / b.rolling(200, min_periods=60).mean() - 1.0,
            "F3": b / b.shift(63) - 1.0,
            "F4": _z(b, 504)}


NARROW = {"F1": lambda s: s < -1.0, "F2": lambda s: s < 0.0,
          "F3": lambda s: s < 0.0, "F4": lambda s: s < -1.0}


def main() -> int:
    rng = np.random.default_rng(SEED)
    out: dict = {}
    br = pd.read_parquet(ROOT / "data" / "cache" / "breadth.parquet")

    # ── canli stack replikasi (ablation ile birebir) ─────────────────────────
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
    stack_pos = (stack.values > 0)

    print("=" * 100)
    print("  RSP/SPY — KARAR-KESICI KONTROLLER (plasebo-bayrak + maruziyet-eslenik kiyas)")
    print("=" * 100)
    fret, base_r = {}, {}
    for a in ("SPX", "NDX"):
        fret[a] = E.fwd_ret(prices[a], idx).values
        base_r[a] = pd.Series(np.concatenate([[0.0], stack.values[:-1]]) * fret[a], index=idx).dropna()
        got = _sh(base_r[a])
        ok = abs(got - EXPECT[a]) <= TOL
        print(f"    replika {a}: {got:+.3f} (hedef {EXPECT[a]:.3f}) {'OK' if ok else 'FAIL -> HALT'}")
        if not ok:
            return 1

    # ── [E] PLASEBO-BAYRAK ───────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"  [E] PLASEBO-BAYRAK — bayragi {NPLACEBO}x dongusel kaydir (duty-cycle+sure yapisi KORUNUR,")
    print("      yalniz ZAMANLAMA bozulur). Gercek dSharpe plasebo dagiliminin neresinde?")
    print("      Karar: persentil %5-%95 arasi -> ZAMANLAMA BILGISI YOK (etki=maruziyet olcegi).")
    print("=" * 100)
    print(f"    {'asset':<6}{'form':<5}{'yon':<8}{'gercek dSh':>12}{'plasebo ort':>13}{'p5':>8}{'p95':>8}"
          f"{'persentil':>11}{'karar':>22}")
    E_res = {}
    for a in ("SPX", "NDX"):
        b = br[RATIO_COL[a]].dropna()
        for fname, s in forms(b).items():
            flag = NARROW[fname](s.reindex(idx, method="ffill")).fillna(False).values
            n = len(idx)
            for mult in (0.5, 1.25):
                fac = np.where(flag & stack_pos, mult, 1.0)
                pos = stack.values * fac
                real = _sh(pd.Series(np.concatenate([[0.0], pos[:-1]]) * fret[a], index=idx).dropna())
                d_real = real - _sh(base_r[a])
                shifts = rng.integers(1, n, size=NPLACEBO)
                ds = np.empty(NPLACEBO)
                for i, k in enumerate(shifts):
                    pf = np.roll(flag, int(k))
                    pp = stack.values * np.where(pf & stack_pos, mult, 1.0)
                    ds[i] = _sh(pd.Series(np.concatenate([[0.0], pp[:-1]]) * fret[a], index=idx).dropna())
                ds = ds - _sh(base_r[a])
                pct = float((ds < d_real).mean())
                verdict = "ZAMANLAMA YOK" if 0.05 <= pct <= 0.95 else "UC -> incele"
                E_res[f"{a}|{fname}|x{mult}"] = {
                    "d_sharpe_real": round(d_real, 4), "placebo_mean": round(float(ds.mean()), 4),
                    "placebo_p5": round(float(np.percentile(ds, 5)), 4),
                    "placebo_p95": round(float(np.percentile(ds, 95)), 4),
                    "percentile": round(pct, 3), "verdict": verdict,
                    "flag_days": int(flag.sum()), "n_days": n}
                print(f"    {a:<6}{fname:<5}{'x'+str(mult):<8}{d_real:>+12.4f}{ds.mean():>+13.4f}"
                      f"{np.percentile(ds,5):>+8.4f}{np.percentile(ds,95):>+8.4f}{pct:>11.1%}{verdict:>22}")
    out["E_placebo"] = E_res

    # ── [F] MARUZIYET-ESLENIK SABIT KIYAS (uzun tarihce) ─────────────────────
    print("\n" + "=" * 100)
    print("  [F] MARUZIYET-ESLENIK KIYAS (2003+/2006+) — 'genis-iken LONG' kuralinin maxDD kazanci")
    print("      sadece az-maruziyetten mi? Ayni ort. maruziyette SABIT pozisyon + 1000 rastgele-blok kural.")
    print("=" * 100)
    print("      NOT: 'genis-iken' burada = NOT(daralma) (daralinca FLAT). Sabit-maruziyet Sharpe'i")
    print("      B&H ile ozdestir (olcek-degismez); ASIL kiyas = ayni duty-cycle'li KAYDIRILMIS bayrak.")
    print(f"    {'asset':<6}{'form':<5}{'expo':>6}{'kural Sh':>10}{'sabit Sh':>10}{'rnd Sh ort':>12}"
          f"{'rnd p95':>9}{'Sh pctl':>9}{'kural dd':>10}{'rnd dd ort':>12}{'dd pctl':>9}")
    F_res, F_p = {}, {}
    for a in ("SPX", "NDX"):
        close = load_price_csv(DESK / LONG_PRICES[a])
        b = br[RATIO_COL[a]].dropna()
        for fname, s in forms(b).items():
            s = s.dropna()
            ii = s.index.intersection(close.index)
            ret = E.fwd_ret(close, ii).values
            cond = (~NARROW[fname](s.reindex(ii))).astype(float).values     # genis-iken LONG
            def _r(p):
                return pd.Series(np.concatenate([[0.0], p[:-1]]) * ret, index=ii).dropna()
            r_rule = _r(cond)
            expo = float(np.nanmean(cond))
            r_const = _r(np.full(len(ii), expo))
            # rastgele-blok kural: ayni maruziyette, bayrak dongusel kaydirmali
            rs, rd = np.empty(NPLACEBO), np.empty(NPLACEBO)
            shifts = rng.integers(1, len(ii), size=NPLACEBO)
            for i, k in enumerate(shifts):
                rr = _r(np.roll(cond, int(k)))
                rs[i], rd[i] = _sh(rr), _dd(rr)
            dd_pct = float((rd < _dd(r_rule)).mean())    # dusuk pctile = kural daha KOTU dd
            sh_pct = float((rs < _sh(r_rule)).mean())    # yuksek pctile = kural zamanlamasi IYI
            F_res[f"{a}|{fname}"] = {
                "rule_sharpe": round(_sh(r_rule), 3), "const_sharpe": round(_sh(r_const), 3),
                "rnd_sharpe_mean": round(float(rs.mean()), 3),
                "rnd_sharpe_p95": round(float(np.percentile(rs, 95)), 3),
                "rule_sharpe_percentile_vs_rnd": round(sh_pct, 3),
                "rule_maxdd": round(_dd(r_rule), 3),
                "const_maxdd": round(_dd(r_const), 3), "rnd_maxdd_mean": round(float(rd.mean()), 3),
                "rule_dd_percentile_vs_rnd": round(dd_pct, 3), "expo": round(expo, 3)}
            F_p[f"{a}|{fname}"] = 1.0 - sh_pct
            print(f"    {a:<6}{fname:<5}{expo:>6.0%}{_sh(r_rule):>+10.3f}{_sh(r_const):>+10.3f}"
                  f"{rs.mean():>+12.3f}{np.percentile(rs,95):>+9.3f}{sh_pct:>9.1%}"
                  f"{100*_dd(r_rule):>+9.0f}%{100*rd.mean():>+11.0f}%{dd_pct:>9.1%}")
    from screen._util import fdr_bh                                  # noqa: E402
    fp = fdr_bh(F_p, alpha=0.05)
    for k in F_res:
        F_res[k]["fdr_pass"] = bool(fp.get(k, False))
    print(f"\n    BH-FDR (alpha=.05, aile={len(F_p)}): Sharpe-zamanlama PASS = "
          f"{[k for k, v in fp.items() if v] or 'HICBIRI'}")
    out["F_expo_matched"] = F_res

    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    n_extreme = sum(1 for v in E_res.values() if v["verdict"] != "ZAMANLAMA YOK")
    print("\n" + "=" * 100)
    print(f"  yazildi -> {OUT_JSON}")
    print(f"  [E] plasebo: {len(E_res)} testin {n_extreme} tanesi uc-persentil "
          f"({'hicbiri -> ZAMANLAMA BILGISI YOK' if n_extreme == 0 else 'incele'})")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
