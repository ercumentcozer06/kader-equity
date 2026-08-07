"""
backtest/research/rsp_spy_breadth_ablation — RSP/SPY (equal-weight vs cap-weight) KONSANTRASYON
sinyalinin kader-equity'ye katkisi var mi? TAM cerceve, ON-ILANLI izgara, varyant avi YOK.

BAGLAM: Emir bir sosyal-medya klibi paylasti ("RSP divided by SPY" grafigini izle). Retail
okumasi: oran duserse (equal-weight geride kalir) piyasa "daralmis"/mega-cap konsantre ->
kirilgan; oran yukselirse "genis katilim" -> saglikli. Bu iddia 4 bacakta test edilir.
NDX analogu QQEW/QQQ.

VERI: data/cache/breadth.parquet (screen/fetch_breadth.py, yfinance adj-close; RSP 2003-05+,
QQEW 2006-05+). NOT: yfinance adj-close temettu-duzeltmeli -> oran, fiyat-only TradingView
grafiginden hafif sapar (RSP temettu verimi SPY'dan yuksek). Detrend edilmis z-formlarinda
bu fark ihmal edilebilir; ham-seviye formu (F4) icin acikca uyarilir.

BACAKLAR:
  A) UZUN-TARIHCE STANDALONE (2003+/2006+): yon-agnostik quintile bucket, 4 sinyal-formu x
     fwd 21/63g; + basit LONG/FLAT kurallarinin B&H'ye karsi paired block-bootstrap kazanma
     olasiligi. Model-disi: "sinyalin KENDI basina bir seyi var mi?"
  B) CANLI-STACK OVERLAY ABLATION (frozen 2019+): tide x froth(8,11,0) x shield(.5,1,.4) x
     esb-gate replikasi. Replika SPX 1.661 / NDX 1.810 tutmazsa HALT.
     ON-ILANLI IZGARA: 4 form x 2 yon = 8 kol x 2 endeks = 16 test. Trim = target>0 iken x0.5.
     KARAR (once ilan): kol ancak (i) SPX'te full VE 2020+ pencerelerinde Sharpe strict >
     ve maxDD no-worse, (ii) yillik PnL farki > 0, (iii) |NW-t(10)| >= 2, (iv) 16-lik ailede
     BH-FDR(alpha=.05) PASS ise DEPLOY-aday. Aksi RED. HICBIR SEY entegre edilmez.
  C) ARAC-SECIMI (vehicle): sinyal, SPX yerine RSP tutmayi mi soyluyor? breadth momentumu
     ileriye-donuk RSP-eksi-SPY getirisini ongoruyor mu (2003+)?
  D) FAZLALIK: sinyalin canli stack bacaklariyla (COR1M-froth, GEX-shield, esb-gate) ve
     mevcut dispersion-froth persentiliyle korelasyonu.

  python backtest/research/rsp_spy_breadth_ablation.py
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
from screen._util import load_price_csv, paired_win_prob, fdr_bh # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "rsp_spy_breadth_ablation_result.json"
DESK = Path(r"C:\Users\admin\Desktop\backtesting")
LONG_PRICES = {"SPX": "SPX_daily.csv", "NDX": "NASDAQ_daily.csv"}
RATIO_COL = {"SPX": "RSP_SPY", "NDX": "QQEW_QQQ"}
NW_LAGS = 10
EXPECT = {"SPX": 1.661, "NDX": 1.810}
TOL = 0.0051


# ── metrikler (three_down_rule_ablation ile birebir) ──────────────────────────
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


def win_slice(r: pd.Series, wname: str) -> pd.Series:
    if wname == "2020+":
        return r[r.index >= "2020-01-01"]
    if wname == "ex20H1":
        return r[(r.index < "2020-01-01") | (r.index > "2020-06-30")]
    return r


# ── ON-ILANLI SINYAL FORMLARI (hepsi PIT: yalniz t'ye kadar veri) ─────────────
def _z(s: pd.Series, win: int) -> pd.Series:
    return (s - s.rolling(win, min_periods=win // 4).mean()) / s.rolling(win, min_periods=win // 4).std()


def forms(b: pd.Series) -> dict[str, pd.Series]:
    """b = oran serisi (RSP/SPY veya QQEW/QQQ). Dusuk/negatif = DARALMA (mega-cap konsantrasyon)."""
    return {
        "F1 z252(oran/MA126)": _z(b / b.rolling(126, min_periods=40).mean(), 252),
        "F2 oran/MA200-1":     b / b.rolling(200, min_periods=60).mean() - 1.0,
        "F3 mom63":            b / b.shift(63) - 1.0,
        "F4 z504(ham seviye)": _z(b, 504),
    }


# ON-ILANLI DARALMA esikleri (form basina), ve simetrik GENISLEME esigi
NARROW = {"F1 z252(oran/MA126)": lambda s: s < -1.0,
          "F2 oran/MA200-1":     lambda s: s < 0.0,
          "F3 mom63":            lambda s: s < 0.0,
          "F4 z504(ham seviye)": lambda s: s < -1.0}
BROAD = {"F1 z252(oran/MA126)": lambda s: s > 1.0,
         "F2 oran/MA200-1":     lambda s: s > 0.0,
         "F3 mom63":            lambda s: s > 0.0,
         "F4 z504(ham seviye)": lambda s: s > 1.0}


def main() -> int:
    out: dict = {}
    print("=" * 104)
    print("  RSP/SPY (equal-vs-cap KONSANTRASYON) — kader-equity katki testi | on-ilanli izgara, varyant avi YOK")
    print("=" * 104)

    br = pd.read_parquet(ROOT / "data" / "cache" / "breadth.parquet")

    # ── [0] DATA CENSUS (sonuclardan ONCE) ───────────────────────────────────
    print("\n  [0] DATA CENSUS")
    census = {}
    for c in ("RSP", "SPY", "QQEW", "QQQ", "RSP_SPY", "QQEW_QQQ"):
        s = br[c].dropna()
        gap = int(pd.Series(s.index).diff().dt.days.max())
        census[c] = {"n": int(len(s)), "start": str(s.index.min().date()),
                     "end": str(s.index.max().date()), "max_gap_days": gap}
        print(f"    {c:<10} N={len(s):>5}  {s.index.min().date()}..{s.index.max().date()}  max-bosluk={gap}g")
    long_close = {}
    for a, fn in LONG_PRICES.items():
        s = load_price_csv(DESK / fn)
        long_close[a] = s
        census[f"{a} close (uzun)"] = {"n": int(len(s)), "start": str(s.index.min().date()),
                                       "end": str(s.index.max().date())}
        print(f"    {a+' close':<10} N={len(s):>5}  {s.index.min().date()}..{s.index.max().date()}")
    out["census"] = census

    sig_long = {a: forms(br[RATIO_COL[a]].dropna()) for a in ("SPX", "NDX")}

    # ── [A] UZUN-TARIHCE STANDALONE ──────────────────────────────────────────
    print("\n" + "=" * 104)
    print("  [A] STANDALONE (uzun tarihce) — yon-agnostik quintile bucket, fwd ORTALAMA getiri %")
    print("      iddia dogruysa: Q1(daralma) belirgin DUSUK, Q5(genis) YUKSEK olmali")
    print("=" * 104)
    print(f"    {'asset':<6}{'form':<22}{'h':>4}{'Q1 dar':>9}{'Q2':>8}{'Q3':>8}{'Q4':>8}{'Q5 genis':>10}"
          f"{'Q5-Q1':>8}{'mono':>6}")
    bucket = {}
    for a in ("SPX", "NDX"):
        close = long_close[a]
        for fname, s in sig_long[a].items():
            s = s.dropna()
            idx = s.index.intersection(close.index)
            sv, cb = s.reindex(idx), close.reindex(idx, method="ffill")
            for h in (21, 63):
                fh = cb.shift(-h) / cb - 1.0
                q = pd.qcut(sv, 5, labels=False, duplicates="drop")
                bq = [100 * float(fh[q == i].mean()) for i in range(5)]
                mono = all(bq[i] <= bq[i + 1] for i in range(4))
                bucket[f"{a}|{fname}|h{h}"] = {"q": [round(x, 2) for x in bq],
                                               "q5_q1": round(bq[4] - bq[0], 2), "mono": mono,
                                               "n": int(len(sv))}
                print(f"    {a:<6}{fname:<22}{h:>4}{bq[0]:>+9.2f}{bq[1]:>+8.2f}{bq[2]:>+8.2f}"
                      f"{bq[3]:>+8.2f}{bq[4]:>+10.2f}{bq[4]-bq[0]:>+8.2f}{'EVET' if mono else '-':>6}")
    out["A_bucket"] = bucket

    print("\n    [A2] BASIT KURAL vs B&H (standalone LONG/FLAT, lag=1, paired block-bootstrap P(kural>B&H))")
    print(f"    {'asset':<6}{'kural':<34}{'Sharpe':>9}{'B&H':>8}{'maxDD':>8}{'B&H dd':>8}{'expo':>7}{'P':>7}")
    a2 = {}
    for a in ("SPX", "NDX"):
        close = long_close[a]
        bh_done = False
        for fname, s in sig_long[a].items():
            s = s.dropna()
            idx = s.index.intersection(close.index)
            sv = s.reindex(idx)
            bh = strat_ret(pd.Series(1.0, index=idx), close)
            if not bh_done:
                print(f"    {a:<6}{'B&H (referans)':<34}{_sh(bh):>+9.3f}{'':>8}{100*_dd(bh):>+7.0f}%{'':>8}{1.0:>7.0%}")
                bh_done = True
            for lab, cond in (("genis-iken LONG (daralma->FLAT)", BROAD[fname]),
                              ("dar-iken LONG (genisleme->FLAT)", NARROW[fname])):
                pos = pd.Series(cond(sv).astype(float).values, index=idx)
                r = strat_ret(pos, close)
                p = paired_win_prob(bh, r)
                a2[f"{a}|{fname}|{lab}"] = {"sharpe": round(_sh(r), 3), "bh": round(_sh(bh), 3),
                                            "maxdd": round(_dd(r), 3), "bh_maxdd": round(_dd(bh), 3),
                                            "expo": round(float(pos.mean()), 3),
                                            "p_win": None if p is None else round(p, 3)}
                print(f"    {'':<6}{fname[:3]+' '+lab:<34}{_sh(r):>+9.3f}{_sh(bh):>+8.3f}"
                      f"{100*_dd(r):>+7.0f}%{100*_dd(bh):>+7.0f}%{float(pos.mean()):>7.0%}"
                      f"{(p if p is not None else float('nan')):>7.0%}")
    out["A2_standalone_rules"] = a2

    # ── [B] CANLI-STACK OVERLAY ABLATION ─────────────────────────────────────
    print("\n" + "=" * 104)
    print("  [B] CANLI-STACK OVERLAY ABLATION (frozen 2019+) — replika dogrulamali")
    print("=" * 104)
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

    base_r = {a: strat_ret(stack, prices[a]) for a in ("SPX", "NDX")}
    print("    replika dogrulama:")
    for a in ("SPX", "NDX"):
        got = _sh(base_r[a])
        ok = abs(got - EXPECT[a]) <= TOL
        print(f"      {a}: base full Sharpe {got:+.3f} (hedef {EXPECT[a]:.3f}) {'OK' if ok else 'FAIL -> HALT'}")
        if not ok:
            print("    HALT: canli-replika taban tutmadi; ablation KOSULMAZ.")
            return 1
    out["base"] = {a: {"sharpe_full": round(_sh(base_r[a]), 3), "maxdd_full": round(_dd(base_r[a]), 3)}
                   for a in ("SPX", "NDX")}

    # sinyal formlari frozen idx'e (ffill) — PIT: t kapanisi, engine lag=1 -> t+1
    sig_f = {a: {k: v.reindex(idx, method="ffill") for k, v in sig_long[a].items()} for a in ("SPX", "NDX")}
    print("\n    sinyal kapsama (frozen pencerede NaN gunleri = overlay notr):")
    for a in ("SPX", "NDX"):
        cov = {k: int(v.notna().sum()) for k, v in sig_f[a].items()}
        print(f"      {a}: " + "  ".join(f"{k[:2]}={n}/{len(idx)}" for k, n in cov.items()))

    # ── kol izgarasi: SIMETRIK TAMAMLAMA (kisma x0.5 + yaslanma x1.25 + uc-kuyruk) ──
    # Gerekce (once ilan): yalniz "kisma" kollari test etmek yukselen ornekte HER overlay'i
    # cezalandirir (PnL kesilir) -> yon-agnostik tasarim yaslanma kolunu da ZORUNLU kilar.
    # Uc-kuyruk: form serisinin PIT z504'u < -2 (F1/F4 zaten z -> kendisi < -2).
    base_maxabs = float(stack.abs().max())
    CLIP = base_maxabs * 1.25
    windows = ["full", "2020+", "ex20H1"]
    print(f"    taban |target| max={base_maxabs:.3f} -> clip tavani {CLIP:.3f}")
    print(f"\n    {'asset':<6}{'kol':<34}" + "".join(f"{w+' Sh':>10}{'dd':>7}" for w in windows)
          + f"{'expo':>7}{'NWt':>7}{'annPnL':>9}{'P':>7}{'karar':>10}")
    arms, pv = {}, {}
    for a in ("SPX", "NDX"):
        row = f"    {a:<6}{'BASE (canli-replika)':<34}"
        for w in windows:
            rr = win_slice(base_r[a], w)
            row += f"{_sh(rr):>+10.3f}{100*_dd(rr):>+6.0f}%"
        print(row + f"{float(stack.abs().mean()):>7.2f}{'':>33}")
        for fname in sig_f[a]:
            s = sig_f[a][fname]
            ext = (s if fname.startswith(("F1", "F4")) else _z(s, 504)) < -2.0
            grid = [("daralma->long x0.5", NARROW[fname](s), 0.5),
                    ("genisleme->long x0.5", BROAD[fname](s), 0.5),
                    ("daralma->long x1.25", NARROW[fname](s), 1.25),
                    ("genisleme->long x1.25", BROAD[fname](s), 1.25),
                    ("UC-daralma(z<-2)->x0.5", ext, 0.5),
                    ("UC-daralma(z<-2)->x1.25", ext, 1.25)]
            for dlab, condser, mult in grid:
                flag = condser.fillna(False)
                fac = pd.Series(np.where(flag.values & (stack.values > 0), mult, 1.0), index=idx)
                ovl = (stack * fac).reindex(idx).clip(lower=-CLIP, upper=CLIP)
                vr = strat_ret(ovl, prices[a])
                diff = (vr - base_r[a]).dropna()
                key = f"{a}|{fname}|{dlab}"
                ent = {"flag_days": int(flag.sum()), "n_days": int(len(idx)),
                       "gross_expo": round(float(ovl.abs().mean()), 3)}
                beats = True
                row = f"    {'':<6}{fname[:3]+' '+dlab:<34}"
                for w in windows:
                    rb, rv = win_slice(base_r[a], w), win_slice(vr, w)
                    shb, shv, ddb, ddv = _sh(rb), _sh(rv), _dd(rb), _dd(rv)
                    ent[w] = {"sharpe": round(shv, 3), "maxdd": round(ddv, 3),
                              "d_sharpe": round(shv - shb, 3), "d_maxdd": round(ddv - ddb, 4)}
                    if w in ("full", "2020+"):
                        beats = beats and (shv > shb) and (ddv >= ddb - 1e-12)
                    row += f"{shv:>+10.3f}{100*ddv:>+6.0f}%"
                t = nw_t(diff)
                ann = float(diff.mean() * 252)
                p = paired_win_prob(base_r[a], vr)
                ent.update({"nw_t_full": round(t, 2), "ann_pnl_diff": round(ann, 4),
                            "p_win": None if p is None else round(p, 3),
                            "gate_sharpe_dd": beats, "gate_pnl": bool(ann > 0),
                            "gate_t": bool(abs(t) >= 2.0)})
                arms[key] = ent
                pv[key] = None if p is None else (1.0 - p)
                pre = beats and ann > 0 and abs(t) >= 2.0
                row += f"{ent['gross_expo']:>7.2f}{t:>+7.2f}{100*ann:>+8.2f}%"
                row += f"{(p if p is not None else float('nan')):>7.0%}{('FDR-bekle' if pre else 'RED'):>10}"
                print(row)

    passed = fdr_bh({k: v for k, v in pv.items() if v is not None}, alpha=0.05)
    for k in arms:
        arms[k]["fdr_pass"] = bool(passed.get(k, False))
        arms[k]["verdict"] = ("DEPLOY-aday" if (arms[k]["gate_sharpe_dd"] and arms[k]["gate_pnl"]
                                                and arms[k]["gate_t"] and arms[k]["fdr_pass"]) else "RED")
    out["B_arms"] = arms
    n_deploy = sum(1 for v in arms.values() if v["verdict"] == "DEPLOY-aday")
    print(f"\n    BH-FDR (alpha=.05, aile={len(pv)} test): PASS={sum(passed.values())}  "
          f"-> DEPLOY-aday kol sayisi = {n_deploy}")

    # ── [C] ARAC-SECIMI: RSP mi SPY mi? ──────────────────────────────────────
    print("\n" + "=" * 104)
    print("  [C] ARAC-SECIMI — breadth momentumu ileriye-donuk RSP-eksi-SPY getirisini ongoruyor mu?")
    print("=" * 104)
    csel = {}
    for a, (num, den) in (("SPX", ("RSP", "SPY")), ("NDX", ("QQEW", "QQQ"))):
        b = br[RATIO_COL[a]].dropna()
        rel = b.pct_change()                                   # gunluk num-eksi-den (log~) getiri
        for fname, s in forms(b).items():
            s = s.dropna()
            ii = s.index.intersection(rel.index)
            sv = s.reindex(ii)
            for h in (21, 63):
                fwd = b.reindex(ii).shift(-h) / b.reindex(ii) - 1.0     # fwd relatif getiri
                q = pd.qcut(sv, 5, labels=False, duplicates="drop")
                bq = [100 * float(fwd[q == i].mean()) for i in range(5)]
                # basit uygulanabilir kural: sinyal>0/genis iken num tut, degilse den tut
                cond = BROAD[fname](sv).astype(float)
                r_rule = (cond.shift(1).fillna(0.0) * rel.reindex(ii)).dropna()   # sadece relatif alfa
                csel[f"{a}|{fname}|h{h}"] = {"q": [round(x, 2) for x in bq],
                                             "q5_q1": round(bq[4] - bq[0], 2),
                                             "rel_alpha_sharpe": round(_sh(r_rule), 3),
                                             "rel_bh_sharpe": round(_sh(rel.reindex(ii).dropna()), 3)}
                if h == 63:
                    print(f"    {a:<6}{fname:<22}h63  Q1{bq[0]:>+7.2f}  Q5{bq[4]:>+7.2f}  Q5-Q1{bq[4]-bq[0]:>+7.2f}"
                          f"   [{num}/{den} relatif-kural Sharpe {_sh(r_rule):+.2f} vs "
                          f"hep-{num} {_sh(rel.reindex(ii).dropna()):+.2f}]")
    out["C_vehicle"] = csel

    # ── [D] FAZLALIK ─────────────────────────────────────────────────────────
    print("\n" + "=" * 104)
    print("  [D] FAZLALIK — sinyal, stack'in mevcut bacaklariyla ne kadar ortusuyor? (frozen pencere)")
    print("=" * 104)
    legs = {"froth (COR1M)": froth, "shield (GEX)": shield, "esb-gate": esb_gate, "tide dir": tdir}
    dcorr = {}
    for a in ("SPX", "NDX"):
        for fname, s in sig_f[a].items():
            for lname, l in legs.items():
                c = float(pd.concat([s, l], axis=1).dropna().corr().iloc[0, 1])
                dcorr[f"{a}|{fname}|{lname}"] = round(c, 3)
    for a in ("SPX", "NDX"):
        for fname in sig_f[a]:
            print(f"    {a:<6}{fname:<22}" + "  ".join(
                f"{ln}={dcorr[f'{a}|{fname}|{ln}']:+.2f}" for ln in legs))
    out["D_redundancy"] = dcorr

    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 104)
    print(f"  yazildi -> {OUT_JSON}")
    print(f"  SONUC: DEPLOY-aday kol = {n_deploy} / {len(arms)}  "
          f"({'ENTEGRASYON YOK' if n_deploy == 0 else 'incele'})")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
