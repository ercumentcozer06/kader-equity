"""
backtest/remeasure/RC2_confound.py — RC2.4 CONFOUND (D5 regresyonu AYNEN, yeni bayraklarla).

TEŞHİS-ONLY: yeni strateji/eşik/parametre/sinyal YOK; P&L üretilmez; trial-ledger'a yazılmaz
(yeni trade-trial değil, mevcut bayrakların confound-decompose'u).

SPEC TEK-KAYNAK = backtest/diagnosis/D5_confound.py — istatistik fonksiyonları ve pencere sabitleri
ORADAN import edilir (logistic_mcfadden / ols_r2 / contingency_2x2 / zscore / dte_from_chains /
RVWIN / TRWIN) → byte-aynı matematik. RC-sabitleri (panel penceresi / yollar / semboller / index-map)
backtest/remeasure/config.py'den (tek-gerçek-kaynak; hardcode yok).

UYGULANAN SERİLER (bayrak = level_series.regime = sign(net_gex)):
  SPY/QQQ LIVE-MATCH + SPY/QQQ FULL-SURFACE + SPX-FULL (index-bayrak, SPY enstrümanına)
  + NDX-FULL (index-bayrak, QQQ; görev-metninde adlandırılmadı → 'simetrik-ek' etiketi; amendment
  index-bayrak ailesi SPX→SPY VE NDX→QQQ olduğundan battery-yorumu için aynı spec'le ölçüldü).

DTE REGRESSOR NOTU (görev: "D5'te ne kullanıldıysa onu uygula, farklıysa belirt"):
  D5'in DTE'si = eski md_{sym}.parquet zincirinin front-expiry'sine gün (tek-expiry fetch; aylık-OPEX +
  ay-sonu döngüsü, 0-25 testere). Bu seri panel tarihlerini %100 kapsıyor → AYNI regressor AYNI kaynaktan
  (değer-değer özdeş) kullanıldı. Yeni ham full-chain'lerin kendi front-DTE'si GÜNLÜK vadeler yüzünden
  dejenere (~0; kanıt sayıları JSON 'raw_frontdte_evidence' alanında) → D5-eşdeğeri DEĞİL, kullanılmadı.
  SPX/NDX index-bayrakları için DTE = eşlenik ETF'in (SPY/QQQ) md-zinciri (aynı OPEX takvimi; belirtildi).
  Duyarlılık: (4)-kapanış regresyonu bir de dte yerine yeni serinin dte2_share kolonuyla koşuldu (variant).

RV20/trail20/intraday için bar kaynağı: kendi-bayraklarda sembolün kendisi; index-bayraklarda eşlenik
trade-ETF (SPX→SPY, NDX→QQQ; bayrak ETF D+1 seansında trade edilir — RC2_battery ile aynı hizalama).

  & C:/Users/admin/Downloads/kader-macro/.venv/Scripts/python.exe backtest/remeasure/RC2_confound.py
→ backtest/remeasure/RC2_confound.json (+ stdout ESKİ→YENİ tablo)
"""
from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "backtest"))
sys.path.insert(0, str(ROOT / "backtest" / "diagnosis"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import config as CFG                      # noqa: E402  (tek-gerçek-kaynak: yollar/pencere/semboller)
import D5_confound as D5                  # noqa: E402  (spec tek-kaynak: istatistik fonksiyonları + RVWIN/TRWIN)
from spine_diagnostic import daily_rth    # noqa: E402  (D+1 RTH OHLC — D5 ile aynı tanım)

# index-bayrak → trade-ETF eşlemesi config'ten türetilir (hardcode yok)
ETF_OF = {s: s for s in CFG.TRADE_SYMS}
ETF_OF.update({idx: etf for etf, idx in CFG.INDEX_FLAG_MAP.items()})

# ölçülecek seriler: (etiket, mode, sym, bayrak-türü, kapsam)
SERIES = [
    ("livematch_spy",   "livematch",   "SPY", "own",        "task"),
    ("livematch_qqq",   "livematch",   "QQQ", "own",        "task"),
    ("fullsurface_spy", "fullsurface", "SPY", "own",        "task"),
    ("fullsurface_qqq", "fullsurface", "QQQ", "own",        "task"),
    ("fullsurface_spx", "fullsurface", "SPX", "index_flag", "task"),
    ("fullsurface_ndx", "fullsurface", "NDX", "index_flag", "simetrik-ek"),
]

# ESKİ referans — DIAGNOSIS.md §D5 + 2026-06-11 taze D5 re-run (byte-tutarlı doğrulandı)
OLD_REF = {
    "source": "backtest/DIAGNOSIS.md §D5 + backtest/diagnosis/D5_confound.py taze re-run 2026-06-11 (aynı sayılar)",
    "SPY": {"n": 236, "pos_rate": 0.45,
            "vol": {"r2_rv20": 0.000, "acc_rv20": 0.55, "r2_vix": 0.149, "acc_vix": 0.66,
                    "r2_rv20_vix": 0.202, "acc_rv20_vix": 0.71, "phi_pg_lowvol": 0.12, "ols_absgex_rv20": 0.028},
            "calendar": {"corr_dte_flag": -0.240, "corr_dte_logabsgex": -0.178, "ols_absgex_dte": 0.032},
            "trend": {"corr_tr20_flag": 0.467, "pg_up": 102, "pg_dn": 4, "ng_up": 86, "ng_dn": 44,
                      "phi": 0.37, "ols_absgex_tr20": 0.126},
            "decompose": {"ols_r2_absgex": 0.278, "gamma_specific_share": 0.722,
                          "pseudo_r2_flag": 0.315, "acc_flag": 0.78, "base_acc": 0.55}},
    "QQQ": {"n": 236, "pos_rate": 0.56,
            "vol": {"r2_rv20": 0.030, "acc_rv20": 0.54, "r2_vix": 0.256, "acc_vix": 0.73,
                    "r2_rv20_vix": 0.261, "acc_rv20_vix": 0.75, "phi_pg_lowvol": 0.32, "ols_absgex_rv20": 0.036},
            "calendar": {"corr_dte_flag": -0.038, "corr_dte_logabsgex": -0.339, "ols_absgex_dte": 0.115},
            "trend": {"corr_tr20_flag": 0.493, "pg_up": 122, "pg_dn": 11, "ng_up": 58, "ng_dn": 45,
                      "phi": 0.41, "ols_absgex_tr20": 0.072},
            "decompose": {"ols_r2_absgex": 0.257, "gamma_specific_share": 0.743,
                          "pseudo_r2_flag": 0.360, "acc_flag": 0.79, "base_acc": 0.56}},
}


def _f(x, nd=3):
    """JSON-temiz float."""
    try:
        v = float(x)
        return round(v, nd) if v == v else None
    except Exception:
        return None


def load_vix():
    """D5 ile aynı kaynak: yerel vixcls.parquet (ağ yok)."""
    vp = CFG.CACHE / "vixcls.parquet"
    v = pd.read_parquet(vp)
    col = "vix" if "vix" in v.columns else v.columns[0]
    return pd.Series(v[col].values, index=pd.to_datetime(v.index).normalize())


def build_panel(mode: str, sym: str):
    """D5.build_confound_panel BİREBİR mantık; fark: (a) yeni level_series (config.level_path),
    (b) panel PANEL_START..PANEL_END'e kırpılır (06-09/10 panel-DIŞI), (c) bar/DTE kaynağı index-bayrakta
    eşlenik ETF, (d) dte2_share kolonu (variant için) taşınır."""
    lv = pd.read_parquet(CFG.level_path(mode, sym)).copy()
    lv.index = pd.to_datetime(lv.index).normalize()
    lv = lv[(lv.index >= pd.Timestamp(CFG.PANEL_START)) & (lv.index <= pd.Timestamp(CFG.PANEL_END))]

    etf = ETF_OF[sym]
    rth = daily_rth(etf)
    rth.index = pd.to_datetime(rth.index).normalize()
    sess = list(rth.index)
    rth_ret = rth["c"].pct_change()

    vix = load_vix()
    dte = D5.dte_from_chains(etf)        # D5'in BİREBİR DTE regressor'u (eski md-chain front-expiry)

    rows = []
    for D in lv.index:
        if D not in rth.index:
            continue
        nxt = [s for s in sess if s > D]
        if not nxt:
            continue
        N = nxt[0]
        o1 = rth.loc[N, "o"]; c1 = rth.loc[N, "c"]
        hist = rth_ret.loc[:D].dropna()
        rv20 = hist.tail(D5.RVWIN).std() * np.sqrt(252) if len(hist) >= D5.RVWIN else np.nan
        cwin = rth["c"].loc[:D]
        tr20 = (cwin.iloc[-1] / cwin.iloc[-1 - D5.TRWIN] - 1) if len(cwin) > D5.TRWIN else np.nan
        vix_d = float(vix.asof(D)) if vix is not None and not vix.empty else np.nan
        dte_d = float(dte.asof(D)) if D in dte.index or (dte.index <= D).any() else np.nan

        r = lv.loc[D].to_dict()
        r.update(dict(D=D, N=N, intraday=c1 / o1 - 1,
                      rv20=rv20, vix=vix_d, dte=dte_d, tr20=tr20))
        rows.append(r)
    p = pd.DataFrame(rows)
    p["flag"] = (p["regime"] > 0).astype(int)
    p["absgex"] = p["net_gex"].abs()
    p["log_absgex"] = np.log(p["absgex"].clip(lower=1.0))
    return p


# ----------------------------------------------------------------- D5 bölümleri (aynı matematik, dict döner)
def sec0_overlap(p):
    """D5.section_decompose_72 — gamma_inv-poz vs vol_only-poz örtüşmesi (atm_iv medyan kovası)."""
    sub = p.dropna(subset=["regime", "atm_iv", "intraday"]).copy()
    reg = sub["regime"].values
    volhigh = sub["atm_iv"].values > np.median(sub["atm_iv"].values)
    cg = np.where(reg > 0, 1, -1)
    cv = np.where(volhigh, -1, 1)
    overlap = float((cg == cv).mean())
    c = D5.contingency_2x2((reg > 0), ~volhigh)
    return {"overlap_gammainv_vs_volonly": _f(overlap), "n": len(sub), "phi": _f(c["phi"], 2),
            "agree_days": c["n11"] + c["n00"]}


def sec1_vol(p):
    """D5.section_vol — bayrak~vol logistic + 2x2 + |net_gex|~RV20 OLS."""
    sub = p.dropna(subset=["flag", "rv20"]).copy()
    n = len(sub)
    out = {"n": n}
    r2_rv, _, acc_rv = D5.logistic_mcfadden(D5.zscore(sub["rv20"].values), sub["flag"].values)
    out["r2_rv20"], out["acc_rv20"] = _f(r2_rv), _f(acc_rv, 2)
    have_vix = sub["vix"].notna().sum() >= 0.8 * n and sub["vix"].std() > 0
    if have_vix:
        sv = sub.dropna(subset=["vix"])
        r2_v, _, acc_v = D5.logistic_mcfadden(D5.zscore(sv["vix"].values), sv["flag"].values)
        r2_b, _, acc_b = D5.logistic_mcfadden(
            np.column_stack([D5.zscore(sv["rv20"].values), D5.zscore(sv["vix"].values)]), sv["flag"].values)
        out.update(r2_vix=_f(r2_v), acc_vix=_f(acc_v, 2), r2_rv20_vix=_f(r2_b), acc_rv20_vix=_f(acc_b, 2))
    volhigh = sub["rv20"].values > np.nanmedian(sub["rv20"].values)
    c = D5.contingency_2x2(sub["flag"].values.astype(bool), ~volhigh)
    out["c2x2_pg_lowvol"] = {k: c[k] for k in ("n11", "n10", "n01", "n00")}
    out["phi_pg_lowvol"], out["agree_pg_lowvol"] = _f(c["phi"], 2), _f(c["agree"], 2)
    out["ols_absgex_rv20"] = _f(D5.ols_r2(D5.zscore(sub["rv20"].values), D5.zscore(sub["log_absgex"].values)))
    return out


def sec2_calendar(p, dte_col="dte"):
    """D5.section_calendar — DTE-bucket sawtooth + korelasyonlar. Bucket sınırları D5.section_calendar
    satır 255-256 ile BİREBİR (spec-kopya; ayar değil)."""
    sub = p.dropna(subset=[dte_col, "flag"]).copy()
    sub["dte_i"] = sub[dte_col].round().astype(int)
    bins = [-1, 2, 5, 9, 14, 99]                      # D5 spec-kopya
    labs = ["0-2", "3-5", "6-9", "10-14", "15+"]      # D5 spec-kopya
    sub["bucket"] = pd.cut(sub["dte_i"], bins=bins, labels=labs)
    table = []
    for lab in labs:
        b = sub[sub["bucket"] == lab]
        if len(b) == 0:
            continue
        table.append({"bucket": lab, "n": len(b), "pos_rate": _f(b["flag"].mean(), 2),
                      "mean_absgex_B": _f(b["absgex"].mean() / 1e9, 2),
                      "med_absgex_B": _f(b["absgex"].median() / 1e9, 2)})
    cf = np.corrcoef(sub["dte_i"].values, sub["flag"].values)[0, 1]
    cm = np.corrcoef(sub["dte_i"].values, sub["log_absgex"].values)[0, 1]
    r2 = D5.ols_r2(sub["dte_i"].values.astype(float), D5.zscore(sub["log_absgex"].values))
    return {"n": len(sub), "buckets": table, "corr_dte_flag": _f(cf), "corr_dte_logabsgex": _f(cm),
            "ols_absgex_dte": _f(r2)}


def sec3_trend(p):
    """D5.section_trend — trailing-20g getiri vs bayrak; '+γ düşüş-trendinde hiç yok' boşluğu 2x2'de."""
    sub = p.dropna(subset=["tr20", "flag"]).copy()
    cf = np.corrcoef(sub["tr20"].values, sub["flag"].values)[0, 1]
    up = sub["tr20"].values > 0
    c = D5.contingency_2x2(sub["flag"].values.astype(bool), up)
    r2 = D5.ols_r2(D5.zscore(sub["tr20"].values), D5.zscore(sub["log_absgex"].values))
    return {"n": len(sub), "corr_tr20_flag": _f(cf),
            "c2x2": {"pg_up": c["n11"], "pg_dn": c["n10"], "ng_up": c["n01"], "ng_dn": c["n00"]},
            "agree": _f(c["agree"], 2), "phi": _f(c["phi"], 2), "ols_absgex_tr20": _f(r2)}


def sec4_decompose(p, dte_col="dte"):
    """D5.section_decompose — çoklu-regresyon kapanış. A) log|net_gex| OLS R²; B) sign-flag logistic pseudo-R²."""
    feats = ["rv20", dte_col, "tr20"]
    if p["vix"].notna().sum() >= 0.8 * len(p) and p["vix"].std() > 0:
        feats = ["rv20", "vix", dte_col, "tr20"]
    subm = p.dropna(subset=feats + ["log_absgex"]).copy()
    Xm = np.column_stack([D5.zscore(subm[f].values) for f in feats])
    ym = D5.zscore(subm["log_absgex"].values)
    r2_mag = D5.ols_r2(Xm, ym)
    univ = {f: _f(D5.ols_r2(D5.zscore(subm[f].values), ym)) for f in feats}
    subf = p.dropna(subset=feats + ["flag"]).copy()
    Xf = np.column_stack([D5.zscore(subf[f].values) for f in feats])
    r2_flag, _, acc_flag = D5.logistic_mcfadden(Xf, subf["flag"].values)
    base = max(subf["flag"].mean(), 1 - subf["flag"].mean())
    return {"feats": feats, "n": len(subm),
            "ols_r2_absgex": _f(r2_mag), "gamma_specific_share": _f(1 - r2_mag), "univariate_r2": univ,
            "pseudo_r2_flag": _f(r2_flag), "acc_flag": _f(acc_flag, 2), "base_acc": _f(base, 2),
            "acc_above_base_pp": _f(100 * (acc_flag - base), 1)}


def raw_frontdte_evidence():
    """Yeni ham full-chain'lerin KENDİ front-DTE'si dejenere mi? (exp − chain-tarihi, min≥0; tüm panel günleri).
    Kanıt: günlük vadeler yüzünden ~0 → D5-eşdeğeri DEĞİL → md-chain DTE'si kullanıldı."""
    out = {}
    end = pd.Timestamp(CFG.PANEL_END)
    for sym in CFG.SYMS:
        vals = []
        for fp in sorted((CFG.RAW_DIR / sym).glob("*.json.gz")):
            d0 = pd.Timestamp(fp.name[:10])
            if d0 > end:
                continue
            with gzip.open(fp, "rt", encoding="utf-8") as f:
                resp = json.load(f)["resp"]
            exp = pd.to_datetime(pd.Series(resp["expiration"]), unit="s", utc=True) \
                .dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
            dd = (exp - d0).dt.days
            dd = dd[dd >= 0]
            if len(dd):
                vals.append(int(dd.min()))
        a = np.array(vals, float)
        out[sym] = {"n_days": len(a), "mean": _f(a.mean(), 2), "median": _f(np.median(a), 1),
                    "max": int(a.max()), "share_le1": _f((a <= 1).mean(), 3)}
    return out


def main():
    results = {
        "config_sha": CFG.config_sha(),
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "script": "backtest/remeasure/RC2_confound.py",
        "spec": {
            "source": "backtest/diagnosis/D5_confound.py (logistic_mcfadden/ols_r2/contingency_2x2/zscore/"
                      "dte_from_chains/RVWIN/TRWIN import — byte-aynı matematik)",
            "panel": [CFG.PANEL_START, CFG.PANEL_END],
            "regressors": "RV20(20g, yıllık) + VIX(vixcls.parquet) + DTE + trailing-20g getiri; hepsi PIT D-EOD",
            "flag": "level_series.regime = sign(net_gex); D-EOD bilinir, D+1 trade (D5 ile aynı)",
            "dte_spec": {
                "d5_kullanimi": "eski md_{sym}.parquet zinciri front-expiry'ye gün (tek-expiry fetch; "
                                "aylık-OPEX+ay-sonu döngüsü, 0-25 testere)",
                "burada": "AYNI seri AYNI kaynaktan (D5.dte_from_chains; panel tarihlerinin %100'ü kapsanır). "
                          "SPX/NDX index-bayrakta eşlenik ETF'in (SPY/QQQ) md-zinciri (aynı OPEX takvimi).",
                "fark_beyani": "Yeni ham full-chain'lerin kendi front-DTE'si günlük vadeler yüzünden dejenere "
                               "(~0; bkz. raw_frontdte_evidence) → D5-eşdeğeri değil, regressor olarak alınmadı. "
                               "Duyarlılık: kapanış-regresyonu dte yerine dte2_share ile de koşuldu (variant).",
            },
            "bars": "RV20/trail20/intraday kaynağı: own-bayrakta sembol; index-bayrakta trade-ETF (SPX→SPY, NDX→QQQ)",
        },
        "old_reference": OLD_REF,
        "series": {},
    }

    print(f"RC2.4 CONFOUND — config_sha={results['config_sha']}  panel {CFG.PANEL_START}→{CFG.PANEL_END}")
    print("spec = D5_confound.py birebir (fonksiyon-import); bayrak = sign(net_gex); regressorlar PIT D-EOD\n")

    for label, mode, sym, kind, scope in SERIES:
        p = build_panel(mode, sym)
        s = {
            "mode": mode, "sym": sym, "flag_kind": kind, "scope": scope, "etf_bars": ETF_OF[sym],
            "n": len(p), "period": [str(p["D"].min().date()), str(p["D"].max().date())],
            "pos_rate": _f(p["flag"].mean(), 2),
            "sec0_overlap": sec0_overlap(p),
            "sec1_vol": sec1_vol(p),
            "sec2_calendar": sec2_calendar(p, "dte"),
            "sec3_trend": sec3_trend(p),
            "sec4_decompose": sec4_decompose(p, "dte"),
            "sec4_variant_dte2share": sec4_decompose(p, "dte2_share"),
        }
        results["series"][label] = s
        d4 = s["sec4_decompose"]; t3 = s["sec3_trend"]
        print(f"  {label:16} n{s['n']}  +γ%{100*s['pos_rate']:.0f}  | pseudoR² {d4['pseudo_r2_flag']:.3f} "
              f"acc %{100*d4['acc_flag']:.0f} (taban %{100*d4['base_acc']:.0f})  | |gex|-R² {d4['ols_r2_absgex']:.3f} "
              f"→ gamma-özgü %{100*d4['gamma_specific_share']:.0f}  | trend-corr {t3['corr_tr20_flag']:+.3f} "
              f"phi {t3['phi']:+.2f}  | 2x2 +γ&up {t3['c2x2']['pg_up']} +γ&dn {t3['c2x2']['pg_dn']} "
              f"−γ&up {t3['c2x2']['ng_up']} −γ&dn {t3['c2x2']['ng_dn']}")

    print("\nham front-DTE kanıtı taranıyor (yeni full-chain'ler, panel günleri)...")
    results["raw_frontdte_evidence"] = raw_frontdte_evidence()
    for sym, e in results["raw_frontdte_evidence"].items():
        print(f"  {sym}: front-DTE mean {e['mean']} / median {e['median']} / max {e['max']} / "
              f"DTE≤1 pay %{100*e['share_le1']:.0f} (n{e['n_days']}) → dejenere, D5-eşdeğeri değil")

    # ---- ESKİ→YENİ kıyas tablosu (stdout)
    print("\nESKİ→YENİ (D-FAZ kırık tek-expiry → onarılmış seriler):")
    hdr = (f"  {'seri':16}{'pseudoR²':>9}{'acc%':>6}{'|gex|kalan%':>12}{'trend-corr':>11}{'+γ&dn':>7}{'phi-tr':>7}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for sym in CFG.TRADE_SYMS:
        o = OLD_REF[sym]
        print(f"  {'ESKİ_' + sym.lower():16}{o['decompose']['pseudo_r2_flag']:>9.3f}"
              f"{100*o['decompose']['acc_flag']:>6.0f}{100*o['decompose']['gamma_specific_share']:>12.0f}"
              f"{o['trend']['corr_tr20_flag']:>+11.3f}{o['trend']['pg_dn']:>7}{o['trend']['phi']:>+7.2f}")
    for label, *_ in [(s[0],) for s in SERIES]:
        s = results["series"][label]
        d4, t3 = s["sec4_decompose"], s["sec3_trend"]
        print(f"  {label:16}{d4['pseudo_r2_flag']:>9.3f}{100*d4['acc_flag']:>6.0f}"
              f"{100*d4['gamma_specific_share']:>12.0f}{t3['corr_tr20_flag']:>+11.3f}"
              f"{t3['c2x2']['pg_dn']:>7}{t3['phi']:>+7.2f}")

    out = HERE / "RC2_confound.json"
    out.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
