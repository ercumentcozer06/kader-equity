"""
backtest/karsan/k_phase1_part2.py — KARSAN VALİDASYONU PART 2 (vol-surface adayları). AYRI FDR FAMILY.
POST-HOC BATCH: bu adaylar Part-1'in çoğunlukla null çıktığı GÖRÜLDÜKTEN sonra seçildi → her anlamlı
sonuç 'zayıf kanıt' (null-sonrası fishing). Part-1'in 28 trial'ına POOL EDİLMEZ; kendi BH-FDR'ı.
C7 = identity-check (leverage effect, tradeable DEĞİL); C8 = confirmatory (term-structure, Tier-1);
C9 = exploratory (skew=f(trend), gürültülü). Pre-registered: window/yön k_config'te sabit. t+1, PIT-clean.
  & <venv> backtest/karsan/k_phase1_part2.py  → results/phase1_part2_report.json + tablo. Sonra DUR.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import k_config as K
import k_stats as KS
import k_data as KD
from k_phase1 import third_fridays, bd_offset_to_events   # aynı tanımlar (tek kaynak)

RNG = K.boot_rng()
TRIALS = []


def reg(cid, desc, p, effect, label, note=""):
    TRIALS.append({"id": cid, "desc": desc, "p": float(p), "effect": effect, "label": label, "note": note})


# ===================== C7 — realized vol up/down asimetrisi (IDENTITY) =====================
def C7(S):
    out = {}
    for asset, key in (("SPX", "SPX_ohlc"), ("NDX", "NDX_ohlc")):
        px = S[key]
        ret = px["c"].pct_change()
        gk = pd.Series(KS.garman_klass(px["o"], px["h"], px["l"], px["c"]), index=px.index)
        fwd5 = gk.shift(-1).rolling(5).mean().shift(-4)         # t+1..t+5 ortalama GK vol
        df = pd.DataFrame({"fwd5": fwd5, "absr": ret.abs(), "down": (ret < 0).astype(float)}).dropna()
        # fwd5 ~ const + |ret| + down  → down katsayısı (move-büyüklüğü kontrollü)
        X = np.column_stack([np.ones(len(df)), df["absr"].values, df["down"].values])
        r = KS.ols_coef_boot(df["fwd5"].values, X, ["const", "absr", "down"], RNG)
        dn = r["coef"]["down"]
        reg(f"C7.H7.{asset}", f"{asset} fwd-5g GK-vol: down-day move-kontrollü fazla", dn["p"],
            f"down β {1e4*dn['beta']:+.1f}bps/gün (t{dn['t']})", "IDENTITY (leverage effect, tradeable DEĞİL)",
            note=f"R²={r['r2']}")
        out[asset] = r
    return out


# ===================== C8 — vol term-structure OpEx etrafında (CONFIRMATORY) =====================
def C8(S):
    vix = S["VIX"]; vsurf = S["VOLSURF"]; vix9d = S["VIX9D"]["c"]
    # slope_a = VIX/VIX3M (yüksek=daha inverted/backwardation); slope_b = VIX9D/VIX (ön-uç stresi)
    sa = (vsurf["vix"] / vsurf["vix3m"]).dropna()                 # 2007+
    sb = (vix9d.reindex(vix.index).ffill(limit=1) / vix).dropna() # 2011+
    out = {}
    for sname, sl in (("VIX/VIX3M", sa), ("VIX9D/VIX", sb)):
        tf = third_fridays(sl.index)
        quad = pd.DatetimeIndex([d for d in tf if d.month in K.QUAD_MONTHS])
        off_m = bd_offset_to_events(sl.index, tf)
        off_q = bd_offset_to_events(sl.index, quad)
        for tag, off in (("monthly", off_m), ("quarterly", off_q)):
            pre = off.between(-4, 0)                              # H8a: into-OpEx slope elevated?
            ha = KS.mean_diff_boot(sl.values, pre.values, RNG)
            z = " [0DTE-flag 2023+]" if sname == "VIX9D/VIX" else ""
            reg(f"C8.H8a.{sname}.{tag}", f"{sname} into-OpEx slope ({tag})", ha["p"],
                f"{ha['obs']:+.4f} (t{ha['t']})", "CONFIRMATORY (term-structure, Tier-1)",
                note=f"n_in={ha['n_in']}; tez: elevated/inverted{z}")
            post = off.between(1, 5)                              # H8b: post-OpEx normalize?
            hb = KS.mean_diff_boot(sl.values, post.values, RNG)
            reg(f"C8.H8b.{sname}.{tag}", f"{sname} post-OpEx slope ({tag})", hb["p"],
                f"{hb['obs']:+.4f} (t{hb['t']})", "CONFIRMATORY",
                note=f"n_in={hb['n_in']}; tez: normalize/geri-dön{z}")
            out[f"{sname}.{tag}"] = {"H8a": ha, "H8b": hb}
    return out


# ===================== C9 — skew = f(trend) (EXPLORATORY) =====================
def C9(S):
    spx = S["SPX_ohlc"]; skew = S["SKEW_VVIX"]["SKEW"].dropna()
    common = spx.index.intersection(skew.index)
    px = spx["c"].reindex(common); sk = skew.reindex(common)
    out = {}
    for win in (21, 63):
        tr = px.pct_change(win)                                  # trailing win-gün return
        df = pd.DataFrame({"skew": sk, "tr": tr}).dropna()
        X = np.column_stack([np.ones(len(df)), df["tr"].values])
        r = KS.ols_coef_boot(df["skew"].values, X, ["const", f"tr{win}"], RNG)
        b = r["coef"][f"tr{win}"]
        reg(f"C9.H9.tr{win}", f"SKEW ~ trailing-{win}g return (up-run→steepen?)", b["p"],
            f"β {b['beta']:+.1f} SKEW-pt /100% (t{b['t']})", "EXPLORATORY (gürültülü)",
            note=f"R²={r['r2']}; pozitif=up-run skew-steepen")
        out[f"tr{win}"] = r
    return out


def main():
    S = KD.load_all()
    print("=" * 96); print("  FAZ 1 — PART 2: vol-surface adayları (AYRI FDR family; POST-HOC batch)"); print("=" * 96)
    results = {}
    print("\n  C7 realized-vol asimetrisi (identity)..."); results["C7"] = C7(S)
    print("  C8 term-structure OpEx (confirmatory)..."); results["C8"] = C8(S)
    print("  C9 skew=f(trend) (exploratory)..."); results["C9"] = C9(S)

    pvals = [t["p"] for t in TRIALS]
    adj = KS.bh_fdr(pvals)                                       # AYRI family — yalnız Part-2
    for t, a in zip(TRIALS, adj):
        t["p_bh"] = float(a); t["pass_bh"] = bool(a < K.FDR_ALPHA)
    results["trials"] = TRIALS; results["n_trials"] = len(TRIALS)
    results["family"] = "PART-2 (Part-1'in 28'ine POOL EDİLMEDİ)"
    (K.KRESULTS / "phase1_part2_report.json").write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"\n  PART-2 TRIAL = {len(TRIALS)}  | AYRI BH-FDR (Part-1'e pool YOK) α={K.FDR_ALPHA}")
    print("=" * 96)
    h = f"  {'trial':26}{'effect':32}{'raw p':>8}{'BH p':>8}{'geç':>5}  etiket"
    print(h); print("  " + "-" * (len(h) - 2))
    for t in TRIALS:
        mark = "✓" if t["pass_bh"] else "·"
        print(f"  {t['id']:26}{t['effect'][:31]:32}{t['p']:>8.3f}{t['p_bh']:>8.3f}{mark:>5}  {t['label'][:30]}")
    print("\n  NOT: POST-HOC batch (Part-1 null görüldükten SONRA) → anlamlı sonuç ZAYIF kanıt.")
    print("  C7=identity (tradeable değil); C8=confirmatory term-structure; C9=exploratory.")
    print("  → results/phase1_part2_report.json. DUR (Faz 2 = Emir onayı + survivor).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
