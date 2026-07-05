"""
backtest/karsan/k_phase1.py — FAZ 1 claim validasyonu (C1-C6). PRE-REGISTERED (k_config kilitli).
Footprint ≠ mekanizma: hiçbir verdict "dealer yaptı" demez; "mekanizmanın öngördüğü fiyat/vol izi tutarlı/değil".
Tüm trial'lar registry'ye → BH-FDR (toplam üzerinden). Sharpe türevi sonuçlarda DSR. t+1 lag, PIT-clean.
  & <venv> backtest/karsan/k_phase1.py   → results/phase1_report.json + verdict tablosu (stdout). Sonra DUR.
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

RNG = K.boot_rng()
TRIALS = []   # her eleman: dict(id, claim, desc, stat, p, effect, note)


def reg(cid, desc, p, effect, note="", extra=None):
    TRIALS.append({"id": cid, "desc": desc, "p": float(p), "effect": effect, "note": note, **(extra or {})})


def third_fridays(index: pd.DatetimeIndex):
    """index içindeki her ayın 3.-Cuma tarihini (varsa o ay) döndürür."""
    idx = pd.DatetimeIndex(index)
    fri = idx[idx.weekday == 4]
    out = []
    for (y, m), grp in pd.Series(fri, index=fri).groupby([fri.year, fri.month]):
        if len(grp) >= 3:
            out.append(grp.iloc[2])
    return pd.DatetimeIndex(out)


def bd_offset_to_events(index, events):
    """Her işgünü için en yakın event'e işgünü-ofseti (neg=önce, 0=event, pos=sonra)."""
    idx = pd.DatetimeIndex(index); ev = pd.DatetimeIndex(sorted(events))
    pos = idx.get_indexer(ev, method="nearest")
    off = pd.Series(np.nan, index=idx)
    for p in pos:
        lo, hi = max(0, p - 12), min(len(idx), p + 12)
        for j in range(lo, hi):
            o = j - p
            if pd.isna(off.iloc[j]) or abs(o) < abs(off.iloc[j]):
                off.iloc[j] = o
    return off


# ============================ C1 — OpEx charm/vanna ramp ============================
def C1(S):
    out = {}
    for asset, key in (("SPX", "SPX_ohlc"), ("NDX", "NDX_ohlc")):
        px = S[key]
        ret = px["c"].pct_change()
        gk = pd.Series(KS.garman_klass(px["o"], px["h"], px["l"], px["c"]), index=px.index)
        tf = third_fridays(px.index)
        quad = pd.DatetimeIndex([d for d in tf if d.month in K.QUAD_MONTHS])
        off_m = bd_offset_to_events(px.index, tf)
        off_q = bd_offset_to_events(px.index, quad)
        for tag, off in (("monthly", off_m), ("quarterly", off_q)):
            # H1a: into-OpEx hafta (offset [-4..0]) return drift vs baseline
            pre = off.between(-4, 0)
            r1 = KS.mean_diff_boot(ret.values, pre.values, RNG)
            reg(f"C1.H1a.{asset}.{tag}", f"{asset} {tag} OpEx-haftası drift (ret)", r1["p"],
                f"{1e4*r1['obs']:+.1f}bps/gün (t{r1['t']})", note=f"n_in={r1['n_in']}")
            # H1a-RV: into-OpEx realized vol (GK) vs baseline
            rv1 = KS.mean_diff_boot(gk.values, pre.values, RNG)
            reg(f"C1.H1a_rv.{asset}.{tag}", f"{asset} {tag} OpEx-haftası GK-vol", rv1["p"],
                f"{1e4*rv1['obs']:+.1f}bps/gün vol (t{rv1['t']})", note=f"n_in={rv1['n_in']}")
            # H1b: post-expiry hafta (offset [+1..+5]) realized vol vs baseline (unwind→vol↑?)
            post = off.between(1, 5)
            rv2 = KS.mean_diff_boot(gk.values, post.values, RNG)
            reg(f"C1.H1b.{asset}.{tag}", f"{asset} {tag} post-OpEx GK-vol (unwind)", rv2["p"],
                f"{1e4*rv2['obs']:+.1f}bps/gün vol (t{rv2['t']})", note=f"n_in={rv2['n_in']}")
            out[f"{asset}.{tag}"] = {"H1a_ret": r1, "H1a_rv": rv1, "H1b_rv": rv2}
    # H1c COVID case-check (kalitatif, STAT DEĞİL)
    spx = S["SPX_ohlc"]; ret = spx["c"].pct_change()
    cc = {}
    for nm, ds in K.COVID_OPEX.items():
        d = pd.Timestamp(ds)
        win = ret[(ret.index >= d - pd.Timedelta("7D")) & (ret.index <= d + pd.Timedelta("10D"))]
        cc[nm] = {"opex": ds, "next5_cumret": round(1e2 * float((1 + win[win.index > d].head(5)).prod() - 1), 2),
                  "prev5_cumret": round(1e2 * float((1 + win[win.index <= d].tail(5)).prod() - 1), 2)}
    out["H1c_covid_qualitative"] = cc
    return out


# ============================ C2 — pinning vs grind ============================
def C2(S):
    spx = S["SPX_ohlc"]; vix = S["VIX"].reindex(spx.index).ffill(limit=1)
    ret = spx["c"].pct_change()
    gk = pd.Series(KS.garman_klass(spx["o"], spx["h"], spx["l"], spx["c"]), index=spx.index)
    # gelecek 21g realized vol (close-to-close std, annualize gereksiz — göreli)
    fwd_rv = ret.rolling(K.C2_FWD_RV_WIN).std().shift(-K.C2_FWD_RV_WIN)
    med = vix.rolling(K.C2_VIX_MED_WIN, min_periods=60).median()
    low_iv = (vix < med).astype(float)
    df = pd.DataFrame({"fwd_rv": fwd_rv, "vix": vix, "low": low_iv}).dropna()
    # H2a: fwd_rv ~ const + vix + low + vix×low (interaction = reflexive compression izi)
    X = np.column_stack([np.ones(len(df)), df["vix"].values, df["low"].values, (df["vix"] * df["low"]).values])
    r = KS.ols_coef_boot(df["fwd_rv"].values, X, ["const", "vix", "low", "vix×low"], RNG)
    inter = r["coef"]["vix×low"]
    reg("C2.H2a.SPX", "SPX düşük-IV → fwd-RV nonlineer (vix×low interaction)", inter["p"],
        f"β_int {inter['beta']:+.4f} (t{inter['t']})", note=f"R²={r['r2']} PROXY: IV=positioning değil")
    # H2b: drawdown asimetrisi — local 60g-peak'ten başlayan düşüşler, IV-rejimine göre worst-1d & gün-to-trough
    peak = spx["c"] == spx["c"].rolling(60).max()
    peaks = spx.index[peak.fillna(False) & low_iv.notna().reindex(spx.index).fillna(False)]
    rows = []
    closes = spx["c"]
    for pk in peaks:
        sub = closes[closes.index >= pk]
        # düşüş: peak'ten sonraki ilk yeni-high'a kadar; trough = min
        run = []
        pkval = closes.loc[pk]
        for d, v in sub.items():
            run.append((d, v))
            if v >= pkval and d != pk:
                break
        seg = pd.Series({d: v for d, v in run})
        if len(seg) < 3:
            continue
        trough_i = int(np.argmin(seg.values))
        dd = float(seg.values[trough_i] / pkval - 1)
        if dd > -0.02:        # sadece anlamlı düşüşler (≥%2)
            continue
        worst_1d = float(seg.pct_change().min())
        days_to_trough = trough_i
        reg_at_peak = "low" if (low_iv.reindex([pk]).iloc[0] == 1) else "high"
        rows.append({"peak": pk, "dd": dd, "worst_1d": worst_1d, "ttt": days_to_trough, "regime": reg_at_peak})
    ddf = pd.DataFrame(rows)
    res2b = {}
    if len(ddf) > 20:
        lowm = (ddf["regime"] == "low").values
        # worst_1d: düşük-IV-başlangıç daha SERT (daha negatif) mı? high daha yavaş/sığ mı (Karsan)
        w = KS.mean_diff_boot(ddf["worst_1d"].values, lowm, RNG)
        reg("C2.H2b_worst1d.SPX", "SPX worst-1d-drop: low-IV-start vs high-IV-start", w["p"],
            f"low−high {1e4*w['obs']:+.1f}bps (t{w['t']})", note=f"n_low={int(lowm.sum())}/n_high={int((~lowm).sum())} PROXY")
        tt = KS.mean_diff_boot(ddf["ttt"].astype(float).values, lowm, RNG)
        reg("C2.H2b_ttt.SPX", "SPX gün-to-trough: low-IV-start vs high-IV-start", tt["p"],
            f"low−high {tt['obs']:+.1f}gün (t{tt['t']})", note="Karsan: high-IV daha YAVAŞ (ttt büyük) bekler")
        res2b = {"worst1d": w, "ttt": tt, "n_episodes": len(ddf),
                 "mean_worst1d_low": float(ddf[ddf.regime=="low"].worst_1d.mean()),
                 "mean_worst1d_high": float(ddf[ddf.regime=="high"].worst_1d.mean()),
                 "mean_ttt_low": float(ddf[ddf.regime=="low"].ttt.mean()),
                 "mean_ttt_high": float(ddf[ddf.regime=="high"].ttt.mean())}
    return {"H2a": r, "H2b": res2b}


# ============================ C3 — skew-slide decomposition ============================
def C3(S):
    spx = S["SPX_ohlc"]; vix = S["VIX"]; skew = S["SKEW_VVIX"]["SKEW"]
    common = spx.index.intersection(vix.index).intersection(skew.index)
    d = pd.DataFrame({"vix": vix.reindex(common), "skew": skew.reindex(common),
                      "ret": spx["c"].reindex(common).pct_change()}).dropna()
    d["dvix"] = d["vix"].diff(); d["dskew"] = d["skew"].diff()
    d = d.dropna()
    X = np.column_stack([np.ones(len(d)), d["ret"].values, d["dskew"].values])
    r = KS.ols_coef_boot(d["dvix"].values, X, ["const", "ret_spx", "dskew"], RNG)
    b = r["coef"]["ret_spx"]
    reg("C3.H3.SPX", "ΔVIX ~ ret_SPX + Δskew (β negatif = down-day→VIX↑ mekanik)", b["p"],
        f"β_ret {b['beta']:+.2f} (t{b['t']}) R²={r['r2']}", note="büyüklük testi, doğru/yanlış değil")
    # down-day decomposition: ΔVIX'in payı ret-terimi tarafından açıklanan (mekanik skew-slide)
    down = d[d["ret"] < 0]
    beta = b["beta"]
    explained = (beta * down["ret"]).sum()
    total = down["dvix"].sum()
    share = float(explained / total) if total != 0 else np.nan
    return {"ols": r, "downday_share_explained_by_return_term": round(share, 3),
            "mean_dvix_downday": float(down["dvix"].mean()), "n_down": len(down)}


# ============================ C4 — FOMC vol-crush ============================
def C4(S):
    fomc = pd.DatetimeIndex([pd.Timestamp(x) for x in K.FOMC_DATES])
    res = {}
    for vol_name, ser in (("VIX9D", S["VIX9D"]["c"]), ("VIX", S["VIX"])):
        ser = ser.dropna()
        deltas, drifts = [], []
        spx_ret = S["SPX_ohlc"]["c"].pct_change()
        for f in fomc:
            if f not in ser.index:
                # en yakın işgünü
                near = ser.index[ser.index.get_indexer([f], method="nearest")[0]]
                if abs((near - f).days) > 3:
                    continue
                f = near
            pos = ser.index.get_loc(f)
            if pos < 1 or pos + 1 >= len(ser):
                continue
            pre = ser.iloc[pos - 1]; post = ser.iloc[pos + 1]
            deltas.append(float(post / pre - 1))     # crush = negatif
            # H4b drift coincidence: FOMC+1 gün SPX getirisi
            fd = ser.index[pos]
            if fd in spx_ret.index:
                rp = spx_ret.index.get_loc(fd)
                if rp + 1 < len(spx_ret):
                    drifts.append(float(spx_ret.iloc[rp + 1]))
        d = np.array(deltas)
        rb = KS.paired_event_boot(d, RNG)
        zdte = "0DTE-flag: VIX9D 2023+ ayrı oku" if vol_name == "VIX9D" else ""
        reg(f"C4.H4a.{vol_name}", f"FOMC pre→post {vol_name} crush (post/pre−1)", rb["p"],
            f"{1e2*rb['obs']:+.1f}% (t{rb['t']}, n={rb['n']})", note=zdte)
        # H4b coincidence (sadece raporla, trial olarak da say)
        dr = np.array(drifts)
        drb = KS.paired_event_boot(dr, RNG) if len(dr) > 10 else None
        if drb:
            reg(f"C4.H4b.{vol_name}", f"FOMC+1 SPX drift (crush↔drift coincidence)", drb["p"],
                f"{1e4*drb['obs']:+.1f}bps (t{drb['t']})", note="coincidence, flow DEĞİL")
        res[vol_name] = {"crush": rb, "drift": drb,
                         "mean_crush_pct": round(1e2 * float(d.mean()), 2),
                         "frac_crushed": round(float((d < 0).mean()), 2)}
    return res


# ============================ C5 — intraday open/close (single-regime) ============================
def C5(S):
    """Gün-bazlı istatistik → gün-üstünde block bootstrap (doğru birim = gün; intraday-OOM yok)."""
    res = {}
    for asset, fn in (("SPY", "alpaca_spy_1m"), ("QQQ", "alpaca_qqq_1m")):
        p = ROOT / "data" / "historical_bars" / f"{fn}.parquet"
        if not p.exists():
            res[asset] = {"error": "1-min yok"}; continue
        b = pd.read_parquet(p)
        ts = pd.to_datetime(b.index.get_level_values("timestamp")).tz_convert("America/New_York")
        df = pd.DataFrame({"c": b["close"].values}, index=ts).sort_index()
        df = df[(df.index.time >= pd.Timestamp("09:30").time()) & (df.index.time < pd.Timestamp("16:00").time())]
        dates = pd.Series(df.index.date, index=df.index)
        df["ret"] = df.groupby(dates.values)["c"].pct_change()
        t = df.index.time
        opn = (t >= pd.Timestamp(K.RTH_OPEN[0]).time()) & (t < pd.Timestamp(K.RTH_OPEN[1]).time())
        cls = (t >= pd.Timestamp(K.RTH_CLOSE[0]).time()) & (t < pd.Timestamp(K.RTH_CLOSE[1]).time())
        oc = opn | cls
        g = pd.DataFrame({"d": dates.values, "ret": df["ret"].values, "abs": np.abs(df["ret"].values),
                          "oc": oc, "cls": cls})
        # gün-başı skalerler
        rows = []
        for d, grp in g.groupby("d"):
            a_oc = grp.loc[grp["oc"], "abs"].mean()
            a_rest = grp.loc[~grp["oc"], "abs"].mean()
            r_cls = grp.loc[grp["cls"], "ret"].mean()
            r_rest = grp.loc[~grp["cls"], "ret"].mean()
            rows.append({"d": pd.Timestamp(d), "voc": a_oc - a_rest, "cdrift": r_cls - r_rest, "a_oc": a_oc})
        D = pd.DataFrame(rows).dropna().set_index("d").sort_index()
        # H5a vol konsantrasyon: gün-içi (oc_abs − rest_abs) > 0?
        v = KS.one_sample_block_boot(D["voc"].values, RNG)
        reg(f"C5.H5a_vol.{asset}", f"{asset} open+close |ret| − gün-içi (vol konsantrasyon)", v["p"],
            f"{1e4*v['obs']:+.1f}bps/dk (t{v['t']})", note="SINGLE-REGIME ~5.8y; gün-bazlı")
        # H5a drift: close-bucket drift − rest
        dcl = KS.one_sample_block_boot(D["cdrift"].values, RNG)
        reg(f"C5.H5a_drift.{asset}", f"{asset} close-bucket drift − rest", dcl["p"],
            f"{1e4*dcl['obs']:+.1f}bps/dk (t{dcl['t']})", note="SINGLE-REGIME")
        # H5b: OpEx haftası interaction — a_oc OpEx-içi vs dışı (gün-bazlı mean-diff)
        tf = third_fridays(pd.DatetimeIndex(D.index))
        off = bd_offset_to_events(pd.DatetimeIndex(D.index), tf)
        is_opex = off.between(-4, 0).values
        ob = KS.mean_diff_boot(D["a_oc"].values, is_opex, RNG)
        reg(f"C5.H5b.{asset}", f"{asset} open+close vol: OpEx-haftası vs değil", ob["p"],
            f"{1e4*ob['obs']:+.1f}bps/dk (t{ob['t']})", note="SINGLE-REGIME, C1 cross-check")
        res[asset] = {"vol_conc": v, "close_drift": dcl, "opex_interaction": ob, "n_days": len(D)}
    return res


# ============================ C6 — GHOST open-imbalance (PROXY, single-regime) ============================
def C6(S):
    res = {}
    for asset, fn in (("SPY", "alpaca_spy_1m"), ("QQQ", "alpaca_qqq_1m")):
        p = ROOT / "data" / "historical_bars" / f"{fn}.parquet"
        if not p.exists():
            res[asset] = {"error": "1-min yok"}; continue
        b = pd.read_parquet(p)
        ts = pd.to_datetime(b.index.get_level_values("timestamp")).tz_convert("America/New_York")
        df = pd.DataFrame({"o": b["open"].values, "h": b["high"].values, "l": b["low"].values,
                           "c": b["close"].values}, index=ts).sort_index()
        df = df[(df.index.time >= pd.Timestamp("09:30").time()) & (df.index.time < pd.Timestamp("16:00").time())]
        df["date"] = pd.to_datetime(df.index.date)
        daily = df.groupby("date").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
        # PROXY hedef = ÖNCEKİ RTH close (LOUD: gerçek GHOST DEĞİL)
        proxy = daily["c"].shift(1)
        rows = []
        for d in daily.index[1:]:
            tgt = proxy.loc[d]
            if pd.isna(tgt):
                continue
            op = daily.loc[d, "o"]; hi = daily.loc[d, "h"]; lo = daily.loc[d, "l"]
            gap = op / tgt - 1
            if abs(gap) < 0.0005:    # flat-overnight (no imbalance) → atla
                continue
            touched = (lo <= tgt <= hi)
            # fade EV: açılışta hedefe doğru gir, hedef dokununca çık (R-multiple ~ gap büyüklüğü)
            if gap > 0:    # gap-up → short toward proxy; touch=kâr (gap kadar), yoksa kapanışa
                ev = (op - daily.loc[d, "c"]) / op if not touched else (op - tgt) / op
            else:          # gap-down → long toward proxy
                ev = (daily.loc[d, "c"] - op) / op if not touched else (tgt - op) / op
            rows.append({"date": d, "gap": gap, "touched": int(touched), "ev": ev, "dir": "up" if gap > 0 else "dn"})
        gdf = pd.DataFrame(rows)
        if len(gdf) < 30:
            res[asset] = {"error": "az olay"}; continue
        ptouch = float(gdf["touched"].mean())
        # fade EV: mean>0? gün-üstünde block bootstrap (gap-günleri ardışık → otokorelasyon)
        ev = gdf["ev"].values
        evb = KS.one_sample_block_boot(ev, RNG)
        shp = float(ev.mean() / ev.std()) if ev.std() > 0 else 0.0
        reg(f"C6.H6.{asset}", f"{asset} GHOST-PROXY fade EV (>0?)", evb["p"],
            f"{1e4*evb['obs']:+.1f}bps/işlem (t{evb['t']}), P(touch)={ptouch:.2f}",
            note="PROXY=önceki-close, gerçek GHOST DEĞİL; SINGLE-REGIME")
        res[asset] = {"p_touch": round(ptouch, 3), "fade_ev": evb, "sharpe_daily": round(shp, 4),
                      "n": len(gdf), "dsr_note": "DSR Faz-2'de N-trial ile; tek-strateji DSR yanıltıcı",
                      "ev_by_dir": {"up": round(1e4*float(gdf[gdf.dir=='up'].ev.mean()),1),
                                    "dn": round(1e4*float(gdf[gdf.dir=='dn'].ev.mean()),1)}}
    return res


def main():
    S = KD.load_all()
    print("=" * 96); print("  FAZ 1 — KARSAN MEKANİZMA VALİDASYONU (footprint testi, edge DEĞİL)"); print("=" * 96)
    results = {}
    print("\n  C1 OpEx ramp..."); results["C1"] = C1(S)
    print("  C2 pinning/grind..."); results["C2"] = C2(S)
    print("  C3 skew-slide..."); results["C3"] = C3(S)
    print("  C4 FOMC crush..."); results["C4"] = C4(S)
    print("  C5 intraday open/close..."); results["C5"] = C5(S)
    print("  C6 GHOST-proxy..."); results["C6"] = C6(S)

    # ---- BH-FDR over ALL trials ----
    pvals = [t["p"] for t in TRIALS]
    adj = KS.bh_fdr(pvals)
    for t, a in zip(TRIALS, adj):
        t["p_bh"] = float(a)
        t["pass_bh"] = bool(a < K.FDR_ALPHA)

    results["trials"] = TRIALS
    results["n_trials"] = len(TRIALS)
    (K.KRESULTS / "phase1_report.json").write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"\n  TOPLAM TRIAL = {len(TRIALS)}  | düzeltme = Benjamini-Hochberg FDR (α={K.FDR_ALPHA})")
    print("=" * 96)
    h = f"  {'trial':22}{'effect':30}{'raw p':>9}{'BH p':>9}{'geçer?':>8}"
    print(h); print("  " + "-" * (len(h) - 2))
    for t in TRIALS:
        mark = "✓" if t["pass_bh"] else "·"
        print(f"  {t['id']:22}{t['effect'][:29]:30}{t['p']:>9.3f}{t['p_bh']:>9.3f}{mark:>8}")
    print("\n  NOT: footprint testi — 'geçer' = mekanizmanın öngördüğü iz veride TUTARLI; flow/dealer kanıtı DEĞİL.")
    print("  C5/C6 = SINGLE-REGIME (~5.8y 1-dk); C6 = PROXY (önceki-close, gerçek GHOST değil). C2 = IV-PROXY.")
    print("  → results/phase1_report.json. DUR (Faz 2 = Emir onayı).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
