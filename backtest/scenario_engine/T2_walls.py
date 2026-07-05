"""
T2_walls.py — T2 DUVAR HOLD/BREAK + VU/VD KASKAD (rejim-koşullu). Spec = se_config.py docstring (KİLİTLİ).

Olaylar (seans N, seviyeler D-EOD):
  CW touch        : 1-dk high >= call_wall ; PW touch: 1-dk low <= put_wall
  BREAK-CONFIRM   : ilk 15-dk bar close > CW (/< PW), bar başlangıcı <= 14:45 ET
  REJECT          : touch var, seans boyunca hiç 15-dk close-through yok
  VU/VD reach     : confirm bar kapanışından SONRA 1-dk high >= vu (/low <= vd)

Setuplar: S1 CW-REJECT-FADE (short), S2 CW-BREAK-MOM (long; A=VU-TP, B=kaskad),
          S3 PW-REJECT-BOUNCE (long), S4 PW-BREAK-MOM (short; A=VD-TP, B=kaskad).
Muhafazakâr eş-bar kuralı: TP-dokunuş ve soft-stop aynı 15-dk barda -> STOP sayılır
(B-varyantında VU/VD-dokunuş + aynı-bar stop-close -> STOP; disable bir sonraki bardan).

Çıktı: results/T2_spy.json + results/T2_qqq.json + stdout tablolar.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import se_config as SE  # noqa: E402
import se_panel  # noqa: E402

CUTOFF = pd.Timestamp(SE.ENTRY_CUTOFF).time()      # 14:45 ET — giriş yapılabilen son 15-dk bar başlangıcı
COST_BPS = SE.COST_RT * 1e4                        # 2.0 bps round-trip
GAP = SE.TP_MIN_GAP                                # 0.001
Z = 1.959963984540054

SETUPS = ["S1_CW_REJECT_FADE", "S2A_CW_BREAK_VU_TP", "S2B_CW_BREAK_CASCADE",
          "S3_PW_REJECT_BOUNCE", "S4A_PW_BREAK_VD_TP", "S4B_PW_BREAK_CASCADE"]
REGROWS = ["+1", "-1", "all"]
PROB_KEYS = ["p_cw_touch", "p_break_given_cw_touch", "p_vu_given_cw_break",
             "p_pw_touch", "p_break_given_pw_touch", "p_vd_given_pw_break"]


# ----------------------------------------------------------------------------- istatistik
def wilson(k: int, n: int):
    if n == 0:
        return None, None, None
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def pcell(k: int, n: int) -> dict:
    p, lo, hi = wilson(k, n)
    return {"n": n, "k": k, "p": p, "wilson_lo": lo, "wilson_hi": hi,
            "flag": "YETERSIZ-N" if n < 10 else ""}


def ev_cell(trs: list, sks: list) -> dict:
    n = len(trs)
    out = {"n": n, "skips": len(sks)}
    if n == 0:
        out.update(hit_pct=None, hit_wilson_lo=None, hit_wilson_hi=None, hit_pct_net=None,
                   mean_gross_bps=None, mean_net_bps=None, t_gross=None, t_net=None,
                   med_mfe_em1=None, med_mae_em1=None, flag="YETERSIZ-N")
        return out
    g = np.array([t["gross"] for t in trs], float)
    nets = g - COST_BPS
    k = int((g > 0).sum())
    p, lo, hi = wilson(k, n)
    sd = float(g.std(ddof=1)) if n >= 2 else None
    se_ = (sd / math.sqrt(n)) if (sd is not None and sd > 0) else None
    mfes = [t["mfe"] for t in trs if t["mfe"] is not None]
    maes = [t["mae"] for t in trs if t["mae"] is not None]
    out.update(hit_pct=p, hit_wilson_lo=lo, hit_wilson_hi=hi,
               hit_pct_net=float((nets > 0).mean()),
               mean_gross_bps=float(g.mean()), mean_net_bps=float(nets.mean()),
               t_gross=(float(g.mean() / se_) if se_ else None),
               t_net=(float(nets.mean() / se_) if se_ else None),
               med_mfe_em1=(float(np.median(mfes)) if mfes else None),
               med_mae_em1=(float(np.median(maes)) if maes else None),
               flag="YETERSIZ-N" if n < 10 else "")
    return out


# ----------------------------------------------------------------------------- motor
def bar_of(ts: pd.Timestamp) -> pd.Timestamp:
    """1-dk barın ait olduğu 09:30-anchored 15-dk barın başlangıcı."""
    m = ts.hour * 60 + ts.minute
    bs = 570 + ((m - 570) // 15) * 15
    return ts.replace(hour=int(bs // 60), minute=int(bs % 60), second=0, microsecond=0, nanosecond=0)


def _stop_hit(close: float, op: str, lvl) -> bool:
    if lvl is None or not np.isfinite(lvl):
        return False
    return {"ge": close >= lvl, "gt": close > lvl, "le": close <= lvl, "lt": close < lvl}[op]


def simulate(bd: pd.DataFrame, md: pd.DataFrame, entry_start: pd.Timestamp, entry_px: float,
             side: int, tp, stop_op: str, stop_lvl, c1: float, em1: float, cascade_lvl=None) -> dict:
    """Girişten sonraki 15-dk barları sırayla işle (se_config eş-bar kuralı: STOP önce).
    cascade_lvl verilirse (varyant B): TP yok; seviye 1-dk intrabar dokununca stop devre dışı, c1'e tut.
    Eş-bar (dokunuş + stop-close) -> STOP (muhafazakâr). Hiçbiri olmazsa c1'de çık."""
    b15 = pd.Timedelta(minutes=15)
    one = pd.Timedelta(minutes=1)
    stop_disabled = False
    exit_px = None
    reason = None
    exit_incl = None  # MFE/MAE penceresinin kapsayıcı son dakikası
    for s, bar in bd[bd.index > entry_start].iterrows():
        bend = s + b15
        bm = md[(md.index >= s) & (md.index < bend)]
        close = float(bar["c"])
        sh = (not stop_disabled) and _stop_hit(close, stop_op, stop_lvl)
        if cascade_lvl is None:
            t_t = None
            if tp is not None and np.isfinite(tp) and len(bm):
                hits = bm.index[bm["h"] >= tp] if side > 0 else bm.index[bm["l"] <= tp]
                t_t = hits[0] if len(hits) else None
            if t_t is not None and sh:
                exit_px, reason, exit_incl = close, "stop_tie", bend - one
                break
            if t_t is not None:
                exit_px, reason, exit_incl = float(tp), "tp", t_t
                break
            if sh:
                exit_px, reason, exit_incl = close, "stop", bend - one
                break
        else:
            lv_t = None
            if np.isfinite(cascade_lvl) and len(bm):
                hits = bm.index[bm["h"] >= cascade_lvl] if side > 0 else bm.index[bm["l"] <= cascade_lvl]
                lv_t = hits[0] if len(hits) else None
            if lv_t is not None and sh:
                exit_px, reason, exit_incl = close, "stop_tie", bend - one
                break
            if lv_t is not None:
                stop_disabled = True
            elif sh:
                exit_px, reason, exit_incl = close, "stop", bend - one
                break
    if exit_px is None:
        exit_px = float(c1)
        reason = "eod_cascade" if stop_disabled else "eod"
        exit_incl = md.index[-1]
    gross = side * (exit_px / entry_px - 1.0) * 1e4
    mfe = mae = None
    if em1 is not None and np.isfinite(em1) and em1 > 0:
        wm = md[(md.index >= entry_start + b15) & (md.index <= exit_incl)]
        if len(wm):
            hi, lo = float(wm["h"].max()), float(wm["l"].min())
            if side > 0:
                mfe, mae = max(0.0, (hi - entry_px) / em1), max(0.0, (entry_px - lo) / em1)
            else:
                mfe, mae = max(0.0, (entry_px - lo) / em1), max(0.0, (hi - entry_px) / em1)
        else:
            mfe = mae = 0.0
    return {"gross": float(gross), "net": float(gross - COST_BPS), "reason": reason,
            "mfe": mfe, "mae": mae, "exit_px": float(exit_px)}


def fnum(x):
    """panel değeri → float ya da None (NaN korumalı)."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


# ----------------------------------------------------------------------------- sembol koşusu
def run_sym(sym: str) -> dict:
    panel = pd.read_parquet(SE.panel_path(sym))
    mins = se_panel.rth_minutes(sym)
    b15 = se_panel.bars15(mins)
    day_m = {d: g for d, g in mins.groupby("date")}
    day_b = {d: g for d, g in b15.groupby("date")}

    devs, trades, skips = [], [], []
    meta = {"n_days": 0, "missing_session_days": 0,
            "cw_break_late_only": 0, "pw_break_late_only": 0,
            "s1_touch_bar_after_cutoff": 0, "s3_touch_bar_after_cutoff": 0,
            "s2a_entry_at_or_above_vu": 0, "s4a_entry_at_or_below_vd": 0,
            "regime_idx_zero_days": 0, "join_verified_days": 0}

    for D, r in panel.iterrows():
        N = pd.Timestamp(r["N"])
        md, bd = day_m.get(N), day_b.get(N)
        if md is None or bd is None or len(bd) == 0:
            meta["missing_session_days"] += 1
            continue
        # JOIN DOĞRULAMA: D-satırı seans N'i (o1..c1) anlatmalı
        for a, b in ((md["o"].iloc[0], r["o1"]), (md["c"].iloc[-1], r["c1"]),
                     (md["h"].max(), r["h1"]), (md["l"].min(), r["l1"])):
            if abs(float(a) - float(b)) > 1e-6:
                raise AssertionError(f"JOIN MISMATCH {sym} D={D.date()} N={N.date()}: {a} vs {b}")
        meta["join_verified_days"] += 1
        meta["n_days"] += 1

        cw, pw = fnum(r["call_wall"]), fnum(r["put_wall"])
        vu, vd = fnum(r["vu"]), fnum(r["vd"])
        ghost, mid_up = fnum(r["ghost"]), fnum(r["mid_up"])
        mid_dn, flip = fnum(r["mid_dn"]), fnum(r["flip"])
        em1, c1 = fnum(r["em1"]), float(r["c1"])
        reg_o, reg_i = int(r["regime_own"]), int(r["regime_idx"])
        if reg_i == 0:
            meta["regime_idx_zero_days"] += 1
        med_vol = float(bd["v"].median())
        base_id = {"regime_own": reg_o, "regime_idx": reg_i, "D": str(pd.Timestamp(D).date()),
                   "N": str(N.date())}

        ev = dict(base_id)
        # ---- CW tarafı: olaylar
        cw_touch_first = cw_confirm = None
        if cw is not None:
            ev["cw_valid"] = True
            tm = md.index[md["h"] >= cw]
            ev["cw_touch"] = bool(len(tm))
            cw_touch_first = tm[0] if len(tm) else None
            ct = bd.index[bd["c"] > cw]
            first_ct = ct[0] if len(ct) else None
            ev["cw_break"] = bool(first_ct is not None and first_ct.time() <= CUTOFF)
            if first_ct is not None and first_ct.time() > CUTOFF:
                meta["cw_break_late_only"] += 1
            if ev["cw_break"]:
                cw_confirm = first_ct
                if vu is not None:
                    after = md[md.index >= cw_confirm + pd.Timedelta(minutes=15)]
                    ev["cw_vu_reach"] = bool((after["h"] >= vu).any())
                else:
                    ev["cw_vu_reach"] = None
        else:
            ev["cw_valid"] = False
        # ---- PW tarafı: olaylar
        pw_touch_first = pw_confirm = None
        if pw is not None:
            ev["pw_valid"] = True
            tm = md.index[md["l"] <= pw]
            ev["pw_touch"] = bool(len(tm))
            pw_touch_first = tm[0] if len(tm) else None
            ct = bd.index[bd["c"] < pw]
            first_ct = ct[0] if len(ct) else None
            ev["pw_break"] = bool(first_ct is not None and first_ct.time() <= CUTOFF)
            if first_ct is not None and first_ct.time() > CUTOFF:
                meta["pw_break_late_only"] += 1
            if ev["pw_break"]:
                pw_confirm = first_ct
                if vd is not None:
                    after = md[md.index >= pw_confirm + pd.Timedelta(minutes=15)]
                    ev["pw_vd_reach"] = bool((after["l"] <= vd).any())
                else:
                    ev["pw_vd_reach"] = None
        else:
            ev["pw_valid"] = False
        devs.append(ev)

        # ---- S1 CW-REJECT-FADE (SHORT @ touch-bar close < CW)
        if cw_touch_first is not None:
            tb = bar_of(cw_touch_first)
            tbar = bd.loc[tb]
            if tb.time() > CUTOFF:
                meta["s1_touch_bar_after_cutoff"] += 1
            elif float(tbar["c"]) < cw:
                entry = float(tbar["c"])
                volc = bool(float(tbar["v"]) > med_vol)
                tp = None
                if ghost is not None and ghost <= entry * (1 - GAP):
                    tp = ghost
                elif mid_up is not None and mid_up <= entry * (1 - GAP):
                    tp = mid_up
                rec = {**base_id, "setup": "S1_CW_REJECT_FADE", "volc": volc}
                if tp is None:
                    skips.append(rec)
                else:
                    trades.append({**rec, **simulate(bd, md, tb, entry, -1, tp, "ge", vu, c1, em1)})

        # ---- S2 CW-BREAK-MOM (LONG @ confirm close)
        if cw_confirm is not None:
            cbar = bd.loc[cw_confirm]
            entry = float(cbar["c"])
            volc = bool(float(cbar["v"]) > med_vol)
            if vu is not None and entry >= vu:
                meta["s2a_entry_at_or_above_vu"] += 1
            recA = {**base_id, "setup": "S2A_CW_BREAK_VU_TP", "volc": volc}
            trades.append({**recA, **simulate(bd, md, cw_confirm, entry, +1, vu, "lt", cw, c1, em1)})
            recB = {**base_id, "setup": "S2B_CW_BREAK_CASCADE", "volc": volc}
            trades.append({**recB, **simulate(bd, md, cw_confirm, entry, +1, None, "lt", cw, c1, em1,
                                              cascade_lvl=vu)})

        # ---- S3 PW-REJECT-BOUNCE (LONG @ touch-bar close > PW)
        if pw_touch_first is not None:
            tb = bar_of(pw_touch_first)
            tbar = bd.loc[tb]
            if tb.time() > CUTOFF:
                meta["s3_touch_bar_after_cutoff"] += 1
            elif float(tbar["c"]) > pw:
                entry = float(tbar["c"])
                volc = bool(float(tbar["v"]) > med_vol)
                tp = None
                if mid_dn is not None and mid_dn >= entry * (1 + GAP):
                    tp = mid_dn
                elif flip is not None and flip >= entry * (1 + GAP):
                    tp = flip
                rec = {**base_id, "setup": "S3_PW_REJECT_BOUNCE", "volc": volc}
                if tp is None:
                    skips.append(rec)
                else:
                    trades.append({**rec, **simulate(bd, md, tb, entry, +1, tp, "le", vd, c1, em1)})

        # ---- S4 PW-BREAK-MOM (SHORT @ confirm close)
        if pw_confirm is not None:
            cbar = bd.loc[pw_confirm]
            entry = float(cbar["c"])
            volc = bool(float(cbar["v"]) > med_vol)
            if vd is not None and entry <= vd:
                meta["s4a_entry_at_or_below_vd"] += 1
            recA = {**base_id, "setup": "S4A_PW_BREAK_VD_TP", "volc": volc}
            trades.append({**recA, **simulate(bd, md, pw_confirm, entry, -1, vd, "gt", pw, c1, em1)})
            recB = {**base_id, "setup": "S4B_PW_BREAK_CASCADE", "volc": volc}
            trades.append({**recB, **simulate(bd, md, pw_confirm, entry, -1, None, "gt", pw, c1, em1,
                                              cascade_lvl=vd)})

    # ------------------------------------------------------------------ olasılık tabloları
    def agg_probs(key: str) -> dict:
        out = {}
        for rg in REGROWS:
            sub = devs if rg == "all" else [e for e in devs if e[key] == int(rg)]
            cells = {}
            den = [e for e in sub if e.get("cw_valid")]
            cells["p_cw_touch"] = pcell(sum(e["cw_touch"] for e in den), len(den))
            tch = [e for e in den if e["cw_touch"]]
            cells["p_break_given_cw_touch"] = pcell(sum(e["cw_break"] for e in tch), len(tch))
            brk = [e for e in den if e["cw_break"] and e.get("cw_vu_reach") is not None]
            cells["p_vu_given_cw_break"] = pcell(sum(e["cw_vu_reach"] for e in brk), len(brk))
            den = [e for e in sub if e.get("pw_valid")]
            cells["p_pw_touch"] = pcell(sum(e["pw_touch"] for e in den), len(den))
            tch = [e for e in den if e["pw_touch"]]
            cells["p_break_given_pw_touch"] = pcell(sum(e["pw_break"] for e in tch), len(tch))
            brk = [e for e in den if e["pw_break"] and e.get("pw_vd_reach") is not None]
            cells["p_vd_given_pw_break"] = pcell(sum(e["pw_vd_reach"] for e in brk), len(brk))
            out[rg] = cells
        return out

    # ------------------------------------------------------------------ setup EV tabloları
    def agg_setups(key: str, vol_only: bool) -> dict:
        out = {}
        for st in SETUPS:
            out[st] = {}
            for rg in REGROWS:
                trs = [t for t in trades if t["setup"] == st
                       and (rg == "all" or t[key] == int(rg))
                       and (not vol_only or t["volc"])]
                sks = [s for s in skips if s["setup"] == st
                       and (rg == "all" or s[key] == int(rg))
                       and (not vol_only or s["volc"])]
                out[st][rg] = ev_cell(trs, sks)
        return out

    res = {
        "meta": meta,
        "prob_table_own": agg_probs("regime_own"),
        "prob_table_idx": agg_probs("regime_idx"),
        "setups_own": agg_setups("regime_own", vol_only=False),
        "setups_own_volsplit": agg_setups("regime_own", vol_only=True),
        "setups_idx": agg_setups("regime_idx", vol_only=False),
        "caveats": CAVEATS,
    }
    return res


CAVEATS = [
    "P(break|touch) BREAK-CONFIRM tanimini kullanir (ilk 15-dk close-through barinin baslangici <= 14:45 ET); "
    "tek close-through'u 14:45'ten SONRA gelen gunler ne break ne reject sayilir (meta.cw/pw_break_late_only).",
    "Varyant B es-bar kurali: VU/VD intrabar dokunus ile ayni 15-dk barin stop-close'u cakisirsa STOP sayilir "
    "(se_config'in muhafazakar TP/SL kuralinin B'ye genisletilmesi); disable ancak dokunus-bari stop'suz bitince baslar.",
    "S1/S3 TP adaylari NaN ise kosul False -> SKIP (panelde 2 gun flip/mid_up NaN olabilir).",
    "regime_idx=0 gunler (idx serisinde D yok) +1/-1 satirlarinin disinda kalir; 'all' satiri onlari icerir "
    "(meta.regime_idx_zero_days).",
    "Volume-confirm o seansin TUM 15-dk barlarinin medyan hacmini kullanir (se_config: betimsel ikili split; "
    "giris aninda PIT degildir).",
    "S2A'da confirm bar VU'nun da otesinde kapanmissa TP=VU giris altinda kalir -> literal VU fill = zarar "
    "(meta.s2a_entry_at_or_above_vu / s4a_entry_at_or_below_vd sayilari).",
    "Touch-bari 14:45'ten sonra baslayan gunlerde S1/S3 girisi yok (skip SAYILMAZ; meta.s1/s3_touch_bar_after_cutoff).",
    "hit_pct gross>0 uzerinden (hit_pct_net ayrica raporlanir); t_net ayni std ile (sabit maliyet kaymasi).",
    "MFE/MAE giris barinin bitiminden cikis dakikasina kadarki 1-dk barlardan, 0'a kirpilmis, em1 birimi.",
    "Es-bar TP+SL -> STOP (stop_tie) se_config geregi; cikis o barin kapanisi.",
]


def fmt(x, d=1):
    return "  None" if x is None else f"{x:6.{d}f}"


def main():
    SE.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_res = {}
    for sym in SE.SYMS:
        res = run_sym(sym)
        all_res[sym] = res
        out = SE.RESULTS_DIR / f"T2_{sym.lower()}.json"
        out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")

        m = res["meta"]
        print(f"\n{'='*100}\n{sym} — T2 duvar hold/break + VU/VD kaskad   (n_days={m['n_days']}, "
              f"join dogrulandi={m['join_verified_days']})\n{'='*100}")
        print(f"meta: cw_late_only={m['cw_break_late_only']} pw_late_only={m['pw_break_late_only']} "
              f"s1_gec_touch={m['s1_touch_bar_after_cutoff']} s3_gec_touch={m['s3_touch_bar_after_cutoff']} "
              f"s2a_entry>=vu={m['s2a_entry_at_or_above_vu']} s4a_entry<=vd={m['s4a_entry_at_or_below_vd']} "
              f"idx0={m['regime_idx_zero_days']}")
        for key, tbl in (("regime_own (PRIMARY)", res["prob_table_own"]),
                         ("regime_idx (SENSITIVITY)", res["prob_table_idx"])):
            print(f"\n--- OLASILIK TABLOSU [{key}] ---")
            print(f"{'hucre':28s} " + " ".join(f"{rg:>24s}" for rg in REGROWS))
            for pk in PROB_KEYS:
                cells = []
                for rg in REGROWS:
                    c = tbl[rg][pk]
                    if c["p"] is None:
                        cells.append(f"{'—':>24s}")
                    else:
                        s = f"{100*c['p']:3.0f}% [{100*c['wilson_lo']:.0f},{100*c['wilson_hi']:.0f}] n={c['n']}"
                        s += "!" if c["flag"] else ""
                        cells.append(f"{s:>24s}")
                print(f"{pk:28s} " + " ".join(cells))
        for name, tbl in (("setups_own [ALL trades]", res["setups_own"]),
                          ("setups_own [VOLUME-CONFIRM]", res["setups_own_volsplit"]),
                          ("setups_idx [ALL trades]", res["setups_idx"])):
            print(f"\n--- SETUP EV [{name}] --- (n | hit% [CI] | gross | net | t_net | MFE/MAE em1 | skip)")
            for st in SETUPS:
                for rg in REGROWS:
                    c = tbl[st][rg]
                    if c["n"] == 0:
                        print(f"{st:22s} {rg:>3s}  n=0 skips={c['skips']}  YETERSIZ-N")
                        continue
                    hit = f"{100*c['hit_pct']:3.0f}% [{100*c['hit_wilson_lo']:.0f},{100*c['hit_wilson_hi']:.0f}]"
                    print(f"{st:22s} {rg:>3s}  n={c['n']:<3d} {hit:>14s}  g={fmt(c['mean_gross_bps'])} "
                          f"n={fmt(c['mean_net_bps'])}  t={fmt(c['t_net'],2)}  "
                          f"mfe={fmt(c['med_mfe_em1'],2)}/mae={fmt(c['med_mae_em1'],2)}  "
                          f"skip={c['skips']} {c['flag']}")
    n_cells = 0
    for sym in SE.SYMS:
        n_cells += 2 * len(REGROWS) * len(PROB_KEYS)              # prob own+idx
        n_cells += 3 * len(SETUPS) * len(REGROWS)                 # setups own + volsplit + idx
    print(f"\nTOPLAM RAPORLANAN HUCRE: {n_cells} (secim yok, hepsi JSON'da)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
