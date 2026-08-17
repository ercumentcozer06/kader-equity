# -*- coding: utf-8 -*-
"""
aicapex_credit_lab — T10: AI-CAPEX TAHVIL EMILIMI, ON-KAYITLI test bataryasi.
Bu dosya SADECE backtest/ altinda yasar; modules/ ve run.py'ye DOKUNMAZ.

IDDIA (Constan): AI capex'i finanse etmek icin ihrac edilen kurumsal tahviller par altinda,
UST'ye spread'leri genisliyor = piyasanin bu ihraci EMMEKTE zorlanmasi ("indigestion").
Ve bu capex TUM EKONOMIYI tasiyor -> finansman kanali tikanirsa boom durur.

PANEL (screen/fetch_aicapex_credit_panel.py): iShares LQD gunluk holdings'ten haftalik
  diff_adj = duration-kontrollu (AI spread - AI-disi IG spread), bp
  ai_below_par = AI tahvillerinin fiyat<100 orani
Panel piyasa-geneli kredi hareketini VE vade farkini kontrol eder (bkz fetch dosyasi).

ON-KAYITLI SORULAR (sonuca bakilmadan yazildi):
  D1 BETIMSEL — Constan'in cumlesi dogru mu? diff_adj ve ai_below_par seviyesi/trendi;
     2022-23 (AI oncesi) taban ile 2024+ kiyasi.
  H1 KREDI -> HISSE (ana hipotez): diff_adj ARTISI ileri AI-hisse getirisini (SMH varsa,
     yoksa NDX) NEGATIF ongoruyor mu? Kredi piyasasi hisseyi onculer (standart).
     Ufuk merdiveni {2,4,8,13} hafta. Yon-agnostik KOVA once, sonra iki yon.
  H2 SEVIYE mi DEGISIM mi: diff_adj'in kendi 52h persentili (seviye) vs 4h degisimi.
  H3 KONTROL: ayni test AI-disi IG spread'i (nonai_spread) ile — eger o da ayni sonucu
     veriyorsa bulgu AI'ya ozgu DEGIL, genel kredi sinyalidir. AYIRT EDICI KAPI.
  H4 MEVCUT MODELE EKLIYOR MU: kader-macro m6 (HY OAS) ile korelasyon; HY OAS'i kontrol
     ettikten sonra diff_adj'in artik bilgisi kaliyor mu (2 degiskenli NW regresyon).

BAR: yon-agnostik kova -> iki yonlu kural -> eslestirilmis blok-bootstrap dSharpe CI +
  blok-permutasyon plasebo + BH-FDR (alpha=0.10) tum kollarda ORTAK aile.
  H3 ayirt ediciligi DUSERSE bulgu "genel kredi" olarak etiketlenir, "AI edge" DENMEZ.
ORNEKLEM UYARISI: haftalik, ~2022+ -> n~240h; 13h ufukta ~18 bagimsiz gozlem. Kucuk.
HICBIR SEY ENTEGRE EDILMEZ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
KM = Path(r"C:\Users\admin\Downloads\kader-macro")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PANEL = ROOT / "data" / "cache" / "aicapex_credit_panel.parquet"
H = (2, 4, 8, 13)
BLOCK_W, N_BOOT, N_PERM, ALPHA = 8, 2000, 200, 0.10

import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("_hc_eq", KM / "backtest" / "hc_battery" / "_hc.py")
_hc = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_hc)


def load_px(name: str, idx) -> pd.Series:
    try:
        import yfinance as yf
        s = yf.download(name, period="6y", interval="1d", progress=False, auto_adjust=True)["Close"]
        if hasattr(s, "columns"):
            s = s.iloc[:, 0]
        return s.dropna().reindex(idx, method="ffill")
    except Exception as e:
        print(f"  ! {name} indirilemedi: {e}")
        return pd.Series(dtype=float)


def hy_oas(idx) -> pd.Series:
    import json
    p = KM / "data" / "fred_cache" / "BAMLH0A0HYM2.json"
    o = json.load(open(p, encoding="utf-8"))["observations"]
    s = pd.Series({pd.Timestamp(x["date"]): pd.to_numeric(x["value"], errors="coerce") for x in o}).dropna()
    s.index = s.index + pd.Timedelta(days=1)                # PIT
    return s.reindex(idx, method="ffill")


def main() -> int:
    if not PANEL.exists():
        print("HALT: panel yok -> screen/fetch_aicapex_credit_panel.py")
        return 2
    p = pd.read_parquet(PANEL).sort_index()
    idx = p.index
    _hc.census({"diff_adj(bp)": p["diff_adj"], "ai_below_par": p["ai_below_par"],
                "nonai_spread(bp)": p["nonai_spread"], "n_ai": p["n_ai"].astype(float)},
               required=("diff_adj(bp)",), title="T10 PANEL SAYIMI", min_rows=40)

    _hc.hdr("D1 BETIMSEL — Constan'in cumlesi dogru mu?")
    yr = p.groupby(p.index.year)[["diff_adj", "ai_spread", "nonai_spread", "ai_below_par",
                                  "nonai_below_par", "ai_price_w", "n_ai"]].mean()
    print(yr.round(2).to_string())
    print(f"\n  SON okuma {idx[-1].date()}: diff_adj={p['diff_adj'].iloc[-1]:.1f}bp  "
          f"(52h persentil={float((p['diff_adj'].iloc[-1] >= p['diff_adj'].tail(52)).mean()):.0%})  "
          f"AI par-alti={p['ai_below_par'].iloc[-1]:.0%} vs AI-disi {p['nonai_below_par'].iloc[-1]:.0%}")

    tgt = {}
    for t in ("SMH", "^NDX"):
        s = load_px(t, idx)
        if len(s.dropna()) > 100:
            tgt[t] = s
    if not tgt:
        print("HALT: hedef fiyat yok")
        return 1

    lvl = _hc.roll_pct(p["diff_adj"], 52)
    chg = p["diff_adj"].diff(4)
    sigs = {"seviye(52h pct)": lvl, "degisim(4h)": chg, "KONTROL nonai_seviye": _hc.roll_pct(p["nonai_spread"], 52)}

    _hc.hdr("H1/H2/H3 YON-AGNOSTIK KOVA — sinyal terzili -> ileri getiri (bps)")
    for sn, sv in sigs.items():
        for tn, px in tgt.items():
            line = f"  {sn:<22}{tn:<7}"
            for h in H:
                fr = (px.shift(-h) / px - 1.0)
                df = pd.concat([sv.rename("s"), fr.rename("r")], axis=1).dropna()
                if len(df) < 60:
                    line += f"  h{h}: -"
                    continue
                q = pd.qcut(df["s"], 3, labels=False, duplicates="drop")
                m = df.groupby(q)["r"].mean() * 10000
                r = _hc.nw_tstat(df["r"], df["s"], lags=h)
                line += f"  h{h}: T1={m.get(0, np.nan):>6.0f} T3={m.get(2, np.nan):>6.0f} t={r['t']:>5.2f}"
            print(line)

    _hc.hdr("H4 MEVCUT MODELE EKLIYOR MU — HY OAS kontrolu")
    hy = hy_oas(idx)
    print(f"  corr(diff_adj, HY OAS) = {float(pd.concat([p['diff_adj'], hy], axis=1).dropna().corr().iloc[0, 1]):+.3f}")
    for tn, px in tgt.items():
        for h in (4, 13):
            fr = (px.shift(-h) / px - 1.0)
            d = pd.concat([fr.rename("y"), p["diff_adj"].rename("ai"), hy.rename("hy")], axis=1).dropna()
            if len(d) < 60:
                continue
            X = np.column_stack([np.ones(len(d)), d["ai"], d["hy"]])
            b, *_ = np.linalg.lstsq(X, d["y"].to_numpy(), rcond=None)
            e = d["y"].to_numpy() - X @ b
            U = e[:, None] * X
            S = U.T @ U
            for l in range(1, h + 1):
                w = 1 - l / (h + 1)
                G = U[l:].T @ U[:-l]
                S += w * (G + G.T)
            V = np.linalg.inv(X.T @ X) @ S @ np.linalg.inv(X.T @ X)
            se = np.sqrt(np.maximum(np.diag(V), 0))
            print(f"  {tn:<7}h={h:<3}n={len(d):<4} beta_AI={b[1]:+.5f} (t={b[1]/se[1]:+.2f})  "
                  f"beta_HY={b[2]:+.5f} (t={b[2]/se[2]:+.2f})")

    _hc.hdr("KOLLAR — long/flat, taban = al-tut; ortak BH-FDR")
    rows = []
    print(f"  {'kol':<40}{'n':>5}{'maruz':>7}{'Sh':>7}{'taban':>7}{'dSh':>7}{'d-p5':>7}{'p':>7}{'plas':>7}")
    for sn, sv in sigs.items():
        for tn, px in tgt.items():
            base = px.pct_change().dropna()
            for sgn, tag in ((-1, "dusuk->long"), (+1, "yuksek->long")):
                sc = (sv - (0.5 if "pct" in sn or "seviye" in sn else 0.0)) * sgn
                sg, pxw = _hc.sig_window(sc, px)
                if len(pxw) < 80:
                    continue
                ret = _hc.long_flat_ret(sg > 0, pxw, exec_lag=1)
                b2 = pxw.pct_change().dropna()
                if len(ret) < 60:
                    continue
                d = _hc.boot_diff_ci(ret, b2, block=BLOCK_W, n_boot=N_BOOT, ppy=52)
                pl = _hc.placebo_p(lambda s: _hc.long_flat_ret(s > 0, pxw), sg,
                                   n_perm=N_PERM, block=BLOCK_W)
                nm = f"{sn}|{tn}|{tag}"
                rows.append({"arm": nm, "d": d, "pl": pl})
                print(f"  {nm:<40}{len(ret):>5}{float((sg > 0).mean()):>7.2f}"
                      f"{_hc.sharpe(ret, 52):>7.3f}{_hc.sharpe(b2, 52):>7.3f}"
                      f"{d['d_sharpe']:>7.3f}{d['p5']:>7.3f}{d['p_le0']:>7.3f}{pl['p']:>7.3f}")
    flags = _hc.bh_fdr([r["d"]["p_le0"] for r in rows], alpha=ALPHA)
    surv = []
    print(f"\n  BH-FDR (alpha={ALPHA}, aile n={len(rows)}):")
    for r, f in zip(rows, flags):
        ok = bool(f and r["pl"]["p"] < 0.10 and r["d"]["p5"] > 0)
        ctrl = "KONTROL" in r["arm"]
        if ok and not ctrl:
            surv.append(r["arm"])
        print(f"    {r['arm']:<40}FDR={'GECTI' if f else 'kaldi':<6}-> "
              f"{'ADAY' if ok and not ctrl else ('KONTROL de gecti -> AI-ozgu DEGIL' if ok and ctrl else 'RED')}")
    print(f"\n  AYAKTA KALAN: {surv if surv else 'YOK'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
