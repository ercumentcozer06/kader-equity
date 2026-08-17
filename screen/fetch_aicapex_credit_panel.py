# -*- coding: utf-8 -*-
"""
fetch_aicapex_credit_panel — T10: AI-CAPEX TAHVIL EMILIMI paneli (Constan iddiasi).

CONSTAN (Excess Returns, 2026-08, kelimesi kelimesine):
  "SpaceX ... is below its IPO price. Google did an $80 billion issuance. Everybody who
   bought that is out money. EVERY CORPORATE BOND THAT'S BEEN ISSUED TO FUND DATA CENTERS
   IS TRADING BELOW PAR ... their SPREAD TO TREASURIES IS WIDER than it's been. There's a
   fair amount of INDIGESTION regarding the absorption by the financial markets of the
   issuance required to pay for the capex which the entire economy depends on."

KAYNAK (2026-08-15'te EMB icin bulunan ucu LQD'ye tasiyoruz — bkz reference_tr_eurobond_ishares_source):
  iShares LQD (IG kurumsal) holdings, BlackRock varnish-api, portfolioId=239566,
  component=holdings & asOfDate=YYYYMMDD -> GUNLUK, auth'suz, ISIN + Price + YTM +
  Duration + Maturity + Coupon + Weight. ~3100 tahvil/gun.

ON-KAYITLI AI-CAPEX SEPETI (veriye BAKILMADAN, Constan'in saydigi gruplar):
  hyperscaler : MICROSOFT, ALPHABET, GOOGLE, AMAZON, META PLATFORMS, ORACLE
  cip/bellek  : BROADCOM, MICRON, INTEL, NVIDIA, ADVANCED MICRO
  veri-merkezi: EQUINIX, DIGITAL REALTY
  DISLANAN: APPLE — nakit-zengin, veri-merkezini BORCLA finanse etmiyor; Constan'in
  mekanizmasi "borclanmak zorunda olanlar". Dahil edilseydi sepet seyrelirdi.

OLCUM (kompozisyon ve piyasa-geneli kredi hareketi KONTROL EDILIR):
  1) her tahvil icin spread_i = YTM_i - UST(vade_i)   [DGS1..DGS30 interpolasyonu]
  2) DIFF     = agirlikli ort spread(AI) - agirlikli ort spread(AI-DISI IG)
     -> piyasa-geneli kredi hareketini duser
  3) DIFF_adj = AYNI GUN AI-disi tahvillerde spread ~ a + b*duration regresyonu kurulur,
     AI tahvillerinin durationlarina UYGULANIR, fark alinir
     -> "AI spread'i genis" gozlemi sadece DAHA UZUN VADELI olmalarindan mi? sorusunu keser
  4) par_alti_pay = AI tahvillerinin fiyat<100 orani (Constan'in birebir cumlesi)
Cikti: data/cache/aicapex_credit_panel.parquet  (date, n_ai, ai_spread, nonai_spread,
       diff, diff_adj, ai_below_par, ai_dur, nonai_dur, ai_price_w)

Kosum: python screen/fetch_aicapex_credit_panel.py [--start 2024-01-03] [--freq W-WED]
Nezaket: 1.1s/istek (EMB avinda kanitlanmis), kalici disk cache -> tekrar kosum agsiz.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache" / "lqd_holdings"
OUT = ROOT / "data" / "cache" / "aicapex_credit_panel.parquet"
KM = Path(r"C:\Users\admin\Downloads\kader-macro")
PID = 239566                       # iShares LQD
URL = ("https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v1/"
       "get-fund-document?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=us-ishares"
       f"&locale=en_US&portfolioId={PID}&component=holdings&userType=individual&asOfDate=%s")
UA = "kader-equity/0.1 (research; emirsancar2003@gmail.com)"
SLEEP = 1.1

AI_PATTERNS = [
    r"MICROSOFT", r"ALPHABET", r"GOOGLE", r"AMAZON\.?COM", r"META PLATFORMS", r"ORACLE",
    r"BROADCOM", r"MICRON", r"INTEL CORP", r"NVIDIA", r"ADVANCED MICRO",
    r"EQUINIX", r"DIGITAL REALTY",
]
AI_RE = re.compile("|".join(AI_PATTERNS), re.I)
UST_TENORS = [("DGS1", 1.0), ("DGS2", 2.0), ("DGS3", 3.0), ("DGS5", 5.0),
              ("DGS7", 7.0), ("DGS10", 10.0), ("DGS20", 20.0), ("DGS30", 30.0)]


def ust_curve() -> pd.DataFrame:
    """kader-macro FRED cache'inden UST egrisi (modules paketi cakismasi yok)."""
    cols = {}
    for sid, yrs in UST_TENORS:
        p = KM / "data" / "fred_cache" / f"{sid}.json"
        if not p.exists():
            continue
        o = json.load(open(p, encoding="utf-8"))["observations"]
        cols[yrs] = pd.Series({pd.Timestamp(x["date"]): pd.to_numeric(x["value"], errors="coerce")
                               for x in o}).dropna()
    return pd.DataFrame(cols).sort_index().ffill()


def fetch_day(d: str) -> pd.DataFrame | None:
    """d = YYYYMMDD. Kalici cache; as-of satiri DOGRULANIR (sahte-tarih tuzagi)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = CACHE / f"lqd_{d}.parquet"
    if cp.exists():
        try:
            return pd.read_parquet(cp)
        except Exception:
            cp.unlink(missing_ok=True)
    try:
        r = requests.get(URL % d, timeout=90, headers={"User-Agent": UA})
    except Exception as e:
        print(f"  ! {d} istek hatasi {e}", flush=True)
        return None
    if r.status_code != 200 or len(r.content) < 5000:
        return None
    txt = r.content.decode("utf-8-sig", errors="ignore")
    lines = txt.splitlines()
    asof = next((l for l in lines[:15] if "Fund Holdings as of" in l), "")
    m = re.search(r'"([A-Z][a-z]{2} \d{1,2}, \d{4})"', asof)
    if not m:
        return None
    got = pd.Timestamp(m.group(1))
    if got.strftime("%Y%m%d") != d:                       # SAHTE-TARIH KAPISI
        return None
    hi = next((i for i, l in enumerate(lines) if l.startswith('"Name"') or l.startswith("Name,")), None)
    if hi is None:
        return None
    df = pd.read_csv(io.StringIO("\n".join(lines[hi:])), engine="python", on_bad_lines="skip")
    need = ["Name", "Price", "YTM (%)", "Duration", "Weight (%)", "Maturity", "Sector"]
    if not all(c in df.columns for c in need):
        return None
    df = df[need].copy()
    for c in ("Price", "YTM (%)", "Duration", "Weight (%)"):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
    df["asof"] = got
    df = df.dropna(subset=["Price", "YTM (%)", "Duration"])
    df = df[(df["Price"] > 20) & (df["Price"] < 200) & (df["YTM (%)"] > 0) & (df["YTM (%)"] < 25)]
    df.to_parquet(cp)
    return df


def summarize(df: pd.DataFrame, curve: pd.DataFrame) -> dict | None:
    d = df["asof"].iloc[0]
    row = curve.reindex(curve.index.union([d])).ffill().loc[d]
    ten = np.array([t for _, t in UST_TENORS if t in row.index], float)
    yld = np.array([row[t] for t in ten], float)
    ok = ~np.isnan(yld)
    if ok.sum() < 3:
        return None
    dur = df["Duration"].clip(0.25, 30).to_numpy()
    ust = np.interp(dur, ten[ok], yld[ok])
    df = df.assign(spread=df["YTM (%)"].to_numpy() - ust)
    ai = df[df["Name"].astype(str).str.contains(AI_RE, na=False)]
    non = df[~df["Name"].astype(str).str.contains(AI_RE, na=False)]
    if len(ai) < 20 or len(non) < 200:
        return None
    w_ai = ai["Weight (%)"].fillna(0).to_numpy()
    w_no = non["Weight (%)"].fillna(0).to_numpy()
    s_ai = float(np.average(ai["spread"], weights=w_ai) if w_ai.sum() > 0 else ai["spread"].mean())
    s_no = float(np.average(non["spread"], weights=w_no) if w_no.sum() > 0 else non["spread"].mean())
    # duration-kontrollu: AI-disi evrende spread ~ a + b*dur, AI durationlarina uygula
    X = np.column_stack([np.ones(len(non)), non["Duration"].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, non["spread"].to_numpy(), rcond=None)
    pred = beta[0] + beta[1] * ai["Duration"].to_numpy()
    diff_adj = float(np.average(ai["spread"].to_numpy() - pred,
                                weights=w_ai if w_ai.sum() > 0 else None))
    return {"date": d, "n_ai": int(len(ai)), "n_non": int(len(non)),
            "ai_spread": s_ai * 100, "nonai_spread": s_no * 100,           # bp
            "diff": (s_ai - s_no) * 100, "diff_adj": diff_adj * 100,
            "ai_below_par": float((ai["Price"] < 100).mean()),
            "nonai_below_par": float((non["Price"] < 100).mean()),
            "ai_dur": float(ai["Duration"].mean()), "nonai_dur": float(non["Duration"].mean()),
            "ai_price_w": float(np.average(ai["Price"], weights=w_ai) if w_ai.sum() > 0 else ai["Price"].mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-03")
    ap.add_argument("--end", default=None)
    ap.add_argument("--freq", default="W-WED")
    a = ap.parse_args()
    end = pd.Timestamp(a.end) if a.end else pd.Timestamp.today().normalize()
    dates = pd.date_range(a.start, end, freq=a.freq)
    curve = ust_curve()
    print(f"LQD AI-CAPEX PANEL | {len(dates)} tarih {dates[0].date()}..{dates[-1].date()} freq={a.freq}")
    print(f"UST egrisi: {curve.shape[1]} tenor {curve.index[0].date()}..{curve.index[-1].date()}")
    rows, miss, t0 = [], 0, time.time()
    for i, d in enumerate(dates, 1):
        ds = d.strftime("%Y%m%d")
        cached = (CACHE / f"lqd_{ds}.parquet").exists()
        df = fetch_day(ds)
        if df is None:
            miss += 1
        else:
            s = summarize(df, curve)
            if s:
                rows.append(s)
        if not cached:
            time.sleep(SLEEP)
        if i % 20 == 0:
            print(f"  {i}/{len(dates)} ok={len(rows)} miss={miss} {time.time()-t0:.0f}s", flush=True)
    if not rows:
        print("HALT: satir yok")
        return 1
    out = pd.DataFrame(rows).set_index("date").sort_index()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT)
    print(f"\nPANEL: n={len(out)} {out.index[0].date()}..{out.index[-1].date()} -> {OUT.name}")
    print(out[["n_ai", "ai_spread", "nonai_spread", "diff_adj", "ai_below_par", "ai_price_w"]].tail(8).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
