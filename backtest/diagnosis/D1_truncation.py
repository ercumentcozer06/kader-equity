"""
backtest/diagnosis/D1_truncation — VERİ SANSÜRÜ (truncation) teşhisi. TEŞHİS-ONLY.

Hipotez ①: tarihsel chain (md_{spy,qqq}.parquet) tarih başına TEK front-monthly expiry içerir →
backtest gamma$, bugünkü 5-vade canlı motorun gördüğünün küçük bir kısmını görüyor olabilir.
③: havuz = ETF (SPY/QQQ) vs index (SPX/NDX) gamma$ oranı. ④: canlı 5-vade bayrak vs backtest 1-vade bayrak.

Hiçbir P&L/strateji üretmez; mevcut level_series/cache OKUNUR. Her sayı = bu script'in çıktısı.
IV = build_level_series ile AYNI yol (_bsiv.implied_vol mid'den), gamma_engine._greeks byte-eş.
Canlı yfinance bacağı (③ bugünkü full chain) gamma_engine konvansiyonuyla aynı (yfinance KENDİ verisi).

  & C:/Users/admin/Downloads/kader-macro/.venv/Scripts/python.exe backtest/diagnosis/D1_truncation.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "screen"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from _bsiv import implied_vol          # noqa: E402  (build_level_series ile AYNI)
from gamma_engine import _greeks       # noqa: E402  (byte-eş greeks)

M = 100
BAND = 0.15
CHAINS = ROOT / "data" / "historical_chains"
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "backtest" / "diagnosis"


# ----------------------------------------------------------------------------- §1
def section1():
    lines = ["## 1. Tarihsel chain yapısı — expiry sayısı / monthly-OPEX testi / DTE konvansiyonu", ""]
    for sym in ("spy", "qqq"):
        df = pd.read_parquet(CHAINS / f"md_{sym}.parquet")
        df["date"] = df["date"].astype(str)
        df["expiration"] = df["expiration"].astype(str)
        n_dates = df["date"].nunique()
        # sembol başına unique expiration
        uniq_exp = df["expiration"].nunique()
        per_date_exp = df.groupby("date")["expiration"].nunique()
        # 3.-Cuma testi: her expiry için weekday + ay-içi-hafta (ceil(day/7))
        exp_dates = sorted(df["expiration"].unique())
        third_fri = 0
        detail = []
        for e in exp_dates:
            dt = pd.Timestamp(e)
            wd = dt.weekday()                       # 4 = Cuma
            week_of_month = (dt.day - 1) // 7 + 1    # 1..5
            is3f = (wd == 4 and week_of_month == 3)
            third_fri += int(is3f)
            detail.append((e, wd, week_of_month, is3f))
        # vade/yıl: 243 gün ~ ? farklı expiry; expiry başına ortalama gün
        span_years = (pd.Timestamp(df["date"].max()) - pd.Timestamp(df["date"].min())).days / 365.25
        exp_per_year = uniq_exp / span_years
        # DTE iki konvansiyon: calendar vs trading-day (busday_count)
        per_date = df.groupby("date")["expiration"].first().reset_index()
        cal_dte = (pd.to_datetime(per_date["expiration"]) - pd.to_datetime(per_date["date"])).dt.days
        bus_dte = np.array([np.busday_count(d, e) for d, e in
                            zip(per_date["date"].values, per_date["expiration"].values)])
        lines += [
            f"### {sym.upper()}",
            f"- unique tarih: {n_dates}; unique expiration (tüm seri): **{uniq_exp}**; "
            f"tarih başına expiry: min {per_date_exp.min()} / med {int(per_date_exp.median())} / max {per_date_exp.max()} "
            f"→ **tarih başına TEK expiry** ({(per_date_exp==1).mean()*100:.0f}%)",
            f"- 3.-Cuma (weekday=Cuma & ay-3.-haftası) olan expiry: **{third_fri}/{uniq_exp}** "
            f"({100*third_fri/uniq_exp:.0f}%) → {'monthly-OPEX zinciri' if third_fri/uniq_exp>0.7 else 'KARIŞIK (weekly dahil)'}",
            f"- expiry/yıl ≈ **{exp_per_year:.1f}** (span {span_years:.2f}y) → "
            f"{'~12/yıl MONTHLY' if 9 <= exp_per_year <= 15 else 'monthly-12 DEĞİL'}",
            f"- DTE **calendar-day** (expiration−date): min {int(cal_dte.min())} / med **{int(cal_dte.median())}** / max {int(cal_dte.max())}",
            f"- DTE **trading-day** (np.busday_count): min {int(bus_dte.min())} / med **{int(np.median(bus_dte))}** / max {int(bus_dte.max())}",
            f"- → 'DTE 0-25 med8' = **calendar-day** konvansiyonu (trading-day med={int(np.median(bus_dte))})",
            "",
        ]
        # ilk birkaç expiry örneği
        ex = ", ".join(f"{e}(wd{wd},hf{wk}{'✓3F' if i3 else ''})" for e, wd, wk, i3 in detail[:6])
        lines.append(f"  örnek expiry: {ex}")
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- §3
def _gamma_dollar_from_chain(rows, S):
    """Σ |gamma$| ve net_gex ($/1%) — gamma_engine/build_level_series byte-eş formül.
    gamma$ (toplam, yön-bağımsız) = Σ gamma·OI·100·S²·0.01 ; net_gex = Σ sgn·(...)."""
    sgn = lambda rt: 1.0 if rt == "C" else -1.0
    tot_g = sum(x["g"] * x["oi"] * M * S * S * 0.01 for x in rows)             # yön-bağımsız toplam
    net = sum(sgn(x["right"]) * x["g"] * x["oi"] * M * S * S * 0.01 for x in rows)
    tot_oi = sum(x["oi"] for x in rows)
    return tot_g, net, tot_oi


def _live_chain_gamma(tick):
    """BUGÜN yfinance full chain (N tüm vade, ±15% bant) → DTE-bucket gamma$ + monthly/non-monthly.
    yfinance KENDİ IV alanı DEĞİL — _bsiv.mid_iv_from_row (bid/ask MID) build_level_series ile tutarlı.
    Index chain (^SPX/^NDX) yfinance'te genelde YOK → varsa ölç, yoksa 'ölçülemedi'."""
    try:
        import yfinance as yf
        from _bsiv import mid_iv_from_row
        from math import log
    except Exception as e:
        return {"err": f"import {e}"}
    t = yf.Ticker(tick)
    try:
        spot = float(t.fast_info["lastPrice"])
    except Exception:
        try:
            spot = float(t.history(period="1d")["Close"].iloc[-1])
        except Exception as e:
            return {"err": f"spot {e}"}
    today = date.today()
    exps = []
    for e in (t.options or []):
        try:
            d = datetime.strptime(e, "%Y-%m-%d").date()
        except Exception:
            continue
        if (d - today).days >= 0:
            exps.append((d, e))
    exps.sort()
    if not exps:
        return {"err": "zincir-yok"}
    buckets = {"0-1": 0.0, "2-5": 0.0, "6-21": 0.0, ">21": 0.0}
    monthly_g = non_monthly_g = 0.0
    front_monthly_g = 0.0           # tarihsel-replikanın gördüğü: ilk monthly expiry tek-başına
    tot_oi = 0.0
    n_exp_used = 0
    first_monthly_done = False
    for d, e in exps:
        dte = (d - today).days
        try:
            oc = t.option_chain(e)
        except Exception:
            continue
        T = max(dte, 0.5) / 365.0
        rows = []
        for df, right in ((oc.calls, "C"), (oc.puts, "P")):
            for _, r in df.iterrows():
                K, oi = r.get("strike"), r.get("openInterest")
                if not K or not oi or oi != oi or abs(K / spot - 1) > BAND:
                    continue
                iv = mid_iv_from_row(r, spot, float(K), T, right)
                if iv is None or iv <= 0:
                    continue
                g, *_ = _greeks(spot, float(K), T, iv, right)
                rows.append({"g": g, "oi": float(oi), "right": right})
        if not rows:
            continue
        n_exp_used += 1
        gE, _net, oiE = _gamma_dollar_from_chain(rows, spot)
        tot_oi += oiE
        b = "0-1" if dte <= 1 else "2-5" if dte <= 5 else "6-21" if dte <= 21 else ">21"
        buckets[b] += gE
        wd, wk = d.weekday(), (d.day - 1) // 7 + 1
        is_monthly = (wd == 4 and wk == 3)
        if is_monthly:
            monthly_g += gE
            if not first_monthly_done:
                front_monthly_g = gE          # tarihsel veri ~ ilk monthly expiry (tek vade)
                first_monthly_done = True
        else:
            non_monthly_g += gE
    total_g = sum(buckets.values())
    return {"spot": spot, "n_exp": n_exp_used, "buckets": buckets, "total_g": total_g,
            "monthly_g": monthly_g, "non_monthly_g": non_monthly_g,
            "front_monthly_g": front_monthly_g, "tot_oi": tot_oi}


def section3():
    lines = ["## 3. BUGÜN full chain (yfinance, tüm vade) → toplam gamma$ + DTE-bucket + monthly dağılımı", ""]
    etf = {}
    for tick in ("SPY", "QQQ"):
        r = _live_chain_gamma(tick)
        etf[tick] = r
        if "err" in r:
            lines.append(f"### {tick}: ölçülemedi — {r['err']}")
            lines.append("")
            continue
        b = r["buckets"]
        tg = r["total_g"] or 1.0
        lines += [
            f"### {tick}  spot {r['spot']:.2f}  ({r['n_exp']} vade kullanıldı, ±15% bant, mid-IV)",
            f"- toplam gamma$ (yön-bağımsız Σ|γ·OI·100·S²·0.01|): **${r['total_g']/1e9:.2f}bn**  (toplam OI {r['tot_oi']/1e6:.2f}M)",
            f"- DTE-bucket: 0-1 ${b['0-1']/1e9:.2f}bn ({100*b['0-1']/tg:.0f}%) | "
            f"2-5 ${b['2-5']/1e9:.2f}bn ({100*b['2-5']/tg:.0f}%) | "
            f"6-21 ${b['6-21']/1e9:.2f}bn ({100*b['6-21']/tg:.0f}%) | "
            f">21 ${b['>21']/1e9:.2f}bn ({100*b['>21']/tg:.0f}%)",
            f"- monthly (3.-Cuma) ${r['monthly_g']/1e9:.2f}bn ({100*r['monthly_g']/tg:.0f}%) vs "
            f"non-monthly ${r['non_monthly_g']/1e9:.2f}bn ({100*r['non_monthly_g']/tg:.0f}%)",
            f"- **TEK SAYI**: tarihsel-veri (tek-front-monthly ${r['front_monthly_g']/1e9:.2f}bn) bugünkü toplam gamma$'ın "
            f"**%{100*r['front_monthly_g']/tg:.0f}**'ini görüyordu  (kalan %{100*(1-r['front_monthly_g']/tg):.0f} sansürlü)",
            "",
        ]
    # ^SPX / ^NDX index chain
    lines.append("### Index chain (^SPX / ^NDX) — ④ havuz için")
    idx = {}
    for tick in ("^SPX", "^NDX"):
        r = _live_chain_gamma(tick)
        idx[tick] = r
        if "err" in r:
            lines.append(f"- {tick}: ölçülemedi — index-chain yfinance'te yok ({r['err']})")
        else:
            lines.append(f"- {tick} spot {r['spot']:.2f} toplam gamma$ **${r['total_g']/1e9:.2f}bn** ({r['n_exp']} vade)")
    lines.append("")
    return "\n".join(lines), etf, idx


# ----------------------------------------------------------------------------- §4
def section4(etf, idx):
    lines = ["## 4. Havuz oranı — index (SPX/NDX) vs ETF (SPY/QQQ) gamma$", ""]
    pairs = [("^SPX", "SPY"), ("^NDX", "QQQ")]
    # yfinance index-OI GÜVENİLMEZ: ^SPX front-expiry total-OI=0 (ayrı probe), bid/ask kapsamı %33-62
    # (ETF %63-66). Index gamma$ ETF'in altında çıkıyor (SPX havuzu SPY'den BÜYÜK olmalı) → ham oran artefakt.
    for ix, ef in pairs:
        ri, re = idx.get(ix, {}), etf.get(ef, {})
        if "err" in ri or "err" in re or not re.get("total_g") or not ri.get("total_g"):
            why = ri.get("err", re.get("err", "?"))
            lines.append(f"- {ix}/{ef}: ölçülemedi — {ix} index-chain yfinance'te yok ({why})")
        else:
            ratio = ri["total_g"] / re["total_g"]
            artefact = ratio < 1.0           # index havuzu ETF'ten KÜÇÜK çıkması fiziksel-olarak olanaksız
            note = (" — **ölçülemedi/ARTEFAKT** (index gamma$ ETF'in altında = yfinance index-OI eksik; "
                    "gerçek SPX/NDX havuzu için CBOE/ORATS/SpotGamma OI lazım)") if artefact else ""
            lines.append(f"- {ix}/{ef} ham gamma$ oranı = **{ratio:.2f}×** "
                         f"(index ${ri['total_g']/1e9:.2f}bn / ETF ${re['total_g']/1e9:.2f}bn, "
                         f"index-OI {ri['tot_oi']/1e3:.0f}k vs ETF {re['tot_oi']/1e3:.0f}k){note}")
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- §5
def section5():
    lines = ["## 5. Canlı snapshot bayrağı vs backtest level_series bayrağı uyumu", ""]
    for sym in ("spy", "qqq"):
        snapdir = CACHE / f"gamma_{sym}"
        lvp = CACHE / f"level_series_{sym}.parquet"
        snaps = sorted(snapdir.glob("*.json")) if snapdir.exists() else []
        if not snaps:
            lines.append(f"- {sym.upper()}: canlı snapshot yok → ④ ileriye-dönük ölçülür")
            continue
        lv = pd.read_parquet(lvp)
        lv.index = pd.to_datetime(lv.index).date
        rows = []
        for f in snaps:
            j = json.load(open(f, encoding="utf-8"))
            d = pd.Timestamp(j.get("as_of")).date()
            live_sign = 1 if (j.get("net_gex_bn") or 0) >= 0 else -1
            bt_sign = int(lv.loc[d, "regime"]) if d in lv.index else None
            rows.append((d, live_sign, bt_sign, j.get("net_gex_bn")))
        overlap = [(d, ls, bs, ng) for d, ls, bs, ng in rows if bs is not None]
        lines.append(f"### {sym.upper()}: {len(snaps)} canlı snapshot, level_series'le örtüşen gün: {len(overlap)}")
        for d, ls, bs, ng in rows:
            tag = "—" if bs is None else ("UYUM" if ls == bs else "ÇELİŞKİ")
            lines.append(f"  {d}: canlı net_gex {ng:+.2f}bn (işaret {ls:+d}) vs backtest regime "
                         f"{bs if bs is not None else 'örtüşme-yok'} → {tag}")
        if overlap:
            agree = sum(1 for _, ls, bs, _ in overlap if ls == bs) / len(overlap)
            lines.append(f"  → bayrak uyum-% = **{100*agree:.0f}%** (n={len(overlap)})")
        else:
            lines.append(f"  → örtüşen gün YOK (canlı snapshot'lar level_series son tarihinden ({lv.index.max()}) sonra) "
                         f"→ ④ uyum ileriye-dönük ölçülür")
        lines.append("")
    return "\n".join(lines)


def main():
    out = ["# D1 — VERİ SANSÜRÜ (truncation) teşhisi", ""]
    out.append(section1())
    s3, etf, idx = section3()
    out.append(s3)
    out.append(section4(etf, idx))
    out.append(section5())
    txt = "\n".join(out)
    print(txt)
    (OUT / "D1_truncation_report.md").write_text(txt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
