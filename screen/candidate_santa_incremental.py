"""
screen/candidate_santa_incremental — Constan KOŞULLU Noel rallisinin TIDE-STACK'e karşı ARTIMSAL testi.

Bağlam: standalone replikasyon (screen/candidate_santa_conditional.py, 2026-06-13) GÜÇLÜ çıktı:
1928+, 1 Kas'ta SPX YTD >= +10% → yıl sonuna tut: n=43, ort +4.54%, isabet %88, 3-dönem stabil,
perm-p<0.001. Bu dosya SONRAKİ adım: edge tide×froth×shield stack'inin ÜSTÜNE bir şey katıyor mu,
yoksa stack o pencerelerde zaten full-long olduğu için absorbe mi ediyor? (OpEx-kapısı emsali:
standalone-gerçek ama model-incremental-nötr → taktik etiket.)

ÖN-KAYITLI TASARIM (sonuçlar görülmeden yazıldı):
  • Nitelik: her yıl, yılın ilk kapanışı → 1 Kas (veya sonraki ilk işgünü) kapanışı YTD >= +10%
    → o yılın Kas-Ara penceresi NİTELİKLİ. Kaynak: data/cache/spx_gspc_long.csv (1927-12-30+).
    NDX niteliği SPX'ten alınır (rejim-durumu); NDX'in KENDİ YTD'siyle bir robustluk satırı ayrıca.
  • Test penceresi: repo standardı 2019+ (frozen tide-stack penceresi). GÜÇ SINIRI dürüstçe
    raporlanır (~5 nitelikli pencere × ~42 gün; fark-günü sayısı basılır).
  • Base = TAM STACK: tide_dir × COR1M-froth(8,11,0) × GEX-shield(0.5,1.0,0.4)
    (finalize_stack ile aynı tek-kaynak modül fonksiyonları; 1.64/1.77 reprodüksiyon sanity basılır).
  • VARYANTLAR (her ikisi {SPX,NDX}, strict BH-FDR, screen._util.paired_win_prob + fdr_bh):
      V1 boost-tam-boy : nitelikli pencerede pos = max(pos, 1.0) (trim'ler kalkar + flat günler
                         dahil tam-boy — Constan'ın koşulsuz-tut kuralının birebir karşılığı)
      V2 boost+0.25    : nitelikli pencerede pos += 0.25, cap 1.0 (kaldıraçsız üst)
      V3 ters-yüz      : nitelik TUTMAYAN yılların Kas-Ara'sında pos × 0.5 (1928+ kanıt: o pencere
                         ort ~0, min −22.7%) — bilgi amaçlı, aynı FDR ailesinde raporlanır
    Her varyant stack'e KARŞI: dSharpe, dMaxDD, paired-bootstrap P(v>b), BH-FDR q (aile = {SPX,NDX}).
    BİLGİ satırları (FDR-verdict DIŞI, post-hoc etiketli): V1t = trim-remove yalnız tide-long
    günlerde (flat override YOK); V1g = V1 + NDX OpEx-günü kapı önceliği (deploy→0 korunur).
  • TIDE-ÖRTÜŞME: nitelikli pencere günlerinde stack ortalama pozisyonu + tide-long gün oranı +
    pos<1 olan gün sayısı (flat vs trim ayrık) — absorpsiyonun doğrudan ölçüsü.
  • ÇİFTE-SAYIM: modules/opex_calendar NDX'i OpEx günü sıfırlıyor; nitelikli Kas+Ara pencerelerinde
    OpEx günleri (Ara = quad-witch) sayılır, V1-vs-V1g farkı raporlanır (boost-kapı çatışması).
VERDICT sözlüğü: INCREMENTAL-PASS (FDR geçti) / ABSORBED (geçmedi + örtüşme yüksek) / MIXED.
Karar (boost-katmanı vs taktik-uyarı-etiketi) Emir'de — bu dosya yalnız ölçer.

Çıktı: stdout (ASCII-only, cp1254-güvenli) + output/santa_incremental_report.txt (utf-8 ayna).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from spine import contract as C, tide as T              # noqa: E402
from backtest import engine as E                         # noqa: E402
from screen._util import load_price_csv, paired_win_prob, fdr_bh   # noqa: E402
from modules.cor1m_froth import froth_factor_series      # noqa: E402
from modules.gex_shield import gex_zscore, shield_factor_series  # noqa: E402
from modules.opex_calendar import third_friday, is_quad_witch    # noqa: E402

DESK = Path(r"C:\Users\admin\Desktop\backtesting")
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "output" / "santa_incremental_report.txt"
YTD_THR = 0.10

_LINES: list[str] = []


def say(line: str = "") -> None:
    _LINES.append(line)
    print(line)


def _sh(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")


def _dd(r):
    eq = (1 + r.dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min())


def strat_ret(pos, close, lag=1):
    idx = pos.index
    ret = E.fwd_ret(close, idx).values
    p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def ytd_nov1_by_year(close: pd.Series) -> pd.Series:
    """Her yil: yilin ilk kapanisi -> 1 Kas (veya sonraki ilk isgunu) kapanisi YTD."""
    out = {}
    for y in sorted(set(close.index.year)):
        yr = close[close.index.year == y]
        pre = yr[yr.index < f"{y}-11-01"]
        nov = yr[yr.index >= f"{y}-11-01"]
        if len(yr) < 100 or pre.empty or nov.empty:
            continue
        out[y] = float(nov.iloc[0] / yr.iloc[0] - 1.0)
    return pd.Series(out)


def main() -> int:
    # ── 0) STACK (base) — finalize_stack ile ayni tek-kaynak modul fonksiyonlari ──
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    cor = pd.read_parquet(CACHE / "corr_pc.parquet")["COR1M"].dropna()
    gex = pd.read_parquet(CACHE / "squeeze_dix_gex.parquet")["gex"].dropna()
    froth = froth_factor_series(cor.reindex(idx, method="ffill"), 8, 11, 0.0)
    zg = gex_zscore(gex).reindex(idx, method="ffill")
    shield = shield_factor_series(zg, 0.5, 1.0, 0.4)
    stack = (tdir * froth * shield).rename("stack_pos")

    bases = {a: strat_ret(stack, prices[a]) for a in ("SPX", "NDX")}
    say("=" * 96)
    say("  CONSTAN KOSULLU NOEL RALLISI — INCREMENTAL over TIDE-STACK (frozen 2019+)")
    say("=" * 96)
    say(f"  sanity (reproduksiyon): stack Sharpe SPX {_sh(bases['SPX']):+.2f} / NDX {_sh(bases['NDX']):+.2f}"
        f"   (beklenen ~ +1.64 / +1.77)")
    say(f"  frozen pencere: {idx.min().date()} -> {idx.max().date()}  ({len(idx)} gun)")

    # ── 1) NITELIK: uzun-tarih SPX cache + NDX robustluk ──
    spx_long = pd.read_csv(CACHE / "spx_gspc_long.csv", parse_dates=["Date"]).set_index("Date")["Close"].dropna().sort_index()
    ytd_spx = ytd_nov1_by_year(spx_long)
    ndx_close = load_price_csv(DESK / "NASDAQ_daily.csv")
    ytd_ndx = ytd_nov1_by_year(ndx_close)

    yrs = sorted(set(idx.year))
    say("")
    say("  [1] NITELIK DURUMU (2019+; kosul: 1 Kas'ta YTD >= +10%)")
    say(f"      {'yil':<6}{'SPX YTD@Kas1':>14}{'nitelikli':>11}{'NDX YTD@Kas1':>14}{'NDX-kendi':>11}{'pencere-gunu':>14}")
    q_years, q_years_ndx = [], []
    for y in yrs:
        if y not in ytd_spx.index:
            continue
        q = ytd_spx[y] >= YTD_THR
        qn = bool(ytd_ndx.get(y, np.nan) >= YTD_THR) if y in ytd_ndx.index else False
        ndays = int(((idx.year == y) & (idx.month >= 11)).sum())
        if ndays == 0:
            continue
        if q:
            q_years.append(y)
        if qn:
            q_years_ndx.append(y)
        say(f"      {y:<6}{ytd_spx[y]*100:>+13.1f}%{('EVET' if q else 'hayir'):>11}"
            f"{(f'{ytd_ndx[y]*100:+.1f}%' if y in ytd_ndx.index else 'n/a'):>14}"
            f"{('EVET' if qn else 'hayir'):>11}{ndays:>14}")

    Q = pd.Series((idx.month >= 11) & np.isin(idx.year, q_years), index=idx)
    NQ = pd.Series((idx.month >= 11) & np.isin(idx.year, [y for y in yrs if y in ytd_spx.index and y not in q_years]), index=idx)
    Qn = pd.Series((idx.month >= 11) & np.isin(idx.year, q_years_ndx), index=idx)
    say(f"      nitelikli yillar (SPX-kosul): {q_years}  -> toplam {int(Q.sum())} gun")
    say(f"      nitelik-tutmayan Kas-Ara    : {[y for y in yrs if y in ytd_spx.index and y not in q_years and ((idx.year==y)&(idx.month>=11)).sum()>0]}"
        f"  -> toplam {int(NQ.sum())} gun")
    say(f"      NDX-kendi-YTD nitelik farki : {sorted(set(q_years) ^ set(q_years_ndx))} (bos = ayni)")
    say(f"      GUC SINIRI: ~{len(q_years)} pencere x ~42 gun = {int(Q.sum())} fark-gunu / {len(idx)} toplam"
        f" ({100*Q.sum()/len(idx):.1f}%) — tek-rejim, dusuk-guc; FDR gecse bile genis CI oku.")

    # ── 2) TIDE-ORTUSME (absorpsiyon olcusu) ──
    say("")
    say("  [2] TIDE-ORTUSME — nitelikli pencere gunlerinde stack zaten ne kadar long?")
    sQ, sA = stack[Q.values], stack
    tQ = tdir[Q.values]
    flatQ = int((tQ == 0).sum())
    trimQ = int(((tQ == 1) & (sQ < 0.999)).sum())
    say(f"      stack ort-pozisyon : nitelikli {sQ.mean():.3f}  vs tum-gunler {sA.mean():.3f}")
    say(f"      tide LONG orani    : nitelikli {100*(tQ == 1).mean():.1f}%  vs tum-gunler {100*(tdir == 1).mean():.1f}%")
    say(f"      pos<1 gun dagilimi : tide-FLAT {flatQ} gun + trim'li {trimQ} gun = {flatQ+trimQ}/{int(Q.sum())}"
        f"  (V1'in fiilen degistirdigi gunler)")
    say(f"      ort froth/shield   : nitelikli {froth[Q.values].mean():.3f}/{shield[Q.values].mean():.3f}"
        f"  vs tum {froth.mean():.3f}/{shield.mean():.3f}")
    say(f"      {'yil':<6}{'gun':>5}{'ort-pos':>9}{'long%':>7}{'flat-gun':>9}{'trim-gun':>9}")
    for y in q_years:
        m = Q.values & (idx.year == y)
        ty, sy = tdir[m], stack[m]
        say(f"      {y:<6}{int(m.sum()):>5}{sy.mean():>9.3f}{100*(ty==1).mean():>6.0f}%"
            f"{int((ty==0).sum()):>9}{int(((ty==1)&(sy<0.999)).sum()):>9}")

    # ── 3) OPEX CIFTE-SAYIM KONTROLU ──
    say("")
    say("  [3] OPEX-KAPISI ETKILESIMI (NDX OpEx gunu deploy->0; nitelikli Aralik = quad-witch)")
    opex_in_q = []
    for y in q_years:
        for mth in (11, 12):
            tf = pd.Timestamp(third_friday(y, mth))
            if tf in idx and Q.loc[tf]:
                opex_in_q.append((tf.date(), is_quad_witch(tf.date())))
    say(f"      nitelikli pencerelerde OpEx gunleri: {len(opex_in_q)} adet "
        f"({sum(1 for _, qw in opex_in_q if qw)} quad-witch)")
    for d, qw in opex_in_q:
        say(f"        {d}{'  [quad-witch]' if qw else ''}")
    opex_mask = pd.Series(False, index=idx)
    for d, _ in opex_in_q:
        opex_mask.loc[pd.Timestamp(d)] = True

    # ── 4) VARYANTLAR — strict BH-FDR (aile = {SPX,NDX} per varyant) ──
    say("")
    say("  [4] INCREMENTAL over STACK — V1/V2/V3 on-kayitli; V1t/V1g bilgi (post-hoc, verdict-disi)")
    pos_v1 = pd.Series(np.where(Q.values, np.maximum(stack.values, 1.0), stack.values), index=idx)
    pos_v2 = pd.Series(np.where(Q.values, np.minimum(stack.values + 0.25, 1.0), stack.values), index=idx)
    pos_v3 = pd.Series(np.where(NQ.values, stack.values * 0.5, stack.values), index=idx)
    pos_v1t = pd.Series(np.where(Q.values & (tdir.values == 1), 1.0, stack.values), index=idx)
    pos_v1g = {"SPX": pos_v1,
               "NDX": pd.Series(np.where(opex_mask.values, 0.0, pos_v1.values), index=idx)}

    variants = [
        ("V1 boost tam-boy (max(pos,1))", {"SPX": pos_v1, "NDX": pos_v1}, True),
        ("V2 boost +0.25 (cap 1.0)",      {"SPX": pos_v2, "NDX": pos_v2}, True),
        ("V3 ters-yuz (nonQ KasAra x0.5)", {"SPX": pos_v3, "NDX": pos_v3}, True),
        ("V1t trim-remove (yalniz long)",  {"SPX": pos_v1t, "NDX": pos_v1t}, False),
        ("V1g = V1 + NDX OpEx-kapi",       pos_v1g, False),
    ]
    say(f"      {'varyant':<34}{'SPX dSh':>9}{'SPX dDD':>9}{'SPX P':>7}{'NDX dSh':>9}{'NDX dDD':>9}{'NDX P':>7}{'FDR':>6}")
    results = {}
    for label, posd, registered in variants:
        res = {}
        for a in ("SPX", "NDX"):
            v = strat_ret(posd[a], prices[a])
            b = bases[a]
            res[a] = {"dsh": _sh(v) - _sh(b), "ddd": _dd(v) - _dd(b), "p": paired_win_prob(b, v)}
        passed = fdr_bh({a: 1.0 - res[a]["p"] for a in res if res[a]["p"] is not None}, alpha=0.05)
        both = "PASS" if all(passed.get(a, False) for a in ("SPX", "NDX")) else "-"
        tag = both if registered else f"({both})"
        say(f"      {label:<34}{res['SPX']['dsh']:>+9.2f}{100*res['SPX']['ddd']:>+8.1f}p{res['SPX']['p']:>7.0%}"
            f"{res['NDX']['dsh']:>+9.2f}{100*res['NDX']['ddd']:>+8.1f}p{res['NDX']['p']:>7.0%}{tag:>6}")
        results[label] = (res, both, registered)
    # NDX-kendi-YTD robustluk satiri (bilgi): V1'i NDX icin Qn ile kur
    pos_v1n = pd.Series(np.where(Qn.values, np.maximum(stack.values, 1.0), stack.values), index=idx)
    vN = strat_ret(pos_v1n, prices["NDX"])
    pN = paired_win_prob(bases["NDX"], vN)
    say(f"      {'V1n NDX-kendi-YTD (bilgi, NDX)':<34}{'':>9}{'':>9}{'':>7}"
        f"{_sh(vN)-_sh(bases['NDX']):>+9.2f}{100*(_dd(vN)-_dd(bases['NDX'])):>+8.1f}p{pN:>7.0%}{'(-)':>6}")
    say("      (dDD birimi: puan; pozitif = maxDD KOTULESTI. P = paired-bootstrap P(varyant>stack).)")

    # ── 5) PENCERE-ICI PnL dokumu (bilgi) ──
    say("")
    say("  [5] NITELIKLI PENCERE PnL DOKUMU (SPX; stack vs V1 vs buy&hold KasAra)")
    say(f"      {'yil':<6}{'B&H':>8}{'stack':>8}{'V1':>8}")
    for y in q_years:
        m = Q.values & (idx.year == y)
        for a in ("SPX",):
            ret = E.fwd_ret(prices[a], idx).values
            p_b = np.concatenate([np.zeros(1), stack.values[:-1]])
            p_v = np.concatenate([np.zeros(1), pos_v1.values[:-1]])
            bh = float(np.nansum(np.where(m, ret, 0.0)))
            sb = float(np.nansum(np.where(m, p_b * ret, 0.0)))
            sv = float(np.nansum(np.where(m, p_v * ret, 0.0)))
            say(f"      {y:<6}{100*bh:>+7.1f}%{100*sb:>+7.1f}%{100*sv:>+7.1f}%")

    # ── 6) VERDICT yardimi ──
    say("")
    say("  [6] OKUMA KILAVUZU (on-kayitli sozluk)")
    say("      INCREMENTAL-PASS = en az bir on-kayitli varyant {SPX,NDX} ikisinde FDR gecti")
    say("      ABSORBED         = hicbiri gecmedi + ortusme yuksek (stack nitelikli pencerede zaten ~full)")
    say("      MIXED            = kismi gecis / varlik-asimetrik sonuc")
    say("=" * 96)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"  rapor yazildi: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
