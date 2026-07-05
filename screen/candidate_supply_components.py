"""
screen/candidate_supply_components — Constan ARZ-BILESENLERI: on-kayitli icerik testi.

ONCEKI FAZ (candidate_net_supply): TOPLAM net-arz KALICILIK-FAIL / SEKULER-ARTEFAKT
(1984+ icerik ~0, 2005+ isaret ters). BU FAZIN HIPOTEZI: NIYET ayristirmasi sinyali kurtarir —
2020 ihraclari kurtarma-sermayesiydi (sonrasi +27..40%), 2021 ihraclari spekulatif arzdi
(sonrasi -12..-15%); toplam net seri ikisini ayiramiyor.

VERI: data/cache/supply_components.parquet (screen/fetch_supply_components.py; 265 ceyrek x 45 kolon).
PIT (kaynak-basina, panelden): ritter q-sonu+7g | spx_bb q-sonu+90g | sifma yil-sonu+30g |
spac yil-sonu+7g | z1 q-bas+165g. NGDP SAAR 4Q-ORTALAMA ile yilliklastirildi (fetch'te; ders #3).

ON-KAYITLI SINYALLER (sonuclardan ONCE tanimlandi; z10y = 40 ceyrek pencere, min 20 — fetch kurali):
  S1 spec_dollar_z : z10y( ratio_ann[sifma_ipo_true + sifma_secondary + spac_proceeds(0-dolgu)] )
                     dolar-bazli spekulatif-ihrac %NGDP; YILLIK 1990-2013 (z ~1994Q4+). Yillik deger
                     4 ceyrege kopyali -> n 4x sismis (dairesel-roll perm kopya-yapisini korur, yine de
                     YILLIK-TEKILLESTIRILMIS dusuk-guc varyant ayrica raporlanir: n~20, min_shift=3).
                     SPAC bosluk-yillari 0-dolgu (tarihsel hacim kucuk; durust etiket).
  S2 spec_count_z  : z10y_ritter_ipo_n_gross_4q — IPO adedi 4Q-toplam (SPAC/CEF/penny DAHIL brut),
                     1965Q3+ n~242, EN UZUN kapsam = ANA BAYRAK ADAYI; 2000 VE 2021'i gorebilir.
  S3 spec_opco_z   : z10y_ritter_ipo_n_net_4q — yalniz operasyonel-sirket IPO adedi, 1980Q3+.
  S4 spec_nonop_z  : z10y(rolling-4Q ritter_ipo_nonop_n) — brut-net farki (SPAC+CEF+penny KARISIK
                     vekil; AYLIK-zamanli tek SPAC-proxy'si). z burada ayni 40q/min20 kuralla hesaplanir.
  B1 bb_level      : ratio4q_spx_bb_pct — S&P500 geri-alim 4Q-toplam %NGDP (2009Q2+; evren S&P500,
                     TUM-piyasa degil — makro-vekil, durust etiket).
  B2 bb_z          : z10y_ratio4q_spx_bb_pct (2014Q1+, n=44 — DUSUK GUC; on-kayitli beklenti:
                     tek basina anlamlilik/FDR BEKLENMEZ).
  TOPLAM-Z (ref)   : z1_z10y_nfc — onceki fazin sinyali, yalniz KIYAS icin (ana sinav tablosu).

ON-KAYITLI HIPOTEZLER:
  H1 SPEKULATIF-IHRAC: S1/S2/S3/S4 YUKSEK -> forward NEGATIF (cilginlik imzasi) => Spearman NEGATIF.
     Merdiven 63/126/252bd (PIT gozlem noktalarinda) + 5-kova MUTLAK getiri (ders #2 rank!=absolute)
     + donem-stabilite: S2 (1965-1989 / 1990-2004 / 2005+), S1 (1994-2003 / 2004-2013),
     S3/S4 (1980-2004 / 2005+), B1 (2009-2016 / 2017+), B2 (tek pencere — n yetersiz, durust).
  H2 BUYBACK: B1/B2 yuksek -> forward POZITIF mi (buyback-yield tezi) yoksa ~0/absorbe mi?
     Iki yonde raporlanir; isareti veri secer.
  H3 KOMPOZISYON-MAKASI: spread = z(S2) - z(B2), 2014+ ortak pencere (n~44, dusuk guc), PIT =
     max(ritter,spx_bb) = q-sonu+90g; ayni ceyreklerde TOPLAM-Z kiyasi — bilesim toplamdan bilgili mi?
  ANA SINAV (BAYRAK AYRISTIRMA): bayrak = z >= 2.0 (duyarlilik z >= 1.5 ayrica). Uclu kriter:
     (a) 2021 YAKALA: >=1 bayrak-ceyregi referans-yili 2021;
     (b) 2000 YAKALA (kapsam izin veriyorsa): >=1 bayrak referans-ceyregi 1999Q4..2000Q4;
     (c) 2020 ATESLEME: hicbir bayragin PIT'i 2020-03-23..2020-12-31 (kurtarma penceresi) icinde olmasin.
     TOPLAM-Z ayni tabloda; tum bayrak-epizotlari -> ilk-PIT'ten fwd-252bd getiri tablosu + 1999-2002 /
     2020-2022 ceyrek-ceyrek z-degerleri (ayrismanin ciplak gozle gorulmesi icin).
  4) 2019+ INCREMENTAL-over-TIDE (trim-only, canli birikimli-carpan sozlesmesi):
     S2 z>{1.5,2.0} -> x{0.75,0.50} (4 kural) + B2 z<-1 -> x{0.75,0.50} (2 kural, geri-alim-cokmesi);
     strict BH-FDR {SPX,NDX} kural-basina. GUC SINIRI (on-kayitli, durust): 2019+ ~28 ceyreklik gozlem,
     bayrak epizotlari az; spx_bb verisi 2024Q4'te bitiyor (son z ffill = bayat, notlu) ->
     FDR-PASS BEKLENMEZ; karar agirligi cok-on-yillik 1-3 + ana-sinavdadir.

ON-KAYITLI VERDICT KURALLARI (sozluk gorevden):
  FLAG-UPGRADE : >=1 spekulatif bilesen-bayragi (S2/S3/S4; S1 kapsami 2013'te bitiyor -> 2020/21
                 kriterlerini gosteremez, uygunluk disi) z>=2.0'de uclu kriteri GECER
                 ((a)+(c) zorunlu, (b) kapsam varsa) VE TOPLAM-Z ayni kriteri GECEMEZ -> panele girer;
                 pozisyon-sinyali YALNIZ madde-4 FDR-PASS verirse onerilir.
  PANEL-ONLY   : uclu gecilemedi AMA bilesen toplamin gosteremedigi tutarli bilgi katiyor
                 (bayrak-ceyrekleri ort fwd-252 orneklem-tabaninin belirgin altinda VEYA z-formda
                 >=1 donemde stabil anlamlilik) -> bilgi paneli, bayrak/pozisyon yok.
  DEAD         : ikisi de yok — bilesenler de bos.

DERSLER (uygulanan): #1 sahte-trend — HER sinyal icin sinyal~zaman Spearman raporu ZORUNLU;
anlamlilik yalniz trend-arindirilmis (z) + donem-stabil formda sayilir. #2 rank!=absolute —
kova MUTLAK getiriler. #3 SAAR 4Q-ORTALAMA. #4 PIT yayin-gecikmesi kaynak-basina ayri.

Cikti: konsol (ASCII) + output/supply_components_report.txt (utf-8).
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
from screen._util import paired_win_prob, fdr_bh         # noqa: E402

CACHE = ROOT / "data" / "cache"
OUT = ROOT / "output"
HORIZONS = (63, 126, 252)
N_PERM, SEED = 2000, 77
RECOVERY_WIN = (pd.Timestamp("2020-03-23"), pd.Timestamp("2020-12-31"))

REPORT: list[str] = []


def say(line: str = "") -> None:
    print(line)
    REPORT.append(line)


def _sh(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")


def strat_ret(pos, close, lag=1):
    idx = pos.index; ret = E.fwd_ret(close, idx).values; p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def load_long_spx() -> pd.Series:
    df = pd.read_csv(CACHE / "spx_gspc_long.csv", parse_dates=["Date"]).set_index("Date")
    return df["Close"].dropna().sort_index()


def fwd_at(close: pd.Series, dates: pd.DatetimeIndex, h: int) -> pd.Series:
    """Her gozlem tarihinde (as-of) h-islem-gunu forward getiri. Ufuk-disi NaN."""
    pos = close.index.searchsorted(dates, side="right") - 1
    out = np.full(len(dates), np.nan)
    ok = (pos >= 0) & (pos + h < len(close))
    cv = close.values
    out[ok] = cv[pos[ok] + h] / cv[pos[ok]] - 1.0
    return pd.Series(out, index=dates)


def spearman_perm(sig: pd.Series, fwd: pd.Series, min_n: int = 40, min_shift: int = 8
                  ) -> tuple[float, float, int]:
    """Spearman + dairesel-blok-permutasyon p (signal'i >=min_shift gozlem kaydir; otokorelasyon
    VE yillik-kopya yapisi korunur)."""
    df = pd.concat([sig.rename("s"), fwd.rename("f")], axis=1).dropna()
    n = len(df)
    if n < min_n or n <= 2 * min_shift:
        return float("nan"), float("nan"), n
    rho = df["s"].corr(df["f"], method="spearman")
    rng = np.random.default_rng(SEED)
    sv, fv = df["s"].values, df["f"].values
    fr = pd.Series(fv).rank().values
    perm = np.empty(N_PERM)
    for i in range(N_PERM):
        k = int(rng.integers(min_shift, n - min_shift))
        sr = pd.Series(np.roll(sv, k)).rank().values
        perm[i] = np.corrcoef(sr, fr)[0, 1]
    p = float((np.abs(perm) >= abs(rho)).mean())
    return float(rho), p, n


def pit_series(panel: pd.DataFrame, col_vals: pd.Series, pit_col: str) -> pd.Series:
    """Sinyali kendi PIT tarihlerine tasi (ceyrek-degeri -> yayin-gunu gozlemi)."""
    df = pd.DataFrame({"v": col_vals.values, "p": panel[pit_col].values}, index=panel.index).dropna()
    s = pd.Series(df["v"].values, index=pd.DatetimeIndex(df["p"]))
    return s[~s.index.duplicated(keep="last")].sort_index()


def episodes(z: pd.Series, thr: float) -> list[dict]:
    """Ardisik z>=thr ceyrekleri epizot olarak grupla (NaN epizodu keser)."""
    on = (z >= thr) & z.notna()
    eps, cur = [], None
    for d in z.index:
        if bool(on.loc[d]):
            if cur is None:
                cur = {"q0": d, "q1": d, "zmax": float(z.loc[d])}
            else:
                cur["q1"] = d
                cur["zmax"] = max(cur["zmax"], float(z.loc[d]))
        elif cur is not None:
            eps.append(cur); cur = None
    if cur is not None:
        eps.append(cur)
    return eps


def main() -> int:
    panel = pd.read_parquet(CACHE / "supply_components.parquet")
    spx = load_long_spx()

    # ── sinyal insasi (on-kayitli tanimlar) ──
    # S1: dolar-bazli spekulatif-ihrac (ayni-satir NGDP paydasi -> oran-toplami = toplam-orani)
    spec_ratio = (panel["ratio_ann_sifma_ipo_true_pct"] + panel["ratio_ann_sifma_secondary_pct"]
                  + panel["ratio_ann_spac_proceeds_pct"].fillna(0.0))
    z_s1 = (spec_ratio - spec_ratio.rolling(40, min_periods=20).mean()) / spec_ratio.rolling(40, min_periods=20).std()
    # S4: nonop 4Q toplam + z (fetch'in 40q/min20 kuraliyla birebir)
    nonop4 = panel["ritter_ipo_nonop_n"].rolling(4, min_periods=4).sum()
    z_s4 = (nonop4 - nonop4.rolling(40, min_periods=20).mean()) / nonop4.rolling(40, min_periods=20).std()

    SIGS = {
        "S1 spec_dollar_z": dict(z=z_s1, lvl=spec_ratio, pit="pit_date_sifma",
                                 eras=(("1994-2003", 1994, 2003), ("2004-2013", 2004, 2013)), spec=True),
        "S2 spec_count_z": dict(z=panel["z10y_ritter_ipo_n_gross_4q"], lvl=panel["ritter_ipo_n_gross_4q"],
                                pit="pit_date_ritter",
                                eras=(("1965-1989", 1965, 1989), ("1990-2004", 1990, 2004), ("2005+", 2005, 2100)),
                                spec=True),
        "S3 spec_opco_z": dict(z=panel["z10y_ritter_ipo_n_net_4q"], lvl=panel["ritter_ipo_n_net_4q"],
                               pit="pit_date_ritter",
                               eras=(("1980-2004", 1980, 2004), ("2005+", 2005, 2100)), spec=True),
        "S4 spec_nonop_z": dict(z=z_s4, lvl=nonop4, pit="pit_date_ritter",
                                eras=(("1980-2004", 1980, 2004), ("2005+", 2005, 2100)), spec=True),
        "B1 bb_level": dict(z=panel["ratio4q_spx_bb_pct"], lvl=panel["ratio4q_spx_bb_pct"],
                            pit="pit_date_spx_bb",
                            eras=(("2009-2016", 2009, 2016), ("2017+", 2017, 2100)), spec=False),
        "B2 bb_z": dict(z=panel["z10y_ratio4q_spx_bb_pct"], lvl=panel["ratio4q_spx_bb_pct"],
                        pit="pit_date_spx_bb", eras=(("2014+", 2014, 2100),), spec=False),
    }
    TOTAL = dict(z=panel["z1_z10y_nfc"], pit="pit_date_z1")

    say("=" * 104)
    say("  CANDIDATE: ARZ-BILESENLERI (Constan ayristirmasi) — on-kayitli icerik testi (plan docstring'de)")
    say("  onceki faz: TOPLAM net-arz KALICILIK-FAIL; hipotez: NIYET ayristirmasi (spekulatif vs kurtarma")
    say("  vs geri-alim) sinyali kurtarir. Bayrak ana-sinavi: 2000+2021 yakala / 2020 atesleme.")
    say("=" * 104)

    # ── 0) SAHTE-TREND TANISI (ders #1, HER sinyal) ──
    say("")
    say("  0) SINYAL~ZAMAN Spearman (sahte-trend tanisi; |rho|>0.5 = sekuler-trend riski ->")
    say("     anlamlilik yalniz z/detrend + donem-stabil formda sayilir)")
    say(f"  {'sinyal':<22}{'seviye-formu':>14}{'z-formu':>10}   not")
    for name, m in SIGS.items():
        notes = []
        for tag, s in (("lvl", m["lvl"]), ("z", m["z"])):
            sv = s.dropna()
            t = pd.Series(np.arange(len(sv), dtype=float), index=sv.index)
            m[f"rho_t_{tag}"] = float(sv.corr(t, method="spearman")) if len(sv) > 20 else float("nan")
        flag = " <-- trend-riski (seviye)" if abs(m["rho_t_lvl"]) > 0.5 else ""
        zflag = " z-formda da trend!" if abs(m["rho_t_z"]) > 0.5 else ""
        say(f"  {name:<22}{m['rho_t_lvl']:>+14.3f}{m['rho_t_z']:>+10.3f}{flag}{zflag}")
    tz = TOTAL["z"].dropna()
    rt_tot = float(tz.corr(pd.Series(np.arange(len(tz), dtype=float), index=tz.index), method="spearman"))
    say(f"  {'TOPLAM z1_z10y_nfc':<22}{'-':>14}{rt_tot:>+10.3f}   (referans)")

    # ── 1) HORIZON-MERDIVENI (PIT gozlem noktalarinda) ──
    say("")
    say("  1) HORIZON-MERDIVENI — PIT gozleminde forward Spearman (blok-perm p, 2000/seed77)")
    say("     H1 beklentisi: S* NEGATIF; H2: B* isareti veri secer (iki yon raporlanir)")
    say(f"  {'sinyal':<22}{'ufuk':>6}{'Spearman':>10}{'perm-p':>8}{'n':>6}   okuma")
    ladder: dict = {}
    for name, m in SIGS.items():
        sig = pit_series(panel, m["z"], m["pit"])
        m["sig_pit"] = sig
        for h in HORIZONS:
            fwd = fwd_at(spx, sig.index, h)
            rho, p, n = spearman_perm(sig, fwd)
            ladder[(name, h)] = (rho, p, n)
            if not np.isfinite(rho):
                say(f"  {name:<22}{h:>5}g{'-':>10}{'-':>8}{n:>6}   n yetersiz")
                continue
            if m["spec"]:
                read = "hipotez-yonu (spek-yuksek=kotu)" if rho < 0 else "TERS yon"
            else:
                read = "buyback-yield yonu (bb-yuksek=iyi)" if rho > 0 else "ters/absorbe yonu"
            star = " *" if p < 0.05 else ("  " if p >= 0.10 else " .")
            say(f"  {name:<22}{h:>5}g{rho:>+10.3f}{p:>8.3f}{n:>6}{star}  {read}")
    # S1 yillik-tekillestirilmis dusuk-guc varyant
    say("")
    say("  1b) S1 YILLIK-TEKIL varyant (4x-kopya sismesi giderilmis; n kucuk, min_shift=3 yil):")
    s1q = pd.DataFrame({"v": z_s1.values, "p": panel["pit_date_sifma"].values}, index=panel.index).dropna()
    s1y = s1q.groupby(s1q.index.year).first()
    s1y_sig = pd.Series(s1y["v"].values, index=pd.DatetimeIndex(s1y["p"])).sort_index()
    for h in HORIZONS:
        fwd = fwd_at(spx, s1y_sig.index, h)
        rho, p, n = spearman_perm(s1y_sig, fwd, min_n=15, min_shift=3)
        say(f"     {h:>4}g  Spearman {rho:+.3f}  perm-p {p:.3f}  n={n} yil" if np.isfinite(rho)
            else f"     {h:>4}g  n={n} yetersiz")

    # ── 2) 5-KOVA MUTLAK GETIRI (ders #2) ──
    say("")
    say("  2) 5-KOVA MUTLAK-GETIRI — kuintil -> ort forward getiri % (PIT; rank != absolute)")
    say(f"  {'sinyal':<22}{'ufuk':>6}{'Q1(dusuk)':>11}{'Q2':>8}{'Q3':>8}{'Q4':>8}{'Q5(yuksek)':>11}{'isabet Q1/Q5':>14}")
    bucket252: dict = {}
    for name, m in SIGS.items():
        sig = m["sig_pit"]
        hs = HORIZONS if name in ("S2 spec_count_z", "B1 bb_level") else (252,)
        for h in hs:
            fwd = fwd_at(spx, sig.index, h)
            df = pd.concat([sig.rename("s"), fwd.rename("f")], axis=1).dropna()
            if len(df) < 25:
                say(f"  {name:<22}{h:>5}g   n={len(df)} yetersiz")
                continue
            try:
                q = pd.qcut(df["s"], 5, labels=False, duplicates="drop")
            except ValueError:
                continue
            nq = int(q.max()) + 1
            b = [100 * df["f"][q == i].mean() for i in range(nq)]
            while len(b) < 5:
                b.append(float("nan"))
            hit = [100 * (df["f"][q == i] > 0).mean() for i in (0, nq - 1)]
            if h == 252:
                bucket252[name] = b
            say(f"  {name:<22}{h:>5}g{b[0]:>+11.1f}{b[1]:>+8.1f}{b[2]:>+8.1f}{b[3]:>+8.1f}{b[4]:>+11.1f}"
                f"{hit[0]:>7.0f}/{hit[1]:<5.0f}")

    # ── 3) DONEM-STABILITE ──
    say("")
    say("  3) DONEM-STABILITE — Spearman per donem (PIT; 'gercek etki kalici olmali')")
    stab: dict = {}
    for name, m in SIGS.items():
        sig = m["sig_pit"]
        for h in (126, 252):
            fwd = fwd_at(spx, sig.index, h)
            cells = []
            for lab, a, b in m["eras"]:
                mask = (sig.index.year >= a) & (sig.index.year <= b)
                rho, p, n = spearman_perm(sig[mask], fwd[mask], min_n=24)
                stab[(name, h, lab)] = (rho, p, n)
                cells.append(f"{lab} {rho:+.2f} p{p:.2f} n{n}" if np.isfinite(rho) else f"{lab} n{n} az")
            say(f"  {name:<22}{h:>4}g  " + " | ".join(cells))

    # ── 4) ANA SINAV — BAYRAK AYRISTIRMA ──
    say("")
    say("  4) ANA SINAV — BAYRAK AYRISTIRMA: bayrak=z>=2.0 (duyarlilik 1.5)")
    say("     kriter: (a) 2021 yakala  (b) 2000 yakala (kapsam varsa)  (c) 2020 kurtarma-penceresinde atesleme")
    # 4a: epizot listeleri + ilk-PIT fwd-252
    all_sigs = {**{k: v for k, v in SIGS.items()}, "TOPLAM z1_z10y_nfc": dict(z=TOTAL["z"], pit=TOTAL["pit"], spec=True)}
    flag_results: dict = {}
    for thr in (2.0, 1.5):
        say("")
        say(f"  4a) epizotlar (z>={thr:.1f}) — referans-ceyrek araligi, ilk-PIT, ilk-PIT'ten fwd-252bd:")
        for name, m in all_sigs.items():
            if not m.get("spec", True) and name.startswith("B"):
                continue  # bayrak sinavi spekulatif tarafta; B* H2'de test edildi
            z = m["z"]
            pit_map = pd.Series(pd.DatetimeIndex(panel[m["pit"]]), index=panel.index)
            eps = episodes(z, thr)
            catch21 = catch00 = fire20 = False
            cov = z.dropna()
            cov_has_2000 = (len(cov) > 0 and cov.index.min() <= pd.Timestamp("1999-10-01"))
            cov_has_2021 = (len(cov) > 0 and cov.index.max() >= pd.Timestamp("2021-01-01"))
            lines = []
            for ep in eps:
                pit0 = pit_map.loc[ep["q0"]]
                f252 = fwd_at(spx, pd.DatetimeIndex([pit0]), 252).iloc[0]
                yr0, yr1 = ep["q0"].year, ep["q1"].year
                qq = z.loc[ep["q0"]:ep["q1"]]
                # kriter isaretleri
                if any(d.year == 2021 for d in qq.index):
                    catch21 = True
                if any(pd.Timestamp("1999-10-01") <= d <= pd.Timestamp("2000-12-31") for d in qq.index):
                    catch00 = True
                # epizot icindeki HER ceyregin kendi PIT'i kurtarma-penceresinde mi
                for d in qq.index:
                    pd_pit = pit_map.loc[d]
                    if RECOVERY_WIN[0] <= pd_pit <= RECOVERY_WIN[1]:
                        fire20 = True
                fs = f"{100*f252:+.1f}%" if np.isfinite(f252) else "(henuz yok)"
                lines.append(f"      {ep['q0'].year}Q{ep['q0'].quarter}-{ep['q1'].year}Q{ep['q1'].quarter}"
                             f"  zmax {ep['zmax']:+.2f}  ilk-PIT {pit0.date()}  fwd-252 {fs}")
            say(f"    {name}  (epizot n={len(eps)})")
            for ln in lines:
                say(ln)
            if not lines:
                say("      (epizot yok)")
            flag_results[(name, thr)] = dict(catch21=catch21, catch00=catch00, fire20=fire20,
                                             cov00=cov_has_2000, cov21=cov_has_2021, n_ep=len(eps))
    # 4b: uclu-kriter ozet tablosu
    say("")
    say("  4b) UCLU-KRITER OZETI (bayrak z>=2.0 ana / z>=1.5 duyarlilik):")
    say(f"  {'sinyal':<22}{'esik':>6}{'(a)2021':>13}{'(b)2000':>13}{'(c)2020-yok':>13}{'UCLU':>7}")
    triple_pass: dict = {}
    for name in all_sigs:
        if name.startswith("B"):
            continue
        for thr in (2.0, 1.5):
            r = flag_results[(name, thr)]
            a = ("EVET" if r["catch21"] else "hayir") if r["cov21"] else "kapsam-disi"
            b = ("EVET" if r["catch00"] else "hayir") if r["cov00"] else "kapsam-disi"
            c = "ATESLEDI" if r["fire20"] else "temiz"
            ok = (r["cov21"] and r["catch21"]) and (not r["fire20"]) and (r["catch00"] if r["cov00"] else True)
            triple_pass[(name, thr)] = ok
            say(f"  {name:<22}{thr:>6.1f}{a:>13}{b:>13}{c:>13}{'GECTI' if ok else '-':>7}")
    # 4c: 1999-2002 + 2020-2022 ceyrek-ceyrek z (ayrisma ciplak goz)
    say("")
    say("  4c) KRITIK PENCERELER — ceyrek-ceyrek z degerleri (ayrisma tablosu):")
    keyq = [d for d in panel.index if (1999 <= d.year <= 2002) or (2020 <= d.year <= 2022)]
    cols = ["S1 spec_dollar_z", "S2 spec_count_z", "S3 spec_opco_z", "S4 spec_nonop_z", "TOPLAM z1_z10y_nfc"]
    say(f"  {'ceyrek':<10}" + "".join(f"{c.split()[0]:>10}" for c in cols) + f"{'fwd252(q-sonu)':>16}")
    qe_map = panel.index + pd.offsets.QuarterEnd(0)
    fwd_qe = fwd_at(spx, pd.DatetimeIndex(qe_map), 252)
    for d in keyq:
        vals = []
        for c in cols:
            z = all_sigs[c]["z"]
            v = z.loc[d] if d in z.index else float("nan")
            vals.append(f"{v:>+10.2f}" if np.isfinite(v) else f"{'-':>10}")
        i = list(panel.index).index(d)
        fv = fwd_qe.iloc[i]
        fs = f"{100*fv:+.1f}%" if np.isfinite(fv) else "-"
        say(f"  {d.year}Q{d.quarter:<6}" + "".join(vals) + f"{fs:>16}")

    # ── 5) H3 KOMPOZISYON-MAKASI ──
    say("")
    say("  5) H3 KOMPOZISYON-MAKASI — spread = z(S2) - z(B2), 2014+ ortak pencere (PIT q-sonu+90g)")
    zs2, zb2 = SIGS["S2 spec_count_z"]["z"], SIGS["B2 bb_z"]["z"]
    common = panel.index[(zs2.notna() & zb2.notna()).values]
    spread = (zs2 - zb2).loc[common]
    pit_bb = pd.Series(pd.DatetimeIndex(panel["pit_date_spx_bb"]), index=panel.index).loc[common]
    sp_sig = pd.Series(spread.values, index=pd.DatetimeIndex(pit_bb)).sort_index()
    tot_sig = pd.Series(TOTAL["z"].loc[common].values,
                        index=pd.DatetimeIndex(panel.loc[common, "pit_date_z1"])).sort_index()
    say(f"     ortak pencere: {common.min().date()} -> {common.max().date()}  n={len(common)} (DUSUK GUC, on-kayitli)")
    say(f"  {'sinyal':<28}{'ufuk':>6}{'Spearman':>10}{'perm-p':>8}{'n':>5}")
    for nm, s in (("spread z(S2)-z(B2)", sp_sig), ("TOPLAM-Z (ayni ceyrekler)", tot_sig)):
        for h in HORIZONS:
            fwd = fwd_at(spx, s.index, h)
            rho, p, n = spearman_perm(s, fwd, min_n=24)
            say(f"  {nm:<28}{h:>5}g{rho:>+10.3f}{p:>8.3f}{n:>5}" if np.isfinite(rho)
                else f"  {nm:<28}{h:>5}g{'-':>10}{'-':>8}{n:>5}")

    # ── 6) 2019+ INCREMENTAL over TIDE ──
    say("")
    say("  6) INCREMENTAL over TIDE (2019+): bilesen-bayrakli trim, strict BH-FDR {SPX,NDX}")
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    say(f"  base: SPX {_sh(bases['SPX']):+.3f} / NDX {_sh(bases['NDX']):+.3f}")
    zq = {}
    for tag, name in (("S2", "S2 spec_count_z"), ("B2", "B2 bb_z")):
        s = SIGS[name]["sig_pit"]
        zq[tag] = s.reindex(s.index.union(idx)).ffill().reindex(idx)
    n_s2_15 = int((zq["S2"] > 1.5).sum()); n_s2_20 = int((zq["S2"] > 2.0).sum())
    n_b2_lo = int((zq["B2"] < -1.0).sum())
    say(f"  guc-siniri (on-kayitli): S2 z>1.5 gun={n_s2_15}, z>2 gun={n_s2_20}; B2 z<-1 gun={n_b2_lo};")
    say(f"  spx_bb 2024Q4'te bitiyor -> B2 son-z ffill (bayat-kuyruk, durust not). FDR-PASS beklenmez.")
    say(f"  {'kural':<34}{'SPX dSh':>9}{'SPX P':>7}{'NDX dSh':>9}{'NDX P':>7}{'FDR':>6}")
    rules = []
    for thr in (1.5, 2.0):
        for lvl in (0.75, 0.50):
            rules.append((f"S2 z>{thr:.1f} (spek-patlama) -> x{lvl:.2f}", "S2", +1, thr, lvl))
    for lvl in (0.75, 0.50):
        rules.append((f"B2 z<-1.0 (geri-alim-cokme) -> x{lvl:.2f}", "B2", -1, 1.0, lvl))
    fdr_any = False
    for label, tag, sign, thr, lvl in rules:
        fac = pd.Series(np.where(sign * zq[tag] > thr, lvl, 1.0), index=idx)
        res = {}
        for a in ("SPX", "NDX"):
            v = strat_ret((tdir * fac).reindex(idx), prices[a])
            res[a] = (paired_win_prob(bases[a], v), _sh(v) - _sh(bases[a]))
        passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
        both = "PASS" if all(passed.get(a, False) for a in ("SPX", "NDX")) else "-"
        fdr_any = fdr_any or (both == "PASS")
        say(f"  {label:<34}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}{both:>6}")

    # ── 7) BULGULAR-OZETI (ON-KAYIT-DISI, betimsel — sonuclar gorulduktan sonra yazildi; ──
    # ──    uclu-kriter/verdict kurallarina DOKUNMAZ, yalniz mekanizmayi aciklar)        ──
    say("")
    say("  7) BULGULAR-OZETI (on-kayit-DISI, betimsel; verdict-kurallari degistirilmedi):")
    zS2, zS3, zS4 = SIGS["S2 spec_count_z"]["z"], SIGS["S3 spec_opco_z"]["z"], SIGS["S4 spec_nonop_z"]["z"]
    w00 = (panel.index >= "1999-10-01") & (panel.index <= "2000-12-31")
    say(f"  7a) 2000'i SAYIM kacirdi: dot-com penceresinde max z — S2 {zS2[w00].max():+.2f} /"
        f" S3 {zS3[w00].max():+.2f} / S4 {zS4[w00].max():+.2f} (hicbiri >=1.5):")
    say("      90'larin tamami sicak IPO on-yiliydi -> 10y taban yuksek, 2000 SAYIM-bazda sira-disi degil.")
    say("      Cilginlik DOLARDA ve fiyatlamadaydi: S1 (dolar) 2000'de z+1.95'e cikti (1.5-bayrak yakaladi,")
    say("      fwd-252 -20.7%) ama S1 kapsami 2013'te bitiyor (SIFMA form-kapisi) -> canli kullanilamaz.")
    say("  7b) S1 dolar-z'si de NIYET-KOR: 2009Q1-Q4'te de bayrak verdi (kurtarma-ihraclari, fwd +18.9%)")
    say("      -> 'yuksek dolar-ihraci = cilginlik' imzasi 2009'da da yaniliyor; ayristirma hipotezinin")
    say("      ozu (niyet) hicbir tek-degiskenli arz-olcusunde yok.")
    say(f"  7c) TEK gercek ayrisma: S3 (yalniz operasyonel-sirket IPO adedi) 2020 kurtarma-penceresinde")
    say(f"      TEMIZ kaldi (2020Q2 {zS3.get(pd.Timestamp('2020-04-01'), float('nan')):+.2f},"
        f" 2020Q3 {zS3.get(pd.Timestamp('2020-07-01'), float('nan')):+.2f}) ve 2021'de atesledi"
        f" (2021Q2 {zS3.get(pd.Timestamp('2021-04-01'), float('nan')):+.2f});")
    say("      TOPLAM-Z ise 2020Q3'te atesledi (PIT 2020-12, fwd +27%) = tam hipotezin ongordugu hata.")
    say("      AMA S3 2000'i kaciriyor (7a) -> uclu-kriter yine FAIL; S3'un 2005+ 252g hucresi (-0.37 p0.00)")
    say("      tek-donem — on-kayitli 'donem-stabil' standardini TEK BASINA karsilamaz.")
    say("  7d) Incremental trimler 2019+ HEP negatif: S2/S4 bayragi Eki-2020'de cok erken geliyor,")
    say("      2021 melt-up'ini kesiyor; 2022 ayisindan tasarruf bunu telafi etmiyor.")

    # ── VERDICT (on-kayitli kurallar, docstring) ──
    say("")
    say("  " + "-" * 100)
    say("  VERDICT (on-kayitli kurallara gore, durust ust-satir):")
    eligible = ("S2 spec_count_z", "S3 spec_opco_z", "S4 spec_nonop_z")
    comp_pass = [n for n in eligible if triple_pass.get((n, 2.0), False)]
    total_pass = triple_pass.get(("TOPLAM z1_z10y_nfc", 2.0), False)
    say(f"  [ana-sinav z>=2.0] bilesen-gecen: {comp_pass if comp_pass else 'YOK'};"
        f" TOPLAM-Z gecti mi: {'EVET' if total_pass else 'HAYIR'}")
    comp_pass15 = [n for n in eligible if triple_pass.get((n, 1.5), False)]
    total_pass15 = triple_pass.get(("TOPLAM z1_z10y_nfc", 1.5), False)
    say(f"  [duyarlilik z>=1.5] bilesen-gecen: {comp_pass15 if comp_pass15 else 'YOK'};"
        f" TOPLAM-Z: {'EVET' if total_pass15 else 'HAYIR'}")
    flag_upgrade = bool(comp_pass) and not total_pass
    # PANEL-ONLY ikincil kriteri: z-formda >=1 donem-stabil anlamlilik (hipotez yonunde) VEYA
    # spek-bayrak ceyrekleri ort fwd-252 orneklem-tabaninin altinda (ana sinyal S2)
    stab_sig = [(k, v) for k, v in stab.items()
                if np.isfinite(v[0]) and v[1] < 0.05 and (v[0] < 0 if k[0].startswith("S") else True)]
    s2sig = SIGS["S2 spec_count_z"]["sig_pit"]
    f252_s2 = fwd_at(spx, s2sig.index, 252)
    base_mean = float(f252_s2.mean())
    flag_mean = float(f252_s2[s2sig >= 2.0].mean()) if (s2sig >= 2.0).any() else float("nan")
    say(f"  [panel-bilgi] S2-bayrak(z>=2) ceyrekleri ort fwd-252 {100*flag_mean:+.1f}% vs orneklem-taban"
        f" {100*base_mean:+.1f}%; donem-stabil z-anlamlilik hucresi n={len(stab_sig)}")
    if flag_upgrade:
        verdict = "FLAG-UPGRADE"
        say(f"  SONUC: FLAG-UPGRADE — {comp_pass} uclu-kriteri gecti, TOPLAM-Z gecemedi -> bilesen-bayragi")
        say("         panele girer. POZISYON-SINYALI: " + ("FDR-PASS var -> onerilir." if fdr_any
            else "FDR-PASS YOK (on-kayitli dusuk guc) -> yalniz panel-bayragi, pozisyon kablosu yok."))
    elif (np.isfinite(flag_mean) and flag_mean < base_mean - 0.02) or stab_sig:
        verdict = "PANEL-ONLY"
        say("  SONUC: PANEL-ONLY — hicbir bilesen-bayragi uclu kriteri gecemedi (2000'i sayim kaciriyor,")
        say("         2020'yi S2/S4/TOPLAM yanlis-atesliyor; tek temiz 2020/21-ayristirici S3 de 2000-FAIL).")
        say("         Ama bilesen tarafinda toplamin gosteremedigi bilgi var: S2-bayrak ceyrekleri taban-alti")
        say("         fwd + S3'un 2020-temiz/2021-yakala ayrismasi + S3 2005+ hucresi. Bu bilgi PANELDE")
        say("         (betimsel, mekanizma-iddiasiz) tutulur; bayrak/pozisyon kablosu ONERILMEZ (FDR yok,")
        say("         incremental negatif).")
    else:
        verdict = "DEAD"
        say("  SONUC: DEAD — bilesenler de bos: ne bayrak-ustunlugu ne panel-bilgisi. Kablolama yok.")
    say(f"  (makine-okur: VERDICT={verdict})")
    say("=" * 104)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "supply_components_report.txt").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    print(f"  rapor -> {OUT / 'supply_components_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
