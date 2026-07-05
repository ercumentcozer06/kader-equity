"""
screen/fetch_supply_demand_balance — K1: Constan ARZ-TALEP DENGE gostergesi (tam panel, BETIMSEL takip).

Emir mandasi: "model toplam hisse ARZINI VE TALEBINI takip etsin", %NGDP normalize (nominal yaniltir).
Bu bir EDGE-CANDIDATE degil; BETIMSEL bir denge-kadrandir (Constan ikizi). Net-arz YON-sinyali tek
basina FAIL etti ([[net_supply]]: sahte-trend; 1984+ ~0; 2005+ ters). Buradaki katma-deger arz-tarafini
TALEP-tarafiyla KARSILASTIRMAK: ayni arz-patlamasi talep guclu ise NOTR, talep zayif ise BEARISH.

DURUSTLUK DUZELTMESI (look-ahead onarimi, 2026-06):
  Bir onceki surum 3 bacagi TAM-ORNEKLEM z (s.mean()/s.std() tum 1960-2026) ile normalize ediyordu
  (SPAC, NGDP-buyume, tide) -> her tarihsel okuma GELECEGI biliyordu. Sizdiran 3 bacaktan 2'si TALEP
  tarafindaydi. Onarim: SPAC -> hazir TRAILING z10y kolonu; NGDP -> genisleyen-pencere z (mp=8);
  tide -> genisleyen-z (mp=4, 2019+ kisa pencere).
  SONUC (kritik): tam-orneklem altinda 2020 vs 2021 'ayraci' +0.59'du; PIT-DURUST altinda -0.08'e
  COKER (fark isaret degistirir) -> bu Z-PANELI 2020'yi 2021'den AYIRT ETMEZ. Ustelik HER IKI yil da
  GUCLU-BEARISH ('ARZ TALEBI EZIYOR') okunur; 2020 hicbir zaman kullanicinin istedigi NOTR/BULLISH
  tarafa dusmez. Gercek 2020-sessizlik ayraci bu z-ortalamasinda DEGIL, K2'nin HAM tide-DUZEY kapisinda
  (tide_score<=+2, dususte) yasar (K2: 2020 +11.7/+8.8/+5.9 = kapinin USTUNDE -> SUSAR; 2021 H2
  +1.3/+1.1/-0.5 = kapinin ALTINDA -> ATESLER). K1 paneli BETIMSEL kalir, ayrac iddiasi K2'ye birakilir.

VERI (hazir parquet'lerden birlestirme; HEPSI %NGDP normalize, mumkun oldugunda z10y trend-arindirilmis):
  data/cache/supply_components.parquet  -> NFC net-arz (z1_z10y_nfc), buyback %NGDP (z10y_ratio4q_spx_bb_pct,
        ratio4q_spx_bb_pct), SIFMA toplam ihrac %NGDP (z10y_ratio_ann_sifma_total_pct),
        SPAC proceeds %NGDP (ratio_ann_spac_proceeds_pct), NGDP (ngdp_saar_bn). MASTER dosya.
  data/cache/ipo_pipeline.parquet       -> S-1/F-1 boru-hatti baskisi (z10y; %NGDP'ye cevrilemez -> z).
  data/cache/net_equity_supply.parquet  -> ratio4q_total (NFC+fin) referans bacagi.
  spine frozen tide (2019+)              -> likidite->risk-istahi TALEP yardimci proxy'si (2020/2021 ayraci).

DENGE TASARIMI:
  ARZ-z   = ortalama( z10y_nfc ; z10y_sifma_total ; z(spac_pct) ; ipo_pipeline_z ; -z10y_buyback )
            (buyback NEGATIF arzdir: buyback artisi = net-arz DUSER -> arz-tarafinda isaret TERS)
  TALEP-z = ortalama( z(ngdp_yoy) ; +z10y_buyback ; z(tide_q) )
            (ANA talep = NGDP-buyumesi + buyback'in sirket-talebi; tide = likidite yardimcisi 2019+)
  net_supply_pressure = ARZ-z - TALEP-z
     YUKSEK  -> arz talebi eziyor -> BEARISH baski
     DUSUK   -> talep arzi eziyor -> BULLISH

Cikti: data/cache/supply_demand_balance.parquet (tam ceyreklik seri, tum bilesenler + denge)
       output/supply_demand_panel.txt (Constan ikizi: son 12 ceyrek arz/talep/denge + bugun + yon + 4 tarihsel uc)
Kosu: repo kokunden -> python -X utf8 screen/fetch_supply_demand_balance.py
Konsol cp1254 -> print ASCII-ONLY; dosyaya utf-8 serbest.
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

CACHE = ROOT / "data" / "cache"
OUT = ROOT / "output"

REPORT: list[str] = []


def say(line: str = "") -> None:
    print(line)
    REPORT.append(line)


def zexpand(s: pd.Series, min_periods: int = 8) -> pd.Series:
    """PIT-DURUST genisleyen-pencere z-skor: her ceyrek YALNIZ o ana kadarki gecmisi gorur
    (tam-orneklem zlevel'in gelecek-sizdirmasini onler). min_periods kadar gozlem birikene
    dek NaN. Bos/sabit -> NaN.

    NOT (look-ahead onarimi): eski zlevel() tum 1960-2026 seri ortalama/std'sini kullaniyordu;
    tarihsel her okuma GELECEGI biliyordu (SPAC z 2021Q1 = +4.87 tam-orneklem vs +6.75 genisleyen).
    Sizan 3 bacaktan 2'si TALEP tarafindaydi -> 2020/2021 ayraci kontamine. Genisleyen-z bunu kapatir.
    """
    s = s.astype(float)
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std()
    z = (s - mu) / sd
    return z.where(sd > 0)


def row_mean(df: pd.DataFrame) -> pd.Series:
    """Satir bazinda mevcut bacaklarin ortalamasi (en az 1 bacak varsa deger; hepsi NaN -> NaN)."""
    return df.mean(axis=1, skipna=True)


def load_tide_quarterly() -> pd.Series:
    """Spine donmus tide skoru -> ceyreklik ortalama (likidite->risk-istahi talep proxy'si, 2019+)."""
    try:
        from spine import contract as C, tide as T
        scores, prices, vector, prov = C.read_frozen()
        ts = T.tide_score_series(scores, vector)
        tq = ts.resample("QS").mean()
        tq.index = pd.to_datetime(tq.index)
        return tq.rename("tide_q")
    except Exception as e:  # tide opsiyonel yardimci; yoksa talep NGDP+buyback ile devam
        say(f"  [uyari] tide proxy okunamadi ({type(e).__name__}); talep NGDP+buyback ile kurulur.")
        return pd.Series(dtype=float, name="tide_q")


def build() -> pd.DataFrame:
    sc = pd.read_parquet(CACHE / "supply_components.parquet")
    ipo = pd.read_parquet(CACHE / "ipo_pipeline.parquet")
    nes = pd.read_parquet(CACHE / "net_equity_supply.parquet")

    df = pd.DataFrame(index=sc.index.copy())
    df.index.name = "qdate"

    # ---- HAM %NGDP bacaklari (betimsel panelde gosterilir) ----
    df["ngdp_saar_bn"] = sc["ngdp_saar_bn"]
    df["ngdp_yoy_pct"] = sc["ngdp_saar_bn"].pct_change(4) * 100.0
    df["nfc_net_pct"] = sc["z1_ratio4q_nfc_pct"]          # NFC net-arz %NGDP (seviye)
    df["buyback_pct"] = sc["ratio4q_spx_bb_pct"]          # SPX buyback %NGDP (seviye)
    df["sifma_total_pct"] = sc["ratio_ann_sifma_total_pct"]
    df["sifma_ipo_pct"] = sc["ratio_ann_sifma_ipo_true_pct"]
    df["sifma_secondary_pct"] = sc["ratio_ann_sifma_secondary_pct"]
    df["spac_proceeds_pct"] = sc["ratio_ann_spac_proceeds_pct"]
    df["total_net_pct"] = nes["ratio4q_total_pct"].reindex(df.index)   # NFC+fin referans
    df["ipo_pipeline_z"] = ipo["z10y"].reindex(df.index)              # S-1/F-1 boru-hatti z (zaten 10y-z)

    tide_q = load_tide_quarterly()
    df["tide_q"] = tide_q.reindex(df.index)

    # ---- z-skor bacaklari (trend-arindirilmis z10y MEVCUTSA o kullanilir; degilse tam-orneklem z) ----
    # ARZ bacaklari (yuksek z = arz baskisi YUKSEK):
    sup = pd.DataFrame(index=df.index)
    sup["nfc_z"] = sc["z1_z10y_nfc"]                                  # NFC net-ihrac 10y-z (PIT-trailing)
    sup["sifma_total_z"] = sc["z10y_ratio_ann_sifma_total_pct"]       # SIFMA toplam ihrac 10y-z (1994-2013, PIT-trailing)
    sup["spac_z"] = sc["z10y_ratio_ann_spac_proceeds_pct"]            # SPAC %NGDP TRAILING z10y (PIT; eski tam-orneklem zlevel sizdiriyordu)
    sup["ipo_pipe_z"] = df["ipo_pipeline_z"]                          # S-1 boru-hatti z (2001+, PIT-trailing)
    sup["neg_buyback_z"] = -sc["z10y_ratio4q_spx_bb_pct"]             # buyback NEGATIF arz (isaret TERS, PIT-trailing)

    # TALEP bacaklari (yuksek z = talep YUKSEK):
    dem = pd.DataFrame(index=df.index)
    dem["ngdp_growth_z"] = zexpand(df["ngdp_yoy_pct"], min_periods=8)  # NGDP buyume PIT genisleyen-z (1961+; eski tam-orneklem sizdiriyordu)
    dem["buyback_z"] = sc["z10y_ratio4q_spx_bb_pct"]                  # buyback = sirketin kendi-hisse talebi (+, PIT-trailing)
    # tide YALNIZ 2019+ (30 ceyrek) -> z-normalize PIT-DURUST OLAMAZ (kisa pencere). Genisleyen-z mp=4
    # ile tutariz ki tam-orneklem sizmasin; AMA 2020 H1 NaN kalir (yetersiz gecmis). Bu yapisal kisit
    # tam da neden bu Z-PANELI 2020/2021'i ayirt EDEMEDIGINI gosterir; gercek ayrac K2'nin HAM tide-DUZEY
    # kapisidir (tide_score<=+2), z-ortalamasi degil. Asagidaki AYRAC NOTU bunu acikca yazar.
    dem["tide_z"] = zexpand(df["tide_q"], min_periods=4)              # likidite->risk-istahi (2019+ yardimci, PIT genisleyen-z)

    # bilesik z = mevcut bacaklarin satir-ortalamasi
    df["supply_z"] = row_mean(sup)
    df["demand_z"] = row_mean(dem)
    # bacak sayilari (panelde seffaflik)
    df["supply_n_legs"] = sup.notna().sum(axis=1)
    df["demand_n_legs"] = dem.notna().sum(axis=1)

    # ---- DENGE: net arz-talep baskisi ----
    df["net_supply_pressure"] = df["supply_z"] - df["demand_z"]

    # tekil z-bacaklari da sakla (denetim / panel)
    for c in sup.columns:
        df["sup_" + c] = sup[c]
    for c in dem.columns:
        df["dem_" + c] = dem[c]

    return df


def fmt(x, w=8, p=2, plus=True) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return f"{'--':>{w}}"
    return f"{x:>+{w}.{p}f}" if plus else f"{x:>{w}.{p}f}"


def direction_word(p: float) -> str:
    """net_supply_pressure -> betimsel yon cumlesi (z-birimi)."""
    if not np.isfinite(p):
        return "veri yetersiz"
    if p >= 1.0:
        return "ARZ TALEBI EZIYOR (guclu bearish baski)"
    if p >= 0.35:
        return "arz hafif onde (bearish-egilim)"
    if p > -0.35:
        return "DENGE (arz~talep, notr)"
    if p > -1.0:
        return "talep hafif onde (bullish-egilim)"
    return "TALEP ARZI EZIYOR (guclu bullish)"


def main() -> int:
    df = build()
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE / "supply_demand_balance.parquet")

    say("=" * 100)
    say("  K1 ARZ-TALEP DENGE GOSTERGESI (Constan ikizi, BETIMSEL) — %NGDP normalize, z10y trend-arindirilmis")
    say(f"  kapsam: {df.index.min().date()} -> {df.index.max().date()}  ({len(df)} ceyrek)")
    say("  ARZ-z   = ort( nfc_net ; sifma_total ; spac ; ipo_pipeline ; -buyback )  [yuksek=arz baskisi]")
    say("  TALEP-z = ort( ngdp_buyume ; +buyback ; tide )                            [yuksek=talep gucu]")
    say("  net_supply_pressure = ARZ-z - TALEP-z   (YUKSEK=bearish / DUSUK=bullish)")
    say("=" * 100)

    # ---- TARIHSEL UCLAR: ceyreklik denge degerlerini ACIKCA raporla ----
    say("")
    say("  TARIHSEL UCLAR (ceyreklik net_supply_pressure; 2020'yi 2021'den ayirt edebiliyor mu?)")
    say(f"  {'ceyrek':<12}{'ARZ-z':>9}{'TALEP-z':>9}{'DENGE':>9}{'arz-bacak':>11}{'talep-bacak':>12}   okuma")
    marks = {
        "2000-dot-com (arz)": ("2000-01-01", "2000-12-31"),
        "2009-kurtarma-ihrac": ("2009-01-01", "2009-12-31"),
        "2020-COVID (arz+talep-guclu)": ("2020-01-01", "2020-12-31"),
        "2021-SPAC-froth (arz+talep-zayif)": ("2021-01-01", "2021-12-31"),
    }
    extreme_vals: dict[str, dict] = {}
    for label, (a, b) in marks.items():
        sub = df.loc[a:b]
        if sub.empty:
            continue
        say(f"  -- {label}")
        # yil ortalamasi (panel ozeti) + ceyrek-ceyrek
        ymean = sub[["supply_z", "demand_z", "net_supply_pressure"]].mean()
        extreme_vals[label] = {
            "supply_z": round(float(ymean["supply_z"]), 3),
            "demand_z": round(float(ymean["demand_z"]), 3),
            "balance": round(float(ymean["net_supply_pressure"]), 3),
        }
        for d, r in sub.iterrows():
            say(f"  {str(d.date()):<12}{fmt(r['supply_z'],9)}{fmt(r['demand_z'],9)}"
                f"{fmt(r['net_supply_pressure'],9)}{int(r['supply_n_legs']):>11}{int(r['demand_n_legs']):>12}"
                f"   {direction_word(r['net_supply_pressure'])}")
        say(f"  {'  >> YIL ORT':<12}{fmt(ymean['supply_z'],9)}{fmt(ymean['demand_z'],9)}"
            f"{fmt(ymean['net_supply_pressure'],9)}{'':>11}{'':>12}   {direction_word(float(ymean['net_supply_pressure']))}")
        say("")

    # 2020 vs 2021 ayrac testi (mansetteki kilit soru)
    if "2020-COVID (arz+talep-guclu)" in extreme_vals and "2021-SPAC-froth (arz+talep-zayif)" in extreme_vals:
        b20 = extreme_vals["2020-COVID (arz+talep-guclu)"]["balance"]
        b21 = extreme_vals["2021-SPAC-froth (arz+talep-zayif)"]["balance"]
        say(f"  >>> AYRAC TESTI: 2020 denge {b20:+.2f} vs 2021 denge {b21:+.2f}  "
            f"(fark {b21 - b20:+.2f})")
        verdict = ("AYIRT EDIYOR: 2021 dengesi 2020'den daha BEARISH (arz-froth + talep-soguma)"
                   if b21 > b20 + 0.25 else
                   "ZAYIF AYRIM: iki yil dengesi yakin -> bu Z-PANELI 2020/2021'i AYIRT ETMEZ")
        say(f"      {verdict}")
        say("      DURUST OKUMA (PIT-onarim sonrasi): iki yil da GUCLU-BEARISH ('ARZ TALEBI EZIYOR') okunur;")
        say("      2020 hicbir zaman NOTR/BULLISH degildir. Bacak-ayristirmasi 'mekanizma'yi DUZELTIR:")
        say("        - TALEP tarafi 2020->2021 ASLINDA YUKSELDI (cogu genisleyen-NGDP-z taban-etkisiyle);")
        say("          tide bacagi DUSTU ama TALEP-toplamini surukleyen o degil. 'Talep-soguma' yanlistir.")
        say("        - Bearish kayma TAMAMEN ARZ tarafindan: SPAC + IPO-boru-hatti z patlamasi.")
        say("      Yani arz-froth GERCEK; ama bu Z-PANELI 2020-rali ile 2021-tepe arasinda yon AYIRT ETMEZ.")
        say("      Gercek 2020-sessizlik ayraci = K2'nin HAM tide-DUZEY kapisi (tide_score<=+2 VE dususte),")
        say("      z-ortalamasi DEGIL: K2'de 2020 +11.7/+8.8/+5.9 (kapi USTU -> susar), 2021H2 +1.3/+1.1/-0.5")
        say("      (kapi ALTI -> atesler). Bu panel BETIMSEL kadrandir; ayrac iddiasi K2'ye aittir.")
    say("")
    say("  DURUST KISIT (pre-2019 talep bacagi): likidite-tide YALNIZ 2019+ mevcut; oncesinde talep =")
    say("  NGDP-buyume + buyback. 2009'da NGDP yoy ~ -3% (cagdas/gecikmeli olcum) talep-z'yi asiri-negatife")
    say("  cekti -> denge en-bearish. Oysa 2009 ileri-getirisi GUCLU idi (dip). Yani pre-2019 derin-resesyon")
    say("  ceyrekleri talep-tarafinda YAPISAL OLARAK asiri-bearish okunur; bu gostergenin bilinen kisitidir")
    say("  (cagdas NGDP ileri-toparlanmayi gormez, tide-proxy o donemde yok).")
    say("  DUZELTME: onceki surum '2020 vs 2021 ayrimi tide sayesinde saglamdir' diyordu -> YANLIS. PIT-")
    say("  durust normalize altinda ayrac +0.59'dan -0.08'e coker (her iki yil da guclu-bearish, fark yon")
    say("  bile degistirir). Bu Z-PANELI yon AYIRT ETMEZ; ayrac K2'nin ham tide-DUZEY kapisindadir.")
    say("  2000/2009/2020/2021 uclarinda denge-SEVIYESI betimseldir, yon-sinyali olarak okunmamalidir.")

    # ---- CONSTAN PANELI: son 12 ceyrek ----
    say("")
    say("  " + "-" * 96)
    say("  CONSTAN PANELI — SON 12 CEYREK (arz / talep / denge, betimsel)")
    say(f"  {'ceyrek':<12}{'NGDP%yoy':>10}{'buyback%':>10}{'IPO-boru-z':>11}{'ARZ-z':>9}{'TALEP-z':>9}{'DENGE':>9}   okuma")
    tail = df.tail(12)
    for d, r in tail.iterrows():
        say(f"  {str(d.date()):<12}{fmt(r['ngdp_yoy_pct'],10,1)}{fmt(r['buyback_pct'],10,2,plus=False)}"
            f"{fmt(r['ipo_pipeline_z'],11)}{fmt(r['supply_z'],9)}{fmt(r['demand_z'],9)}"
            f"{fmt(r['net_supply_pressure'],9)}   {direction_word(r['net_supply_pressure'])}")

    # ---- BUGUNKU OKUMA + YON CUMLESI ----
    last = df.dropna(subset=["net_supply_pressure"]).iloc[-1]
    last_d = df.dropna(subset=["net_supply_pressure"]).index[-1]
    say("")
    say("  " + "-" * 96)
    say(f"  BUGUNKU OKUMA (son tam ceyrek {last_d.date()}):")
    say(f"    ARZ-z   {last['supply_z']:+.2f}  ({int(last['supply_n_legs'])} bacak)"
        f"    TALEP-z {last['demand_z']:+.2f}  ({int(last['demand_n_legs'])} bacak)")
    say(f"    DENGE (net_supply_pressure) = {last['net_supply_pressure']:+.2f} z-birimi")
    say(f"    YON: {direction_word(float(last['net_supply_pressure']))}")
    # son ceyrek bilesenleri betimsel cumle
    sup_hi = "yuksek" if last["supply_z"] > 0.5 else ("dusuk" if last["supply_z"] < -0.5 else "norm")
    dem_hi = "guclu" if last["demand_z"] > 0.5 else ("zayif" if last["demand_z"] < -0.5 else "norm")
    bb = last["buyback_pct"]
    bb_str = f"%{bb:.2f}NGDP" if np.isfinite(bb) else "(buyback verisi gecikmeli, son tam ceyrek 2025Q3)"
    say(f"    OZET: arz-tarafi {sup_hi} (NFC-net %{last['nfc_net_pct']:+.2f}NGDP, IPO-boru z {last['ipo_pipeline_z']:+.2f}), "
        f"talep-tarafi {dem_hi} (NGDP %{last['ngdp_yoy_pct']:.1f}yoy, buyback {bb_str}).")
    say("")
    say("  NOT: bu gosterge BETIMSEL bir denge-kadrandir (Constan ikizi), edge/yon-sinyali DEGILDIR.")
    say("       net-arz tek-basina FAIL etti; katma-deger arzi TALEP-baglamiyla okumaktir.")
    say("=" * 100)

    (OUT / "supply_demand_panel.txt").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    print(f"  parquet -> {CACHE / 'supply_demand_balance.parquet'}")
    print(f"  panel   -> {OUT / 'supply_demand_panel.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
