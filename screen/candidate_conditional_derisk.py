"""
screen/candidate_conditional_derisk — K2: KOSULLU forward de-risk (arz-asiri VE talep-zayif AYNI ANDA).

ARKA PLAN / NEDEN KOSUL:
  TEK-ARZ yon-sinyali backtest'te FAIL'di (candidate_net_supply: sahte-trend/sekuler-artefakt;
  candidate_supply_components: bilesen-ayristirmasi da NIYET-KOR). En kritik somut hata: arz-z
  2020 H2'de yuksekti (kurtarma-ihraclari) AMA o pencerede forward +%8..+%27'lik rali vardi ->
  arz-tek trim 2020'de ATESLER ve raliyi keser. BU FAZIN HIPOTEZI: arz-yuksekligini TALEP-zayifligi
  ile KOSULLAMAK 2020'yi susturur (talep guclu = rali) ama 2021 tepesini (arz-yuksek + talep-cozulur)
  yakalar. Bu bir ALFA iddiasi DEGIL; rejim-degisim SIGORTASIDIR (kucuk, trim-only, rebound-safe).

ON-KAYITLI KURAL (sonuclardan ONCE; parametreler SABIT — asagidaki sabitlerde dondurulmus):
  ATESLEME (ikisi BIRDEN):
    (1) ARZ-ASIRI : z(net-arz %NGDP) = z10y_nfc (net_equity_supply.parquet) >= SUPPLY_Z_THR (=+1.0),
                    PIT pub-lag +165g parquet'te (zaten); ceyreklik -> gunluk ffill (canli gozlem).
    (2) TALEP-ZAYIF: donmus tide-stack'ten (spine sozlesmesi) tureyen, ASAGIDAKILERDEN BIRI:
          (a) tide_dir == 0 (likidite FLAT — spine'in kendi yon-sinyali soguk), VEYA
          (b) tide_score <= DEMAND_WEAK_LEVEL (=+2.0) VE tide_score son DECLINE_LB (=63) islem-gununde
              DUSUSTE (tide_score - tide_score.shift(63) < 0).
        Mantik: 2020 H2'de tide_score +5.9..+11.7 (GUCLU, level-kapisi tutar -> SUSAR); 2021 H2'de
        tide_score +1.1..+1.4'e coker (zayif VE dususte -> ATESLER). Salt 'dususte' yetmez (2020'de
        de mean-reversion var) -> DUZEY kapisi (<=+2) ayrimi yapan unsurdur.
        Ucuncu (opsiyonel) bacak NGDP-buyume-yavaslamasi RAPORLANIR ama KULLANILMAZ: 2020 taban-etkisi
        2021 YoY'yi yapay POZITIF/guclu gosterir (kontamine proxy) -> dahil edilmez (durust not).
  AKSIYON: nitelikli (fire) gunlerde SPX/NDX deploy x TRIM (=0.85). Trim-only, rebound-safe; canli
           birikimli-carpan sozlesmesiyle uyumlu (tide_dir x faktor; faktor<=1).
  TEST PENCERESI: donmus tide-stack'in oldugu 2019+ (FROZEN snapshot).

ZORUNLU DOGRULAMALAR (rapora acik):
  1) KRITIK — 2020 SESSIZLIGI: 2020 H2'de (arz-yuksek AMA talep-guclu) kural ATESLEMEMELI. Ceyrek-ceyrek
     PIT-asof tablo (arz-z, tide_score, dir, fire?). Gercek-guclu-talep ceyrekleri (Haz/Eyl/Ara PIT,
     tide +5.9..+11.7, dir=1) SUSAR mi? (Pre-secim Eki26-Kas3 mini-cozulmesi gunlerinde tide_dir
     KENDISI 0'a dustu -> o gunler ayrica raporlanir; bunlar 'guclu-talep raliyi kesme' degil,
     spine'in kendi FLAT-gunleridir.) Eger GERCEK guclu-talep ceyreklerinde atesliyorsa -> NO-WIRE.
  2) 2021 ATESLEME: 2021 tepesinde (arz-yuksek + talep-zayiflar) atesler mi; o gunlerin forward
     getirisi NEGATIF mi (trim hakli mi)?
  3) INCREMENTAL over TIDE (2019+): base = tide_dir; variant = tide_dir x trim-faktor.
     dSharpe / dMaxDD / paired-win-prob (block-boot) / BH-FDR {SPX,NDX}; fire-gun sayisi + tarihleri.
     n DURUST: forward-only dogasi geregi AZ; FDR-PASS muhtemelen beklenmez (acikca yazilir).

VERDICT (on-kayitli):
  WIRE-FORWARD ANCAK (a) 2020-sessiz (gercek-guclu-talep ceyreklerinde fire YOK) VE (b) 2021-atesler
    (negatif-forward pencerede) VE (c) 2019+ net-zararsiz (dSharpe >= -0.03 her iki varlikta) ise.
  Aksi NO-WIRE + neden. Bu bir alfa degil rejim-degisim-sigortasidir; backtest-destegi forward-only/n-az
    (durustce yazilir).

Cikti: konsol (ASCII-only) + output/conditional_derisk_report.txt (utf-8).
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

# ── ON-KAYITLI SABITLER (sonuclardan ONCE dondurulmus; oynama YOK) ──
SUPPLY_Z_THR = 1.0          # arz-asiri esigi (z10y_nfc >= +1.0)
DEMAND_WEAK_LEVEL = 2.0     # talep-zayif duzey-kapisi (tide_score <= +2.0)
DECLINE_LB = 63             # talep-dusus geriye-bakis (islem-gunu)
TRIM = 0.85                 # nitelikli gunde deploy carpani (trim-only)
DSHARPE_FLOOR = -0.03       # zararsizlik esigi (her iki varlik)
# gercek-guclu-talep 2020 H2 PIT-asof ceyrek gozlemleri (kritik sessizlik testi referansi)
STRONG_2020_PITS = ("2020-06-15", "2020-09-14", "2020-12-14")

REPORT: list[str] = []


def say(line: str = "") -> None:
    print(line)
    REPORT.append(line)


def _sh(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")


def _maxdd(r):
    r = r.dropna()
    if len(r) < 5:
        return float("nan")
    eq = (1.0 + r).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def strat_ret(pos, close, lag=1):
    idx = pos.index
    ret = E.fwd_ret(close, idx).values
    p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def asof(series: pd.Series, d: pd.Timestamp):
    sub = series[series.index <= d]
    return sub.iloc[-1] if len(sub) else float("nan")


def fwd_from(close: pd.Series, d: pd.Timestamp, h: int) -> float:
    sub = close[close.index <= d]
    if len(sub) == 0:
        return float("nan")
    pos = close.index.get_loc(sub.index[-1])
    if isinstance(pos, slice):
        pos = pos.stop - 1
    if pos + h >= len(close):
        return float("nan")
    return float(close.iloc[pos + h] / close.iloc[pos] - 1.0)


def build_signals():
    """Donmus stack + arz-z'den (PIT) gunluk arz_hi / talep_zayif / fire serilerini kur."""
    scores, prices, vector, prov = C.read_frozen()
    ts = T.tide_score_series(scores, vector)
    tdir = T.tide_dir_series(ts)
    idx = ts.index

    sup = pd.read_parquet(CACHE / "net_equity_supply.parquet").dropna(subset=["z10y_nfc"])
    zpit = pd.Series(sup["z10y_nfc"].values, index=pd.DatetimeIndex(sup["pit_date"])).sort_index()
    zpit = zpit[~zpit.index.duplicated(keep="last")]
    zd = zpit.reindex(zpit.index.union(idx)).ffill().reindex(idx)   # gunluk ffill (PIT-asof)

    supply_hi = (zd >= SUPPLY_Z_THR)
    ts_decl = (ts - ts.shift(DECLINE_LB)) < 0
    demand_weak = (tdir == 0) | ((ts <= DEMAND_WEAK_LEVEL) & ts_decl)
    fire = supply_hi & demand_weak

    # NGDP YoY (opsiyonel betimsel bacak; KULLANILMAZ)
    ngdp = sup["ngdp_saar_bn"]
    ngdp_yoy = ngdp.pct_change(4) * 100.0
    ngdp_yoy_slow = (ngdp_yoy.diff() < 0)

    return dict(scores=scores, prices=prices, vector=vector, ts=ts, tdir=tdir, idx=idx,
                zd=zd, supply_hi=supply_hi, demand_weak=demand_weak, fire=fire,
                ngdp_yoy=ngdp_yoy, ngdp_yoy_slow=ngdp_yoy_slow, sup=sup)


def main() -> int:
    S = build_signals()
    ts, tdir, idx, zd = S["ts"], S["tdir"], S["idx"], S["zd"]
    supply_hi, demand_weak, fire = S["supply_hi"], S["demand_weak"], S["fire"]
    prices = S["prices"]
    spx = prices["SPX"].dropna()

    say("=" * 104)
    say("  CANDIDATE K2: KOSULLU forward de-risk (arz-asiri VE talep-zayif AYNI ANDA -> kucuk trim)")
    say(f"  on-kayitli kural: z10y_nfc >= +{SUPPLY_Z_THR:.1f} VE (tide_dir==0 VEYA "
        f"(tide_score <= +{DEMAND_WEAK_LEVEL:.1f} VE {DECLINE_LB}g-dususte)) -> deploy x{TRIM:.2f}")
    say(f"  test penceresi: 2019+ (FROZEN stack {idx.min().date()} -> {idx.max().date()})")
    say("  AMAC: alfa DEGIL rejim-degisim-sigortasi; tek-arz-sinyali 2020'de yanlis-atesler -> kosul kurtarir mi?")
    say("=" * 104)

    # ─────────────────────────────────────────────────────────────────────────────
    # 1) KRITIK — 2020 SESSIZLIGI
    # ─────────────────────────────────────────────────────────────────────────────
    say("")
    say("  1) KRITIK -- 2020 SESSIZLIGI (arz-yuksek AMA talep-guclu -> ATESLEMEMELI)")
    say(f"  {'PIT-asof':<12}{'arz-z':>8}{'tide_score':>12}{'dir':>5}{'talep-zayif':>12}{'ATES?':>7}   okuma")
    fired_strong_2020 = []
    # ceyreklik PIT gozlemleri (2020 dort ceyrek)
    pit2020 = ["2020-03-16", "2020-06-15", "2020-09-14", "2020-12-14"]
    for q in pit2020:
        d = pd.Timestamp(q)
        z = asof(zd, d); t = asof(ts, d); dr = asof(tdir, d)
        dw = bool(asof(demand_weak.astype(float), d)); f = bool(asof(fire.astype(float), d))
        strong = (q in STRONG_2020_PITS)
        if f and strong:
            fired_strong_2020.append(q)
        note = ("GUCLU-TALEP (susmali)" if strong else "spine-FLAT/gecis")
        say(f"  {q:<12}{z:>+8.2f}{t:>+12.2f}{dr:>5.0f}{('EVET' if dw else 'hayir'):>12}"
            f"{('ATESLEDI' if f else 'sustu'):>7}   {note}")
    # gun-bazli 2020 fire tarihleri (varsa)
    f2020 = fire[(idx.year == 2020) & fire.values].index
    say("")
    if len(f2020) == 0:
        say("  2020 gun-bazli fire: HIC YOK -> tam sessiz.")
    else:
        say(f"  2020 gun-bazli fire: {len(f2020)} gun ({f2020.min().date()} .. {f2020.max().date()}).")
        # bu gunlerde tide_dir neydi? (guclu-talep raliyi kesme mi, yoksa spine-FLAT gunleri mi)
        dir_on_fire = tdir.reindex(f2020)
        n_dir0 = int((dir_on_fire == 0).sum())
        say(f"     bu {len(f2020)} gunun {n_dir0}'inde tide_dir KENDISI 0 (spine zaten FLAT/gecis) -> ")
        say(f"     'guclu-talep raliyi kesme' DEGIL; geri kalan {len(f2020)-n_dir0} gun dir=1.")
        say(f"     ornek tide_score'lar: {', '.join(f'{ts.loc[x]:+.1f}' for x in f2020[:8])}")
    say("")
    silence_ok = (len(fired_strong_2020) == 0)
    say(f"  >> 2020-SESSIZLIK SONUCU: gercek-guclu-talep ceyreklerinde (Haz/Eyl/Ara PIT) fire = "
        f"{fired_strong_2020 if fired_strong_2020 else 'YOK'} -> {'SESSIZ (PASS)' if silence_ok else 'YANLIS-ATESLER (FAIL)'}")

    # ─────────────────────────────────────────────────────────────────────────────
    # 2) 2021 ATESLEME
    # ─────────────────────────────────────────────────────────────────────────────
    say("")
    say("  2) 2021 ATESLEME (arz-yuksek + talep-zayiflar; forward NEGATIF mi -> trim hakli mi)")
    say(f"  {'PIT-asof':<12}{'arz-z':>8}{'tide_score':>12}{'dir':>5}{'ATES?':>7}{'SPX-fwd63':>11}{'SPX-fwd126':>12}")
    pit2021 = ["2021-03-15", "2021-06-15", "2021-09-13", "2021-12-13"]
    for q in pit2021:
        d = pd.Timestamp(q)
        z = asof(zd, d); t = asof(ts, d); dr = asof(tdir, d); f = bool(asof(fire.astype(float), d))
        f63 = fwd_from(spx, d, 63); f126 = fwd_from(spx, d, 126)
        say(f"  {q:<12}{z:>+8.2f}{t:>+12.2f}{dr:>5.0f}{('ATESLEDI' if f else 'sustu'):>7}"
            f"{100*f63:>+10.1f}%{100*f126:>+11.1f}%")
    f2021 = fire[(idx.year == 2021) & fire.values].index
    fire_2021_ok = len(f2021) > 0
    # 2021 fire gunlerinin forward dagilimi
    if fire_2021_ok:
        fwd63_fire = pd.Series([fwd_from(spx, d, 63) for d in f2021], index=f2021).dropna()
        say("")
        say(f"  2021 fire: {len(f2021)} gun ({f2021.min().date()} .. {f2021.max().date()}); "
            f"bu gunlerin fwd-63g ort {100*fwd63_fire.mean():+.1f}%  isabet(neg) {100*(fwd63_fire<0).mean():.0f}%")
    say(f"  >> 2021-ATESLEME SONUCU: {'ATESLER' if fire_2021_ok else 'ATESLEMEZ'} "
        f"({len(f2021)} gun) -> {'PASS' if fire_2021_ok else 'FAIL'}")

    # ─────────────────────────────────────────────────────────────────────────────
    # 3) INCREMENTAL over TIDE (2019+)
    # ─────────────────────────────────────────────────────────────────────────────
    say("")
    say("  3) INCREMENTAL over TIDE (2019+): base = tide_dir ; variant = tide_dir x trim-faktor")
    fac = pd.Series(np.where(fire.values, TRIM, 1.0), index=idx)
    say(f"  fire-gun toplami (tum 2019+): {int(fire.sum())} / {len(idx)} ({100*fire.mean():.1f}%)")
    say(f"  fire-yillari: " + ", ".join(f"{yr}:{int(fire[idx.year==yr].sum())}"
        for yr in range(2019, 2027) if int(fire[idx.year == yr].sum()) > 0))
    say("")
    say(f"  {'varlik':<6}{'base-Sh':>9}{'var-Sh':>9}{'dSharpe':>9}{'base-maxDD':>12}{'var-maxDD':>11}"
        f"{'dMaxDD':>9}{'win-prob':>10}{'1-p':>7}")
    res = {}
    pvals = {}
    for a in ("SPX", "NDX"):
        base = strat_ret(tdir, prices[a])
        var = strat_ret((tdir * fac).reindex(idx), prices[a])
        bsh, vsh = _sh(base), _sh(var)
        bdd, vdd = _maxdd(base), _maxdd(var)
        wp = paired_win_prob(base, var)
        dsh = vsh - bsh
        res[a] = dict(bsh=bsh, vsh=vsh, dsh=dsh, bdd=bdd, vdd=vdd, ddd=vdd - bdd, wp=wp)
        if wp is not None:
            pvals[a] = 1.0 - wp
        say(f"  {a:<6}{bsh:>+9.3f}{vsh:>+9.3f}{dsh:>+9.3f}{bdd:>+12.3f}{vdd:>+11.3f}"
            f"{vdd - bdd:>+9.3f}{(wp if wp is not None else float('nan')):>10.2f}"
            f"{(1.0 - wp if wp is not None else float('nan')):>7.2f}")
    passed = fdr_bh(pvals, alpha=0.05)
    fdr_str = ", ".join(f"{a}:{'PASS' if passed.get(a, False) else '-'}" for a in ("SPX", "NDX"))
    say(f"  BH-FDR {{SPX,NDX}} (alpha 0.05): {fdr_str}")
    say(f"  NOT (durust): fire forward-only + 2021-2022 yogun (n az, tek-rejim) -> FDR-PASS BEKLENMEZ;")
    say(f"    bu test 'tide ile catismayan/zararsiz mi' sorusudur, alfa-anlamliligi degil.")

    # 3b) fire tarihleri (ozet)
    say("")
    say("  3b) ATESLEME PENCERELERI (ardisik fire bloklari):")
    on = fire.values
    blocks = []
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            blocks.append((idx[i], idx[j], j - i + 1))
            i = j + 1
        else:
            i += 1
    for a0, a1, n in blocks:
        say(f"      {a0.date()} .. {a1.date()}  ({n} gun)")

    # ─────────────────────────────────────────────────────────────────────────────
    # 3c) NGDP-yavaslama bacagi (BETIMSEL — KULLANILMADI; neden dahil edilmedigi)
    # ─────────────────────────────────────────────────────────────────────────────
    say("")
    say("  3c) NGDP-buyume-yavaslamasi (opsiyonel ucuncu bacak -- BETIMSEL, kurala DAHIL DEGIL):")
    ny = S["ngdp_yoy"]
    for q in ["2020-09-14", "2020-12-14", "2021-09-13", "2021-12-13"]:
        d = pd.Timestamp(q)
        v = asof(ny, d)
        say(f"      {q}: NGDP YoY {v:+.1f}%  -> 2020 taban-etkisi 2021 YoY'yi yapay GUCLU gosterir")
    say("      => proxy KONTAMINE (2021 melt-up'inda NGDP-YoY hala +%10) -> talep-zayif sinyalini")
    say("         TERS verir; bu yuzden kuralda KULLANILMADI (tide_score-tabanli proxy daha temiz).")

    # ─────────────────────────────────────────────────────────────────────────────
    # VERDICT
    # ─────────────────────────────────────────────────────────────────────────────
    harmless = all(res[a]["dsh"] >= DSHARPE_FLOOR for a in ("SPX", "NDX"))
    wire = silence_ok and fire_2021_ok and harmless
    say("")
    say("  " + "-" * 100)
    say("  VERDICT (on-kayitli kriterler):")
    say(f"  (a) 2020-sessiz (gercek-guclu-talep ceyreklerinde fire yok): {'EVET' if silence_ok else 'HAYIR'}")
    say(f"  (b) 2021-atesler ({len(f2021)} gun): {'EVET' if fire_2021_ok else 'HAYIR'}")
    say(f"  (c) 2019+ zararsiz (dSharpe >= {DSHARPE_FLOOR:+.2f} her iki varlik): "
        f"SPX {res['SPX']['dsh']:+.3f} / NDX {res['NDX']['dsh']:+.3f} -> {'EVET' if harmless else 'HAYIR'}")
    if wire:
        say(f"  SONUC: WIRE-FORWARD -- her uc kriter saglandi. Onerilen boyut: deploy x{TRIM:.2f} (kucuk,")
        say("         trim-only, rebound-safe). Bu bir ALFA degil rejim-degisim-SIGORTASIDIR; backtest-destegi")
        say("         forward-only/n-az (2021-22 tek-rejim) -> canli FORWARD-izleme ile dogrulanmali. FDR-PASS")
        say("         beklenmedi (dusuk-guc, on-kayitli). Deger: 2021-tarzi arz+talep-cozulme tekrarinda kucuk")
        say("         derisk; 2020-tarzi arz+guclu-talep raliyi KESMEZ (kanitlandi).")
    else:
        reasons = []
        if not silence_ok:
            reasons.append(f"2020 guclu-talep ceyreklerinde YANLIS-ATESLER ({fired_strong_2020})")
        if not fire_2021_ok:
            reasons.append("2021'de ATESLEMEZ")
        if not harmless:
            reasons.append(f"2019+ NET-ZARARLI (dSharpe SPX {res['SPX']['dsh']:+.3f}/NDX {res['NDX']['dsh']:+.3f} < {DSHARPE_FLOOR})")
        say(f"  SONUC: NO-WIRE -- neden: {'; '.join(reasons)}.")
    say(f"  (makine-okur: VERDICT={'WIRE-FORWARD' if wire else 'NO-WIRE'}; "
        f"silence2020={'PASS' if silence_ok else 'FAIL'}; fire2021={'PASS' if fire_2021_ok else 'FAIL'}; "
        f"harmless={'PASS' if harmless else 'FAIL'})")
    say("=" * 104)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "conditional_derisk_report.txt").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    print(f"  rapor -> {OUT / 'conditional_derisk_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
