"""
screen/candidate_net_supply — Constan'in HISSE NET-ARZI edge'i: icerik testi (ON-KAYITLI).

Veri: data/cache/net_equity_supply.parquet (screen/fetch_net_equity_supply.py; Z.1 transactions,
NCBCEBQ027S = NFC ana motor; FBCELIQ027S fin bacagi ETF-yaratimi KONTAMINE -> total referans-varyant).
PIT: pub-lag +165g (pit_date); kaydirmasiz varyant robustluk icin ayrica raporlanir.

ON-KAYITLI HIPOTEZ (sonuclardan once, Constan):
  NET-ARZ DUSUK/NEGATIF (buyback > ihrac, arz daraliyor) -> forward POZITIF;
  NET-ARZ YUKSEK (ihrac patlamasi, 2021 tarzi)            -> forward NEGATIF.
  => beklenen Spearman(arz, fwd) NEGATIF. Iki yon de raporlanir; isareti VERI secer.

ON-KAYITLI TEST PLANI (yavas-sinyal disiplini [[slow-signal-horizon]]; sinyali fark-alip oldurme YOK):
  1) HORIZON-MERDIVENI: ceyreklik PIT gozlem noktalarinda 63/126/252bd forward Spearman
     (SPX uzun tarihce data/cache/spx_gspc_long.csv 1927+; arz kapsami 1952Q3+ -> n~290).
     Sinyaller: ratio4q_nfc (seviye, %NGDP) + z10y_nfc (10y rolling z = trend-arindirilmis,
     [[spurious-trend]] korumasi) + kontamine ratio4q_total (referans).
     p-degeri: dairesel blok-permutasyon (signal'i >=8 ceyrek kaydir, 2000 perm, seed 77) —
     252bd ufuk ceyreklik gozlemde 4x ortusur, naive-p kullanilmaz.
  2) 5-KOVA MUTLAK-GETIRI ([[rank!=absolute]]): sinyal kuintilleri -> ort fwd-252bd + isabet.
  3) DONEM-STABILITE: 1950-79 / 1980-2004 / 2005+ ayri Spearman (gercek etki kalici olmali).
  4) 2019+ INCREMENTAL-over-TIDE: z-esikli yavas-tilt, trim-only (canli birikimli carpan sozlesmesi):
     z>+1 -> x0.75 / x0.50 (hipotez yonu: ihrac patlamasinda trim) VE z<-1 -> x0.75 / x0.50
     (ters yon, isareti veri secsin), strict BH-FDR {SPX,NDX}.
     GUC SINIRI (durustce, on-kayitli): ceyreklik seri 2019+ ~29 gozlem; |z|>1 epizodlari az ->
     bu pencerede FDR-PASS BEKLENMEZ; karar agirligi 1-3'teki cok-on-yillik iceriktedir.

DENETIM EKI (2026-06-13, ON-KAYIT-DISI — adversarial denetim sonrasi eklendi; plan metni yukarida
DEGISTIRILMEDI, yalniz uygulanan kriterler durustce raporlanir):
  5) TANISAL: (a) sinyal-vs-takvim-zamani Spearman (sahte-trend kontrolu, [[spurious-trend]]);
     (b) 1984+ alt-donem (geri-alimin var oldugu TUM donem); (c) 2005+ 252g 5-kova;
     (d) tam-orneklem kovalarin donem-bilesimi (donem-karismasi ayristirmasi).
  VERDICT: on-kayitli kriterlerin (ozellikle madde 3 stabilite: 'gercek etki kalici olmali')
     durust ust-satir uygulamasi. SONUC (bu kosu): KALICILIK-FAIL / SEKULER-ARTEFAKT — detay rapor sonunda.

Cikti: konsol (ASCII) + output/net_supply_report.txt (utf-8).
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
N_PERM, SEED, MIN_SHIFT_Q = 2000, 77, 8
PERIODS = (("1950-1979", 1950, 1979), ("1980-2004", 1980, 2004), ("2005+", 2005, 2100))

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
    """Her gozlem tarihinde (as-of, ffill) h-islem-gunu forward getiri. Son ufuk-disi NaN."""
    pos = close.index.searchsorted(dates, side="right") - 1
    out = np.full(len(dates), np.nan)
    ok = (pos >= 0) & (pos + h < len(close))
    cv = close.values
    out[ok] = cv[pos[ok] + h] / cv[pos[ok]] - 1.0
    return pd.Series(out, index=dates)


def spearman_perm(sig: pd.Series, fwd: pd.Series) -> tuple[float, float, int]:
    """Spearman + dairesel-blok-permutasyon p (signal'i >=MIN_SHIFT_Q ceyrek kaydir; otokorelasyon korunur)."""
    df = pd.concat([sig.rename("s"), fwd.rename("f")], axis=1).dropna()
    n = len(df)
    if n < 40:
        return float("nan"), float("nan"), n
    rho = df["s"].corr(df["f"], method="spearman")
    rng = np.random.default_rng(SEED)
    sv, fv = df["s"].values, df["f"].values
    fr = pd.Series(fv).rank().values
    perm = np.empty(N_PERM)
    for i in range(N_PERM):
        k = int(rng.integers(MIN_SHIFT_Q, n - MIN_SHIFT_Q))
        sr = pd.Series(np.roll(sv, k)).rank().values
        perm[i] = np.corrcoef(sr, fr)[0, 1]
    p = float((np.abs(perm) >= abs(rho)).mean())
    return float(rho), p, n


def main() -> int:
    sup = pd.read_parquet(CACHE / "net_equity_supply.parquet")
    spx = load_long_spx()
    sup = sup.dropna(subset=["ratio4q_nfc_pct"])
    pit = pd.DatetimeIndex(sup["pit_date"])

    say("=" * 100)
    say("  CANDIDATE: NET HISSE ARZI (Constan) — on-kayitli icerik testi (plan docstring'de)")
    say(f"  kapsam: {sup.index.min().date()} -> {sup.index.max().date()} ({len(sup)} ceyrek), PIT pub-lag +165g")
    say("  hipotez: arz DUSUK/NEG -> fwd POZITIF => beklenen Spearman NEGATIF (isareti veri secer)")
    say("=" * 100)

    # ── 1) HORIZON-MERDIVENI ──
    say("")
    say("  1) HORIZON-MERDIVENI — ceyreklik PIT gozleminde forward Spearman (blok-perm p, 2000/seed77)")
    say(f"  {'sinyal':<22}{'ufuk':>6}{'Spearman':>10}{'perm-p':>8}{'n':>6}   okuma")
    signals = {
        "ratio4q_nfc (seviye)": sup["ratio4q_nfc_pct"],
        "z10y_nfc (detrend)": sup["z10y_nfc"],
        "ratio4q_total (kontamine)": sup["ratio4q_total_pct"],
    }
    fwd_pit = {h: fwd_at(spx, pit, h) for h in HORIZONS}
    ladder = {}
    for name, s in signals.items():
        sig = pd.Series(s.values, index=pit)
        for h in HORIZONS:
            rho, p, n = spearman_perm(sig, fwd_pit[h])
            ladder[(name, h)] = (rho, p, n)
            read = ("hipotez-yonu (arz-yuksek=kotu)" if rho < 0 else "TERS yon (arz-yuksek=iyi)")
            star = " *" if p < 0.05 else ("  " if p >= 0.10 else " .")
            say(f"  {name:<22}{h:>5}g{rho:>+10.3f}{p:>8.3f}{n:>6}{star}  {read}")

    # robustluk: kaydirmasiz (pub-lag yok) — ana sinyal, 252bd
    say("")
    say("  robustluk (kaydirmasiz, pub-lag YOK; sadece referans — canli kullanim DAIMA +165g):")
    qd = sup.index
    fwd_q252 = fwd_at(spx, qd, 252)
    for name in ("ratio4q_nfc (seviye)", "z10y_nfc (detrend)"):
        sig = pd.Series(signals[name].values, index=qd)
        rho, p, n = spearman_perm(sig, fwd_q252)
        say(f"    {name:<24} 252g  Spearman {rho:+.3f}  perm-p {p:.3f}  n={n}")

    # ── 2) 5-KOVA MUTLAK GETIRI ──
    say("")
    say("  2) 5-KOVA MUTLAK-GETIRI (rank != absolute) — kuintil -> ort forward getiri (PIT)")
    say(f"  {'sinyal':<22}{'ufuk':>6}{'Q1(dusuk-arz)':>14}{'Q2':>8}{'Q3':>8}{'Q4':>8}{'Q5(yuksek-arz)':>15}{'isabet-Q1/Q5':>14}")
    for name in ("ratio4q_nfc (seviye)", "z10y_nfc (detrend)"):
        sig = pd.Series(signals[name].values, index=pit)
        for h in HORIZONS:
            df = pd.concat([sig.rename("s"), fwd_pit[h].rename("f")], axis=1).dropna()
            try:
                q = pd.qcut(df["s"], 5, labels=False, duplicates="drop")
            except ValueError:
                continue
            b = [100 * df["f"][q == i].mean() for i in range(5)]
            hit = [100 * (df["f"][q == i] > 0).mean() for i in (0, 4)]
            say(f"  {name:<22}{h:>5}g{b[0]:>+14.1f}{b[1]:>+8.1f}{b[2]:>+8.1f}{b[3]:>+8.1f}{b[4]:>+15.1f}"
                f"{hit[0]:>7.0f}/{hit[1]:<5.0f}")

    # ── 3) DONEM-STABILITE ──
    say("")
    say("  3) DONEM-STABILITE — Spearman per donem (PIT)")
    say(f"  {'sinyal':<22}{'ufuk':>6}" + "".join(f"{lab:>18}" for lab, _, _ in PERIODS))
    for name in ("ratio4q_nfc (seviye)", "z10y_nfc (detrend)"):
        sig = pd.Series(signals[name].values, index=pit)
        for h in (126, 252):
            cells = []
            for lab, a, b in PERIODS:
                m = (sig.index.year >= a) & (sig.index.year <= b)
                rho, p, n = spearman_perm(sig[m], fwd_pit[h][m])
                cells.append(f"{rho:+.2f} p{p:.2f} n{n}" if np.isfinite(rho) else f"n{n} az")
            say(f"  {name:<22}{h:>5}g" + "".join(f"{c:>18}" for c in cells))

    # ── 3b) TANISAL (on-kayit-DISI, descriptive): modern-donem ihrac-patlamasi epizodlari ──
    say("")
    say("  3b) TANISAL (on-kayit-disi, sadece betimsel): 2000+ z10y_nfc > +1 ceyrekleri -> fwd-252bd")
    zsig = pd.Series(sup["z10y_nfc"].values, index=pit)
    f252 = fwd_pit[252]
    ep = zsig[(zsig.index.year >= 2000) & (zsig > 1.0)]
    for d, v in ep.items():
        fv = f252.get(d, float("nan"))
        say(f"      PIT {d.date()}  z {v:+.2f}  fwd-252g {100*fv:+.1f}%" if np.isfinite(fv)
            else f"      PIT {d.date()}  z {v:+.2f}  fwd-252g (henuz yok)")

    # ── 4) 2019+ INCREMENTAL over TIDE ──
    say("")
    say("  4) INCREMENTAL over TIDE (2019+): z-esikli yavas-tilt, trim-only, strict BH-FDR {SPX,NDX}")
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    say(f"  base: SPX {_sh(bases['SPX']):+.3f} / NDX {_sh(bases['NDX']):+.3f}")
    zq = {}
    for tag, col in (("nfc", "z10y_nfc"), ("total", "z10y_total")):
        s = pd.Series(sup[col].values, index=pit).sort_index()
        zq[tag] = s.reindex(s.index.union(idx)).ffill().reindex(idx)
        n_hi = int((zq[tag] > 1).sum()); n_lo = int((zq[tag] < -1).sum())
        nq19 = int((pit >= idx.min()).sum())
        say(f"  guc-siniri[{tag}]: 2019+ ceyreklik gozlem n={nq19}; z>+1 gun={n_hi}, z<-1 gun={n_lo}"
            f"  -> dusuk guc, FDR-PASS beklenmez (on-kayitli)")
    say(f"  {'kural':<30}{'SPX dSh':>9}{'SPX P':>7}{'NDX dSh':>9}{'NDX P':>7}{'FDR':>6}")
    rules = []
    for tag in ("nfc", "total"):
        for sign, ystr in ((+1, "z>+1 (ihrac-patlamasi)"), (-1, "z<-1 (derin-daralma)")):
            for lvl in (0.75, 0.50):
                rules.append((f"{tag} {ystr} -> x{lvl:.2f}", tag, sign, lvl))
    for label, tag, sign, lvl in rules:
        fac = pd.Series(np.where(sign * zq[tag] > 1.0, lvl, 1.0), index=idx)
        res = {}
        for a in ("SPX", "NDX"):
            v = strat_ret((tdir * fac).reindex(idx), prices[a])
            res[a] = (paired_win_prob(bases[a], v), _sh(v) - _sh(bases[a]))
        passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
        both = "PASS" if all(passed.get(a, False) for a in ("SPX", "NDX")) else "-"
        say(f"  {label:<30}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}{both:>6}")

    # ── 5) TANISAL (ON-KAYIT-DISI; adversarial denetim 2026-06-13) ──
    say("")
    say("  5) TANISAL (on-kayit-DISI, denetim 2026-06-13): sahte-trend + donem-karismasi kontrolleri")
    sig_lvl = pd.Series(sup["ratio4q_nfc_pct"].values, index=pit)
    tnum = pd.Series(np.arange(len(sig_lvl), dtype=float), index=pit)
    rho_t = float(sig_lvl.corr(tnum, method="spearman"))
    say(f"  5a) sinyal-vs-takvim-zamani Spearman {rho_t:+.3f} -> sinyal ~ sekuler trend (1984 oncesi yapisal POZITIF,")
    say("      sonrasi yapisal NEGATIF). Dairesel-kaydirma permutasyonu monoton-trend hizalanmasini KORUMAZ ->")
    say("      tam-orneklemdeki seviye p=0.006* sahte-trend (Granger-Newbold) riskine ACIK; durust kanit =")
    say("      donem-stabilite (3) + z10y detrend (1) — ikisi de ANLAMSIZ.")
    m84 = sig_lvl.index.year >= 1984
    for h in (126, 252):
        rho, p, n = spearman_perm(sig_lvl[m84], fwd_pit[h][m84])
        say(f"  5b) 1984+ (geri-alimin var oldugu TUM donem) {h}g: Spearman {rho:+.3f}  perm-p {p:.3f}  n={n}"
            "  -> icerik SIFIR")
    m05 = sig_lvl.index.year >= 2005
    df05 = pd.concat([sig_lvl[m05].rename("s"), fwd_pit[252][m05].rename("f")], axis=1).dropna()
    q05 = pd.qcut(df05["s"], 5, labels=False, duplicates="drop")
    b05 = [100 * df05["f"][q05 == i].mean() for i in range(5)]
    say(f"  5c) 2005+ 252g kova Q1(dusuk-arz)->Q5: " + "  ".join(f"{x:+.1f}" for x in b05)
        + f"  (n={len(df05)})")
    say("      -> Q1 EN KOTU kova (derin geri-alim tepeleri 2007-08 GFC-oncesine denk) = mutlak-getiri")
    say("      hipotezi MODERN donemde TERS.")
    dfall = pd.concat([sig_lvl.rename("s"), fwd_pit[252].rename("f")], axis=1).dropna()
    qall = pd.qcut(dfall["s"], 5, labels=False, duplicates="drop")
    base_pre = 100 * dfall["f"][dfall.index.year < 1984].mean()
    base_post = 100 * dfall["f"][dfall.index.year >= 1984].mean()
    say(f"  5d) tam-orneklem 252g kova donem-bilesimi (era tabanlari: 1984-oncesi {base_pre:+.1f}% / 1984+ {base_post:+.1f}%):")
    q_means = {}
    for i in range(5):
        sub = dfall[qall == i]
        sh84 = 100 * float((sub.index.year >= 1984).mean())
        q_means[i] = 100 * sub["f"].mean()
        say(f"      Q{i+1}: ort {q_means[i]:+.1f}%  n={len(sub)}  1984+ uyelik-payi {sh84:.0f}%")
    say("      -> Q1/Q2 tamamen 1984+ (yuksek-taban era), Q4/Q5 agirlikla 1984-oncesi (dusuk-taban era):")
    say(f"      mansetteki Q1({q_means[0]:+.1f}%) > Q5({q_means[4]:+.1f}%) farki buyuk olcude DONEM-KARISMASI;")
    say(f"      ustelik Q1 kendi era tabaninin ({base_post:+.1f}%) ALTINDA -> 'dusuk-arz primi' tam-orneklemde bile yok.")

    say("")
    say("  OKU: karar agirligi 1-3 (cok-on-yillik icerik); 4 yalniz 'tide ile catisma var mi' sorusudur.")
    say("")
    say("  " + "-" * 96)
    say("  VERDICT (on-kayitli kriterlere gore, durust ust-satir):")
    z_ps = [ladder[("z10y_nfc (detrend)", h)][1] for h in HORIZONS]
    say(f"  [1] horizon-merdiveni: seviye her ufukta negatif AMA sinyal~zaman {rho_t:+.2f} -> tam-orneklem")
    say(f"      anlamliligi sahte-trend riskli; trend-arindirilmis z10y HICBIR ufukta anlamli degil"
        f" (p {min(z_ps):.2f}-{max(z_ps):.2f}). KANIT YETERSIZ.")
    say(f"  [2] kovalar: tam-orneklem Q1>Q5 = donem-karismasi (5d); 2005+ icinde Q1 EN KOTU ({b05[0]:+.1f}%)")
    say("      -> mutlak-getiri hipotezi modern donemde TERSINE. FAIL.")
    say("  [3] donem-stabilite ('gercek etki kalici olmali'): FAIL — hicbir alt-donem anlamli degil;")
    say("      2005+ isaret POZITIFE doner; 1984+ butunu ~0 (5b). ON-KAYITLI KALICILIK KRITERI SAGLANMADI.")
    say("  [4] incremental-over-tide: FDR-PASS yok (on-kayitli dusuk-guc; bilgi degeri sinirli).")
    say("  SONUC: KALICILIK-FAIL / SEKULER-ARTEFAKT. Tam-orneklem 'anlamliligi' trend + donem-karismasi")
    say("         urunu; geri-alim doneminin kendi icinde (1984+) icerik SIFIR, 2005+ isaret TERS.")
    say("         CANLI EDGE YOK — kablolama ONERILMEZ. Seri yalniz BETIMSEL panel olarak arsivde kalir")
    say("         (output/net_supply_panel.txt; panelde mekanizma/destek iddiasi YAPILMAZ).")
    say("=" * 100)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "net_supply_report.txt").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    print(f"  rapor -> {OUT / 'net_supply_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
