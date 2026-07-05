"""
backtest/diagnosis/D5_confound — CONFOUND HARİTASI (TEŞHİS-ONLY).

Soru: directional-GEX 'bayrağı' (level_series.regime = sign(net_gex)) ve büyüklüğü (|net_gex|)
gerçekten gamma-mikroyapı bilgisi mi taşıyor, yoksa BİLİNEN confound'ların (vol-rejimi / OPEX-takvim /
piyasa-trendi) bir yeniden-paketlemesi mi? disentangle.py'deki '%72 örtüşme' (gamma_inv vs vol_only
pozisyon) confound-şüphesinin kaynağı → burada decompose ediliyor.

PIT KURALI: bayrak D-EOD'de bilinir, D+1 seansında trade edilir. Tüm confound regressor'ları (RV20, VIX,
DTE, trailing-20g getiri) D-EOD veya öncesinde bilinen değerlerle ölçülür (look-ahead yok). Intraday getiri
= D+1 RTH c1/o1−1 (spine_diagnostic.build_panel ile aynı tanım).

ÖLÇÜLEN:
  (1) bayrak ~ vol      : logistic-fit pseudo-R² (McFadden) + 2x2 contingency (bayrak × vol-yüksek/düşük)
  (2) bayrak ~ takvim   : DTE-bucket × ortalama-bayrak / ortalama-|net_gex| sawtooth tablosu
  (3) bayrak ~ trend    : trailing-20g getiri vs bayrak (korr + contingency)
  (4) KAPANIŞ           : |net_gex| (ve sign-flag) varyansının ~%kaçı vol+takvim+trend ile (çoklu-reg R²)
                          → 'gamma'ya-özgü kalan pay' = 1−R². Sayıyla.

TEŞHİS-ONLY: yeni strateji/parametre/reweight YOK. level_series + bars OKUNUR, P&L üretilmez.
statsmodels/sklearn YOK → logistic ve OLS numpy/scipy ile elle (Newton-Raphson + lstsq).
  & <kader-macro venv python> backtest/diagnosis/D5_confound.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backtest"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from spine_diagnostic import daily_rth  # noqa: E402  (D+1 RTH OHLC, aynı tanım)

CACHE = ROOT / "data" / "cache"
BARS = ROOT / "data" / "historical_bars"
CHAINS = ROOT / "data" / "historical_chains"
RVWIN = 20      # trailing realized-vol penceresi (gün)
TRWIN = 20      # trailing-getiri penceresi (gün)


# ----------------------------------------------------------------------------- istatistik yardımcıları
def zscore(x):
    x = np.asarray(x, float)
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 0 else np.zeros_like(x)


def logistic_mcfadden(X, y, n_iter=50):
    """y∈{0,1} ~ logit(Xβ). Newton-Raphson. Dönen: (pseudo-R² McFadden, β, accuracy).
    X intercept'siz verilir; burada eklenir."""
    y = np.asarray(y, float)
    Xd = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    beta = np.zeros(Xd.shape[1])
    for _ in range(n_iter):
        eta = Xd @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        W = p * (1 - p)
        W = np.clip(W, 1e-9, None)
        grad = Xd.T @ (y - p)
        H = Xd.T @ (Xd * W[:, None])
        try:
            step = np.linalg.solve(H + 1e-8 * np.eye(H.shape[0]), grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    eta = Xd @ beta
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
    eps = 1e-12
    ll = np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    pbar = np.clip(y.mean(), eps, 1 - eps)
    ll0 = np.sum(y * np.log(pbar) + (1 - y) * np.log(1 - pbar))
    r2 = 1 - ll / ll0 if ll0 != 0 else 0.0
    acc = ((p >= 0.5).astype(float) == y).mean()
    return float(r2), beta, float(acc)


def ols_r2(X, y):
    """y ~ [1, X] OLS, R² döner (intercept dahil). X: (n,k) veya (n,)."""
    y = np.asarray(y, float)
    Xa = np.atleast_2d(np.asarray(X, float))
    if Xa.shape[0] != len(y):
        Xa = Xa.T
    if Xa.ndim == 1:
        Xa = Xa[:, None]
    Xd = np.column_stack([np.ones(len(y)), Xa])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    yhat = Xd @ beta
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(r2)


def contingency_2x2(a, b):
    """a,b: bool dizileri. 2x2 sayım + phi-korrelasyon + örtüşme%."""
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    n11 = int(np.sum(a & b)); n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b)); n00 = int(np.sum(~a & ~b))
    n = n11 + n10 + n01 + n00
    agree = (n11 + n00) / n if n else float("nan")
    # phi
    num = n11 * n00 - n10 * n01
    den = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    phi = num / den if den > 0 else 0.0
    return dict(n11=n11, n10=n10, n01=n01, n00=n00, agree=agree, phi=float(phi))


# ----------------------------------------------------------------------------- panel kurulumu
def dte_from_chains(sym):
    """Her D-tarihi için zincirden days-to-expiry (front-expiry, level_series'le aynı kaynak).
    PIT: D-EOD'de bilinir."""
    ch = pd.read_parquet(CHAINS / f"md_{sym.lower()}.parquet")
    g = ch.groupby("date").agg(exp=("expiration", "first"))
    dte = (pd.to_datetime(g["exp"].values) - pd.to_datetime(g.index)).days
    s = pd.Series(np.asarray(dte, float), index=pd.to_datetime(g.index))
    s.index = s.index.normalize()
    return s


def build_confound_panel(sym):
    """level_series (bayrak+|net_gex|, D-EOD) + PIT confound regressor'ları + D+1 intraday getiri."""
    lv = pd.read_parquet(CACHE / f"level_series_{sym.lower()}.parquet").copy()
    lv.index = pd.to_datetime(lv.index).normalize()

    # --- D+1 intraday getiri (spine_diagnostic ile aynı: RTH c1/o1−1) + günlük RTH kapanış (trend/RV için)
    rth = daily_rth(sym)
    rth.index = pd.to_datetime(rth.index).normalize()
    sess = list(rth.index)
    # günlük RTH-kapanış getirisi (trailing RV/trend hesapları PIT olsun diye sadece <=D kullanılır)
    rth_ret = rth["c"].pct_change()

    # --- VIX (D-EOD): yerel cache (vixcls.parquet) tercih; ağ yok. cache yoksa yfinance ^VIX dene.
    vix = None
    vp = CACHE / "vixcls.parquet"
    if vp.exists():
        v = pd.read_parquet(vp)
        col = "vix" if "vix" in v.columns else v.columns[0]
        vix = pd.Series(v[col].values, index=pd.to_datetime(v.index).normalize())
    else:
        try:
            import yfinance as yf
            yv = yf.download("^VIX", start="2025-05-01", end="2026-06-12", progress=False)
            vix = yv["Close"]["^VIX"] if isinstance(yv["Close"], pd.DataFrame) else yv["Close"]
            vix.index = pd.to_datetime(vix.index).normalize()
        except Exception as e:
            print(f"  [uyarı] VIX alınamadı ({e}) → VIX bacağı atlanacak")

    # --- DTE (D-EOD, zincirden)
    dte = dte_from_chains(sym)

    rows = []
    for D in lv.index:
        if D not in rth.index:
            continue
        nxt = [s for s in sess if s > D]
        if not nxt:
            continue
        N = nxt[0]
        o1 = rth.loc[N, "o"]; c1 = rth.loc[N, "c"]
        # PIT trailing realized-vol (D dahil son RVWIN günlük getiri std, yıllık)
        hist = rth_ret.loc[:D].dropna()
        rv20 = hist.tail(RVWIN).std() * np.sqrt(252) if len(hist) >= RVWIN else np.nan
        # PIT trailing-20g kümülatif getiri (D dahil)
        cwin = rth["c"].loc[:D]
        tr20 = (cwin.iloc[-1] / cwin.iloc[-1 - TRWIN] - 1) if len(cwin) > TRWIN else np.nan
        vix_d = float(vix.asof(D)) if vix is not None and not vix.empty else np.nan
        dte_d = float(dte.asof(D)) if D in dte.index or (dte.index <= D).any() else np.nan

        r = lv.loc[D].to_dict()
        r.update(dict(D=D, N=N, intraday=c1 / o1 - 1,
                      rv20=rv20, vix=vix_d, dte=dte_d, tr20=tr20))
        rows.append(r)
    p = pd.DataFrame(rows)
    # bayrak: regime = sign(net_gex) ∈ {−1,+1}; ham büyüklük |net_gex| ve onun logu
    p["flag"] = (p["regime"] > 0).astype(int)          # 1 = +γ, 0 = −γ
    p["absgex"] = p["net_gex"].abs()
    p["log_absgex"] = np.log(p["absgex"].clip(lower=1.0))
    return p


# ----------------------------------------------------------------------------- raporlama
def section_decompose_72(sym, p):
    """disentangle.py'deki '%72 örtüşme' = gamma_inv-pozisyonu vs vol_only-pozisyonu örtüşmesi.
    İki pozisyon da ±gap olduğundan, örtüşme TAM OLARAK flag-kovası ile vol-kovasının aynı
    yön-koşulunu seçtiği günlere eşittir. disentangle'la BYTE-AYNI tanım: vol_only = atm_iv>medyan.
    Bu örtüşmeyi mekanik olarak decompose eder (yeni P&L üretmeden — sadece pozisyon-işaretleri)."""
    print(f"\n  ── (0) '%72 ÖRTÜŞME' DECOMPOSE  [{sym}]  (disentangle.py: gamma_inv-poz vs vol_only-poz)")
    sub = p.dropna(subset=["regime", "atm_iv", "intraday"]).copy()
    gap_sign_proxy = None  # gap işareti panelde yok; örtüşme gap'ten BAĞIMSIZ (ikisi de ±gap → işaret aynıysa örtüşür)
    reg = sub["regime"].values                       # net_gex işareti (+1/−1)
    volhigh = sub["atm_iv"].values > np.median(sub["atm_iv"].values)
    # gamma_inv pozisyon-koşulu: +γ→+gap, −γ→−gap  ⇒ çarpan c_g = sign(reg)
    cg = np.where(reg > 0, 1, -1)
    # vol_only pozisyon-koşulu: düşük-vol→+gap, yüksek-vol→−gap ⇒ çarpan c_v
    cv = np.where(volhigh, -1, 1)
    overlap = (cg == cv).mean()  # iki pozisyon (±gap) aynı işaretli ⇒ örtüşür (gap ortak çarpan, sadeleşir)
    print(f"    pozisyon-örtüşmesi (gamma_inv vs vol_only) = %{100*overlap:.0f}  "
          f"(disentangle.py raporu ile aynı tanım, n{len(sub)})")
    # decompose: örtüşme = +γ&düşük-vol + −γ&yüksek-vol (ikisi de aynı yön seçer)
    c = contingency_2x2((reg > 0), ~volhigh)
    print(f"    decompose → örtüşen günler: (+γ & düşük-vol) {c['n11']} + (−γ & yüksek-vol) {c['n00']} "
          f"= {c['n11']+c['n00']}/{len(sub)}")
    print(f"               örtüşmeyen     : (+γ & yüksek-vol) {c['n10']} + (−γ & düşük-vol) {c['n01']} "
          f"= {c['n10']+c['n01']}/{len(sub)}")
    print(f"    YORUM: örtüşme = flag-kovası ile atm_iv-vol-kovasının yön-eşleşmesi (phi {c['phi']:+.2f}); "
          f"gap ortak çarpan olduğu için sadeleşir.")
    print(f"           %{100*overlap:.0f} = flag'ın atm_iv-vol-bucket ile {('güçlü' if abs(c['phi'])>0.3 else 'orta-zayıf')} "
          f"ilişkisinin pozisyon-yansıması — ama bağımsız-coin baz çizgisi de %50 (rastgele örtüşme).")
    return overlap, c


def section_vol(sym, p):
    print(f"\n  ── (1) BAYRAK ~ VOL  [{sym}]  (PIT: RV20 ve VIX D-EOD'de bilinir)")
    sub = p.dropna(subset=["flag", "rv20"]).copy()
    n = len(sub)
    # logistic: flag ~ z(RV20) [+ z(VIX) varsa]
    have_vix = sub["vix"].notna().sum() >= 0.8 * n and sub["vix"].std() > 0
    r2_rv, _, acc_rv = logistic_mcfadden(zscore(sub["rv20"].values), sub["flag"].values)
    print(f"    logistic flag~RV20      : pseudo-R²(McFadden) {r2_rv:.3f}  accuracy %{100*acc_rv:.0f}  (n{n})")
    if have_vix:
        sv = sub.dropna(subset=["vix"])
        r2_v, _, acc_v = logistic_mcfadden(zscore(sv["vix"].values), sv["flag"].values)
        r2_both, _, acc_both = logistic_mcfadden(
            np.column_stack([zscore(sv["rv20"].values), zscore(sv["vix"].values)]), sv["flag"].values)
        print(f"    logistic flag~VIX       : pseudo-R²(McFadden) {r2_v:.3f}  accuracy %{100*acc_v:.0f}  (n{len(sv)})")
        print(f"    logistic flag~RV20+VIX  : pseudo-R²(McFadden) {r2_both:.3f}  accuracy %{100*acc_both:.0f}")
    else:
        print(f"    [VIX bacağı atlandı — yeterli VIX verisi yok]")
    # 2x2 contingency: bayrak(+γ) × vol-yüksek (RV20 medyan-üstü)
    volhigh = sub["rv20"].values > np.nanmedian(sub["rv20"].values)
    c = contingency_2x2(sub["flag"].values.astype(bool), ~volhigh)  # +γ vs düşük-vol (tez: +γ↔düşük-vol)
    print(f"    2x2  (+γ × DÜŞÜK-vol): +γ&loV {c['n11']}  +γ&hiV {c['n10']}  −γ&loV {c['n01']}  −γ&hiV {c['n00']}")
    print(f"         örtüşme(+γ↔loV / −γ↔hiV) %{100*c['agree']:.0f}  phi {c['phi']:+.2f}  "
          f"({'vol-rejimi bayrağı güçlü açıklıyor' if abs(c['phi'])>0.3 else 'zayıf-orta bağ'})")
    # |net_gex| ~ vol (büyüklük de vol-proxy mi?)
    r2m = ols_r2(zscore(sub["rv20"].values), zscore(sub["log_absgex"].values))
    print(f"    OLS  log|net_gex|~RV20  : R² {r2m:.3f}  (büyüklük vol ile ne kadar açıklanıyor)")
    return r2_rv, c


def section_calendar(sym, p):
    print(f"\n  ── (2) BAYRAK ~ OPEX-TAKVİM  [{sym}]  (DTE = front-expiry'ye gün; sawtooth)")
    sub = p.dropna(subset=["dte", "flag"]).copy()
    sub["dte_i"] = sub["dte"].round().astype(int)
    # DTE-bucket: 0-2, 3-5, 6-9, 10-14, 15+
    bins = [-1, 2, 5, 9, 14, 99]
    labs = ["0-2", "3-5", "6-9", "10-14", "15+"]
    sub["bucket"] = pd.cut(sub["dte_i"], bins=bins, labels=labs)
    print(f"    {'DTE-bucket':<10} {'n':>4} {'+γ-oran':>8} {'ort|net_gex|(B$)':>18} {'med|net_gex|(B$)':>18}")
    for lab in labs:
        b = sub[sub["bucket"] == lab]
        if len(b) == 0:
            continue
        fr = b["flag"].mean()
        am = b["absgex"].mean() / 1e9
        md = b["absgex"].median() / 1e9
        print(f"    {lab:<10} {len(b):>4} {fr:>8.2f} {am:>18.2f} {md:>18.2f}")
    # korr: DTE vs flag, DTE vs |net_gex|
    cf = np.corrcoef(sub["dte_i"].values, sub["flag"].values)[0, 1]
    cm = np.corrcoef(sub["dte_i"].values, sub["log_absgex"].values)[0, 1]
    r2_mag = ols_r2(sub["dte_i"].values.astype(float), zscore(sub["log_absgex"].values))
    print(f"    corr DTE~flag {cf:+.3f}  |  corr DTE~log|net_gex| {cm:+.3f}  |  OLS log|net_gex|~DTE R² {r2_mag:.3f}")
    return cf, r2_mag


def section_trend(sym, p):
    print(f"\n  ── (3) BAYRAK ~ TREND  [{sym}]  (PIT trailing-20g getiri, D-EOD)")
    sub = p.dropna(subset=["tr20", "flag"]).copy()
    cf = np.corrcoef(sub["tr20"].values, sub["flag"].values)[0, 1]
    # contingency: +γ × yukarı-trend (tr20>0)
    up = sub["tr20"].values > 0
    c = contingency_2x2(sub["flag"].values.astype(bool), up)
    print(f"    corr trail20~flag {cf:+.3f}  (tez: yukarı-trend → call-wall kırılmaz → +γ?)")
    print(f"    2x2 (+γ × YUKARI-trend): +γ&up {c['n11']}  +γ&dn {c['n10']}  −γ&up {c['n01']}  −γ&dn {c['n00']}")
    print(f"         örtüşme %{100*c['agree']:.0f}  phi {c['phi']:+.2f}  "
          f"({'trend bayrağı açıklıyor' if abs(c['phi'])>0.3 else 'zayıf-orta bağ'})")
    r2 = ols_r2(zscore(sub["tr20"].values), zscore(sub["log_absgex"].values))
    print(f"    OLS  log|net_gex|~trail20 : R² {r2:.3f}")
    return cf, c


def section_decompose(sym, p):
    print(f"\n  ── (4) KAPANIŞ — ÇOKLU-REGRESYON DECOMPOSE  [{sym}]")
    feats = ["rv20", "dte", "tr20"]
    if p["vix"].notna().sum() >= 0.8 * len(p) and p["vix"].std() > 0:
        feats = ["rv20", "vix", "dte", "tr20"]
    # --- TARGET A: |net_gex| (log)  -- sürekli büyüklük
    subm = p.dropna(subset=feats + ["log_absgex"]).copy()
    Xm = np.column_stack([zscore(subm[f].values) for f in feats])
    ym = zscore(subm["log_absgex"].values)
    r2_mag = ols_r2(Xm, ym)
    print(f"    A) log|net_gex| ~ [{'+'.join(feats)}]  (OLS, n{len(subm)})")
    print(f"       R² = {r2_mag:.3f}  →  GAMMA'YA-ÖZGÜ KALAN PAY = 1−R² = {1-r2_mag:.3f}  (%{100*(1-r2_mag):.0f})")
    # tek-tek katkı (univariate R²)
    for f in feats:
        print(f"         tek {f:<5} R² {ols_r2(zscore(subm[f].values), ym):.3f}")
    # --- TARGET B: sign-flag (binom) -- logistic pseudo-R²
    subf = p.dropna(subset=feats + ["flag"]).copy()
    Xf = np.column_stack([zscore(subf[f].values) for f in feats])
    r2_flag, _, acc_flag = logistic_mcfadden(Xf, subf["flag"].values)
    print(f"    B) sign-flag ~ [{'+'.join(feats)}]  (logistic, n{len(subf)})")
    print(f"       pseudo-R²(McFadden) = {r2_flag:.3f}  accuracy %{100*acc_flag:.0f}  →  "
          f"GAMMA'YA-ÖZGÜ KALAN (1−pseudoR²) = {1-r2_flag:.3f}")
    base = max(subf["flag"].mean(), 1 - subf["flag"].mean())
    print(f"       (taban-oran accuracy %{100*base:.0f}; confound-model {'+' if acc_flag>base else ''}{100*(acc_flag-base):.0f}pp)")
    return r2_mag, r2_flag


def run(sym):
    print("=" * 100)
    print(f"  D5 CONFOUND HARİTASI — {sym}")
    print("=" * 100)
    p = build_confound_panel(sym)
    print(f"  panel n={len(p)} gün  ({p['D'].min().date()} → {p['D'].max().date()})  "
          f"+γ-oran %{100*p['flag'].mean():.0f}  "
          f"RV20-kapsam {p['rv20'].notna().mean()*100:.0f}%  VIX-kapsam {p['vix'].notna().mean()*100:.0f}%  "
          f"DTE-kapsam {p['dte'].notna().mean()*100:.0f}%")
    section_decompose_72(sym, p)
    section_vol(sym, p)
    section_calendar(sym, p)
    section_trend(sym, p)
    r2_mag, r2_flag = section_decompose(sym, p)
    return p, r2_mag, r2_flag


def main():
    out = {}
    for sym in ("SPY", "QQQ"):
        _, r2m, r2f = run(sym)
        out[sym] = (r2m, r2f)
        print()
    print("=" * 100)
    print("  ÖZET — gamma'ya-özgü kalan pay (confound-dışı):")
    for sym, (r2m, r2f) in out.items():
        print(f"    {sym}: |net_gex| büyüklük → kalan %{100*(1-r2m):.0f}  |  sign-flag → kalan (1−pseudoR²) {1-r2f:.3f}")
    print("  OKU: kalan-pay büyükse (örn >%70) bayrak vol+takvim+trend'in tekrarı DEĞİL → ayrı bilgi olabilir.")
    print("       kalan-pay küçükse confound'lar bayrağı büyük oranda yeniden-paketliyor.")
    print("  SINIR: 236 gün / tek-rejim pencere; confound R²'leri in-sample, forward'da değişebilir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
