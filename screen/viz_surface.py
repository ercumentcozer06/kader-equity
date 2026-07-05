"""
screen/viz_surface — vol surface GÖRSELLEŞTİRME (yfinance, free). 4 panel PNG:
  (1) 3D IV yüzeyi (moneyness × DTE × IV)   (2) skew eğrileri (IV vs moneyness, expiry başına)
  (3) term-structure (ATM IV vs DTE)        (4) GEX-by-strike (bar) + spot + gamma-flip
  & <venv python> screen\viz_surface.py [SPY|QQQ]
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from math import erf, exp, log, pi, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import yfinance as yf                  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401

try:                                                 # TEK kanonik IV inversiyonu (gamma_engine/surface_yf ile aynı)
    from _bsiv import implied_vol                     # noqa: E402
except ImportError:
    from screen._bsiv import implied_vol              # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
R, Q = 0.04, 0.013
TARGET_DTE = [7, 30, 60, 90, 150, 250]
MNY = np.round(np.arange(0.82, 1.121, 0.02), 2)


def bs_gamma(S, K, T, s):
    if T <= 0 or s <= 0 or S <= 0:
        return 0.0
    d1 = (log(S/K) + (R-Q+s*s/2)*T)/(s*sqrt(T))
    return exp(-Q*T) * (exp(-d1*d1/2)/sqrt(2*pi)) / (S*s*sqrt(T))


# bs_price / implied_vol → screen/_bsiv (TEK kaynak; gamma_engine+surface_yf ile byte-aynı)


def main():
    tick = (sys.argv[1] if len(sys.argv) > 1 else "SPY").upper()
    idx_lbl, mult = {"SPY": ("SPX", 10), "QQQ": ("NDX", 41)}.get(tick, (tick, 1))
    t = yf.Ticker(tick)
    try:
        spot = float(t.fast_info["lastPrice"])
    except Exception:
        spot = float(t.history(period="1d")["Close"].iloc[-1])
    today = date.today()

    # expiry seç
    avail = []
    for e in t.options:
        d = datetime.strptime(e, "%Y-%m-%d").date(); dte = (d-today).days
        if dte >= 2:
            avail.append((dte, e))
    exps = sorted({min(avail, key=lambda x: abs(x[0]-tg)) for tg in TARGET_DTE})

    dtes, iv_grid, skews, gex_by_k = [], [], {}, {}
    for dte, e in exps:
        T = dte/365.0
        try:
            oc = t.option_chain(e)
        except Exception:
            continue
        ks, ivs = [], []
        for df, right in ((oc.calls, "C"), (oc.puts, "P")):
            for _, r in df.iterrows():
                k, bid, ask, oi = r.get("strike"), r.get("bid"), r.get("ask"), r.get("openInterest")
                if not k or abs(k/spot-1) > 0.20 or not bid or not ask or bid <= 0 or ask <= 0:
                    continue
                iv = implied_vol((bid+ask)/2, spot, float(k), T, right)   # mid'den hesaplanan IV (yahoo IV'sine güvenme)
                if iv is None:
                    continue
                # surface: OTM tarafı (put<spot, call>spot)
                if (right == "P" and k <= spot) or (right == "C" and k >= spot):
                    ks.append(float(k)); ivs.append(iv)
                g = bs_gamma(spot, float(k), T, iv) * float(oi or 0) * 100 * spot*spot*0.01
                gex_by_k[float(k)] = gex_by_k.get(float(k), 0.0) + (g if right == "C" else -g)
        if len(ks) < 5:
            continue
        order = np.argsort(ks); ks = np.array(ks)[order]; ivs = np.array(ivs)[order]
        row = np.interp(MNY*spot, ks, ivs*100)            # fixed-moneyness IV (%)
        iv_grid.append(row); dtes.append(dte); skews[dte] = (MNY, row)

    if not iv_grid:
        print("  [!] grid boş (veri bayat/yok).")
        return 1
    iv_grid = np.array(iv_grid)            # [dte × moneyness]
    med_atm = float(np.median([skews[d][1][len(MNY)//2] for d in dtes]))
    stale = med_atm < 5
    asof = datetime.now().strftime("%Y-%m-%d %H:%M")

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f"{tick} (≈{idx_lbl} {spot*mult:.0f})  vol surface — {asof}"
                 + ("   ⚠ VERİ BAYAT (pre-market) — kapanışta çek" if stale else ""),
                 fontsize=14, color=("red" if stale else "black"))

    # (1) 3D surface
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    X, Yv = np.meshgrid(MNY, dtes)
    ax1.plot_surface(X, Yv, iv_grid, cmap="viridis", edgecolor="none", alpha=0.9)
    ax1.set_xlabel("moneyness (K/S)"); ax1.set_ylabel("DTE (gün)"); ax1.set_zlabel("IV %")
    ax1.set_title("IV yüzeyi (moneyness × tenor)"); ax1.view_init(elev=25, azim=-60)

    # (2) skew curves
    ax2 = fig.add_subplot(2, 2, 2)
    for d in dtes:
        m, iv = skews[d]; ax2.plot(m, iv, marker="o", ms=3, label=f"{d}g")
    ax2.axvline(1.0, color="grey", ls="--", lw=0.8)
    ax2.set_xlabel("moneyness"); ax2.set_ylabel("IV %"); ax2.set_title("Skew eğrileri (put-skew = sol↑)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # (3) term structure (ATM IV)
    ax3 = fig.add_subplot(2, 2, 3)
    atm = [skews[d][1][len(MNY)//2] for d in dtes]
    ax3.plot(dtes, atm, marker="s", color="darkblue")
    ax3.set_xlabel("DTE (gün)"); ax3.set_ylabel("ATM IV %")
    ax3.set_title(f"Term-structure ({'contango↑' if atm[-1] > atm[0] else 'backwardation↓'})"); ax3.grid(alpha=0.3)

    # (4) GEX by strike + spot + flip
    ax4 = fig.add_subplot(2, 2, 4)
    ks = np.array(sorted(gex_by_k)); gv = np.array([gex_by_k[k]/1e9 for k in ks])
    ax4.bar(ks, gv, width=(ks.max()-ks.min())/len(ks)*0.8,
            color=np.where(gv >= 0, "seagreen", "indianred"))
    ax4.axvline(spot, color="black", ls="-", lw=1.2, label=f"spot {spot:.0f}")
    cum = np.cumsum(gv); flipidx = np.where(np.diff(np.sign(cum)))[0]
    if len(flipidx):
        ax4.axvline(ks[flipidx[0]], color="purple", ls="--", lw=1.2, label=f"γ-flip ~{ks[flipidx[0]]:.0f}")
    ax4.set_xlabel(f"{tick} strike"); ax4.set_ylabel("GEX $bn/1% (call+ / put−)")
    ax4.set_title("Dealer GEX by strike (yeşil=long-γ)"); ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = ROOT / "output" / f"surface_{tick.lower()}_{today.isoformat()}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"  saved -> {out}   (med ATM-IV {med_atm:.1f}%{' ⚠BAYAT' if stale else ' ✓'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
