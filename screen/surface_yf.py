"""
screen/surface_yf — equity vol surface + GEX, PROFESYONEL CBOE verisinden (eski adı yfinance'tan kalma).

2026-06-17 CBOE-GEÇİŞİ: yfinance SPY×10 proxy yerine gerçek CBOE delayed_quotes (_SPX index / QQQ ETF).
  • SURFACE : IV(moneyness × tenor) → ATM IV, 25-delta put/call IV (BS-delta), RISK-REVERSAL (skew), term-structure
  • GEX     : net dealer gamma ($/1%) + gamma-flip — gamma_engine ile AYNI reçete (monthly + put-overweight 1.3 +
              BS-repricing) → tek doğruluk kaynağı, iki ledger çatışmaz.

Sınır: CBOE delayed = CANLI snapshot (tarihsel chain YOK) → backtest için forward-collect (snapshot
data/cache/surface_<tick>/<date>.json günlük biriktir). BETİMSEL panel (skew/IV + flip); tradeable sinyal DEĞİL.

  & <kader-equity venv python> screen/surface_yf.py [SPY|QQQ]
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from math import erf, isfinite, log, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:                                                 # script (screen/) ya da paket
    from _cboe_lib import load_rows as cboe_load_rows, flip_bs, bs_gamma, PUT_WEIGHT   # noqa: E402
except ImportError:
    from screen._cboe_lib import load_rows as cboe_load_rows, flip_bs, bs_gamma, PUT_WEIGHT  # noqa: E402

TARGET_DTE = [7, 30, 60, 90, 180]
R, Q = 0.04, 0.013          # gamma_engine/_cboe_lib ile aynı
BAND = 0.20                 # ±20% moneyness (smile için geniş; uzak-OTM çöp IV elenir)
# PUT_WEIGHT _cboe_lib'den import edilir (TEK-KAYNAK; gamma_engine ile AYNI)
CFG = {"SPY": ("_SPX", "SPX", 1), "QQQ": ("QQQ", "NDX", 41)}


def _Nd(x): return 0.5 * (1 + erf(x / sqrt(2)))


def _d1(S, K, T, s): return (log(S / K) + (R - Q + s * s / 2) * T) / (s * sqrt(T))


def bs_delta(S, K, T, s, right):
    if T <= 0 or s <= 0 or S <= 0 or K <= 0:
        return 0.0
    d = _d1(S, K, T, s)
    return _Nd(d) if right == "C" else -_Nd(-d)


def main() -> int:
    arg = (sys.argv[1] if len(sys.argv) > 1 else "SPY").upper()
    cboe_sym, idx_lbl, mult = CFG.get(arg, (arg, arg, 1))
    try:
        spot, crows = cboe_load_rows(cboe_sym, band=BAND)
    except Exception as e:                                          # noqa: BLE001
        print(f"  [!] CBOE çekilemedi ({cboe_sym}): {type(e).__name__}: {str(e)[:120]}")
        return 1
    if not crows:
        print(f"  [!] CBOE zincir boş ({cboe_sym}).")
        return 1
    if not isfinite(spot) or spot <= 0:                  # savunma (kök guard _cboe_lib'de): NaN/0 spot -> junk snapshot YAZMA
        print(f"  [!] geçersiz spot ({cboe_sym}): {spot}")
        return 1
    today = date.today()

    by_exp = {}
    for r in crows:
        iv = r.get("iv")
        if not iv or iv <= 0:
            continue
        by_exp.setdefault(r["expd"], []).append(r)
    avail = sorted(((e - today).days, e) for e in by_exp if (e - today).days >= 2)
    if not avail:
        print("  [!] uygun expiry yok (>=2 DTE).")
        return 1
    picks = []
    for tgt in TARGET_DTE:
        best = min(avail, key=lambda x: abs(x[0] - tgt))
        if best not in picks:
            picks.append(best)
    picks = sorted(set(picks))

    print(f"  {idx_lbl} {spot*mult:.0f}  (CBOE {cboe_sym})   {datetime.now(timezone.utc):%Y-%m-%d %H:%MZ}")
    print(f"\n  {'DTE':>5}{'ATM_IV':>8}{'25dPut':>8}{'25dCall':>9}{'RR_skew':>9}")

    surface = {}
    for dte, e in picks:
        T = max(dte, 1) / 365.0
        rows = by_exp[e]
        calls = [x for x in rows if x["cp"] == "C"]
        puts = [x for x in rows if x["cp"] == "P"]
        if not rows:
            continue
        atm = min(rows, key=lambda x: abs(x["K"] - spot))
        for x in rows:
            x["_d"] = bs_delta(spot, x["K"], T, x["iv"], x["cp"])
        p25 = min((x for x in puts), key=lambda x: abs(x["_d"] + 0.25), default=None)
        c25 = min((x for x in calls), key=lambda x: abs(x["_d"] - 0.25), default=None)
        rr = (p25["iv"] - c25["iv"]) * 100 if (p25 and c25) else None
        surface[f"{dte}d"] = {"atm_iv": round(atm["iv"]*100, 2) if atm else None,
                              "put25_iv": round(p25["iv"]*100, 2) if p25 else None,
                              "call25_iv": round(c25["iv"]*100, 2) if c25 else None,
                              "rr_skew": round(rr, 2) if rr is not None else None}
        print(f"  {dte:>5}{(atm['iv']*100):>8.1f}{(p25['iv']*100 if p25 else float('nan')):>8.1f}"
              f"{(c25['iv']*100 if c25 else float('nan')):>9.1f}{(rr if rr is not None else float('nan')):>+9.2f}")

    # GEX + flip — gamma_engine ile AYNI reçete (monthly + put-overweight + BS-repricing)
    # flip/net = gamma_engine ile BİREBİR aynı küme: monthly ∩ |K/spot-1|<=0.15 (smile 0.20'de kalır)
    flip, _net = flip_bs(crows, spot, row_filter=lambda r: r["is_monthly"] and abs(r["K"]/spot - 1) <= 0.15,
                         sign=lambda cp: 1.0 if cp == "C" else -PUT_WEIGHT, use_cboe_gamma=False)
    mo = [r for r in crows if r["is_monthly"] and r.get("iv") and abs(r["K"]/spot - 1) <= 0.15]
    net_gex = sum((1.0 if r["cp"] == "C" else -PUT_WEIGHT) * bs_gamma(spot, r["K"], r["T"], r["iv"])
                  * r["oi"] * 100 * spot * spot * 0.01 for r in mo)
    if flip:
        regime = "SHORT gamma (hareket-amplifikasyon)" if spot < flip else "LONG gamma (vol-bastırma/pin)"
    else:
        regime = "NEUTRAL gamma (flip ±%15 taramada yok — belirsiz)"
    print(f"\n  GEX (CBOE, monthly+pw{PUT_WEIGHT}): net {net_gex/1e9:+.2f} $bn/1%  → dealer {regime}")
    print((f"  gamma-flip (zero-gamma) ≈ {idx_lbl} {flip*mult:.0f}") if flip else "  gamma-flip: scan dışı")
    print("  NOT: BETİMSEL (CBOE EOD OI, vendor-match); flip gamma_engine ile aynı reçete. Sinyal DEĞİL.")

    if not [k for k, v in surface.items() if v.get("atm_iv")]:
        print("  [X] VERİ ÇÖP: geçerli expiry yok. Snapshot YAZILMADI.")
        return 1
    snap = {"as_of": today.isoformat(), "ts": datetime.now(timezone.utc).isoformat(), "underlying": cboe_sym,
            "index": idx_lbl, "spot": round(spot, 2), "surface": surface,
            "gex_net_bn_per_1pct": round(net_gex / 1e9, 3), "gamma_flip": round(flip, 2) if flip else None,
            "source": "cboe", "note": "CBOE delayed_quotes; flip=monthly+pw1.3 (gamma_engine ile aynı reçete)"}
    outdir = ROOT / "data" / "cache" / f"surface_{arg.lower()}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{today.isoformat()}.json").write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"\n  snapshot → {outdir / (today.isoformat() + '.json')}  (CBOE forward-collector)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
