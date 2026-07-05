"""
[DEAD / SUPERSEDED 2026-06-17] BU SCRIPT KULLANILMIYOR — hiçbir çağıran YOK. surface_spy/ cache'ini artık
CBOE'li screen/surface_yf.py üretir (gamma_flip + gex alanlarıyla). Bu IBKR sürümü GEX/flip ÜRETMEZ (delayed
feed'de OI yok) ve yanlışlıkla geri-bağlanırsa surface_spy snapshot'ını gamma'sız IBKR yüzeyiyle EZER. YENİDEN BAĞLAMA.

screen/vol_surface_live — GERÇEK equity vol surface (IBKR, kader-silver/vol_surface.py pattern'i).

Underlying = SPY (ETF; SPX index delayed feed'de YOK = CBOE index aboneliği gerek. SPY ETF delayed
ÇALIŞIR — SLV gibi. SPY yüzey geometrisi = SPX, SPY≈SPX/10). Skaler endeks DEĞİL — asıl geometri:

  IV(moneyness × tenor) ızgarası → ATM IV, 25-delta put/call IV, RISK-REVERSAL (skew), term-structure, smile.

Delayed feed (type 3): modelGreeks (IV/delta/gamma) çalışır, OI çalışmaz (GEX için OPRA aboneliği lazım).
Snapshot data/cache/surface_spy/<date>.json = forward-collector başlangıcı (per-strike geçmiş yüzey
free/IBKR'de YOK → backtest için ya paralı tarih (ORATS/OptionMetrics) ya bu forward-biriktirme).
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KMR = Path(r"C:\Users\admin\Downloads\kader-macro")
sys.path.insert(0, str(KMR))            # yalnız kader-macro (modules._ibkr + ib_insync); ROOT path'e GİRMEZ
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import yaml                                              # noqa: E402
from dotenv import load_dotenv                           # noqa: E402
from modules import _ibkr                                # noqa: E402

MONEYNESS = [0.85, 0.88, 0.91, 0.94, 0.97, 1.00, 1.03, 1.06, 1.10]
TARGET_DTE = [7, 30, 60, 90, 180]
SETTLE = 12


def _spot(ib, contract) -> float | None:
    tk = ib.reqMktData(contract, "", False, False)
    ib.sleep(10)
    for f in ("last", "close", "marketPrice"):
        v = getattr(tk, f, None)
        v = v() if callable(v) else v
        if v and v == v and v > 0:
            ib.cancelMktData(contract)
            return float(v)
    ib.cancelMktData(contract)
    return None


def _hv_from_bars(bars, n: int) -> float | None:
    closes = [b.close for b in bars if b.close]
    if len(closes) < n + 1:
        return None
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    use = rets[-n:]
    return round(statistics.stdev(use) * math.sqrt(252) * 100, 2) if len(use) > 1 else None


def _pick_expiries(expirations: list[str]) -> list[str]:
    today = date.today()
    avail = []
    for s in sorted(set(expirations)):
        try:
            d = datetime.strptime(s, "%Y%m%d").date()
        except ValueError:
            continue
        dte = (d - today).days
        if dte >= 3:
            avail.append((dte, s))
    out = []
    for tgt in TARGET_DTE:
        if avail:
            best = min(avail, key=lambda x: abs(x[0] - tgt))
            if best[1] not in [o[1] for o in out]:
                out.append(best)
    return [s for _, s in sorted(out)]


def main() -> int:
    load_dotenv(KMR / ".env")
    cfg = yaml.safe_load((KMR / "config.yaml").read_text(encoding="utf-8"))
    cfg.setdefault("ibkr", {})["client_id"] = 48         # benzersiz clientId (kader-macro 12; zombie'den de kaç)
    cfg["ibkr"]["market_data_type"] = 4                  # delayed-FROZEN: off-hours'da son delayed snapshot'ı verir
    from ib_insync import Stock, Option                  # noqa: E402

    ib = _ibkr.connect(cfg)
    if ib is None:
        print("  [!] IBKR bağlanamadı (Gateway açık mı?).")
        return 1
    try:
        ib.reqMarketDataType(cfg.get("ibkr", {}).get("market_data_type", 3))
        spy = Stock("SPY", "SMART", "USD")
        ib.qualifyContracts(spy)
        spot = _spot(ib, spy)                              # delayed snapshot (close/last)
        src = "ibkr_spy_delayed"
        if not spot:                                       # fallback: ES e-mini /10 (AYNI bağlantı)
            try:
                from ib_insync import ContFuture
                es = ContFuture("ES", "CME", currency="USD")
                ib.qualifyContracts(es)
                es_px = _spot(ib, es)
                if es_px:
                    spot, src = es_px / 10.0, "es_emini/10"
            except Exception:
                pass
        if not spot:
            print("  [!] SPY spot alınamadı (delayed snapshot + ES fallback boş; market-data aboneliği sınırlı).")
            return 1
        hv20 = hv60 = None                                 # historical market-data service 162 erroring → HV atla
        print(f"  SPY spot {spot:.2f} ({src})  ≈SPX {spot*10:.0f}   [HV: historical aboneliği yok, atlandı]")

        chains = ib.reqSecDefOptParams("SPY", "", "STK", spy.conId)
        chain = max(chains, key=lambda c: len(c.expirations))
        exps = _pick_expiries(list(chain.expirations))
        all_strikes = sorted(chain.strikes)
        strikes = sorted({min(all_strikes, key=lambda s: abs(s - m * spot)) for m in MONEYNESS})
        print(f"  chain={chain.tradingClass}@{chain.exchange}  expiries={exps}  "
              f"strikes={len(strikes)} ({strikes[0]:.0f}..{strikes[-1]:.0f})")

        opts = [Option("SPY", e, k, r, "SMART", tradingClass=chain.tradingClass, multiplier="100", currency="USD")
                for e in exps for k in strikes for r in ("C", "P")]
        ib.qualifyContracts(*opts)
        for o in opts:
            ib.reqMktData(o, "", False, False)
        ib.sleep(SETTLE)

        rows = []
        for o in opts:
            mg = ib.ticker(o).modelGreeks
            if mg and mg.impliedVol and mg.impliedVol > 0:
                rows.append({"exp": o.lastTradeDateOrContractMonth, "k": o.strike, "right": o.right,
                             "iv": mg.impliedVol, "delta": mg.delta})
        for o in opts:
            ib.cancelMktData(o)
        print(f"  greeks alınan kontrat: {len(rows)}/{len(opts)}")
        if not rows:
            print("  [!] modelGreeks boş (delayed settle yetmedi → SETTLE arttır).")
            return 1

        today = date.today()
        print(f"\n  {'DTE':>5}{'ATM_IV':>8}{'25dPut':>8}{'25dCall':>9}{'RR(skew)':>10}   {'smile (moneyness:IV)':<48}")
        surface = {}
        for e in exps:
            dte = (datetime.strptime(e, "%Y%m%d").date() - today).days
            er = [x for x in rows if x["exp"] == e]
            calls = [x for x in er if x["right"] == "C"]
            puts = [x for x in er if x["right"] == "P"]
            if not calls:
                continue
            atm = min(calls, key=lambda x: abs(x["k"] - spot))
            pput = [x for x in puts if x.get("delta") is not None]
            pcall = [x for x in calls if x.get("delta") is not None]
            p25 = min(pput, key=lambda x: abs(x["delta"] + 0.25)) if pput else None
            c25 = min(pcall, key=lambda x: abs(x["delta"] - 0.25)) if pcall else None
            rr = (p25["iv"] - c25["iv"]) * 100 if (p25 and c25) else None     # put-call skew = kuyruk fiyatı
            smile = "  ".join(f"{x['k']/spot:.2f}:{x['iv']*100:.0f}" for x in sorted(er, key=lambda z: z["k"])
                              if x["right"] == ("P" if x["k"] < spot else "C"))
            surface[f"{dte}d"] = {"atm_iv": round(atm["iv"]*100, 2),
                                  "put25_iv": round(p25["iv"]*100, 2) if p25 else None,
                                  "call25_iv": round(c25["iv"]*100, 2) if c25 else None,
                                  "rr_skew": round(rr, 2) if rr is not None else None}
            print(f"  {dte:>5}{atm['iv']*100:>8.1f}{(p25['iv']*100 if p25 else float('nan')):>8.1f}"
                  f"{(c25['iv']*100 if c25 else float('nan')):>9.1f}{(rr if rr is not None else float('nan')):>+10.2f}   {smile[:46]}")

        atm_front = surface[list(surface)[0]]["atm_iv"] if surface else None
        iv_hv = round(atm_front / hv20, 2) if (atm_front and hv20) else None
        print(f"\n  ATM_IV(front) {atm_front}  /  HV20 {hv20}  =  IV/HV {iv_hv}  "
              f"({'VOL_RICH' if iv_hv and iv_hv>1.2 else 'VOL_CHEAP' if iv_hv and iv_hv<0.9 else 'FAIR'})")

        snap = {"as_of": today.isoformat(), "ts": datetime.now(timezone.utc).isoformat(),
                "underlying": "SPY", "spot": round(spot, 2), "hv20": hv20, "hv60": hv60, "surface": surface,
                "feed": "ibkr_delayed_type3", "note": "per-strike modelGreeks; OI yok (OPRA gerek → GEX kısıtlı)"}
        outdir = ROOT / "data" / "cache" / "surface_spy"
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"{today.isoformat()}.json").write_text(json.dumps(snap, indent=2), encoding="utf-8")
        print(f"  snapshot → {outdir / (today.isoformat() + '.json')}  (forward-collector: günlük biriktir)")
        return 0
    except Exception as e:
        print(f"  [!] FAILED: {type(e).__name__}: {e}")
        return 1
    finally:
        _ibkr._safe_disconnect(ib)


if __name__ == "__main__":
    raise SystemExit(main())
