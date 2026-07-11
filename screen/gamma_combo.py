"""
screen/gamma_combo — SpotGamma-Combo tarzi BIRLESIK gamma (endeks + ETF kompleksi toplami; BETIMSEL).

Profesyonel standart (SpotGamma Combos; Emir karari 2026-07-11): tek komplekse bakma, iki kompleksin
DOLAR-gammasini topla. Yontem: her kompleksin net-GEX$(h) egrisi AYNI goreli-hareket izgarasinda
(h ∈ ±%15, 121 nokta; flip_bs defaults) hesaplanir — dolar dolar toplanir (birim-donusumu gerekmez) —
birlesik egrinin sifir-gecisi = COMBO FLIP, spot'taki degeri = COMBO NET. Recete kartlarla birebir:
monthly + BS-repricing + pw1.3 (_cboe_lib tek-kaynak). DIVERGENT = iki bilesenin isareti zit.

Ciktilar: data/cache/gamma_combo_{spx,ndx}/YYYY-MM-DD.json  (cockpit READ-ONLY okur)
  & <venv python> screen/gamma_combo.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "screen"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import _cboe_lib as L  # noqa: E402

# combo -> [(CBOE sembol, etiket), (..)]; ilk eleman ENDEKS (flip o olcekte raporlanir)
PAIRS = {"spx": [("_SPX", "SPX-endeks"), ("SPY", "SPY-ETF")],
         "ndx": [("_NDX", "NDX-endeks"), ("QQQ", "QQQ-ETF")]}
LO, HI, N = -0.15, 0.15, 121               # flip_bs defaults — ayni izgara
SIGN = lambda cp: 1.0 if cp == "C" else -L.PUT_WEIGHT  # noqa: E731  (vendor recetesi, tek-kaynak pw)


def _curve(sym: str):
    """Kompleksin net-GEX$(h) egrisi (monthly+pw, BS-repricing) + spot + spot'taki net."""
    spot, rows = L.load_rows(sym, band=0.15)
    use = [r for r in rows if r["iv"] and r["is_monthly"]]
    if not use:
        return spot, None, 0.0
    hs_grid = np.linspace(LO, HI, N)
    def net(hs_abs):
        return sum(SIGN(r["cp"]) * L.bs_gamma(hs_abs, r["K"], r["T"], r["iv"]) * r["oi"] * 100
                   * hs_abs * hs_abs * 0.01 for r in use)
    curve = np.array([net(spot * (1 + p)) for p in hs_grid])
    return spot, curve, float(net(spot))


def combo(tag: str) -> dict | None:
    parts, curves = [], []
    for sym, lbl in PAIRS[tag]:
        try:
            spot, curve, net_spot = _curve(sym)
        except Exception as e:
            print(f"  [!] {lbl} alinamadi: {type(e).__name__}: {str(e)[:60]} — combo {tag} SKIP")
            return None
        if curve is None:
            print(f"  [!] {lbl} monthly-satiri yok — combo {tag} SKIP")
            return None
        parts.append({"label": lbl, "spot": round(spot, 2), "net_gex_bn": round(net_spot / 1e9, 3),
                      "regime": "LONG GAMMA" if net_spot >= 0 else "SHORT GAMMA"})
        curves.append(curve)
    total = curves[0] + curves[1]
    net_bn = float(parts[0]["net_gex_bn"] + parts[1]["net_gex_bn"])
    hs_grid = np.linspace(LO, HI, N)
    flip_pct = None
    for (p0, g0), (p1, g1) in zip(zip(hs_grid, total), zip(hs_grid[1:], total[1:])):
        if (g0 <= 0 <= g1) or (g0 >= 0 >= g1):
            flip_pct = p0 + (p1 - p0) * (0 - g0) / (g1 - g0) if g1 != g0 else p0
            break
    idx_spot = parts[0]["spot"]
    snap = {"as_of": date.today().isoformat(), "ts": datetime.now(timezone.utc).isoformat(),
            "combo": tag.upper(), "methodology": "combo-sum-monthly-bs-pw" + str(L.PUT_WEIGHT),
            "net_gex_bn": round(net_bn, 3),
            "regime": "LONG GAMMA" if net_bn >= 0 else "SHORT GAMMA",
            "flip": round(idx_spot * (1 + flip_pct), 1) if flip_pct is not None else None,
            "flip_pct_from_spot": round(flip_pct * 100, 2) if flip_pct is not None else None,
            "index_spot": idx_spot, "components": parts,
            "divergent": parts[0]["regime"] != parts[1]["regime"]}
    outdir = ROOT / "data" / "cache" / f"gamma_combo_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{snap['as_of']}.json").write_text(json.dumps(snap, indent=2), encoding="utf-8")
    dv = "  <<DIVERGENT>>" if snap["divergent"] else ""
    print(f"  COMBO {tag.upper():3}: {snap['regime']}  net {net_bn:+.2f}bn  flip {snap['flip']} "
          f"({snap['flip_pct_from_spot']:+.1f}% spot'tan)  [" +
          " | ".join(f"{p['label']} {p['net_gex_bn']:+.2f}bn" for p in parts) + f"]{dv}")
    return snap


def main() -> int:
    print(f"GAMMA COMBO (endeks+ETF birlesik; SpotGamma-tarzi; betimsel)  {date.today()}")
    ok = [combo(t) for t in PAIRS]
    return 0 if any(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
