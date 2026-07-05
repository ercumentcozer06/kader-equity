"""screen/calibrate_pw — VENDOR-DRIFT GUARD (pw=put-overweight re-fit aracı).

Emir'in InSillico/SpotGamma-klonu indikatörü zamanla bizden ±%0.5+ kaçarsa: o varlığın GÜNCEL GEX FLIP'ini
buraya ver → bu araç hedefe en iyi uyan put-overweight'i (pw) bulur ve mevcut _cboe_lib.PUT_WEIGHT'in
tolerans dışına çıkıp çıkmadığını söyler. DRIFT çıkarsa önerilen pw'yi _cboe_lib.PUT_WEIGHT'e yaz (tek-kaynak).

Kullanım:  python screen/calibrate_pw.py _SPX 7550        (ya da:  QQQ 745)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "screen"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import _cboe_lib as L


def best_pw(sym: str, target: float, band: float = 0.15):
    """Monthly ∩ ±band havuzunda pw'yi [1.0,1.6] tarayıp hedef flip'e en yakın olanı döndürür."""
    spot, rows = L.load_rows(sym, band=0.15)
    best = None
    for pw in np.arange(1.0, 1.601, 0.05):
        flip, _ = L.flip_bs(rows, spot,
                            row_filter=lambda r: r["is_monthly"] and abs(r["K"] / spot - 1) <= band,
                            sign=(lambda cp, w=float(pw): 1.0 if cp == "C" else -w))
        if flip is None:
            continue
        err = abs(flip - target)
        if best is None or err < best[1]:
            best = (round(float(pw), 2), err, round(flip, 1))
    return spot, best


def main() -> int:
    if len(sys.argv) < 3:
        print("kullanım: python screen/calibrate_pw.py <SYM> <hedef_flip>   (örn: _SPX 7550 | QQQ 745)")
        return 1
    sym, target = sys.argv[1], float(sys.argv[2])
    spot, best = best_pw(sym, target)
    if not best:
        print(f"{sym}: flip bulunamadı (veri?).")
        return 1
    pw, err, flip = best
    cur = L.PUT_WEIGHT
    drift = (abs(pw - cur) >= 0.05) and (err > target * 0.004)
    print(f"{sym} spot {spot:.1f} | hedef flip {target:.0f} | en iyi pw={pw} → flip {flip:.0f} "
          f"(hata {err:.1f} = hedefin %{100*err/target:.2f}'i)")
    print(f"mevcut _cboe_lib.PUT_WEIGHT = {cur}  →  " +
          (f"DRIFT: önerilen pw={pw}, _cboe_lib.PUT_WEIGHT'i güncelle (gamma_engine+surface_yf otomatik alır)."
           if drift else "tolerans içinde, dokunma."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
