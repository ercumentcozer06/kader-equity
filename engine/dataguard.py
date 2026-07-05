"""
engine/dataguard — GÖREV 6b. yfinance TEK boru, sanity yok → kasıtlı/kazara bozuk veriyle trade üretmeyi
ENGELLE. Snapshot'ı a priori bound'lara + iç/çapraz tutarlılığa karşı doğrular; herhangi bir kapı patlarsa
ok=False → motor "VERİ ÇÖP" brief'i basar, TRADE ÜRETMEZ (stale ile aynı disiplin, Bible).

Bound'lar A PRİORİ (optimize edilmedi): spot 1..100000, ön ATM-IV %3..150, exp_move 0..%10·spot, gex_flip/
wall'lar spot'un ±%20'si, |net_gex| < $1000bn. ÇAPRAZ: gamma-snapshot exp_move_1d ≈ spot·(surface ATM-IV)·
√(1/252) (yarı..2× tolerans) — iki ayrı yfinance çekiminin tutarlılığı (biri bayat/bozuksa yakalar). ŞEMA:
beklenen anahtarlar+tip yoksa → şema-değişti kapısı.
"""
from __future__ import annotations

from math import sqrt

# a priori sınırlar (optimize edilmedi)
SPOT_MIN, SPOT_MAX = 1.0, 100000.0
IV_MIN, IV_MAX = 5.0, 150.0              # ön ATM-IV % (surface_yf'in <%5 'şüpheli' eşiğiyle aynı; ham-IV ~%3 saçmalığını yakalar)
EM_MIN_FRAC = 0.001                      # H5: 1g expected-move en az spot'un %0.1'i (absürd-küçük em = bozuk veri → dev RR)
EM_MAX_FRAC = 0.10                       # 1g expected-move spot'un en fazla %10'u
WALL_BAND = 0.20                         # flip/wall spot'un ±%20'si
GEX_ABS_MAX = 1000.0                     # |net GEX| $bn üst sınır
EM_CONSISTENCY = (0.5, 2.0)             # exp_move vs IV-türevli expected-move tolerans bandı

_GAMMA_KEYS = {"as_of": str, "spot": (int, float), "net_gex_bn": (int, float)}
_SURF_KEYS = {"as_of": str, "spot": (int, float), "surface": dict}


def _schema_ok(d: dict, spec: dict, fails: list, tag: str) -> bool:
    ok = True
    for k, typ in spec.items():
        if k not in d or d[k] is None or not isinstance(d[k], typ):
            fails.append(f"{tag}: '{k}' yok/yanlış-tip (şema değişti?)"); ok = False
    return ok


def _front_atm_iv(surface: dict | None):
    if not surface:
        return None
    s = surface.get("surface", {}) or {}
    items = [(int(k.rstrip("d")), v.get("atm_iv")) for k, v in s.items() if v.get("atm_iv")]
    return sorted(items)[0][1] if items else None        # en kısa-DTE ATM-IV (%)


def validate(gamma: dict | None, surface: dict | None) -> dict:
    """Döndürür {ok, fails:[...], checks:{...}}. ok=False → trade üretme."""
    fails: list = []
    if not gamma and not surface:
        return {"ok": False, "fails": ["snapshot YOK (gamma+surface ikisi de eksik)"], "checks": {}}

    g = gamma or {}
    svf = surface or {}
    if gamma:
        _schema_ok(g, _GAMMA_KEYS, fails, "gamma")
    if surface:
        _schema_ok(svf, _SURF_KEYS, fails, "surface")

    spot = g.get("spot") or svf.get("spot")
    if spot is None or not (SPOT_MIN < float(spot) < SPOT_MAX):
        fails.append(f"spot aralık-dışı: {spot}")
        spot = None

    atm_iv = _front_atm_iv(svf)
    if atm_iv is not None and not (IV_MIN <= float(atm_iv) <= IV_MAX):
        fails.append(f"ön ATM-IV aralık-dışı: {atm_iv}% (yfinance ham-IV/bayat?)")

    if spot is not None:
        em = g.get("exp_move_1d")
        if em is not None and not (EM_MIN_FRAC * float(spot) < float(em) < EM_MAX_FRAC * float(spot)):
            fails.append(f"exp_move_1d aralık-dışı: {em} (%{EM_MIN_FRAC*100:.1f}–%{EM_MAX_FRAC*100:.0f}·spot dışı)")
        for nm in ("gex_flip", "call_wall", "put_wall", "hvl", "ghost"):    # G1+#3: hvl/ghost de bant kapsamında
            w = g.get(nm)
            if w is not None and abs(float(w)/float(spot) - 1) > WALL_BAND:
                fails.append(f"{nm} spot'tan >%{WALL_BAND*100:.0f} uzak: {w} vs spot {spot}")
    ng = g.get("net_gex_bn")
    if ng is not None and (abs(float(ng)) > GEX_ABS_MAX):
        fails.append(f"net_gex_bn aşırı: {ng}")

    # ÇAPRAZ tutarlılık: gamma.exp_move_1d ≈ spot·(surface ATM-IV)·√(1/252)
    cross = None
    if spot is not None and atm_iv is not None and g.get("exp_move_1d"):
        em_iv = float(spot) * (float(atm_iv) / 100.0) * sqrt(1 / 252)
        ratio = float(g["exp_move_1d"]) / em_iv if em_iv else None
        cross = round(ratio, 2) if ratio else None
        if ratio is not None and not (EM_CONSISTENCY[0] <= ratio <= EM_CONSISTENCY[1]):
            fails.append(f"çapraz tutarsız: exp_move {g['exp_move_1d']} vs IV-türevli {em_iv:.2f} "
                         f"(oran {ratio:.2f}, biri bayat/bozuk?)")

    return {"ok": len(fails) == 0, "fails": fails,
            "checks": {"spot": spot, "front_atm_iv": atm_iv, "exp_move_1d": g.get("exp_move_1d"),
                       "net_gex_bn": ng, "em_cross_ratio": cross}}
