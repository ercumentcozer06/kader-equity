"""
backtest/remeasure/RC1_determinism.py — RC1 DETERMİNİZM + SAĞLIK KAPISI (teşhis-only, idempotent/yeniden-koşulabilir).

(1) DETERMİNİZM: 16 yeni parquet (CFG.level_path) vs data/cache/archive_157g/ aynı-isimli eskiler.
    ORTAK tarihlerde sayısal kolonlarda max|fark| + relative-fark; girdi (raw_chains append-only) ve
    kod-numerikleri değişmediği için fark ~0 beklenir (rel tol 1e-9). Seri-başına PASS/FAIL.
    Ek dürüstlük: cache dosyası archive kopyasıyla AYNI build ise (rebuild inmemiş) determinizm karşılaştırması
    triviyal olur → build_status olarak raporlanır (meta.json + tarih-aralığı kanıtı).
(2) SAĞLIK (yeni seri üzerinde): ≥230 gün; call_wall≥spot & put_wall≤spot ~%100; flip-bulunma ≥%95; regime ∈ {−1,+1}.

FAIL varsa nedeni (hangi kolon, hangi tarihler, kaç gün) JSON'a yazılır — sessiz devam yok.
→ backtest/remeasure/RC1_determinism.json  (config_sha + per-seri detay + summary)
   exit code 0 = 16/16 PASS, 1 = en az bir FAIL.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as CFG  # noqa: E402  (FAZ-R tek-gerçek-kaynak)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---- RC1 kapı-spec'i (görev-tanımı eşikleri; config'te yer almayan GATE sabitleri — model/strateji sabiti DEĞİL) ----
MODES = ["livematch", "fullsurface"]          # CFG.level_path(mode,...) sözleşmesinin iki modu
COMPARE_COLS = ["net_gex", "flip", "call_wall", "put_wall", "hvl",
                "max_pain", "atm_iv", "em1", "gamma_dollar"]   # determinizm sayısal kolon kümesi (görev-spec)
REL_TOL = 1e-9                                # relative float-toleransı
ABS_FLOOR = 1e-12                             # 0-yakını için mutlak taban
MIN_DAYS = 230                                # sağlık: seri uzunluğu alt sınırı
WALL_OK_MIN = 0.99                            # call_wall≥spot & put_wall≤spot payı (~%100)
FLIP_FOUND_MIN = 0.95                         # flip-bulunma payı
REGIME_ALLOWED = {-1, 1}                      # regime ±1

OUT_JSON = CFG.REMEASURE_DIR / "RC1_determinism.json"


def _f(x):
    """JSON-safe float."""
    if x is None:
        return None
    x = float(x)
    return None if (np.isnan(x) or np.isinf(x)) else x


def compare_series(new: pd.DataFrame, old: pd.DataFrame) -> dict:
    """ORTAK tarihlerde COMPARE_COLS determinizm karşılaştırması; kolon-başına max|fark| + ihlal detayı."""
    common = new.index.intersection(old.index)
    res = {"n_common": int(len(common)), "n_new": int(len(new)), "n_old": int(len(old)),
           "cols": {}, "fail_cols": [], "pass": True}
    if len(common) == 0:
        res["pass"] = False
        res["fail_cols"] = ["<NO_COMMON_DATES>"]
        return res
    a_all, b_all = new.loc[common], old.loc[common]
    for c in COMPARE_COLS:
        if c not in new.columns or c not in old.columns:
            res["cols"][c] = {"error": "kolon eksik", "in_new": c in new.columns, "in_old": c in old.columns}
            res["fail_cols"].append(c)
            res["pass"] = False
            continue
        a = pd.to_numeric(a_all[c], errors="coerce").astype(float)
        b = pd.to_numeric(b_all[c], errors="coerce").astype(float)
        both_nan = a.isna() & b.isna()
        one_nan = a.isna() ^ b.isna()
        num = (~a.isna()) & (~b.isna())
        diff = (a[num] - b[num]).abs()
        denom = np.maximum(a[num].abs(), b[num].abs())
        viol = diff > (ABS_FLOOR + REL_TOL * denom)          # numpy.isclose tarzı: atol + rtol*max(|a|,|b|)
        max_abs = _f(diff.max()) if len(diff) else 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = diff / np.where(denom > 0, denom, np.nan)
        max_rel = _f(np.nanmax(rel.values)) if len(diff) and not rel.isna().all() else 0.0
        bad_dates = sorted(set(diff[viol].index.strftime("%Y-%m-%d")) |
                           set(a[one_nan].index.strftime("%Y-%m-%d")))
        col_pass = (int(viol.sum()) == 0) and (int(one_nan.sum()) == 0)
        res["cols"][c] = {
            "max_abs_diff": max_abs, "max_rel_diff": max_rel,
            "n_viol": int(viol.sum()), "n_one_sided_nan": int(one_nan.sum()),
            "n_both_nan": int(both_nan.sum()), "bad_dates": bad_dates[:10],
            "pass": col_pass,
        }
        if not col_pass:
            res["fail_cols"].append(c)
            res["pass"] = False
    return res


def health_series(df: pd.DataFrame) -> dict:
    """Yeni seri üzerinde sağlık kapısı."""
    n = len(df)
    cw_nn = df["call_wall"].notna() & df["spot"].notna()
    pw_nn = df["put_wall"].notna() & df["spot"].notna()
    cw_ok = float((df.loc[cw_nn, "call_wall"] >= df.loc[cw_nn, "spot"]).mean()) if cw_nn.any() else None
    pw_ok = float((df.loc[pw_nn, "put_wall"] <= df.loc[pw_nn, "spot"]).mean()) if pw_nn.any() else None
    flip_found = float(df["flip"].notna().mean()) if n else 0.0
    reg_vals = sorted(int(v) for v in df["regime"].dropna().unique()) if "regime" in df.columns else []
    reg_ok = set(reg_vals) <= REGIME_ALLOWED and len(reg_vals) > 0
    checks = {
        "n_days": {"value": n, "min": MIN_DAYS, "pass": n >= MIN_DAYS},
        "call_wall_ge_spot": {"share": _f(cw_ok), "n_null": int((~cw_nn).sum()), "min": WALL_OK_MIN,
                              "pass": cw_ok is not None and cw_ok >= WALL_OK_MIN},
        "put_wall_le_spot": {"share": _f(pw_ok), "n_null": int((~pw_nn).sum()), "min": WALL_OK_MIN,
                             "pass": pw_ok is not None and pw_ok >= WALL_OK_MIN},
        "flip_found": {"share": _f(flip_found), "min": FLIP_FOUND_MIN, "pass": flip_found >= FLIP_FOUND_MIN,
                       "missing_dates": sorted(df.index[df["flip"].isna()].strftime("%Y-%m-%d"))[:10]},
        "regime_pm1": {"values": reg_vals, "pass": bool(reg_ok)},
    }
    checks["pass"] = all(v["pass"] for k, v in checks.items() if isinstance(v, dict))
    return checks


def build_status(new_p: Path, old_p: Path, new: pd.DataFrame, old: pd.DataFrame) -> dict:
    """Rebuild gerçekten indi mi? meta.json + tarih-aralığı + içerik kanıtı (dürüstlük katmanı)."""
    meta_p = Path(str(new_p) + ".meta.json")
    meta = None
    if meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except Exception:
            meta = {"error": "meta okunamadı"}
    extends = bool(new.index.max() > old.index.max() or len(new) > len(old))
    identical = bool(len(new) == len(old) and new.index.equals(old.index) and new.equals(old))
    if meta and extends:
        status = "REBUILT"
    elif identical and meta is None:
        status = "STALE_PENDING"  # cache == archive kopyası, rebuild henüz inmemiş
    else:
        status = "AMBIGUOUS"
    return {"status": status, "meta_json": meta is not None,
            "built_utc": (meta or {}).get("built_utc"), "meta_config_sha": (meta or {}).get("config_sha"),
            "new_mtime": datetime.fromtimestamp(new_p.stat().st_mtime).isoformat(timespec="seconds"),
            "identical_to_archive": identical, "extends_archive": extends}


def main() -> int:
    out = {
        "script": "backtest/remeasure/RC1_determinism.py",
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha": CFG.config_sha(),
        "gate_spec": {"compare_cols": COMPARE_COLS, "rel_tol": REL_TOL, "abs_floor": ABS_FLOOR,
                      "min_days": MIN_DAYS, "wall_ok_min": WALL_OK_MIN, "flip_found_min": FLIP_FOUND_MIN,
                      "regime_allowed": sorted(REGIME_ALLOWED)},
        "series": {},
    }
    n_pass = 0
    fails = []
    print(f"=== RC1 DETERMİNİZM + SAĞLIK KAPISI  (config_sha={out['config_sha']}) ===")
    for mode in MODES:
        for sym in CFG.SYMS:
            for hyg in (True, False):
                new_p = CFG.level_path(mode, sym, hygiene=hyg)
                old_p = CFG.ARCHIVE_157 / new_p.name
                key = new_p.stem.replace("level_series_", "")
                rec = {"file": new_p.name}
                if not new_p.exists() or not old_p.exists():
                    rec.update({"pass": False, "reason": f"dosya eksik: new={new_p.exists()} old={old_p.exists()}"})
                    out["series"][key] = rec
                    fails.append((key, rec["reason"]))
                    print(f"  {key:32s} FAIL  {rec['reason']}")
                    continue
                new = pd.read_parquet(new_p).sort_index()
                old = pd.read_parquet(old_p).sort_index()
                det = compare_series(new, old)
                hea = health_series(new)
                bst = build_status(new_p, old_p, new, old)
                ok = bool(det["pass"] and hea["pass"])
                rec.update({"build": bst, "determinism": det, "health": hea, "pass": ok})
                out["series"][key] = rec
                if ok:
                    n_pass += 1
                else:
                    why = []
                    if not det["pass"]:
                        why.append(f"determinizm: {det['fail_cols']}")
                    for ck, cv in hea.items():
                        if isinstance(cv, dict) and not cv["pass"]:
                            why.append(f"sağlık.{ck}={cv.get('value', cv.get('share', cv.get('values')))}")
                    if bst["status"] == "STALE_PENDING":
                        why.append("build_status=STALE_PENDING (cache==archive kopyası; rebuild inmemiş)")
                    rec["reason"] = "; ".join(why)
                    fails.append((key, rec["reason"]))
                mx = max((c.get("max_abs_diff") or 0.0) for c in det["cols"].values() if "max_abs_diff" in c) \
                    if det["cols"] else None
                print(f"  {key:32s} {'PASS' if ok else 'FAIL'}  n={len(new):3d}g common={det['n_common']:3d} "
                      f"max|Δ|={mx if mx is not None else float('nan'):.3g} build={bst['status']}"
                      + ("" if ok else f"  ← {rec['reason']}"))
    out["summary"] = {"n_pass": n_pass, "n_total": 16, "pass_16_16": n_pass == 16,
                      "fails": [{"series": k, "reason": r} for k, r in fails]}
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nSONUÇ: {n_pass}/16 PASS → {OUT_JSON}")
    if fails:
        print("FAIL nedenleri:")
        for k, r in fails:
            print(f"  - {k}: {r}")
    return 0 if n_pass == 16 else 1


if __name__ == "__main__":
    raise SystemExit(main())
