"""
RC3_assert.py — RC3 SENTEZ ön-koşul ASSERT'i.
Kontrol: backtest/remeasure/RC2_*.json + data/cache/level_series_{livematch,fullsurface}_*.meta.json
config_sha alanlari HEPSI ayni mi ve == config.config_sha()?
Cikti: RC3_assert.json (config_sha metadata dahil). Tutarsizlik varsa exit 1 + dosya-dosya rapor.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

EXPECTED = config.config_sha()


def main() -> int:
    rows = []
    ok = True

    rc2_files = sorted(glob.glob(str(config.REMEASURE_DIR / "RC2_*.json")))
    meta_files = sorted(
        glob.glob(str(config.CACHE / "level_series_livematch_*.parquet.meta.json"))
        + glob.glob(str(config.CACHE / "level_series_fullsurface_*.parquet.meta.json"))
    )

    for f in rc2_files + meta_files:
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            sha = d.get("config_sha", "<MISSING>")
        except Exception as e:  # fail-loud per-file
            sha = f"<READ-ERROR: {e}>"
        match = sha == EXPECTED
        ok = ok and match
        rows.append({"file": str(Path(f).relative_to(config.ROOT)), "config_sha": sha, "match": match})

    out = {
        "config_sha": EXPECTED,
        "script": "backtest/remeasure/RC3_assert.py",
        "n_rc2": len(rc2_files),
        "n_meta": len(meta_files),
        "all_match": ok,
        "files": rows,
    }
    (config.REMEASURE_DIR / "RC3_assert.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"expected config_sha = {EXPECTED}")
    print(f"checked: {len(rc2_files)} RC2_*.json + {len(meta_files)} .meta.json")
    if ok:
        print("ASSERT PASS: tum config_sha alanlari ayni ve config ile esit.")
        return 0
    print("ASSERT FAIL — tutarsiz dosyalar:")
    for r in rows:
        if not r["match"]:
            print(f"  {r['file']}: {r['config_sha']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
