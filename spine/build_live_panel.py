"""
spine/build_live_panel — CANLI m0..m11 panelini AYRI SUBPROCESS'te üretir (kalıcı izolasyon).

NEDEN (2026-07-07 kalıcı çözüm): reconstruct_live eskiden kader-macro'yu ANA process'e in-process import
ediyordu (`_build_live_panel` → sys.path.insert(macro_repo) + `from backtest import data_loader` →
macro'nun `from modules import _fred`'i). Ana process equity'nin KENDİ `modules` paketini zaten import
etmiş olduğundan (run.py:42 `from modules.opex_calendar import ...`), macro'nunkini gölgeliyordu →
ImportError → 46 GÜN sessiz frozen-fallback (bkz [[kader_equity_modules_shadow_stale_2026_07_07]]).

Bu script macro işini AYRI bir interpreter'da koşar: child'ın sys.modules'ü temiz başlar, sys.path[0]'a
macro_repo sokulunca `from modules import _fred` DOĞRUDAN macro'ya çözülür — gölge YAPISAL OLARAK imkansız.
Bu, gold/silver'ın ZATEN kullandığı kanıtlı desen (kader-gold/run_daily.py:58,62 build_panel.py subprocess).

Çıktı: <out_json> dosyasına {as_of, columns=MODS, index[], data[][], input_stale[]} yazar. Panel = mevcut
_build_live_panel'in AYNISI (aynı run_pit_signals + _inject_m3_m6) → BYTE-AYNI tide (Sharpe değişmez).
Fail-closed: herhangi bir hata → traceback stderr'e + exit 1 → parent bunu STALE sayar (asla sessiz-frozen).

Kullanım (reconstruct_live bağladığında):  python spine/build_live_panel.py <macro_repo> <out_json>
Bu dosya bağlanana kadar İNERT'tir (hiçbir yerden import/exec edilmez).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# script olarak koşulunca equity ROOT'u yola ekle → `from spine.reconstruct import ...` çözülsün.
# (spine/__init__.py boş; reconstruct.py tepe-seviyede `modules` import ETMEZ → child temiz kalır.)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("kullanım: build_live_panel.py <macro_repo> <out_json>", file=sys.stderr)
        return 2
    macro_repo = Path(argv[0])
    out_json = Path(argv[1])

    # Mevcut in-process mantığın AYNISINI çağır (kod tekrarı yok → byte-parity tanım gereği).
    import pandas as pd  # noqa: E402
    from spine.reconstruct import _build_live_panel, MODS  # noqa: E402

    df, input_stale = _build_live_panel(macro_repo)         # macro import BURADA, temiz child'da → gölge yok
    df = df[MODS]
    payload = {
        "as_of": str(df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1]),
        "columns": list(MODS),
        "index": [str(ix.date() if hasattr(ix, "date") else ix) for ix in df.index],
        "data": [[None if pd.isna(v) else float(v) for v in row] for row in df.values],
        "input_stale": input_stale,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()                               # fail-closed: parent exit!=0 görür → STALE damgalar
        raise SystemExit(1)
