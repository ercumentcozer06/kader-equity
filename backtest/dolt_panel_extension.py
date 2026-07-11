"""
backtest/dolt_panel_extension — GERCEK mid (DoltHub) ile panel-uzatma v2 + infidelite AYRISTIRMASI.

Alpaca-proxy denemesi (alpaca_panel_extension) iki ikameyi AYNI ANDA yapiyordu (mid→close, OI→hacim)
ve ortusmede RED yedi. DoltHub gercek bid/ask verdigi icin artik ayristirilabilir:
  A) dolt-mid × md-OI     (sadece ortusme): mid-kaynagi kalitesi IZOLE — OI gercek, tek fark mid kaynagi.
     Beklenti: uyum yuksekse dolt kotasyonlari guvenilir; dusukse dolt verisi kirli → B okunmaz.
  B) dolt-mid × alpaca-HACIM (2024-01-18→): kalan tek ikame OI→hacim. Ortusmede md'yi reproduce
     ediyorsa UZATMA gecerli (hacim-agirlik caveat'iyle); etmiyorsa bedava yol resmen kapali.
Seviye formulleri build_level_series._levels_for_day (byte-es), panel/testler spine ile ayni.
→ data/cache/level_series_{sym}_doltvol.parquet (B varyanti)
  & <kader-macro venv python> backtest/dolt_panel_extension.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from build_level_series import _levels_for_day, _daily_spot        # noqa: E402
from alpaca_panel_extension import build_panel_from, panel_line, SPLIT  # noqa: E402

KEY = ["date", "expiration", "strike", "right"]


def build_levels(ch: pd.DataFrame, sym: str) -> pd.DataFrame:
    spot = _daily_spot(sym)
    out = []
    for d, g in ch.groupby("date"):
        dd = pd.Timestamp(d).date()
        if dd not in spot.index:
            continue
        lv = _levels_for_day(g, float(spot[dd]))
        if lv is None:
            continue
        lv["date"] = pd.Timestamp(d); lv["spot"] = float(spot[dd])
        out.append(lv)
    return pd.DataFrame(out).set_index("date").sort_index()


def agree_line(px: pd.DataFrame, md: pd.DataFrame, tag: str) -> None:
    j = px.join(md, how="inner", lsuffix="_px", rsuffix="_md")
    if not len(j):
        print(f"  {tag}: ortusme YOK"); return
    ag = (j["regime_px"] == j["regime_md"]).mean()
    rc = j["net_gex_px"].corr(j["net_gex_md"], method="spearman")
    fd = ((j["flip_px"] - j["flip_md"]).abs() / j["spot_md"]).median()
    cw = ((j["call_wall_px"] - j["call_wall_md"]).abs() / j["spot_md"]).median()
    pw = ((j["put_wall_px"] - j["put_wall_md"]).abs() / j["spot_md"]).median()
    print(f"  {tag} ({len(j)}g): rejim-uyum %{100*ag:.0f}, net_gex rank-corr {rc:+.2f}, "
          f"medyan |Δ|/spot: flip %{100*fd:.2f} cw %{100*cw:.2f} pw %{100*pw:.2f}")


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["strike"] = pd.to_numeric(df["strike"]).astype(float)
    df["right"] = df["right"].astype(str).str[0].str.upper()
    return df


def main() -> int:
    print("DOLT PANEL UZATMASI v2 — gercek-mid ayristirmasi\n")
    for sym in ("SPY", "QQQ"):
        print(f"=== {sym} ===")
        dolt = _norm(pd.read_parquet(ROOT / "data" / "historical_chains" / f"dolt_chain_{sym.lower()}.parquet"))
        dolt = dolt[dolt["mid"] > 0]
        mdch = _norm(pd.read_parquet(ROOT / "data" / "historical_chains" / f"md_{sym.lower()}.parquet"))
        alp = _norm(pd.read_parquet(ROOT / "data" / "historical_chains" / f"alpaca_chain_{sym.lower()}.parquet"))
        alp = alp[alp["volume"] > 0]
        md_lv = pd.read_parquet(ROOT / "data" / "cache" / f"level_series_{sym.lower()}.parquet")
        per = dolt.groupby("date").size()
        print(f"  dolt census: {len(per)} gun ({per.index.min().date()}→{per.index.max().date()}), "
              f"kontrat/gun medyan {int(per.median())}")

        # A) dolt-mid × md-OI — mid-kaynagi kalitesi (ortusme)
        a = dolt.merge(mdch[KEY + ["open_interest"]], on=KEY, how="inner")
        lv_a = build_levels(a, sym)
        agree_line(lv_a, md_lv, "A dolt-mid × md-OI   ")

        # B) dolt-mid × alpaca-hacim — uzatilabilir kombo
        b = dolt.merge(alp[KEY + ["volume"]], on=KEY, how="inner").rename(columns={"volume": "open_interest"})
        lv_b = build_levels(b, sym)
        lv_b.to_parquet(ROOT / "data" / "cache" / f"level_series_{sym.lower()}_doltvol.parquet")
        agree_line(lv_b[lv_b.index >= SPLIT], md_lv, "B dolt-mid × alp-hacim")

        print("  PANEL (M1/M2/M3, double-confirm):")
        panel_line(build_panel_from(md_lv, sym), "ortusme-GERCEK(md)")
        panel_line(build_panel_from(lv_a, sym), "ortusme-A(mid-izole)")
        panel_line(build_panel_from(lv_b[lv_b.index >= SPLIT], sym), "ortusme-B(hacim)")
        panel_line(build_panel_from(lv_b[lv_b.index < SPLIT], sym), "UZATMA-B-2024→25")
        print()
    print("  Okuma: A-satiri md'ye yakinsa dolt kotasyonu saglam (mid sorunu cozuldu). B ortusmede")
    print("  GERCEK'i reproduce ediyorsa uzatma satiri gecerli; etmiyorsa kalan fark = OI→hacim ikamesi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
