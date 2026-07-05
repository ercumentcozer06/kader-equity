"""
RC2.7 — HAVUZ ORANI + INDEX-vs-OWN BAYRAK UYUMU (TEŞHİS-ONLY, P&L YOK).

(a) FULL-SURFACE gamma_dollar oranı: INDEX/ETF (SPX/SPY, NDX/QQQ) günlük seri
    (ortak tarihler, PANEL_START..PANEL_END) -> medyan / p5 / p95 + ilk-son değer.
(b) INDEX-bayrak (SPX-full regime) vs OWN-bayrak (SPY-full regime) günlük
    işaret-uyum %'si + çelişen-gün sayısı (ve NDX-vs-QQQ).

Tüm sabitler config.py'den; hardcode yok. Çıktı: RC2_pool.json (config_sha'lı).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def load_panel(sym: str) -> pd.DataFrame:
    """FULL-SURFACE serisi, pre-registered panel penceresine kırpılmış."""
    df = pd.read_parquet(config.level_path("fullsurface", sym))
    df = df.loc[(df.index >= config.PANEL_START) & (df.index <= config.PANEL_END)]
    return df


def pool_ratio(idx_sym: str, etf_sym: str) -> dict:
    """(a) gamma_dollar oranı INDEX/ETF, ortak tarihlerde."""
    idx = load_panel(idx_sym)
    etf = load_panel(etf_sym)
    common = idx.index.intersection(etf.index)
    r = (idx.loc[common, "gamma_dollar"] / etf.loc[common, "gamma_dollar"]).astype(float)
    return {
        "pair": f"{idx_sym}-full / {etf_sym}-full",
        "n_common_days": int(len(common)),
        "ratio_median": float(np.median(r.values)),
        "ratio_p5": float(np.percentile(r.values, 5)),
        "ratio_p95": float(np.percentile(r.values, 95)),
        "ratio_first": {"date": str(common[0].date()), "value": float(r.iloc[0])},
        "ratio_last": {"date": str(common[-1].date()), "value": float(r.iloc[-1])},
    }


def flag_agreement(idx_sym: str, etf_sym: str) -> dict:
    """(b) INDEX-bayrak vs OWN-bayrak günlük işaret-uyumu (FULL-SURFACE regime)."""
    idx = load_panel(idx_sym)
    etf = load_panel(etf_sym)
    common = idx.index.intersection(etf.index)
    fi = idx.loc[common, "regime"].astype(int)
    fo = etf.loc[common, "regime"].astype(int)
    agree = (fi == fo)
    disagree_dates = [str(d.date()) for d in common[~agree.values]]
    return {
        "pair": f"INDEX-flag {idx_sym}-full vs OWN-flag {etf_sym}-full",
        "n_common_days": int(len(common)),
        "agreement_pct": float(100.0 * agree.mean()),
        "n_disagree_days": int((~agree).sum()),
        "index_pos_share_pct": float(100.0 * (fi > 0).mean()),
        "own_pos_share_pct": float(100.0 * (fo > 0).mean()),
        "disagree_dates": disagree_dates,
    }


def main() -> None:
    pairs = [(v, k) for k, v in config.INDEX_FLAG_MAP.items()]  # [("SPX","SPY"),("NDX","QQQ")]
    out = {
        "config_sha": config.config_sha(),
        "script": "backtest/remeasure/RC2_pool.py",
        "note": "TEŞHİS-ONLY: havuz-oranı + bayrak-uyumu; P&L yok. FULL-SURFACE serileri, "
                f"panel {config.PANEL_START}..{config.PANEL_END}.",
        "a_pool_ratio": {f"{i}/{e}": pool_ratio(i, e) for i, e in pairs},
        "b_flag_agreement": {f"{i}-vs-{e}": flag_agreement(i, e) for i, e in pairs},
    }
    out_path = config.REMEASURE_DIR / "RC2_pool.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
