"""
engine/brief — FAZ-4 motoru: model → STATE → DECISION → TRADE → RISK → günlük BRIEF (saf-sinyal; execution Emir'de).

Akış: kilitli model (tide × COR1M-froth × GEX-shield) yön/konviksiyon verir; canlı gamma/vol state piyasayı okur;
beyin rejime en uygun TEK ifadeyi (directional/options) + vade seçer; trade somut bileti kurar; risk hesaba
ölçekler; brief tek rapor basar. Tazelik kapısı: snapshot bayatsa GÜNCEL-ÇAĞRI-DEĞİL uyarısı (Bible).

Kullanım:  python -m engine.brief            (terminal)
           python -m engine.brief --json     (+ output/kader_equity_brief_YYYYMMDD.json)
           python -m engine.brief --ticker QQQ
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import yaml                                                      # noqa: E402
from config import load_config                                  # noqa: E402
from engine import state as S, decision as D, trade as TR, risk as RK   # noqa: E402

CFG_ACC = ROOT / "engine" / "config_accounts.yaml"


def build_brief(ticker: str = "SPY") -> dict:
    cfg = load_config()
    acc = yaml.safe_load(CFG_ACC.read_text(encoding="utf-8"))
    model, st, meta = S.build_state(cfg, ticker)
    block = meta.get("data_junk") or model.get("overlay_block")    # 6b veri-çöp VEYA H3 overlay fail-safe
    if block:                                          # → TRADE ÜRETME (fail-loud)
        why = "VERİ ÇÖP" if meta.get("data_junk") else f"OVERLAY FAIL-SAFE: {model.get('overlay_block_reason')}"
        meta["block_reason"] = why
        dec = {"direction": model.get("direction", "?"), "conviction": 0.0, "horizon": "—",
               "vehicle": {"class": "stand_aside", "expression": f"{why} → işlem yok", "instrument": None},
               "regime": {}, "rationale": why}
        trd = {"class": "stand_aside", "ticket": None, "note": why}
        rsk = {"action": "STAND-ASIDE", "dollar_risk": 0.0, "note": f"{why} → risk 0"}
    else:
        dec = D.decide(model, st, acc)
        trd = TR.construct(dec, st, acc)
        rsk = RK.size(dec, trd, acc)
    # CANLI KİTAP (delta-one) — RİSK-1: $1000'da opsiyon yerine ETF directional; opsiyon = paper-forward
    lb = acc.get("live_book", {}) or {}
    idxk = meta["index"]
    etf = (lb.get("etf_map", {}) or {}).get(ticker, "SPLG" if ticker == "SPY" else "QQQM")
    live_book = {
        "mode": lb.get("mode", "delta_one"), "etf": etf,
        "position": model.get("position_target"), "direction": model.get("direction"),
        "halt_pct": (lb.get("max_dd_halt_pct", {}) or {}).get(idxk),
        "options_paper": bool(((acc.get("accounts", {}) or {}).get("options", {}) or {}).get("paper_forward")),
        "unlock_usd": lb.get("options_unlock_min_account_usd"),
    }
    # B1/B4: FTMO eval translator (eval.enabled ise) — model exposure → lot + delta-emir + stop/limit
    eval_line, policy_flag = None, None
    ev = (acc.get("accounts", {}) or {}).get("eval", {}) or {}
    if ev.get("enabled"):
        from engine import position_translator as PT
        policy_flag = PT.check_policy(acc)                          # mid-eval eval_pos değişimi → FLAG
        idx_spot = (st.get("spot") or 0) * meta["mult"]            # SPY: spot zaten _SPX index-native (mult=1); QQQ: ETF×41 (prev_close≈spot placeholder, FLAG)
        eval_line = PT.from_config(model_deploy=float(model.get("deploy_fraction", 0.0)),
                                   spot=idx_spot, prev_close=idx_spot, acc_cfg=acc)
    return {"computed_at": datetime.now(timezone.utc).isoformat(), "ticker": ticker,
            "index": meta["index"], "model": model, "state": st, "meta": meta,
            "decision": dec, "trade": trd, "risk": rsk, "live_book": live_book,
            "eval_order": eval_line, "policy_flag": policy_flag}


def _render(b: dict) -> None:
    st, dec, trd, rsk, meta = b["state"], b["decision"], b["trade"], b["risk"], b["meta"]
    mult = meta["mult"]; idx = meta["index"]; spot = st.get("spot")
    print("=" * 76)
    if meta.get("data_junk"):
        print("  ⛔ KADER-EQUITY BRIEF — VERİ ÇÖP: İŞLEM ÜRETİLMEDİ")
        for f in meta.get("data_fails", [])[:5]:
            print(f"    • {f}")
        print("    canlı için: gamma_engine.py + surface_yf.py kapanışta YENİDEN çek.")
    elif meta["stale"]:
        print("  ⚠ KADER-EQUITY BRIEF — VERİ BAYAT: GÜNCEL ÇAĞRI DEĞİL")
        print(f"  • snapshot {meta['snapshot_as_of']} ({meta['snapshot_age_days']}g eski) / model {meta['model_call_status']}")
        print("    canlı için: gamma_engine.py + surface_yf.py çalıştır (snapshot tazele).")
    else:
        print(f"  KADER-EQUITY GÜNLÜK BRIEF — {b['ticker']} (≈{idx})")
    print("=" * 76)
    sp = f"{spot:.2f}" if spot else "—"
    print(f"  PİYASA   spot {sp} (≈{idx} {spot*mult:.0f})" if spot else "  PİYASA   spot —")
    print(f"           gamma {st.get('gamma_regime') or '—'}  | net-GEX {st.get('net_gex_bn')}$bn  "
          f"flip {st.get('gex_flip')}  | call-wall {st.get('call_wall')}  put-wall {st.get('put_wall')}  "
          f"maxpain {st.get('max_pain')}")
    print(f"           exp-move ±{st.get('exp_move_1d')}  | ATM-IV {st.get('atm_iv')}%  skew(RR) {st.get('rr_skew')}  "
          f"term {st.get('ts_ratio')}  | COR1M {st.get('cor1m')}")
    _vrp = st.get('vrp')
    print(f"           VRP {('—' if _vrp is None else f'{_vrp:+.1f}')} vol-puanı "
          f"(impl30g {st.get('atm_iv_30d')}% − rv {st.get('realized_vol')}%)  "
          f"{'→ vol pahalı (prim-sat lehine)' if (_vrp is not None and _vrp>2) else '→ vol ucuz (konveksite lehine)' if (_vrp is not None and _vrp<0) else ''}")
    print("-" * 76)
    print(f"  MODEL    {b['model']['direction']}  (tide {b['model'].get('tide_score'):+.1f}, "
          f"konviksiyon/pozisyon {dec['conviction']:.2f})")
    lbk = b.get("live_book", {})
    if lbk.get("mode") == "delta_one":
        halt = lbk.get("halt_pct")
        print(f"  CANLI    DELTA-ONE → {lbk['etf']}  pozisyon {lbk.get('position')} {lbk.get('direction')}  | "
              f"max-DD halt {halt}% (gap riski = açık kabul) | opsiyon: PAPER-forward (≥${lbk.get('unlock_usd')} unlock)")
        print("-" * 76)
        _pdte = (b.get("trade", {}) or {}).get("ticket", {}) or {}
        print(f"  [PAPER-ENGINE, DTE {_pdte.get('dte', 21)}g≥21] opsiyon ifadesi (canlı DEĞİL, ledger expression bacağı):")
    print(f"  KARAR    yön {dec['direction']} | vade {dec['horizon']} | ARAÇ → {dec['vehicle']['class']}")
    print(f"           {dec['vehicle']['expression']}")
    print("-" * 76)
    t = trd.get("ticket")
    if t is None:
        print(f"  TRADE    — STAND-ASIDE ({trd.get('note')})")
    else:
        print(f"  TRADE    {trd['class']}  [{t.get('instrument') or t.get('structure')}]")
        print(f"           {t.get('levels')}")
        if t.get("rr"):
            print(f"           R:R ≈ {t['rr']}  (giriş {t.get('entry')} / stop {t.get('stop')} / hedef {t.get('target')})")
    print(f"  RİSK     {rsk.get('action')} | havuz {rsk.get('pool','—')} (${rsk.get('account_size_usd','?')}) | "
          f"$risk ≈ {rsk.get('dollar_risk')}")
    print(f"           {rsk.get('note')}")
    if rsk.get("futures_note"):
        print(f"           ({rsk['futures_note']})")
    # FORWARD — sinyal-PnL vs ifade-PnL drag (GÖREV 1): sinyal mi çürüdü, ifade mi kanıyor
    try:
        from validation import ledger as _L
        ds = _L.drag_summary()
        if ds["n_calls"]:
            sp = "—" if ds["cum_signal_pnl"] is None else f"{ds['cum_signal_pnl']*100:+.2f}%"
            ep = "—(ifade yok)" if ds["cum_expression_pnl"] is None else f"{ds['cum_expression_pnl']*100:+.2f}%"
            dg = "—" if ds["cum_drag"] is None else f"{ds['cum_drag']*100:+.2f}%"
            print("-" * 76)
            print(f"  FORWARD  {ds['n_calls']} çağrı | sinyal-PnL {sp} ({ds['n_signal_marked']} işaretli) | "
                  f"ifade-PnL {ep} | drag {dg}")
            if ds.get("n_permanent_gap"):    # M1: KALICI-GAP açıkça görünür (örneklem sessizce küçülmedi)
                print(f"           (KALICI-GAP: {ds['n_permanent_gap']} gün atlandı — 2026-06-23..07-02, machine-off; kurtarılamaz)")
    except Exception:
        pass
    # MODÜL-ATTRIBUTION (GÖREV 6a): m9/m5/m2 skor + 21g-corr + alarm (m9 vektörün ~%56'sı)
    try:
        from validation import attribution as _A
        print(_A.render())
    except Exception:
        pass
    print("=" * 76)
    print("  Saf-sinyal — execution sende. (Placeholder Midas sermayesi; gerçek hesap girilince ölçeklenir.)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="kader-equity günlük trade-advisory brief")
    ap.add_argument("--ticker", default="SPY", choices=["SPY", "QQQ"])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    b = build_brief(a.ticker)
    if not a.quiet:
        _render(b)
    if a.json:
        out = ROOT / "output"; out.mkdir(exist_ok=True)
        f = out / f"kader_equity_brief_{datetime.now(timezone.utc):%Y%m%d}_{a.ticker}.json"
        f.write_text(json.dumps(b, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  JSON → {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
