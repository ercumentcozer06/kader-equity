# GEX strategy registry

This registry separates predictive claims from measurement diagnostics. A failed
test in one row does not reject a different horizon or information set.

## Canonical discretionary playbook (user specification, 2026-07-16)

These are the actual setup families. The generic academic hypothesis “positive
GEX causes fading/reversal” is **not** itself one of these setups.

| ID | Canonical setup | Essential structure | Closest old test | Exact historical test status |
|---|---|---|---|---|
| P1 | Call-wall fade | Rejection/failure at call wall; short with tight invalidation beyond wall; high convexity | `scenario_engine/T2_walls.py::S1` | Partial only: old target was ghost/midpoint and stop was VU; not the canonical risk geometry |
| P2 | Negative-GEX cascade short | Negative GEX regime plus downside cascade/acceptance; short continuation | `intraday_gex_v0.py::R2`, flip failed-reclaim context | Not run exactly; old R2 is an unimplemented stub and flip study is only a context proxy |
| P3 | Forced buying long | Accepted break above call wall; long; narrow stop back behind call wall | `T2_walls.py::S2` | Partial: old entry used 15-minute close; variants forced VU target or EOD/cascade handling, so they do not reject the canonical setup |
| P4 | Forced selling short | Accepted break below put wall; short; narrow stop back above put wall | `T2_walls.py::S4` | Partial for the same reason as P3 |
| P5 | GEX pinning | If neither wall resolves, price is attracted to a selected GEX/ghost/max-pain level | `scenario_engine/T1_ghost.py` | Not tested exactly: T1 is opening-gap-to-ghost only; no unresolved-range condition or competing-target selector |
| P6 | PDH/PDL magnets | Previous-day high/low used as entry trigger and/or target | None found | Not tested |

The next canonical backtest must lock the undefined parts before unblinding:
acceptance/rejection confirmation, entry timestamp, stop buffer, target/exit,
negative-GEX definition, cascade trigger, pin-target selection, PDH/PDL priority,
maximum holding time and same-bar stop/target ordering.

## A. Aggregate GEX regime strategies

| ID | Claim / rule | Information available at trade time | Existing implementation | Status before master rerun |
|---|---|---|---|---|
| A1 | High/positive GEX is standalone long equity alpha | Prior EOD SqueezeMetrics GEX | `screen/candidate_gex.py` | Historically weak as standalone alpha |
| A2 | Deep negative GEX is a fragility state; reduce long exposure | Prior EOD 252-day GEX z-score; trim formula `(1-.5*clip(-z-1,0,3)).clip(.4,1)` | `modules/gex_shield.py`, `screen/candidate_gex.py` | Useful mainly as a tail-risk filter |
| A3 | Binary flip regime: full exposure above zero GEX, reduced exposure below | Prior EOD GEX sign | `backtest/gex_swing/T1_flipgate.py` | Standalone drawdown benefit; stack ablation previously worse |
| A4 | Asymmetric flip gate: reduce more when GEX<0 and trend/TIDE bearish | Prior EOD GEX sign plus PIT trend/TIDE | `backtest/gex_swing/T1_flipgate.py` | Previously tested; must retain model-ablation distinction |
| A5 | Alternative fragility transforms: percentile, low-and-falling momentum, GEX/price z-score | Prior EOD history | `screen/candidate_gex_eng.py` | Candidate family; multiple-testing burden applies |
| A6 | Joint DIX–GEX stress is a stronger shield | Prior EOD DIX and GEX | `screen/candidate_gex_dix.py` | Candidate filter, not clean GEX-only attribution |
| A7 | Deep negative GEX should reverse the portfolio short rather than merely trim | Prior EOD GEX z-score | `backtest/gex_swing/T4_short.py` | High-risk hypothesis; evaluate standalone and in-stack |
| A8 | Delay a fresh long entry up to three days while GEX z<-1 | Prior EOD GEX plus PIT model entry | `backtest/gex_playbook_v0.py` | Execution overlay; opportunity-cost test required |
| A9 | Positive GEX can re-enter long while the main model is flat | Prior EOD GEX sign plus PIT model state | `screen/candidate_reentry.py` | Additive-alpha claim, separate from shielding |

## B. Gamma-feedback / directional strategies

| ID | Claim / rule | Correct test | Existing implementation | Status |
|---|---|---|---|---|
| B1 | Academic regime hypothesis: positive gamma causes intraday reversal; negative gamma causes momentum | `t-1` GEX × same-day early return predicting later return | `backtest/gex_master/literature_intraday.py` | New literature-aligned diagnostic; not the canonical call-wall fade setup |
| B2 | Negative gamma raises subsequent intraday realized volatility | `t-1` GEX predicting later-session RV, controlling early RV | `gex_price_test.py` plus new master harness | Existing test was mostly volatility-level validation; master version fixes horizon |
| B3 | Overnight gap fades in +gamma and follows in -gamma | Prior EOD chain regime; next open gap known; trade open→close | `backtest/remeasure/RC2_battery.py::gamma_txt` | Corrected full-chain sample showed no robust edge |
| B4 | Inverse of textbook gap rule | Same as B3 | `RC2_battery.py::gamma_inv` | Diagnostic falsification; initial apparent edge collapsed after full-chain repair |
| B5 | Always follow/fade the overnight gap | Gap known at open | `RC2_battery.py::hep_mom/hep_rev` | Gamma-free controls, not GEX strategies |
| B6 | Conditional early-return momentum only in deep negative GEX | `t-1 z<=-1`, sign(09:30–10:30), trade 10:30→close | Master harness | New directly tradeable implementation of B1 |
| B7 | Conditional early-return reversal only in high positive GEX | `t-1 z>=+1`, opposite sign(09:30–10:30), trade 10:30→close | Master harness | New directly tradeable implementation of B1 |
| B8 | Regime switch: B6 in negative extremes, B7 in positive extremes | Same as B6/B7 | Master harness | New; neutral zone stays flat |
| B9 | Continuous gamma-feedback sizing | `clip(-z,-1,1) * sign(early return)` | Master harness | Sensitivity, not primary rule |

## C. Flip and option-level strategies

| ID | Claim / rule | Existing implementation | Main limitation |
|---|---|---|---|
| C1 | Spot above gamma flip long / below flip flat | `screen/candidate_flip_directional.py` | Old front-month proxy; about one year; derived sign |
| C2 | Flip-distance z-score controls exposure | `candidate_flip_directional.py` | Level estimation error dominates near flip |
| C3 | Crossing the flip triggers a ±3-day risk trim | `candidate_flip_directional.py` | Event overlap and small sample |
| C4 | Proximity within expected move of flip is a risk state | `candidate_flip_directional.py` | Expected-move and flip both model-derived |
| C5 | Breakdown below flip then reclaim is a long reversal | `backtest/scenario_engine/T3_flip.py` | About one year; must compare with time-of-day-matched control |
| C6 | Breakdown below flip without reclaim continues lower | `T3_flip.py` failed-event diagnostic | Context test, not the same trade as C5 |

## D. Wall / level execution strategies

| ID | Rule | Existing implementation | Required conditioning |
|---|---|---|---|
| D1 | Call-wall rejection: fade after touch/rejection | `backtest/scenario_engine/T2_walls.py` | Gamma regime, entry cutoff, volume confirmation |
| D2 | Call-wall confirmed break: long toward next upper level (VU) | `T2_walls.py` | Correct PIT wall and VU |
| D3 | Call-wall break cascade: hold after VU | `T2_walls.py` | Tail concentration and stop logic |
| D4 | Put-wall rejection: buy bounce | `T2_walls.py` | Same controls as D1 |
| D5 | Put-wall confirmed break: short toward next lower level (VD) | `T2_walls.py` | Same controls as D2 |
| D6 | Put-wall break cascade: hold after VD | `T2_walls.py` | Same controls as D3 |
| D7 | Generic wall touch fades in +gamma, breaks out in -gamma | `RC2_battery.py::M3_setup` | Full-chain corrected sample, approximately one year |
| D8 | Positive gamma and spot between HVL/call wall: fade to HVL | `backtest/intraday_gex_v0.py` | Stub was never run; rule needs unambiguous entry/stop |
| D9 | Negative gamma and break below flip: momentum short | `intraday_gex_v0.py` | Stub was never run; overlaps C6 but has a different entry rule |

## E. Expiry-flow and higher-Greek hypotheses

| ID | Claim | Existing implementation | Evidence/data classification |
|---|---|---|---|
| E1 | Pre-OPEX positive drift is stronger in +gamma | `backtest/gex_swing/T3_opex.py` | Calendar proxy only; not measured vanna |
| E2 | Post-OPEX weakness/rebound differs by gamma regime | `T3_opex.py` | Calendar proxy only; not measured charm |
| E3 | Vanna exposure predicts returns conditional on spot and IV shocks | None with trustworthy history | Requires historical signed chain/participant proxy and intraday IV surface |
| E4 | Charm exposure predicts time-decay hedge flow, especially near expiry | None with trustworthy history | Same requirement; daily EOD is insufficient for intraday timing |
| E5 | 0DTE net gamma changes late-day momentum/reversal | No long-history implementation | Requires intraday 0DTE trades/quotes and robust signing |
| E6 | Vomma/volga, veta, speed, color, zomma, ultima add directional information | No valid local backtest | Exploratory only; preregister after sufficient forward/paid history |

## F. Measurement and falsification tests (not alpha)

- Own ETF chain versus SPX/NDX index-derived regime.
- Front-five-expiry live-match versus full-surface all-expiry regime.
- SqueezeMetrics sign agreement, especially on large absolute-GEX days.
- IV hygiene, zero bid/crossed market removal, DTE concentration and strike-band sensitivity.
- Negative GEX versus realized-volatility relationship.
- Naive call-minus-put OI gamma versus truly signed dealer inventory. The former is a proxy and must never be labelled observed dealer GEX.

## Locked evaluation order

1. B1/B2/B6–B9 on the 1,400-day minute-bar overlap (primary missing test).
2. A1–A9 on the 2011–2026 aggregate series and the frozen-stack ablations.
3. B3–B5 and D7 on the corrected full-chain panel.
4. C1–C6 and D1–D6 on the option-level panel with matched controls.
5. E1–E2 as calendar proxies, clearly labelled; E3–E6 only after adequate data exists.

Primary decisions use untouched date splits, HAC inference, 5-day block bootstrap,
1/2/5 bps cost sensitivity, and family-level Benjamini–Hochberg correction. QQQ
using SPX aggregate GEX is a correlated sensitivity check, not a second independent
replication.
