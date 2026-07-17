# GEX master audit and backtest report

Run date: 2026-07-16. This report inventories every local GEX-related strategy,
including the Claude-era work, and separates a predictive trade from a risk
measurement or a data-quality diagnostic.

## Playbook correction

The canonical trade playbook is: call-wall fade, negative-GEX cascade short,
forced buying above the call wall, forced selling below the put wall, unresolved
range GEX/ghost/max-pain pinning, and PDH/PDL magnets. The phrase “positive-GEX
fade” elsewhere in this audit names an academic aggregate-regime hypothesis. It is
not a seventh discretionary setup and must not be confused with call-wall fade.

The Claude-era scenario engine did **not** test the full canonical playbook. It
tested modified proxies:

- Call-wall rejection entered on a 15-minute close back below the wall, targeted
  ghost/midpoint, and stopped at VU.
- Forced buying/selling entered on a 15-minute confirmation close, then either
  forced VU/VD as target or used an EOD cascade variant.
- Ghost testing was an opening gap-to-ghost trade, not a “neither wall resolved”
  pinning state.
- No exact negative-GEX cascade or PDH/PDL strategy exists in the old harness.

Consequently, proxy failures below reject those exact old implementations only;
they do not reject the canonical setups. A new test requires locked entry, stop,
target, confirmation and same-bar ordering rules for all six setups.

That canonical remeasurement has now been completed. See `CANONICAL_SPEC.md` and
`CANONICAL_REPORT.md`. In brief: forced selling is the best forward candidate but
is not statistically confirmed; the locked call-wall fade and nearest-level
pinning rules are negative; negative-GEX cascade is adverse but severely
underpowered; forced buying is neutral; PDH/PDL acceptance is more promising than
mechanical rejection/fade, especially on QQQ downside breaks.

## Executive decision

The current evidence does **not** support GEX as a standalone directional trading
signal. It does support GEX as a volatility/fragility variable. The cleanest new
test—prior-EOD GEX interacting with the first trading hour—does not show a robust
SPY momentum/reversal edge. Negative GEX does, however, predict higher subsequent
intraday realized volatility even after early-session volatility is controlled.

Current production implication:

- Keep GEX as a trim/risk-context input, with conservative attribution.
- Do not turn deep negative GEX into an automatic short.
- Do not activate flip-direction, the old VU/VD-target implementations,
  flip-reclaim, positive-GEX re-entry, or continuous regime-switch strategies.
- Preserve the daily full option-chain archive. Exact signed GEX, vanna, charm and
  0DTE hypotheses still need longer or paid historical chain/order-flow data.

## Data and point-in-time discipline

| Dataset | Coverage used | Honest use |
|---|---:|---|
| SqueezeMetrics SPX DIX/GEX | 2011-05-02–2026-06-08, 3,798 EOD rows | Long-regime swing tests and `t-1` market-gamma proxy |
| SPY Alpaca 1-minute RTH | 2020-09-01–2026-06-10; 1,334 complete sessions after bucket-quality filters | Primary intraday feedback test |
| QQQ Alpaca 1-minute RTH | 2020-09-03–2026-06-10; 906 complete sessions | Correlated sensitivity, not independent replication |
| Corrected full-chain level panel | 2025-06-13–2026-06-08; 235–237 usable days | Flip/wall/gap tests; short single-regime history |
| Daily raw CBOE archive | Current forward collection | Future exact chain remeasurement; not a historical backfill |

For the new intraday test, the GEX observation date is strictly earlier than the
traded session. First-hour information ends at 10:30 ET; positions then earn the
10:30–16:00 return. The code asserts the strict date inequality. Half days and
sessions with incomplete fixed buckets are removed.

The fixed evaluation uses HAC(5), 5-day circular block bootstrap, 0/1/2/5 bps
round-trip cost sensitivity, and these date splits:

- Discovery: through 2023-12-31.
- Validation: 2024.
- Test: 2025-01-01 onward.

## 1. Literature-aligned intraday gamma feedback

Primary regression:

`later_return = a + b*early_return + c*GEX_z(t-1) + d*early_return*GEX_z(t-1) + early_RV`

The theory predicts `d < 0`: high gamma should promote reversal and low gamma
should promote continuation.

| Asset / future window | Interaction beta | HAC t | raw p | BH-FDR p | Decision |
|---|---:|---:|---:|---:|---|
| SPY 10:30–16:00 | -0.0306 | -0.39 | 0.6946 | 0.7118 | No directional evidence |
| SPY 15:00–16:00 | -0.0358 | -1.14 | 0.2548 | 0.3821 | No directional evidence |
| QQQ 10:30–16:00 | -0.0260 | -0.37 | 0.7118 | 0.7118 | No directional evidence |
| QQQ 15:00–16:00 | -0.0603 | -2.21 | 0.0273 | 0.0545 | Nominal only; fails family FDR |

The sign is theory-consistent in all four regressions, but effect size and
precision are inadequate. More importantly, the SPY extreme-cell slopes are all
positive: negative extreme +0.171 (`t=0.88`), neutral +0.064 (`t=0.98`), positive
extreme +0.201 (`t=1.25`). Positive GEX did not create the predicted reversal.

### Trade implementations, untouched 2025–2026 test period

Numbers below include 2 bps round-trip cost. Mean bps is averaged over every test
session, including flat days; active mean is per trade day.

| Rule | Asset | Active days | Mean/day | Active mean | Sharpe | t | 5-day bootstrap 95% mean CI | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Negative-extreme momentum (`z<=-1`) | SPY | 67 | +2.33 bp | +11.94 bp | +0.55 | +0.65 | [-1.71,+7.76] bp | Interesting OOS cell, not confirmed |
| Positive-extreme reversal (`z>=+1`) | SPY | 66 | -0.25 bp | -1.32 bp | -0.20 | -0.23 | [-2.00,+1.85] bp | Reject |
| Extreme regime switch | SPY | 133 | +2.07 bp | +5.36 bp | +0.47 | +0.55 | [-2.45,+7.64] bp | Not confirmed |
| Continuous regime switch | SPY | 344 | -1.34 bp | -1.34 bp | -0.29 | -0.34 | [-6.82,+4.91] bp | Reject |
| Negative-extreme momentum | QQQ | 48 | +1.13 bp | +6.62 bp | +0.22 | +0.23 | [-4.63,+8.33] bp | Not confirmed |
| Positive-extreme reversal | QQQ | 58 | -0.03 bp | -0.14 bp | -0.02 | -0.02 | [-2.71,+3.21] bp | Reject |
| Extreme regime switch | QQQ | 106 | +1.10 bp | +2.92 bp | +0.20 | +0.21 | [-5.46,+8.66] bp | Not confirmed |
| Continuous regime switch | QQQ | 281 | -3.52 bp | -3.53 bp | -0.60 | -0.63 | [-11.50,+5.03] bp | Reject |

The SPY negative-gamma momentum cell is the only directional result worth forward
monitoring, but it is not deployable evidence. Its block-bootstrap interval crosses
zero, and the combined switch was negative in discovery (-2.58 bp/day) and
validation (-3.13 bp/day) before becoming positive in the test period. That is
regime instability, not a stable backtest pass.

## 2. Volatility and fragility channel

Regression: `log(rest-of-session RV) ~ GEX_z(t-1) + log(first-hour RV)`.

| Asset | GEX beta | HAC t | raw p | BH-FDR p | Decision |
|---|---:|---:|---:|---:|---|
| SPY | -0.0334 | -3.22 | 0.00130 | 0.00390 | Pass |
| QQQ | -0.0466 | -3.66 | 0.000254 | 0.00152 | Correlated confirmation |

Lower prior-EOD GEX predicts higher later-session realized volatility. This is the
strongest result in the whole audit and agrees with the older price test: the
corrected own-chain proxy had next-day realized volatility near 1.0% in negative
gamma versus 0.6% in positive gamma, while the professional aggregate series had
about 1.5% versus 0.7% in its much smaller negative-GEX cell.

The 2011–2026 leading-fragility test also survives a more honest formulation:

- GEX z-score adds about 2.1% incremental R-squared for forward 21-day drawdown
  beyond trailing realized volatility.
- HAC t is 2.86; non-overlapping partial correlation is 0.229.
- It fired before all 20 worst forward-drawdown events, but was earlier than the
  realized-vol trigger in only 8 of 16 events where both fired.
- Verdict: real but small marginal risk information, not independent alpha.

## 3. Aggregate swing and portfolio overlays

### Standalone exposure controls, 2011–2026

| Variant | Sharpe | Max DD | Cumulative return | Decision |
|---|---:|---:|---:|---|
| Always long | 0.737 | -33.9% | +444.1% | Benchmark |
| Existing z-shield | 0.757 | -27.4% | +380.7% | Risk/return trade-off |
| Binary negative-GEX floor 0.4 | 0.790 | -21.2% | +364.6% | Strongest standalone drawdown control |
| Binary floor 0.5 | 0.788 | -21.8% | +379.1% | Similar |
| Binary floor 0.6 | 0.782 | -24.0% | +393.2% | Similar |
| GEX + 200-day-trend asymmetric floor | 0.800 | -22.0% | +386.5% | Best standalone Sharpe, but model overlap matters |

These are exposure transformations on a long equity premium, not proof that GEX
predicts return direction. Standalone high-GEX long/flat Sharpe was only 0.54 for
SPX and 0.72 for NDX; GEX z-score IC was negative at 1- and 21-day horizons.

### Frozen-stack ablation, 2019 onward

| Asset | Existing stack | Binary flip gate | Asymmetric gate | Decision |
|---|---:|---:|---:|---|
| SPX Sharpe / max DD | 1.636 / -13.2% | 1.593 / -16.5% | 1.585 / -16.8% | Keep existing stack |
| NDX Sharpe / max DD | 1.773 / -15.6% | 1.696 / -19.7% | 1.692 / -20.0% | Keep existing stack |

Alternative z, percentile, momentum, and GEX/price regime transformations all
failed the strict dual-asset FDR rule. GEX+DIX stress also failed. The existing
primary z-shield improved raw Sharpe and drawdown versus bare TIDE, but its strict
FDR pass was false (SPX superiority probability 88%, NDX 95%). It should be treated
as a conservative risk choice rather than newly discovered alpha.

### Short, delay, and re-entry rules

- Automatic shorting in deep negative GEX is rejected. At `z<=-1`, full-short
  Sharpe was 0.34 versus 0.74 always-long and 0.78 shield; in the frozen SPX stack
  it reduced Sharpe from 1.64 to 0.97. Flat was consistently safer than short.
- Delaying a new TIDE long entry for up to three days when `z<-1` changed only eight
  days and raised SPX/NDX Sharpe by +0.07/+0.03, but did not improve the prop-eval
  pass or kill rates. Keep as research-only.
- Positive-GEX re-entry while TIDE is flat is rejected: Sharpe delta was -0.41 SPX
  and -0.45 NDX, with about 1% superiority probability.

## 4. Corrected full-chain gap/regime battery

This panel uses every expiry rather than the originally truncated single-expiry
measurement. There are 235–236 daily observations. No P&L member achieves robust,
stable evidence after 20-trial DSR accounting.

- Textbook gap rule (+gamma fade / -gamma follow): SPY Sharpe from -1.33 to -0.06
  depending on flag; QQQ from -1.27 to +0.34. Unstable and rejected.
- Inverse gamma rule: SPY -0.71 to +0.57; QQQ -0.89 to +0.72. Unstable and rejected.
- Gamma-free gap controls also vary by instrument and holdout; they explain why an
  apparent gamma rule can be a generic gap effect.
- Wall touch fade/breakout (`M3_setup`): index-flag Sharpe -1.96 SPY and -2.36 QQQ.
  Rejected.

This supersedes the original single-expiry result. That original calculation
discarded roughly 89–90% of gamma exposure; the corrected result is the valid one.

## 5. Flip strategies

### Spot versus derived flip

The 243-day front-month proxy is rejected as directional alpha. The long/flat rule
did not beat buy-and-hold, and every overlay failed strict FDR. SPX stack deltas
ranged from -0.01 to -1.61 Sharpe; NDX had isolated small positives but no dual-asset
pass. No live directional module was created.

### Flip breakdown and reclaim

| Asset | Reclaims | Net return to close | t (gross) | Time-matched excess | Decision |
|---|---:|---:|---:|---:|---|
| SPY | 51 | -5.35 bp | -0.78 | -3.15 bp gross | Reject |
| QQQ | 36 | +9.19 bp | +1.24 | +10.70 bp gross | Interesting, underpowered |
| Pooled, correlated | 87 | +0.67 bp | +0.59 | — | No pass |

The QQQ cell is not enough to deploy: only 36 events, no independent OOS period,
and SPY has the opposite sign. Failed reclaim/breakdown events were negative to the
close in both assets, but t statistics were only about -1.3/-1.4 overall.

## 6. Old wall and next-level proxy strategies

All values use prior-EOD levels and 2 bps cost.

| Setup | SPY net / t | QQQ net / t | Decision |
|---|---:|---:|---|
| Call-wall rejection fade | +5.3 bp / +0.50, n=13 | +4.4 bp / +0.47, n=17 | Too few; no evidence |
| Call-wall break to forced VU target | -15.6 / -3.16, n=51 | -22.1 / -4.74, n=59 | Strong reject of this target rule; not canonical forced buying |
| Call-wall break EOD/cascade variant | -7.5 / -0.97 | +2.1 / +0.24 | No evidence; not an exact canonical test |
| Put-wall rejection bounce | -9.6 / -1.06, n=21 | -0.7 / -0.06, n=20 | Reject |
| Put-wall break to forced VD target | -20.0 / -6.58, n=99 | -31.9 / -7.05, n=95 | Strong reject of this target rule; not canonical forced selling |
| Put-wall break EOD/cascade variant | -1.3 / -0.22 | -2.2 / -0.26 | No evidence; not an exact canonical test |

The apparently high probability of reaching VU/VD after a confirmed break is not
the same as profitable entry-to-target execution. Entries occur after confirmation,
and the fixed target/stop geometry loses heavily. A separate volume-gamma artifact
battery confirms that naive breakout profits are mostly generic gap/breakout
behavior: placebo walls perform at least as well and gamma-label permutations do
not remove the effect. This says the old VU/VD geometry is bad; it does not decide
the high-convexity wall-break trade with the user's intended narrow invalidation.

## 7. OPEX, vanna, charm and higher Greeks

The existing OPEX study is **not** a vanna/charm measurement. It is a calendar
proxy conditioned on aggregate GEX:

- Standalone overlay Sharpe 0.68 versus 0.74 buy-and-hold; it loses in all seven
  reported subperiod comparisons.
- Frozen-stack Sharpe moves SPX 1.64→1.67 and NDX 1.77→1.81, with unchanged max
  drawdown. About 72% of affected post-OPEX/negative-GEX days were already reduced
  by the existing shield.
- Verdict: TIDE/stack absorbed; no new independent calendar sleeve.

No honest local history currently supports E3–E6: signed vanna exposure, charm
decay flow, 0DTE gamma, or vomma/veta/speed/color/zomma/ultima directional alpha.
The live engine can calculate several Greeks from current chains, but calculating a
Greek is not equivalent to observing dealer position sign or having a backtestable
history. These hypotheses remain open rather than rejected.

## 8. Status of every strategy family

| Family | Final status | What remains useful |
|---|---|---|
| Aggregate GEX standalone direction | Rejected | None as alpha |
| Negative-GEX fragility trim | Supported as risk information; strict alpha FDR fails | Conservative exposure reduction |
| Binary/asymmetric flip gates | Standalone protection, rejected in current stack | Diagnostic only |
| Deep-negative automatic short | Rejected | Flat/trim instead |
| Entry delay | Weak/research-only | Forward monitor without deploying |
| Positive-GEX re-entry | Rejected | None |
| Intraday feedback regime switch | Rejected currently | Monitor negative-GEX momentum cell forward |
| GEX → subsequent volatility | Passed | Sizing, risk limits, expected range |
| Gap × gamma regime | Rejected | Generic gap controls |
| Spot/flip direction and proximity | Rejected | Descriptive market map |
| Flip reclaim | Mixed/underpowered | Forward event archive |
| Canonical call-wall fade | Not tested exactly; old proxy underpowered | Rebuild with canonical stop/target |
| Canonical negative-GEX cascade short | Not tested | Requires locked trigger and exit |
| Canonical forced buying/selling | Not tested exactly | Rebuild around wall invalidation; old forced VU/VD targets rejected |
| Canonical unresolved-range pinning | Not tested | Need target selector across GEX/ghost/max-pain |
| Canonical PDH/PDL magnets | Not tested | Add prior-day levels to PIT panel |
| Old VU/VD target breakouts | Strongly rejected | None under that execution rule |
| Old wall EOD/cascade variants | Rejected/no evidence | Do not equate with canonical cascade setup |
| OPEX × gamma proxy | Absorbed by existing stack | Context only |
| Vanna/charm/0DTE/higher Greeks | Not yet testable honestly | Continue full-chain collection or buy history |

## Reproducibility

- Full strategy registry: `backtest/gex_master/STRATEGY_REGISTRY.md`
- New primary harness: `backtest/gex_master/literature_intraday.py`
- Machine-readable output: `backtest/gex_master/results/literature_intraday.json`
- Saved PIT panels: `backtest/gex_master/results/panel_spy.parquet` and
  `panel_qqq.parquet`
- Existing rerun outputs: `backtest/gex_swing/results`,
  `backtest/remeasure/RC2_battery_results.json`, and
  `backtest/scenario_engine/results`

All listed executable families were rerun successfully in the current environment.
`backtest/intraday_gex_v0.py` remains a historical stub and was not promoted into a
result: its two ideas are already tested more precisely by the scenario-engine wall
and flip-break families, while its promised CPI/FOMC-filtered prop-simulation data
loader is still unimplemented. Treating its synthetic helper tests as a real
backtest would be false precision.
