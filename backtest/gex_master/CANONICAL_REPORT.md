# Canonical GEX playbook backtest

Run date: 2026-07-16. Exact rules are frozen in `CANONICAL_SPEC.md`; machine
results are in `results/canonical_results.json` and every simulated variant fill is
in `results/canonical_trades.parquet`.

## Scope and limitations

- Option-level PIT panel: 2025-06-16–2026-06-10, approximately 237 sessions.
- SPY and QQQ one-minute RTH paths.
- All option levels come from the preceding EOD row `D`; trades occur on `N>D`.
- Five-minute signals, fill at next one-minute open, conservative stop-first rule.
- Primary round-trip cost 2 bps; 1/5 bps sensitivity.
- Train/holdout boundary: 2026-02-02.
- 3,000 five-session block-bootstrap samples per reported cell.
- Full output contains 5,296 variant trade records.

This history is sufficient to falsify badly performing execution definitions, but
not to establish a stable positive edge across market regimes. The rules were
written after related old proxy outputs had been observed. The chronological
holdout is useful robustness evidence, not a pristine never-seen test.

“GEX level” in the pinning setup is interpreted as gamma flip. If the intended
level is different, that definition needs a separate rerun.

## Primary results

All means below are net of 2 bps. `PF` is profit factor. A confidence interval
crossing zero means the observed positive average is not statistically confirmed.

| Setup | Asset | Full n | Full mean | t | PF | Block-bootstrap 95% CI | Holdout n / mean | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Call-wall fade, 2R | SPY | 42 | -4.46 bp | -2.36 | 0.44 | [-7.55,-1.14] | 19 / -6.50 | Reject exact rule |
| Call-wall fade, 2R | QQQ | 36 | -6.05 | -1.57 | 0.49 | [-14.43,+2.29] | 9 / -20.44 | Reject exact rule |
| Negative-GEX cascade, index regime, 3R | SPY | 14 | -12.63 | -1.35 | 0.44 | [-24.10,-0.27] | 9 / -11.92 | Very small n; adverse |
| Negative-GEX cascade, index regime, 3R | QQQ | 8 | -27.14 | -5.61 | 0.00 | n<10 | 7 / -23.91 | Very small n; 0 winners |
| Forced buying, 2R | SPY | 36 | -1.18 | -0.54 | 0.82 | [-5.75,+3.54] | 17 / -3.83 | No edge |
| Forced buying, 2R | QQQ | 36 | -1.32 | -0.46 | 0.83 | [-6.06,+4.24] | 9 / +7.06 | Unstable/underpowered |
| Forced selling, 2R | SPY | 72 | +1.52 | +0.62 | 1.20 | [-3.36,+6.56] | 29 / +0.02 | Candidate, not confirmed |
| Forced selling, 2R | QQQ | 63 | +3.54 | +0.90 | 1.37 | [-4.49,+12.91] | 26 / +5.92 | Best canonical cell, not confirmed |
| Unresolved pinning, nearest target 11:00 | SPY | 87 | -4.20 | -1.70 | 0.66 | [-8.72,+0.68] | 30 / -1.14 | Reject primary selector |
| Unresolved pinning, nearest target 11:00 | QQQ | 76 | -1.58 | -0.39 | 0.89 | [-7.34,+4.48] | 31 / +5.84 | Unstable; no pass |
| PDH acceptance long, 2R | SPY | 97 | -1.13 | -0.86 | 0.82 | [-4.18,+2.14] | 35 / +0.07 | No edge |
| PDH acceptance long, 2R | QQQ | 86 | +2.88 | +1.24 | 1.39 | [-1.84,+7.81] | 36 / -0.25 | In-sample only |
| PDH rejection short, 2R | SPY | 94 | -3.99 | -2.61 | 0.54 | [-7.04,-1.00] | 30 / -6.81 | Reject 2R fade |
| PDH rejection short, 2R | QQQ | 90 | -2.17 | -0.83 | 0.79 | [-6.08,+2.38] | 34 / -4.84 | Reject 2R fade |
| PDL acceptance short, 2R | SPY | 68 | -1.82 | -0.77 | 0.79 | [-6.51,+3.70] | 35 / -2.43 | No edge |
| PDL acceptance short, 2R | QQQ | 61 | +3.33 | +0.79 | 1.30 | [-4.36,+11.51] | 25 / +2.75 | Candidate, underpowered |
| PDL rejection long, 2R | SPY | 70 | -4.34 | -2.15 | 0.52 | [-9.04,+0.35] | 37 / -4.84 | Reject 2R fade |
| PDL rejection long, 2R | QQQ | 64 | -3.28 | -1.18 | 0.71 | [-8.25,+1.93] | 31 / -0.47 | Reject/no edge |

No positive primary cell passes family-level Benjamini–Hochberg correction. This
is not “there are no interesting cells”; it means none currently clears the bar
for an independently supported live rule.

## P1 — Call-wall fade

Definition: call-wall touch plus five-minute close back inside by `0.02×EM`, next
minute short, stop `0.10×EM` above wall.

The 2R version is negative in both assets. SPY is especially clear: 28 stops,
11 targets and 3 EOD exits; block bootstrap is entirely negative. QQQ records 27
stops against 9 targets. The damage is concentrated in early signals:

- QQQ 09:xx: -15.9 bp, `n=14`, `t=-3.41`.
- SPY 09:xx: -6.7 bp, `n=15`.
- Conditioning on positive index GEX does not rescue it: SPY +gamma cell -4.9 bp,
  QQQ +gamma -5.6 bp.

Alternative exits also fail to produce a repeatable cross-asset result:

- SPY EOD -2.31 bp; max-pain +0.58; ghost +0.66 with only 17 events.
- QQQ EOD -5.57; max-pain -8.15; ghost -7.25.
- Equal-weight active-symbol portfolio: -6.47 bp over 66 event days; block CI
  [-10.76,-2.32]. Holdout: -11.48 bp.

Decision: the locked five-minute rejection with a `0.10×EM` wall stop is rejected.
This does not prove every discretionary call-wall fade is invalid; it says the
tested mechanical confirmation is stopped too often to deliver convexity.

## P2 — Negative-GEX cascade short

Definition: negative index gamma, accepted put-wall break, then another five-minute
close below the break-bar low. Stop above put wall; primary target 3R.

- SPY: `n=14`, 11 stops, 2 targets, 1 EOD; -12.63 bp.
- QQQ: `n=8`, 7 stops, 1 EOD and no winner; -27.14 bp.
- Own-regime and 2R variants produce isolated positive cells, but samples are only
  8–18 events and flip sign between train/holdout or target choices.

Decision: insufficient event count for a universal conclusion, but the specified
two-stage cascade confirmation is demonstrably poor in the available sample.
Continue archiving; do not deploy this implementation.

## P3 — Forced buying

Definition: five-minute accepted close above call wall, next-minute long, stop
`0.10×EM` behind call wall.

| Exit | SPY full mean | QQQ full mean | Important holdout observation |
|---|---:|---:|---|
| 1R | -0.18 bp | -2.37 bp | Both weak |
| 2R primary | -1.18 | -1.32 | SPY -3.83; QQQ +7.06 but only 9 |
| 3R | -0.84 | -0.21 | QQQ holdout +6.99, only 9 |
| EOD | -3.67 | +1.52 | QQQ holdout +13.74, only 9 |
| PDH target | -1.91, n=11 | +4.64, n=15 | Too few |
| VU target | -0.73 | -2.21 | No support |

Across both active symbols, the 2R rule averages -0.56 bp over 64 event days; its
holdout average is +0.30 bp over only 24 days. QQQ has a favorable late subperiod,
but SPY does not replicate it and the sample is tiny.

Decision: not rejected as strongly as call-wall fade, but there is no full-sample
edge. QQQ forced-buying is a forward-monitor candidate, not a live strategy.

## P4 — Forced selling

Definition: accepted five-minute close below put wall, next-minute short, stop
`0.10×EM` above put wall.

This is the strongest family:

- SPY 2R: +1.52 bp, `n=72`, PF 1.20; holdout approximately flat.
- QQQ 2R: +3.54 bp, `n=63`, PF 1.37; holdout +5.92 bp.
- QQQ 1R/2R/3R are all positive; SPY 2R/3R are positive.
- QQQ performs mainly before noon: 09:xx +5.2 bp, 10:xx +9.0, 11:xx +3.3;
  later cells are negative and very small.
- Negative gamma is not necessary in this sample. QQQ index -gamma averages
  +6.8 bp but +gamma still +2.7; SPY -gamma +0.8 and +gamma +2.1.

Reasons it still fails promotion:

- Bootstrap intervals cross zero for both assets.
- Equal-weight active-symbol portfolio: +1.57 bp, `t=0.69`, CI [-3.07,+7.17].
- Holdout portfolio: +0.87 bp, `t=0.23`.
- At 5 bps cost, SPY becomes -1.48 bp and QQQ falls to +0.54.
- Top-three winners exceed total net profit, meaning the remaining trades lose in
  aggregate. The apparent edge is tail-dependent.

Decision: best forward candidate. Preserve exactly this 2R definition and collect
more data without retuning. It is not statistically ready for live risk.

## P5 — Unresolved-range pinning

Definition: neither wall touched before 11:00, then trade toward the nearest
eligible flip/ghost/max-pain target inside the wall range, with `0.50×EM` stop.

- SPY primary: -4.20 bp, PF 0.66; 53 stops, 25 targets, 9 EOD.
- QQQ primary: -1.58 bp, PF 0.89; 43 stops, 20 targets, 13 EOD.
- SPY flip-only is the least bad at -1.38 bp; all full-sample SPY selectors are
  negative.
- QQQ holdout primary is +5.84 bp, but train is -6.69 bp.
- Moving decision to 12:00 and widening stop to `0.75×EM` remains negative in the
  full sample.
- Active-symbol portfolio: -4.39 bp; block CI [-7.94,-0.24].

Decision: “unresolved range automatically pins to nearest option level” is not
supported. Pinning may require an additional compression/order-flow condition;
target proximity alone is insufficient.

## P6 — PDH/PDL magnets

The cleanest pattern is continuation/acceptance doing better than rejection/fade:

- PDH rejection short 2R loses in both assets; SPY block CI is fully negative.
- PDL rejection long 2R loses in both assets.
- QQQ PDH acceptance long is +2.88 bp full but -0.25 holdout.
- QQQ PDL acceptance short is +3.33 full and +2.75 holdout, but only 25 holdout
  events and its confidence interval is wide.
- SPY does not confirm either acceptance edge.

Exit sensitivity matters. PDH-rejection held to EOD becomes +2.27 bp SPY and
+5.68 QQQ in full history, but QQQ holdout turns -1.14. This is an exploratory
observation, not a 2R fade pass. QQQ PDL-acceptance 3R produces +4.39 bp versus
+3.33 at 2R; SPY remains flat/negative.

Using PDH/PDL as targets for the option-wall setups does not help consistently:

- Forced-buying→PDH: SPY -1.91 bp (`n=11`), QQQ +4.64 (`n=15`).
- Forced-selling→PDL: SPY -0.92 (`n=41`), QQQ -4.51 (`n=38`).
- Call-wall-fade→PDL: SPY -1.57, QQQ worse than the primary fade.

Decision: reject mechanical PDH/PDL rejection fades. Keep QQQ PDL acceptance
short as a forward candidate alongside forced selling; do not promote it yet.

## Cost, concentration and robustness conclusion

Tight wall stops do create large nominal R multiples on winners, but most tested
fade/pinning setups hit the tight stop too frequently. Positive candidates remain
sensitive to realistic cost and a handful of large winners:

- Forced selling is the only canonical family with positive primary means in both
  instruments.
- QQQ PDL acceptance short is directionally consistent but not replicated by SPY.
- No primary positive cell survives family-level FDR.
- All positive holdout cells have wide block-bootstrap uncertainty.

## Final playbook status

| Setup | Status now | Next action |
|---|---|---|
| Call-wall fade | Rejected under locked mechanical rule | Do not retune on this sample; collect discretionary labels if the human setup uses extra context |
| Negative-GEX cascade short | Underpowered and adverse | Forward collect exact events; no deployment |
| Forced buying | Neutral; QQQ late-sample hint | Freeze 2R rule and forward monitor |
| Forced selling | Best candidate, not confirmed | Highest-priority forward validation; retain 2R and pre-noon diagnostic without optimizing |
| Unresolved pinning | Rejected as nearest-level-only rule | Requires a new, independently motivated compression filter before retest |
| PDH/PDL rejection | Rejected | Do not use as mechanical fade |
| PDH/PDL acceptance | Mixed; QQQ downside acceptance interesting | Forward monitor PDL acceptance short; no live risk yet |

The honest conclusion is narrower than “GEX setups do not work.” With the
available one-year option-level history, the repeatable-looking behavior is on the
downside acceptance/forced-selling side. Fade and unconditional pinning definitions
perform poorly. More history is required before deciding whether the positive
downside-continuation cells are genuine or simply features of this particular
2025–2026 regime.

## Prospective proof gate

The unresolved candidates are now frozen in `FORWARD_VALIDATION_SPEC.md` and
scored by `forward_validation.py`. The scheduled task
`KaderEquity_GEX_ForwardEOD` runs daily at 00:20 Europe/Istanbul with
StartWhenAvailable enabled. It captures a physically separate post-close CBOE
chain, freezes EOD levels, refreshes Alpaca minute bars and updates
`results/forward_validation.json`.

No candidate can pass before 200 new prospective trades. A pass additionally
requires positive mean after 5 bps, 99% block-bootstrap lower bound above zero,
PF>=1.20, positive first and second halves and limited top-five concentration.
Based on the small historical effects, conventional 99%-confidence/90%-power
requirements are much larger: roughly 2,409 SPY forced-selling events, 1,017 QQQ
forced-selling events and 1,277 QQQ PDL-acceptance events if the measured effects
remain stable. This is why buying multi-year true-OI history is materially faster
than waiting for forward data.

The free 2024–2026 Alpaca option-volume proxy was explicitly rejected as proof:
its overlap sign agreement is only 60–64%, wall errors are roughly 0.8–1.4% of
spot, and it produces positive breakout results where the true-OI panel is
negative. More proxy rows would create false precision.

## Post-unblinding amendment: liquidity sweeps

The original P6 `PDL acceptance short` is a downside-continuation rule: price
closes below PDL and the strategy shorts the next-minute open. It is not a
liquidity-sweep fade. These two hypotheses must not be described as the same
trade:

- PDL breach followed by a close back above PDL: sweep/reclaim, tested long.
- PDL break that remains accepted below the level (or fails a retest):
  continuation, tested short.

After this distinction was requested, a separate exploratory amendment tested
a 5-minute wick of at least `0.05 x EM`, a close back inside by `0.02 x EM`,
next-minute entry, and a stop `0.05 x EM` beyond the wick. It did not overwrite
the frozen canonical family.

| Setup and exit | Asset | n | Mean after 2 bp | t | Holdout n / mean |
|---|---:|---:|---:|---:|---:|
| PDL sweep/reclaim long, 2R | SPY | 46 | -1.32 bp | -0.39 | 26 / -2.46 bp |
| PDL sweep/reclaim long, 2R | QQQ | 31 | -7.84 bp | -1.90 | 14 / -8.81 bp |
| PDH sweep/reject short, 1R | SPY | 33 | -2.80 bp | -1.50 | 15 / -3.51 bp |
| PDH sweep/reject short, 1R | QQQ | 43 | +5.36 bp | +2.23 | 17 / +10.15 bp |

The PDL liquidity-sweep long is therefore adverse under this exact mechanical
definition. QQQ PDH sweep/reject short is a new exploratory lead, but only 43
events exist and the rule was examined after the canonical results were known;
it requires prospective confirmation and SPY does not replicate it. Full
variant output is in `results/liquidity_sweep_results.json`.

Forced selling already represents the downside cascade thesis. The separate
negative-GEX cascade rule adds another confirmed five-minute close and is better
understood as a later, stricter entry variant, not an independent economic
setup. In this sample it is rare and adverse, so it should not be counted as a
second confirmation of forced selling.

Forced buying is not simply empty: QQQ has an open-ended right-tail signature.
Its EOD rule averages +1.52 bp in 36 events and +13.74 bp in the nine-event
holdout, while 7/36 trades reach at least 5R intraday MFE. SPY does not replicate
the effect (EOD -3.67 bp full sample). Because only ten QQQ trades survive to EOD
and minute OHLC cannot establish the ordering of a stop and the bar high, this
is a gamma-squeeze candidate, not validated edge.
