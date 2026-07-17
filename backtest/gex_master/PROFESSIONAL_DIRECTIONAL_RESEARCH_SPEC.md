# Professional directional GEX research protocol

Frozen design date: 2026-07-16. This document separates mechanism validation,
event prediction, trade construction and portfolio validation. Historical work
seen before this date is exploratory. Only observations collected prospectively
from 2026-07-17 can be called pristine out-of-sample evidence.

## Core decision model

The model must never convert a regime label directly into a direction. It must
update filtered (not hindsight-smoothed) regime probabilities as information
arrives, then estimate an outcome distribution for each eligible setup:

`P(regime | information through t) -> P(outcome | regime, setup, information through t)`

An order is allowed only when expected value after costs and an uncertainty
penalty is positive. Direction comes from observed price/flow confirmation;
gamma describes whether that move is more likely to be absorbed or amplified.

Suggested latent states are containment/long-gamma, unstable/short-gamma,
transition-near-flip, upside-flow-amplification, downside-flow-amplification and
event-dominated. These are probabilities, not hard labels. The live model must
use one-sided filtering; a two-sided HMM smoother is forbidden because it leaks
future observations into the inferred state.

## Layer 1: validate the economic mechanisms

These tests do not contain entries, stops or profit targets.

1. Volatility: does prior-known signed and absolute GEX predict subsequent
   30/60-minute and rest-of-day realized variance after controlling for
   early-session variance, VIX/IV, gap, scheduled events and day-of-week?
2. Feedback: does `past return x signed GEX` predict the next return with the
   expected negative interaction (negative gamma continuation, positive gamma
   reversal)? Test 5, 15, 30, 60-minute and last-hour horizons.
3. Liquidity interaction: does the feedback effect strengthen as estimated
   hedge demand rises relative to underlying dollar volume/depth?
4. Expiry interaction: is the effect stronger for 0DTE/1DTE gamma, near the
   close, and near high-gamma strikes?
5. Transition: do range, autocorrelation and tail probability change around the
   gamma flip without treating the flip crossing itself as directional?

Primary outputs are coefficients with HAC errors, monotonic dose-response
plots, block-bootstrap intervals, posterior distributions and stability by
year/instrument. A useful regime variable must predict distributions out of
sample before it is allowed into a trade model.

## Layer 2: validate level and path hypotheses

Every wall encounter is one event that branches into mutually exclusive states.
No branch may be selected using information after its decision time.

### Call wall

- Rejection: price trades at/through the wall, then closes back below the
  predeclared buffer. Candidate direction: short fade.
- Acceptance: price closes above the buffer and remains above for the declared
  confirmation window or passes a retest. Candidate direction: long.
- Upside amplification: acceptance plus negative/transition gamma, positive
  signed call-delta flow, sufficient gamma/ADV and no wall migration invalidating
  the old level. This is the gamma-squeeze/forced-buying hypothesis.
- Ambiguous: neither acceptance nor rejection; no trade.

### Put wall

- Rejection/reclaim: sweep below followed by a close back above. Candidate long.
- Acceptance: close below plus hold/failed retest. Candidate short.
- Downside amplification: acceptance plus negative/transition gamma, negative
  delta flow and sufficient gamma/ADV. This is forced selling/cascade.
- Ambiguous: no trade.

For each branch estimate first the conditional path, not one preferred exit:
future 5/15/30/60-minute and EOD returns, MFE/MAE, probability of 1R/2R/3R before
1R adverse, time-to-barrier, close location, realized variance and tail loss.
Use survival/competing-risk models for target-versus-stop ordering. Only after
the path result is stable may fixed target, trailing, EOD or option-spread
execution be compared.

## Layer 3: Bayesian directional model

### Prior-known features

- Signed GEX, absolute GEX and GEX percentile.
- Estimated hedge notional divided by underlying ADV/depth.
- Spot distance to call wall, put wall, flip and major gamma nodes, scaled by
  expected move and current realized volatility.
- Wall concentration, expiry concentration and 0DTE/1DTE share.
- Overnight gap, IV term structure/skew, VIX, prior realized volatility,
  scheduled macro/earnings/OpEx flags and prior-day wall migration.

### Intraday updates

- Wall rejection, acceptance or failed-retest state.
- Returns and realized volatility through time t, VWAP displacement, volume,
  breadth and spot/futures liquidity.
- Signed option delta flow, call/put opening-flow inference, IV/skew movement and
  live wall migration. If these data are absent, the model must expose the
  missingness and reduce confidence rather than impute a bullish/bearish signal.

Use a hierarchical model with partial pooling across SPX/ES, SPY and QQQ rather
than assuming they are identical. A practical first version is a Bayesian
mixture-of-experts: a filtered regime model gates regularized logistic/barrier
models for rejection, acceptance and amplification. More complex models are
allowed only if they beat this baseline prospectively.

For a candidate trade calculate the posterior distribution of net payoff:

`EV = P(target)*target_payoff - P(stop)*stop_loss + E(other exits) - costs`

Trade only if the lower credible bound of EV exceeds zero and predicted edge is
large enough to survive the adverse cost/slippage scenario. Size must decrease
with posterior uncertainty and predicted tail risk; a regime probability alone
can never create a position.

## Layer 4: honest validation

1. Maintain an immutable hypothesis registry. Count every tried feature,
   threshold, horizon and exit, including failed ones.
2. Freeze event definitions before outcome inspection. Parameter grids must be
   coarse and economically motivated, not optimized strike by strike.
3. Use expanding or rolling walk-forward estimation. Train only on dates before
   the traded observation. Refit cadence is frozen.
4. Purge observations whose outcome windows overlap a validation fold and apply
   a time embargo. Cluster inference and resampling by session/week because
   simultaneous SPY/QQQ/setup trades are not independent.
5. Report untouched prospective performance separately from historical
   pseudo-OOS. Never relabel an already inspected period as holdout.
6. Include realistic spread, fees, slippage, next-bar fills, stop-first ambiguity,
   missed fills and capacity. For options, reconstruct tradable bid/ask paths;
   spot returns are not sufficient evidence of option P&L.
7. Correct the full research family for multiple testing (BH/FDR for mechanism
   claims; Deflated Sharpe/false-discovery accounting for selected strategies).
8. Require effect stability across time, nearby parameter values, costs and at
   least one related instrument. Report top-day concentration and leave-one-event
   sensitivity.
9. Before deployment require shadow trading, then a small-risk sequential test.
   Posterior deterioration or a structural data break automatically reduces
   size or disables the setup.

## Data tiers and what each tier can prove

- Existing 2020-2026 lagged SqueezeMetrics GEX plus minute bars: suitable for
  volatility and coarse feedback tests, not intraday wall/squeeze attribution.
- Existing roughly one-year true-OI wall panel: suitable for exploratory static
  level event studies, underpowered for conditional regime interactions.
- Daily forward EOD chains: protects point-in-time integrity and validates
  overnight-stable levels, but cannot reconstruct live 0DTE positioning.
- Professional directional research requires timestamped option trades/quotes,
  intraday IV/Greeks, opening/closing inference, current expiry mix, SPX and ES
  data, and underlying liquidity. Several years are preferable; forward data
  must continue indefinitely for drift monitoring.

## Existing evidence at freeze time

The literature-aligned t-1 GEX test covers SPY from 2020-09-01 through
2026-06-10 (`n=1,334`) and QQQ for `n=906` sessions. After controlling for
early realized volatility, higher GEX predicts lower rest-of-day realized
volatility:

- SPY: beta -0.0334, HAC t -3.22, p 0.0013.
- QQQ: beta -0.0466, HAC t -3.66, p 0.00025.
- Both survive the existing six-cell BH/FDR adjustment.

The broad directional interaction is not established for rest-of-day returns
(SPY t -0.39; QQQ t -0.37). QQQ has a last-hour interaction in the predicted
direction (t -2.21, raw p 0.027), but its FDR-adjusted value is approximately
0.054 and it is not a deployable standalone rule. This supports using GEX as a
probabilistic volatility/path regime and requiring a separate directional
trigger.

## Promotion gates

A mechanism may enter the directional model only after sign-correct validation
and stability. A setup may enter shadow trading only if its net posterior EV is
positive under adverse costs in multiple walk-forward paths. Live risk requires
pristine prospective evidence, positive first/second halves, low winner
concentration and no dependence on one regime episode. No historical result is
described as certainty.
