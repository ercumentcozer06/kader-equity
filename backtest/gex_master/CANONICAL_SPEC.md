# Canonical GEX playbook — locked test specification

Declared before running the new canonical event engine on 2026-07-16. The old
scenario-engine outputs were already known, so this is an exploratory remeasurement
with a chronological holdout, not a pristine never-seen-data experiment.

## Shared execution rules

- Universe: SPY and QQQ.
- Option levels: session `D` EOD levels trade only session `N>D`.
- Bars: 5-minute signals anchored at 09:30 ET; fills occur at the next 1-minute
  bar open. This prevents filling at the same close that creates confirmation.
- Signal window: 09:35–14:30 ET. Positions close no later than 15:59.
- Acceptance buffer: `0.02 × prior-EOD one-day expected move (EM)` beyond a level.
- Tight invalidation: `0.10 × EM` behind the broken/rejected level.
- If stop and target occur in the same one-minute OHLC bar, stop wins.
- Round-trip cost: primary 2 bps; sensitivity 1 and 5 bps.
- One trade per setup per session. Different setup families may overlap and are
  never pooled as independent evidence without a same-day cluster warning.
- Primary chronological holdout: level date `D >= 2026-02-02`, matching the
  existing corrected full-chain battery.
- Gamma regime primary: index full-surface flag (`SPX→SPY`, `NDX→QQQ`). Own-chain
  regime is a sensitivity because the S&P option pool is index-dominated.

## P1 — Call-wall fade

First 5-minute bar whose high touches/exceeds the call wall and whose close is at
least `0.02×EM` back below it. Enter short at next minute open. Stop is
`call_wall + 0.10×EM`.

- Primary target: 2R.
- Robustness: 1R, 3R, EOD, ghost target, gamma-flip target, max-pain target and
  PDL target, each only when it lies below entry.
- This setup is not conditioned on positive aggregate GEX.

## P2 — Negative-GEX cascade short

Negative gamma plus accepted put-wall break. First 5-minute close below
`put_wall - 0.02×EM` is the break bar; cascade requires the next 5-minute close
also below the break-bar low. Enter short at the next minute open. Stop is
`put_wall + 0.10×EM`.

- Primary target: 3R.
- Robustness: 2R, EOD, PDL target.
- Primary regime: index flag negative; own-regime negative is sensitivity.

## P3 — Forced buying

First 5-minute close above `call_wall + 0.02×EM`, with the previous 5-minute
close not already above that threshold. Enter long next minute. Stop is
`call_wall - 0.10×EM`.

- Primary target: 2R.
- Robustness: 1R, 3R, EOD, PDH target and next-upper-strike/VU target.
- No gamma-sign condition: the wall acceptance itself defines the setup.

## P4 — Forced selling

Mirror of P3 below `put_wall - 0.02×EM`. Enter short next minute; stop is
`put_wall + 0.10×EM`.

- Primary target: 2R.
- Robustness: 1R, 3R, EOD, PDL target and next-lower-strike/VD target.

## P5 — Unresolved-range pinning

At 11:00 ET, neither call wall nor put wall may have been touched since 09:30.
Candidate targets are gamma flip, ghost and max pain that lie strictly inside the
put-wall/call-wall range. “GEX level” is interpreted as gamma flip.

- Primary selector: candidate nearest to the 11:00 tradable price.
- Enter toward the target at 11:00 open; require distance at least `0.20×EM`.
- Target is the selected level; stop is `0.50×EM` adverse from entry.
- Robustness: target-specific flip/ghost/max-pain cells, 12:00 decision time and
  `0.75×EM` stop.

## P6 — PDH/PDL magnets

PDH and PDL are computed from the immediately preceding RTH session; hence they
are point-in-time known.

Entry models:

- PDH rejection short: touch PDH then close `0.02×EM` below; stop above PDH.
- PDL rejection long: touch PDL then close `0.02×EM` above; stop below PDL.
- PDH acceptance long: close `0.02×EM` above; stop below PDH.
- PDL acceptance short: close `0.02×EM` below; stop above PDL.

Each entry model uses 2R primary, plus 1R/3R/EOD robustness. PDH/PDL are also
tested as alternative targets for P1–P4 when the relevant level lies ahead of the
entry.

## Statistical output

For every primary and robustness cell: event count, net mean/median, hit rate,
profit factor, mean R, t-statistic, five-session block-bootstrap 95% CI and
probability mean>0, max drawdown, top-three-day P&L concentration, train/holdout,
hour, gamma regime, cost sensitivity and SPY/QQQ split. Family-level primary
p-values receive Benjamini–Hochberg correction. Cells with fewer than 30 trades
are explicitly underpowered regardless of point estimate.
