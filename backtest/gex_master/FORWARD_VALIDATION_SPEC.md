# Canonical GEX prospective validation gate

Frozen on 2026-07-16 after the canonical historical study. No parameter changes
are permitted until each candidate is passed or failed.

## Candidates

1. Forced selling, SPY, 2R.
2. Forced selling, QQQ, 2R.
3. Forced buying, QQQ, 2R.
4. PDL acceptance short, QQQ, 2R.

Rules, fills, stop buffer and cost are exactly those in `CANONICAL_SPEC.md`.

## Date classification

- Shadow extension: 2026-06-11 through 2026-07-16. These dates were outside the
  old option-level panel, but precede formal declaration; useful, not prospective.
- Prospective validation: level date on or after 2026-07-17. Only this segment can
  pass the final gate.

## Strong pass gate

A candidate passes only when all conditions hold:

- At least 200 prospective trades.
- Net mean after 5 bps round-trip cost is positive.
- Five-session block-bootstrap one-sided confidence is at least 99% that mean is
  positive (equivalently lower 99% bound above zero).
- Profit factor after 5 bps is at least 1.20.
- First-half and second-half prospective means are both positive after 5 bps.
- Top five winning sessions contribute no more than 35% of total net profit.
- For the forced-selling family, both SPY and QQQ must individually satisfy the
  sign and 95% bootstrap condition; QQQ alone cannot promote the family.

This is strong statistical evidence, not “100% certainty.” A future market process
can always change.

## Failure gate

- Hard fail after at least 100 prospective trades if the 95% block-bootstrap upper
  bound is at or below zero after 2 bps.
- Hard fail after 200 trades if mean after 5 bps is non-positive or PF<1.0.
- Otherwise status remains `COLLECTING`; there is no discretionary override.

## Data integrity

- Prior-day live CBOE gamma-level ledger only.
- Next-session Alpaca IEX one-minute RTH path.
- SPX-labelled levels used for SPY are scaled by previous SPY close / ledger spot.
- A session requires at least 350 RTH minute bars; partial current days are ignored.
- Missing level, IV, wall, prior session or complete next session produces no trade,
  never an imputed fill.
