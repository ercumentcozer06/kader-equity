# GEX / option research data inventory

Last verified: 2026-07-16. All new collection is append-only and uses authorized APIs.

## Active prospective collection

| Dataset | Coverage / cadence | Contents | Important limitation |
|---|---|---|---|
| Alpaca option surface | SPY, QQQ; every 5 minutes during the US session | Bid/ask, last trade, IV, delta/gamma/theta/vega/rho, contract OI and OI date, short-dated daily OHLCV, local higher Greeks | Basic plan is indicative, not consolidated OPRA; volume is delayed and not signed |
| Alpaca EOD surface | SPY, QQQ; 00:30 Europe/Istanbul, after the completed US session | Wider strike range and up to 365 DTE; creates frozen next-session levels | EOD rows before 16:20 New York are rejected by the exporter |
| OCC participant volume | SPY, QQQ, SPX, NDX; daily | Customer/Firm/Market-Maker call/put volume by exchange, normalized to one row per product/day | No buy/sell or open/close flag; not strike-level OI |
| Alpaca spot bars | SPY, QQQ; refreshed after EOD | One-minute OHLCV, currently 2020-09-01 onward | Free IEX feed, not SIP |

Higher Greeks calculated locally: vanna, charm, vomma/volga, veta, speed, color,
zomma and ultima. Their stored units are explicit in the column names. The dealer
exposure sign is a testable naive convention (+call / -put), not observed dealer
inventory.

## Recovered pre-existing / Claude-era data

| Dataset | Verified coverage | Research use |
|---|---|---|
| SqueezeMetrics DIX/GEX cache | 3,798 daily rows, 2011-05-02 through 2026-06-08 | Long-history aggregate mechanism and regime tests |
| MarketData raw chains | SPY 261, QQQ 258, SPX 256, NDX 256 files; 2025-06-13 through 2026-07-15/09 | Historical strike-level OI, IV and first-order Greeks |
| Normalized MarketData chains | SPY 117,712 rows; QQQ 101,280 rows; 2025-06-13 through 2026-06-08 | Wall/flip/pinning candidate reconstruction |
| OptionsDX QQQ EOD | 144 monthly files, 2012-01 through 2023-12 | EOD quote/IV/Greeks/volume tests; no OI, so it cannot independently reconstruct true GEX |
| Alpaca option daily bars | SPY 130,042; QQQ 103,046 rows, 2024-01-18 through 2026-05-15 | Contract-volume and price proxy tests; no OI |
| Spot minute bars | SPY 560,420; QQQ 536,617 rows, 2020-09-01 onward | Intraday event paths, entries, stops and targets |

The OCC backfill now contains 1,908 normalized rows: 477 sessions for each of
SPY, QQQ, SPX and NDX from 2024-07-25 through 2026-07-15.

## What free data does not solve

- A pristine multi-year, point-in-time SPY/QQQ/SPX/NDX intraday OPRA quote and
  trade archive cannot be reconstructed from the free sources found.
- Historical aggressor side, dealer identity and open/close flags are not
  present. Dealer positioning therefore remains an inferred latent variable.
- IBKR currently returns the SPX contract definitions but no SPX option quotes,
  OI or Greeks because the account lacks the needed market-data subscription.
  IBKR also does not provide historical data for expired options.
- OCC's free daily aggregate OI report is market-wide, not strike-level.
- CBOE's delayed-quote webpage explicitly prohibits automated extraction. The
  two old CBOE scraping scheduled tasks are disabled and superseded by the
  authorized Alpaca pipeline.

These gaps do not block prospective validation. They do limit the strength of
claims from old history: use the recovered data for discovery and robustness,
then require confirmation on the frozen forward ledger before trading a setup.

## Files and operations

- Collector: `screen/collect_option_research.py`
- Higher Greeks: `screen/option_research_greeks.py`
- Surface features: `backtest/gex_master/build_dynamic_option_features.py`
- OCC collector: `screen/fetch_occ_participant_volume.py`
- Forward export: `screen/export_authorized_gex_forward.py`
- End-to-end scheduler wrapper: `screen/run_option_research_pipeline.py`
- Data health audit: `screen/audit_option_research_data.py`
- Frozen test specification: `backtest/gex_master/PROFESSIONAL_DIRECTIONAL_RESEARCH_SPEC.md`
- Dependencies: `requirements-option-research.txt`

Scheduled tasks:

- `KaderEquity_OptionResearch_Intraday`: every 5 minutes; calendar/time gated.
- `KaderEquity_OptionResearch_EOD`: daily 00:30 Europe/Istanbul; exchange-calendar gated.
- `KaderEquity_OCCParticipant_Daily`: daily 09:10 Europe/Istanbul; retries the recent publication window.
