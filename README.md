# optengine

> Backtester and reproduction scripts for Daniel Gatto's options work, published on [daru.finance](https://www.daru.finance).

A from-scratch options backtester and the analysis scripts that reproduce the
results in the paper *When the Spread Is the Edge: A post-cost, multiple-testing
audit of systematic options strategies*.

The engine prices and hedges multi-leg option structures, fills every leg at the
real quoted bid/ask, runs rolling walk-forward optimization with out-of-sample
scoring only, and writes a full per-trade ledger for every run. The scripts in
`tools/` build the strategy corpus, apply the deflated-Sharpe and FDR screens,
and produce each table and figure in the paper. The scripts in `verify/` are the
correctness gates (pricing, Greeks, no-look-ahead, walk-forward).

## Layout

```
optengine/      the engine package
  config.py     all paths and data locations (resolved from environment)
  pricing.py    Black-Scholes / Black-76 / Bachelier closed forms
  american.py   Cox-Ross-Rubinstein binomial pricer
  surface.py    arbitrage-checked SVI volatility surface
  position.py   multi-leg position and path P&L
  costs.py      the quoted-spread fill / commission model
  signals.py    causal daily conditioning features (shifted one day)
  archetypes.py the 35 strategy archetypes
  wfo.py        walk-forward driver (in-sample tune, out-of-sample score)
  fan.py        parallel fan-out over instruments
  data/         the daily Greeks-store loader
tools/          corpus build, statistics, and the per-result scripts
verify/         correctness gates
```

## Install

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or newer.

## Data

The source data is licensed and is not distributed here. Point the engine at
your own copy with environment variables; nothing else needs editing.

| Variable | Contents |
|---|---|
| `OPTENGINE_GREEKS` | daily equity-option Greeks store (per-day zip archives) |
| `OPTENGINE_PANEL`  | prebuilt daily surface panel (parquet) |
| `OPTENGINE_SAMPLE` | single-file Greeks sample for the validation checks |
| `OPTENGINE_FOP_S3` | futures-options 1-minute TAQ bucket template with a `{year}` placeholder |
| `OPTENGINE_AWS_PROFILE` | named credentials profile for the requester-pays pull |

Working directories (`pertkr/`, `findings/`, `fop_pertkr/`, ...) default under
the repo root and can be redirected with `OPTENGINE_PERTKR`,
`OPTENGINE_FINDINGS`, and the matching variables in `optengine/config.py`.

## Reproducing the paper

The equity arm runs end to end from `tools/run_mega.sh` (build chains, run the
35-archetype corpus, deflate, assemble portfolios, report). Each result also has
a standalone script:

| Paper artifact | Script |
|---|---|
| Central figure + family survival table | `tools/killer.py` |
| Strategy corpus (35 archetypes x instruments) | `tools/corpus.py` |
| Effective-trials + deflated Sharpe | `tools/deflate.py` |
| Survivor / non-hindsight portfolios, bootstrap CIs | `tools/portfolio.py`, `tools/portfolio_ci.py` |
| Two-sided vega split | `tools/vega_breakdown.py` |
| Power and calibration of the deflation | `tools/power_audit.py` |
| Trade-level factor decomposition (MKT, MKT^2, VRP) | `tools/theory_decomp.py` |
| Cross-venue build and corpus | `tools/fop_build.py`, `tools/fop_finalize.py`, `tools/fop_corpus.py` |
| 2022 stress and the rates arm | `tools/covid_rates_isolation.py` |
| Crypto crash-window illustration | `tools/covid_crash_illustration.py` |
| Cross-section of option returns + placebo | `tools/cross_sectional.py`, `tools/char_panel.py`, `tools/replication.py` |
| Covered-call decomposition | `tools/covered_call_etf.py` |
| Spread-capture frontier | `tools/spread_capture_curve.py` |
| Dispersion / implied correlation | `tools/dispersion.py` |
| Model-free variance swaps | `tools/varswap.py` |
| SVI surface relative value | `tools/surface_rv_precompute.py`, `tools/surface_rv_corpus.py` |
| Box-spread financing | `tools/boxspread.py` |
| Factor spanning | `tools/factor_spanning.py` |
| Defined-risk income | `tools/defined_risk.py` |
| Universal left tail | `tools/tail_compendium.py` |
| Intraday 0DTE | `tools/odte_pull.py`, `tools/odte_intraday.py`, `tools/odte_study.py` |
| Romano-Wolf, hedge frequency, regimes | `tools/romano_wolf.py`, `tools/hedge_freq.py`, `tools/regime_conditioning.py` |

`tools/README.md` describes every script in one line.

## Validation

`verify/` holds the correctness gates: put-call parity to `1e-13`, Greeks
against the vendor to machine precision, the American binomial against vendor
in-the-money puts, the delta-hedge P&L, the walk-forward driver, and a
pollute-and-verify no-look-ahead check that injects a forward peek and confirms
the harness detects the inflation. See `verify/README.md`.

## Cost and no-look-ahead

Every fill crosses the real quoted half-spread (entry and exit, each leg) plus
commission. P&L marks to the actual quotes, so the pricing model affects only
the Greeks and any no-quote fallback, never a realized fill. Every signal,
selection, and mark uses only the decision-time snapshot, and the walk-forward
procedure itself is leak-tested rather than assumed causal.
