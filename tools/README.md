# tools

Corpus build, statistics, and one script per result. Run from the repo root,
e.g. `python tools/killer.py`. Scripts that need licensed data read it through
`optengine/config.py`; see the top-level README for the environment variables.

## Data build

| Script | Purpose |
|---|---|
| `build_pertkr.py` | build per-ticker option chains from the daily Greeks store |
| `build_pertkr_breadth.py` | extend the build to the full liquid cross-section, RAM-throttled |
| `char_panel.py` | build the per-name characteristic panel used by the cross-sectional sorts |
| `fop_build.py` | pull and distill futures-options 1-minute TAQ into per-root chains |
| `fop_finalize.py` | finalize the futures-option chains into engine format |
| `fop_pull_all.py` | drive the futures-option pull across roots and years |
| `odte_pull.py` | pull the intraday ES weekly roots |
| `surface_rv_precompute.py` | precompute the SVI surface richness panel |

## Corpus and screens

| Script | Purpose |
|---|---|
| `corpus.py` | run the 35 archetypes across all instruments, write per-trade ledgers |
| `corpus_conditioned.py` | corpus variant with signal-conditioned entries |
| `fop_corpus.py` | run the corpus across the 16 futures-option roots |
| `deflate.py` | effective-trials estimate and deflated Sharpe across the corpus |
| `romano_wolf.py` | correlation-aware Romano-Wolf stepdown, the second multiplicity method |
| `factor_spanning.py` | principal-component spanning test on the archetype panel |
| `power_audit.py` | power and calibration of the deflation gate |

## Portfolios and decomposition

| Script | Purpose |
|---|---|
| `portfolio.py` | survivor and non-hindsight books, equal-weight, daily delta-hedged |
| `portfolio_broad.py` | the broad net-positive book |
| `portfolio_ci.py` | bootstrap confidence intervals on the book Sharpe and beta |
| `theory_decomp.py` | trade-level factor decomposition on MKT, MKT^2, and the variance premium |
| `vega_breakdown.py` | split the cross-section by vega sign |

## Literature

| Script | Purpose |
|---|---|
| `cross_sectional.py` | the characteristic-sorted option long-shorts |
| `replication.py` | the replication table with the random-noise placebo |
| `covered_call_etf.py` | the index buy-write decomposition |

## Robustness battery

| Script | Purpose |
|---|---|
| `dispersion.py` | dispersion against a vega-matched single-name basket |
| `varswap.py` | model-free variance swaps from a 1/K^2 strip of OTM options |
| `spread_capture_curve.py` | survival at every spread-capture fraction, re-deflated without a re-run |
| `surface_rv_corpus.py` | selling rich options versus the fitted SVI smile |
| `boxspread.py` | box-spread financing across both directions |
| `defined_risk.py` | defined-risk (capped) structures versus naked selling |
| `tail_compendium.py` | the realized left tail across asset classes |
| `odte_intraday.py` | intraday ladder distiller for the 0DTE arm |
| `odte_study.py` | the intraday 0DTE backtest |
| `hedge_freq.py` | invariance to delta-hedge cadence |
| `regime_conditioning.py` | conditioning and re-deflating within VIX terciles |
| `covid_rates_isolation.py` | the 2022 stress and the Treasury-options arm |
| `covid_crash_illustration.py` | the crypto crash-window illustration |
| `diag_nq_units.py` | the NQ unit-anomaly sanity check |

## Figures and reporting

| Script | Purpose |
|---|---|
| `killer.py` | the central gross-vs-net figure and the family survival table |
| `report.py` | per-run summary statistics from a ledger |
| `robust.py` | corpus-level robustness summary |
| `definitive_numbers.py` | the headline numbers quoted in the text |
| `validate_surface.py` | surface-fit quality across the universe |
| `run_mega.sh` | chains the equity build, corpus, deflation, portfolios, and report |
