# verify

Correctness gates for the engine. Run from the repo root, e.g.
`python verify/check_european_parity.py`. The checks that compare against vendor
values read a Greeks sample through `OPTENGINE_SAMPLE`.

| Script | Gate |
|---|---|
| `check_european_parity.py` | put-call parity residual to `1e-13` on European prices |
| `check_greeks_vs_algoseek.py` | engine Greeks against the vendor to machine precision |
| `check_american.py` | Cox-Ross-Rubinstein convergence and the in-the-money American put premium against vendor values |
| `check_hedge.py` | delta-hedge P&L on a short straddle with real costs |
| `check_leak.py` | no-look-ahead: inject a forward peek and confirm the harness detects the inflation |
| `check_wfo.py` | the walk-forward driver tunes in-sample and scores out-of-sample only |
| `check_loader.py` | the daily Greeks-store loader returns one causal snapshot per date |
| `check_slice.py` | a single archetype on one name over a short window |
| `check_fan.py` | the parallel fan-out matches the serial path |
