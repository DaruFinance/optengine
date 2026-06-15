"""optengine — options-strategy backtest engine.

Large-scale empirical testing of options strategies across OPRA (US equity/index
options) and listed options-on-futures (CME), under real transaction costs,
rolling walk-forward optimization, and strict no-lookahead.

Modules:
  - pricing + Greeks ............. optengine.pricing
  - arbitrage-checked SVI surface  optengine.surface
  - American binomial pricer ..... optengine.american
  - multi-leg position + P&L ..... optengine.position
  - cost / fill model ............ optengine.costs
  - conditioning signals ......... optengine.signals
  - walk-forward driver .......... optengine.wfo
"""
import os as _os
# Cap every BLAS/OMP/numba thread pool BEFORE numpy is imported by any submodule,
# so a run stays single-threaded and shares a host politely.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

__version__ = "0.0.1"
