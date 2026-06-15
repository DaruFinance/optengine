"""Non-hindsight + VIX-excluded portfolio robustness — closes the review gap that the
beta/tail decomposition was shown only on the HINDSIGHT-selected survivor book (cells
picked by raw t-stats the deflation already rejects).

Here we rebuild the equal-weight monthly book over (a) ALL net-positive cells (NO
significance selection at all — the broadest non-hindsight book), and VIX-excluded
variants of both that broad book and the t>2.5 book (VIX's spot-hedge is an
idealization and VIX is the single strongest cell). If beta~1 and the deep drawdown
PERSIST on the broad, non-selected, VIX-free book, the disguised-equity-beta finding
is a property of the short-vol SPACE, not an artifact of the cherry-pick or of VIX.
Reuses the exact estimators in portfolio.py (ledger/portfolio_monthly/spy_monthly/report).
"""
import os
import numpy as np, pandas as pd
from portfolio import ledger, portfolio_monthly, spy_monthly, report, OUT


def robust_tail(name, port):
    """cumprod drawdown is undefined once a monthly return-on-premium breaches -100%
    (short options lose MORE than premium collected -> (1+r)<=0). Report the robust,
    cumprod-free tail instead: worst month, CVaR5, and the count of ruin months."""
    r = port.values
    n = len(r)
    k = max(1, int(0.05 * n))
    cvar = float(np.sort(r)[:k].mean())
    ruin = int((r <= -1.0).sum())
    print(f"  [robust tail] worst_month={r.min():.1%}  CVaR5={cvar:.1%}  "
          f"months_breaching_-100%(ruin)={ruin}/{n}  mean_monthly={r.mean():.2%}")


def main():
    led = ledger()
    s = pd.read_csv(os.path.join(OUT, "corpus_deflated.csv"))
    allpos = set(zip(s[s["oos_net"] > 0]["instrument"], s[s["oos_net"] > 0]["archetype"]))
    t25 = set(zip(s[s["t"] > 2.5]["instrument"], s[s["t"] > 2.5]["archetype"]))
    novix = lambda cells: {c for c in cells if c[0] != "VIX"}
    mkt = spy_monthly()
    for nm, cells in [("ALL net-positive (non-hindsight, %d cells)" % len(allpos), allpos),
                      ("ALL net-positive EX-VIX (%d cells)" % len(novix(allpos)), novix(allpos)),
                      ("t>2.5 survivors EX-VIX (%d cells)" % len(novix(t25)), novix(t25))]:
        p = portfolio_monthly(led, cells)
        report(nm, p, mkt)
        robust_tail(nm, p)


if __name__ == "__main__":
    main()
