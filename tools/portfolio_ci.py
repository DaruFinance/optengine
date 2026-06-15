"""Bootstrap CI on the non-hindsight book's headline beta + Sharpe (closes a review gap:
the disguised-equity-beta claim rested on point estimates). Resamples the 106 monthly
(portfolio, SPY) pairs with replacement (B=2000, fixed seed) and reports 2.5/50/97.5
percentiles of OLS beta and annualized Sharpe. Reuses portfolio.py / portfolio_broad.py.
RAM-light, read-only.
"""
import os
import numpy as np, pandas as pd
from portfolio import ledger, portfolio_monthly, spy_monthly, OUT

B = 2000


def boot(name, port, mkt):
    j = pd.DataFrame({"p": port, "m": mkt}).dropna()
    p = j["p"].to_numpy(); m = j["m"].to_numpy()
    n = len(p)
    rng = np.random.RandomState(12345)
    betas, sharpes = np.empty(B), np.empty(B)
    for b in range(B):
        idx = rng.randint(0, n, n)
        pb, mb = p[idx], m[idx]
        betas[b] = np.polyfit(mb, pb, 1)[0]
        sd = pb.std(ddof=1)
        sharpes[b] = np.sqrt(12) * pb.mean() / sd if sd > 0 else 0.0
    qb = np.percentile(betas, [2.5, 50, 97.5])
    qs = np.percentile(sharpes, [2.5, 50, 97.5])
    print(f"[{name}] n={n} months, B={B}")
    print(f"   beta(SPY):   median {qb[1]:+.2f}   95% CI [{qb[0]:+.2f}, {qb[2]:+.2f}]   P(beta>0.5)={np.mean(betas>0.5):.3f}")
    print(f"   ann Sharpe:  median {qs[1]:+.2f}   95% CI [{qs[0]:+.2f}, {qs[2]:+.2f}]   P(Sharpe>0)={np.mean(sharpes>0):.3f}")


def main():
    led = ledger()
    s = pd.read_csv(os.path.join(OUT, "corpus_deflated.csv"))
    allpos = set(zip(s[s["oos_net"] > 0]["instrument"], s[s["oos_net"] > 0]["archetype"]))
    novix = {c for c in allpos if c[0] != "VIX"}
    mkt = spy_monthly()
    boot("ALL net-positive (non-hindsight, %d)" % len(allpos), portfolio_monthly(led, allpos), mkt)
    boot("ALL net-positive EX-VIX (%d)" % len(novix), portfolio_monthly(led, novix), mkt)


if __name__ == "__main__":
    main()
