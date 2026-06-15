"""Intraday short-dated ES options (the 0DTE-adjacent arm).

The hostile review's load-bearing gap is data resolution: the daily engine marks once at 15:59 and cannot
touch intraday/0DTE. Equity OPRA intraday is NOT in the accessible Algoseek subscription (only daily greeks),
but CME futures-options 1-min TAQ IS — including the ES daily/weekly option roots (E1A..E5D, EW1..), which give
0-1 DTE index-vol options. This pulls those 1-min zips (same machinery as the daily FO pull) and distills an
INTRADAY LADDER (last 2-sided quote at/before each ladder time) instead of a single 15:59 snapshot, so we can
enter/hedge/exit intraday — the resolution the daily engine throws away.

Stage 1 here: the intraday distiller + a one-(root,date) validator. Stage 2 (separate): the intraday backtester.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import ODTE_RAW, FOP_S3_BUCKET

import os, sys, io, zipfile, time, datetime as dt
import numpy as np, pandas as pd
from fop_build import s3, _NEED, TMP   # reuse the proven requester-pays awscli wrapper + schema

OUT = ODTE_RAW
# RTH-ish ET ladder; TZ of LocalTimeBarStart is verified empirically by the validator below.
LADDER = ["09:35", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30",
          "13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "15:55"]


def _scan_ladder(txt, ix, ladder):
    """One reversed pass: for each ladder time, the last 2-sided quote at/before it. Returns
    {snap -> (cp,K,month,year,bid,ask)} for the snaps this contract has a quote for."""
    nl = txt.find("\n")
    if nl < 0:
        return {}
    cp_i, k_i, m_i, y_i, lt_i, b_i, a_i = ix
    lines = txt[nl + 1:].split("\n")
    targets = sorted(ladder, reverse=True)
    out = {}; ti = 0
    for ln in reversed(lines):
        if ti >= len(targets) or not ln:
            continue
        f = ln.split(",")
        if len(f) <= a_i:
            continue
        try:
            bid = float(f[b_i]); ask = float(f[a_i])
        except ValueError:
            continue
        if not (bid > 0 and ask >= bid):
            continue
        t = f[lt_i][11:16] if len(f[lt_i]) >= 16 else f[lt_i][:5]   # HH:MM from ISO or HH:MM:SS
        # assign this bar to every still-unfilled target it satisfies (t <= target)
        while ti < len(targets) and t <= targets[ti]:
            out[targets[ti]] = (f[cp_i], float(f[k_i]), f[m_i], int(f[y_i]), bid, ask)
            ti += 1
    return out


def distill_intraday(blob, ladder=LADDER):
    rows = []
    z = zipfile.ZipFile(io.BytesIO(blob)); ix = None
    for name in z.namelist():
        if not name.endswith(".csv"):
            continue
        try:
            txt = z.read(name).decode("latin-1")
        except Exception:
            continue
        if ix is None:
            nl = txt.find("\n")
            if nl < 0:
                continue
            h = txt[:nl].split(",")
            try:
                ix = tuple(h.index(c) for c in _NEED)
            except ValueError:
                continue
        try:
            d = _scan_ladder(txt, ix, ladder)
        except Exception:
            d = {}
        for snap, (cp, K, mon, yr, bid, ask) in d.items():
            rows.append((snap, cp, K, mon, yr, bid, ask))
    return rows


def pull_one(root, date, year):
    dd = os.path.join(TMP, f"odte_{root}_{date}")
    os.makedirs(dd, exist_ok=True)
    base = f"s3://{FOP_S3_BUCKET.format(year=year)}/{date}/{root}/"
    if s3("cp", base, dd, "--recursive", "--quiet").returncode != 0:
        return []
    rows = []
    for f in os.listdir(dd):
        if f.endswith(".zip"):
            try:
                with open(os.path.join(dd, f), "rb") as fh:
                    rows += distill_intraday(fh.read())
            except Exception:
                pass
    import shutil; shutil.rmtree(dd, ignore_errors=True)
    return rows


def validate(root, date, year):
    t0 = time.time()
    rows = pull_one(root, date, year)
    if not rows:
        print(f"{root} {date}: NO DATA"); return
    df = pd.DataFrame(rows, columns=["snap", "cp", "Strike", "Month", "ExpYear", "bid", "ask"])
    print(f"{root} {date}: {len(df)} intraday quote-rows, {time.time()-t0:.0f}s")
    print(f"  snap times present: {sorted(df.snap.unique())}")
    print(f"  contracts/snap: {df.groupby('snap').size().describe()[['min','mean','max']].round(0).to_dict()}")
    print(f"  strikes: {df.Strike.min():.0f}..{df.Strike.max():.0f} | months: {sorted(df.Month.unique())} | cp: {sorted(df.cp.unique())}")
    # spread at ATM-ish across the day (a quick intraday liquidity read)
    mid = 0.5 * (df.bid + df.ask)
    df["rel"] = (df.ask - df.bid) / mid.replace(0, np.nan)
    print(f"  median rel-spread by snap (first/mid/last): "
          f"{df.groupby('snap')['rel'].median().iloc[[0, len(df.snap.unique())//2, -1]].round(3).to_dict()}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="E1A"); ap.add_argument("--date", default="20250516"); ap.add_argument("--year", default="2025")
    a = ap.parse_args()
    validate(a.root, a.date, a.year)
