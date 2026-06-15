"""optengine.data.greeks_store — loader for the local Algoseek daily Greeks store.

Layout: {ROOT}/{YYYY}/{YYYYMMDD}.zip, each member {TICKER[0]}/{TICKER}.csv = one
row per option contract with vendor American-BSM IV + Greeks + bid/mid/ask + the
underlying mid, snapped 15:59 ET. ~5,700 tickers/day, 2017-2026, fully local.

Reads ONLY the requested member from each daily zip (no full extract). Causal by
construction: a row dated D uses only D's snapshot.
"""
from __future__ import annotations

from optengine.config import GREEKS_ROOT, PERTKR

import os, zipfile
import numpy as np, pandas as pd

ROOT = GREEKS_ROOT

# Keep only what the engine/validation needs; DROP the heavy *Time string columns,
# Ticker/TradeDate/MidRho. float64->float32 + categoricals => ~4x less RAM/chain
# (critical for the 16-core fan-out: a full chain-year drops ~2.4GB -> ~0.6GB).
_USECOLS = ["CallPut", "OptionStyle", "Strike", "Expiration", "YearsToMaturity",
            "DaysToMaturity", "UnderLastMidPrice", "LastBidPrice", "LastMidPrice",
            "LastAskPrice", "MidImpliedVol", "MidTheoPrice", "MidDelta", "MidGamma",
            "MidTheta", "MidVega", "ImpliedVolConvergence"]
_F32 = ["Strike", "YearsToMaturity", "UnderLastMidPrice", "LastBidPrice",
        "LastMidPrice", "LastAskPrice", "MidImpliedVol", "MidTheoPrice", "MidDelta",
        "MidGamma", "MidTheta", "MidVega"]


def _zip_path(date) -> str:
    d = pd.Timestamp(date)
    return os.path.join(ROOT, f"{d.year:04d}", f"{d.strftime('%Y%m%d')}.zip")


def available_dates(year) -> list[str]:
    p = os.path.join(ROOT, f"{int(year):04d}")
    if not os.path.isdir(p):
        return []
    return sorted(f[:8] for f in os.listdir(p) if f.endswith(".zip"))


_CACHE = {}
_CACHE_MAX = 700  # ~one ticker-year of trimmed chains; bounded for RAM safety


_PK_DIR = PERTKR
_PK = {}          # ticker -> (full_df, sorted asof_i array)
_PK_MAX = 3       # bound: few full-history tickers in RAM at once


def _load_pk_full(path):
    df = pd.read_parquet(path)
    ei = df["expiry_i"].astype("int64").clip(lower=10000101).astype(str)
    df["expiry"] = pd.to_datetime(ei, format="%Y%m%d", errors="coerce")
    df["spread"] = (df["LastAskPrice"] - df["LastBidPrice"]).astype("float32")
    return df, df["asof_i"].to_numpy()


def load_greeks_day(date, ticker):
    """One ticker's chain on one date, or None. FAST PATH = per-ticker parquet
    (pertkr/<ticker>.parquet loaded once + searchsorted slice); else the daily zip."""
    yi = int(pd.Timestamp(date).strftime("%Y%m%d"))
    pkp = os.path.join(_PK_DIR, f"{ticker}.parquet")
    if os.path.exists(pkp):
        ent = _PK.get(ticker)
        if ent is None:
            if len(_PK) >= _PK_MAX:
                _PK.clear()
            ent = _load_pk_full(pkp)
            _PK[ticker] = ent
        full, aof = ent
        lo = int(np.searchsorted(aof, yi, "left"))
        hi = int(np.searchsorted(aof, yi, "right"))
        return full.iloc[lo:hi] if hi > lo else None
    key = (str(yi), str(ticker))
    if key in _CACHE:
        return _CACHE[key]
    df = _load_uncached(date, ticker)
    if len(_CACHE) < _CACHE_MAX:
        _CACHE[key] = df
    return df


def clear_cache():
    """Drop the chain caches (call between tickers in a fan worker to bound RAM)."""
    _CACHE.clear()
    _PK.clear()


def _load_uncached(date, ticker):
    zp = _zip_path(date)
    if not os.path.exists(zp):
        return None
    member = f"{ticker[0].upper()}/{ticker}.csv"
    try:
        with zipfile.ZipFile(zp) as z, z.open(member) as fh:
            df = pd.read_csv(fh, usecols=lambda c: c in _USECOLS)
    except (KeyError, zipfile.BadZipFile):
        return None
    if not len(df):
        return None
    for c in _F32:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df["cp"] = np.where(df["CallPut"].astype(str).str.upper().str[0] == "C", 1, -1).astype(np.int8)
    df["expiry"] = pd.to_datetime(df["Expiration"].astype("Int64").astype(str),
                                  format="%Y%m%d", errors="coerce")
    df["asof_date"] = pd.Timestamp(str(date))
    df["DaysToMaturity"] = pd.to_numeric(df.get("DaysToMaturity"), errors="coerce").astype("float32")
    df["spread"] = (df["LastAskPrice"] - df["LastBidPrice"]).astype("float32")
    df["spread_pct"] = (df["spread"] / df["LastMidPrice"].replace(0, np.nan)).astype("float32")
    for c in ("CallPut", "OptionStyle", "ImpliedVolConvergence"):
        if c in df:
            df[c] = df[c].astype("category")
    df.drop(columns=["Expiration"], inplace=True)
    return df


def load_greeks_range(ticker, start, end, dates=None):
    """Concatenate one ticker's daily chains over [start, end] (inclusive)."""
    if dates is None:
        yrs = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
        dates = [d for y in yrs for d in available_dates(y)]
    s = pd.Timestamp(start).strftime("%Y%m%d")
    e = pd.Timestamp(end).strftime("%Y%m%d")
    out = []
    for d in dates:
        if d < s or d > e:
            continue
        df = load_greeks_day(d, ticker)
        if df is not None and len(df):
            out.append(df)
    return pd.concat(out, ignore_index=True) if out else None


_FRAME_COLS = ["asof_i", "expiry_i", "cp", "Strike", "DaysToMaturity", "MidDelta",
               "LastBidPrice", "LastMidPrice", "LastAskPrice", "UnderLastMidPrice",
               "converged", "MidImpliedVol"]


def build_array_frames(ticker):
    """Fast vectorized {yyyymmdd: array-frame} from per-ticker parquet — one read + all-
    numpy per-day slicing (parquet is sorted by asof_i, so each day's rows are contiguous).
    Replaces the slow per-day pandas converge+arrayify (≈20× faster). RAM: drops the full
    df after extracting numpy columns."""
    p = os.path.join(_PK_DIR, f"{ticker}.parquet")
    if not os.path.exists(p):
        return {}
    df = pd.read_parquet(p, columns=_FRAME_COLS)
    aof = df["asof_i"].to_numpy()
    cp = df["cp"].to_numpy(np.int8); K = df["Strike"].to_numpy(np.float32)
    dte = df["DaysToMaturity"].to_numpy(np.float32); de = df["MidDelta"].to_numpy(np.float32)
    bid = df["LastBidPrice"].to_numpy(np.float32); mid = df["LastMidPrice"].to_numpy(np.float32)
    ask = df["LastAskPrice"].to_numpy(np.float32); und = df["UnderLastMidPrice"].to_numpy(np.float64)
    exp = df["expiry_i"].to_numpy(np.int32); conv = df["converged"].to_numpy(bool)
    iv = df["MidImpliedVol"].to_numpy(np.float32)
    del df
    base = conv & (iv > 0) & (mid > 0) & (bid >= 0) & (ask > 0)
    uniq, starts = np.unique(aof, return_index=True)
    starts = np.append(starts, len(aof))
    frames = {}
    for k in range(len(uniq)):
        lo, hi = int(starts[k]), int(starts[k + 1])
        m = base[lo:hi]
        if not m.any():
            continue
        frames[str(int(uniq[k]))] = {
            "cp": cp[lo:hi][m], "K": K[lo:hi][m], "dte": dte[lo:hi][m],
            "delta": de[lo:hi][m], "bid": bid[lo:hi][m], "mid": mid[lo:hi][m],
            "ask": ask[lo:hi][m], "exp": exp[lo:hi][m],
            "under": float(np.median(und[lo:hi][m])),
        }
    return frames
