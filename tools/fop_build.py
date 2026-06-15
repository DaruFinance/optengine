"""Futures-options venue builder: stream-and-distill the Algoseek 1min-TAQ FO history
into per-root daily option chains (the pertkr analogue), with REAL bid/ask.

Per (root, date): pull each expiry zip -> take the 15:59-ET close-bar NBBO per strike ->
put-call-parity forward F (Black-76) -> Black-76 implied vol + delta -> append to the
root's daily chain. The raw zip is DELETED after distill (disk stays ~one zip; only the
compact chains persist). Resumable: dates already in the chain are skipped.

Snapshot = 15:59 ET (matches the OPRA 15:59 convention). Costs/multiplier/hedge are the
engine's job; this only builds the quotable chain.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import FOP_RAW, FOP_TMP, FOP_S3_BUCKET

import os, sys, io, glob, shutil, zipfile, subprocess, argparse, datetime as dt
import numpy as np, pandas as pd
from optengine.pricing import forward_from_parity, implied_vol, black76_greeks

OUT = FOP_RAW      # raw snapshots; fop_finalize -> fop_pertkr (engine format)
TMP = FOP_TMP
# Named credentials profile for the requester-pays object store (set your own).
PROFILE = os.environ.get("OPTENGINE_AWS_PROFILE", "")
MONTHS = {"F":1,"G":2,"H":3,"J":4,"K":5,"M":6,"N":7,"Q":8,"U":9,"V":10,"X":11,"Z":12}


def third_friday(y, m):
    d = dt.date(y, m, 1)
    fri = d + dt.timedelta((4 - d.weekday()) % 7)   # first Friday
    return fri + dt.timedelta(14)                    # third Friday


def _near_expiry(zipname, date_i, lo=-1, hi=5):
    """Keep only zips whose contract-month is lo..hi months ahead of the snapshot date
    (covers all archetype DTEs incl. calendar back-month ~3mo; drops far-dated LEAPS).
    Zip name = <root><monthcode><yeardigit>.zip, e.g. ESH4, BTCF3, OZMN3. Unparseable -> keep."""
    n = zipname[:-4] if zipname.endswith(".zip") else zipname
    if len(n) < 2 or n[-2] not in MONTHS or not n[-1].isdigit():
        return True
    dy, dm = date_i // 10000, (date_i // 100) % 100
    cy = (dy // 10) * 10 + int(n[-1])
    if cy < dy - 1:
        cy += 10
    ma = (cy - dy) * 12 + (MONTHS[n[-2]] - dm)
    return lo <= ma <= hi


def s3(*args):
    return subprocess.run(["aws", "s3", *args, "--profile", PROFILE, "--request-payer", "requester"],
                          capture_output=True, text=True, timeout=300)


_CL = None      # lazy per-process boto3 client (fork-safe: created on first use in each child)


def _client():
    global _CL
    if _CL is None:
        import boto3
        from botocore.config import Config
        _CL = boto3.Session(profile_name=PROFILE).client(
            "s3", config=Config(max_pool_connections=24, retries={"max_attempts": 4, "mode": "adaptive"}))
    return _CL


_NEED = ("CallPut", "Strike", "Month", "ExpirationYear", "LocalTimeBarStart", "CloseBidPrice", "CloseAskPrice")


def _scan_csv(txt, ix, snap):
    """Pandas-free: reversed line-scan for the last 2-sided quote at/before `snap` (the 15:59-ET
    bar sits near the end of each contract's session, so this early-breaks in a few iterations)."""
    nl = txt.find("\n")
    if nl < 0:
        return None
    cp_i, k_i, m_i, y_i, lt_i, b_i, a_i = ix
    lines = txt[nl + 1:].split("\n")
    fb = None
    for ln in reversed(lines):
        if not ln:
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
        if fb is None:
            fb = f                       # last valid quote anywhere (fallback)
        if f[lt_i][:5] <= snap:
            return (f[cp_i], float(f[k_i]), f[m_i], int(f[y_i]), bid, ask)
    if fb is not None:
        return (fb[cp_i], float(fb[k_i]), fb[m_i], int(fb[y_i]), float(fb[b_i]), float(fb[a_i]))
    return None


def distill_zip(blob, snap="15:59"):
    rows = []
    z = zipfile.ZipFile(io.BytesIO(blob))
    ix = None
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
            r = _scan_csv(txt, ix, snap)
        except Exception:
            r = None
        if r is not None:
            rows.append(r)
    return rows


def build_root(root, dates, year, workers_note=""):
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    out_path = os.path.join(OUT, f"{root}.parquet")
    have = set()
    if os.path.exists(out_path):
        have = set(pd.read_parquet(out_path, columns=["asof_i"])["asof_i"].unique().tolist())
    allrows, ndone = [], 0
    for date in dates:
        di = int(date)
        if di in have:
            continue
        base = f"s3://{FOP_S3_BUCKET.format(year=year)}/{date}/{root}/"
        dd = os.path.join(TMP, f"{root}_{date}")
        shutil.rmtree(dd, ignore_errors=True); os.makedirs(dd, exist_ok=True)
        # ONE recursive pull -> awscli fetches the date's ~all expiry zips CONCURRENTLY (latency win)
        if s3("cp", base, dd, "--recursive", "--quiet").returncode != 0:
            shutil.rmtree(dd, ignore_errors=True); continue
        day_rows = []
        for zf in glob.glob(os.path.join(dd, "*.zip")):
            try:
                day_rows += distill_zip(open(zf, "rb").read())
            except Exception:
                pass
        shutil.rmtree(dd, ignore_errors=True)
        if day_rows:
            c = pd.DataFrame(day_rows, columns=["cp", "K", "mcode", "ey", "bid", "ask"]); c["asof_i"] = di
            allrows.append(c); ndone += 1
        if ndone and len(allrows) >= 25:                 # incremental checkpoint (resumable + visible)
            _flush(out_path, allrows); allrows = []
            print(f"{root} {year}: {ndone} dates done", flush=True)
    if allrows:
        _flush(out_path, allrows)
    return out_path


def _flush(out_path, allrows):
    if not allrows:
        return
    new = pd.concat(allrows, ignore_index=True)
    if os.path.exists(out_path):
        new = pd.concat([pd.read_parquet(out_path), new], ignore_index=True)
    new.drop_duplicates(["asof_i", "mcode", "ey", "cp", "K"]).sort_values(
        ["asof_i", "mcode", "ey", "cp", "K"]).to_parquet(out_path, compression="zstd", index=False)


def list_dates(root, year):
    cl = _client()
    bucket = f"us-futures-options-1min-taq-{year}"
    dates, tok = [], None
    while True:
        kw = dict(Bucket=bucket, Delimiter="/", RequestPayer="requester")
        if tok:
            kw["ContinuationToken"] = tok
        r = cl.list_objects_v2(**kw)
        for p in r.get("CommonPrefixes", []):
            d = p["Prefix"].strip("/")
            if d.isdigit():
                dates.append(d)
        if r.get("IsTruncated"):
            tok = r.get("NextContinuationToken")
        else:
            break
    return sorted(dates)


def build_one(args):
    """Pull+distill ONE (root,date) -> fop_raw/<root>/<date>.parquet via boto3 (in-process, reused
    client = NO per-task subprocess spawn), parallel zip downloads to memory. Resumable, race-free,
    atomic write."""
    root, date, year = args
    outdir = os.path.join(OUT, root)
    fp = os.path.join(outdir, f"{date}.parquet")
    if os.path.exists(fp):
        return 0
    bucket = f"us-futures-options-1min-taq-{year}"; prefix = f"{date}/{root}/"
    try:
        resp = _client().list_objects_v2(Bucket=bucket, Prefix=prefix, RequestPayer="requester")
    except Exception:
        return 0
    names = [o["Key"].split("/")[-1] for o in resp.get("Contents", []) if o["Key"].endswith(".zip")]
    keep = [n for n in names if _near_expiry(n, int(date))]      # skip far-dated LEAPS (archetypes use <=~90 DTE)
    if not keep:
        return 0
    os.makedirs(TMP, exist_ok=True)
    dd = os.path.join(TMP, f"{root}_{date}")
    shutil.rmtree(dd, ignore_errors=True); os.makedirs(dd, exist_ok=True)
    inc = []
    for n in keep:
        inc += ["--include", n]
    try:
        # boto3 listed cheaply; awscli multipart-pulls only the near-expiry zips (the data lever)
        if s3("cp", f"s3://{bucket}/{prefix}", dd, "--recursive", "--quiet", "--exclude", "*", *inc).returncode != 0:
            return 0
        rows = []
        for zf in glob.glob(os.path.join(dd, "*.zip")):
            try:
                rows += distill_zip(open(zf, "rb").read())
            except Exception:
                pass
        if not rows:
            return 0
        os.makedirs(outdir, exist_ok=True)
        c = pd.DataFrame(rows, columns=["cp", "K", "mcode", "ey", "bid", "ask"]); c["asof_i"] = int(date)
        c.to_parquet(fp + ".tmp", compression="zstd", index=False); os.replace(fp + ".tmp", fp)
        return len(c)
    finally:
        shutil.rmtree(dd, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="ES")
    ap.add_argument("--year", default="2023")
    ap.add_argument("--ndates", type=int, default=10)
    a = ap.parse_args()
    dates = list_dates(a.root, a.year)[:a.ndates]
    print(f"building {a.root} {a.year}: {len(dates)} dates {dates[:3]}...")
    build_root(a.root, dates, a.year)
