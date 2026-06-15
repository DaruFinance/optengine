"""optengine.fan — RAM-safe parallel fan-out of WFOs across (instrument x archetype).

Pinned to cores 0-15 by the launcher (taskset -c 0-15); each worker is single-threaded
(inherited OMP_NUM_THREADS=1) so W workers use W of the 16 cores. maxtasksperchild=1 frees
each task's memory; clear_cache() bounds each worker to one ticker's working set; a
/proc/meminfo floor check aborts before launch if headroom is short.
"""
from __future__ import annotations
import multiprocessing as mp
import pandas as pd
from .data import greeks_store as G
from . import archetypes as _arch
from .wfo import wfo


def ram_available_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1024 / 1024
    except Exception:
        return 999.0


def _task(t):
    inst, arch_name, grid, exit_grid, years = t
    build = getattr(_arch, arch_name)
    G.clear_cache()
    dates = []
    for y in years:
        dates += G.available_dates(y)
    try:
        oos, _wins, nc = wfo(inst, dates, build, grid, exit_grid=exit_grid,
                             is_len=126, oos_len=63, obj="rrr", delta_hedge=True)
        if oos is None or not len(oos):
            r = dict(instrument=inst, archetype=arch_name, n=0, oos_net=0.0, ncombos=nc)
        else:
            nnet = oos["net"]
            r = dict(instrument=inst, archetype=arch_name, n=len(oos),
                     oos_net=round(float(nnet.sum())), win=round(float((nnet > 0).mean()), 2),
                     med=round(float(nnet.median())), worst=round(float(nnet.min())), ncombos=nc)
    except Exception as e:
        r = dict(instrument=inst, archetype=arch_name, error=repr(e)[:120])
    G.clear_cache()
    return r


def fan(tasks, workers=12, ram_floor_gb=10.0):
    avail = ram_available_gb()
    if avail < ram_floor_gb:
        raise RuntimeError(f"RAM available {avail:.1f} GB < floor {ram_floor_gb} GB — aborting to protect WSL")
    workers = min(workers, len(tasks))
    with mp.Pool(workers, maxtasksperchild=1) as pool:
        res = pool.map(_task, tasks)
    return pd.DataFrame(res)
