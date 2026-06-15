"""Path configuration for optengine.

Every data source and output location resolves from an environment variable,
falling back to a directory under the repository root. No machine-specific path
is baked into the source, so the engine runs unchanged anywhere once the
variables (or the default layout) point at the data.

Source data is licensed and is not distributed with the code. Point the
``OPTENGINE_*`` variables at your own copy:

    OPTENGINE_GREEKS    daily equity-option Greeks store (per-day zip archives)
    OPTENGINE_PANEL     prebuilt daily surface panel (parquet)
    OPTENGINE_SAMPLE    single-file Greeks sample used by the validation checks
    OPTENGINE_FOP_S3    futures-options 1-minute TAQ bucket template, with a
                        ``{year}`` placeholder, e.g. ``my-fop-taq-{year}``

Working directories (built locally by the pipeline) default under the repo root
and can be redirected with OPTENGINE_PERTKR, OPTENGINE_FINDINGS, and the rest.
"""
from __future__ import annotations
import os
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[1])


def _path(env: str, *default_parts: str) -> str:
    return os.environ.get(env) or os.path.join(REPO_ROOT, *default_parts)


# --- Licensed source data (not distributed) ---------------------------------
GREEKS_ROOT     = _path("OPTENGINE_GREEKS", "data", "greeks")
SURFACE_PANEL   = _path("OPTENGINE_PANEL", "data", "surface_panel.parquet")
ALGOSEEK_SAMPLE = _path("OPTENGINE_SAMPLE", "data", "sample_greeks.csv.gz")

# Futures-options object-store bucket template (holders set this to their own).
FOP_S3_BUCKET   = os.environ.get("OPTENGINE_FOP_S3", "")

# --- Working directories (built locally) ------------------------------------
PERTKR     = _path("OPTENGINE_PERTKR", "pertkr")
FINDINGS   = _path("OPTENGINE_FINDINGS", "findings")
LEDGERS    = _path("OPTENGINE_LEDGERS", "findings", "ledgers")
RICHNESS   = _path("OPTENGINE_RICHNESS", "richness")
FOP_PERTKR = _path("OPTENGINE_FOP_PERTKR", "fop_pertkr")
FOP_RAW    = _path("OPTENGINE_FOP_RAW", "fop_raw")
ODTE_RAW   = _path("OPTENGINE_ODTE_RAW", "odte_raw")
FOP_TMP    = _path("OPTENGINE_FOP_TMP", ".fop_tmp")

CHAR_PANEL = os.path.join(FINDINGS, "char_panel.parquet")
