#!/bin/bash
# Mega-run driver: pertkr breadth build -> 35x682 corpus -> deflate -> portfolio(+broad)
# -> report -> robust. Pinned to cores 0-15 with single-thread BLAS so it shares a box
# politely. Stops on first failure; logs each stage to findings/MEGA_STATUS.md so progress
# is monitorable. Resumable: the breadth build skips already-built tickers.
set -e
set -o pipefail   # so `python | tee` returns python's exit status (else a failed stage is masked by tee)
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PIN="taskset -c 0-15"
ST=findings/MEGA_STATUS.md
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a $ST; }

avail(){ awk '/MemAvailable/{printf "%.0f",$2/1024/1024}' /proc/meminfo; }
need_ram(){ local need=$1; local a=$(avail); if [ "$a" -lt "$need" ]; then log "ABORT: avail ${a}G < ${need}G needed"; exit 1; fi; }

log "=== MEGA-RUN START (avail $(avail)G) ==="

# 1) breadth build to ~682 (skip if already there)
NB=$(ls pertkr/*.parquet 2>/dev/null | wc -l)
if [ "$NB" -lt 600 ]; then
  need_ram 22
  log "STAGE build_pertkr_breadth (have $NB tickers)"
  $PIN python3 tools/build_pertkr_breadth.py --workers 12 --chunk 130 --min_avail_gb 14 2>&1 | tee -a $ST
else
  log "STAGE build_pertkr_breadth SKIP (have $NB tickers)"
fi

# 2) corpus: 35 archetypes x all instruments
need_ram 18
log "STAGE corpus (35 archetypes x $(ls pertkr/*.parquet|wc -l) instruments)"
$PIN python3 tools/corpus.py --workers 12 --ram_floor 10 2>&1 | tee -a $ST

# 3) deflation (RAM-safe Gram N_eff)
log "STAGE deflate"
$PIN python3 tools/deflate.py 2>&1 | tee -a $ST

# 4) portfolios: hindsight survivor books + non-hindsight + ex-VIX
log "STAGE portfolio"
$PIN python3 tools/portfolio.py 2>&1 | tee -a $ST
( cd tools && $PIN python3 portfolio_broad.py 2>&1 ) | tee -a $ST

# 5) report + robustness
log "STAGE report"; $PIN python3 tools/report.py 2>&1 | tee -a $ST || true
log "STAGE robust"; $PIN python3 tools/robust.py 2>&1 | tee -a $ST || true

log "=== MEGA-RUN DONE (avail $(avail)G) ==="
