#!/bin/bash
# Daily AgentNick fine-tune wrapper.
#
# Schedule: triggered by cron at 19:30 UTC (= 1am IST) every day.
# Window:   8 hours max — hard-killed at 03:30 UTC (= 9am IST).
#
# Procurement GPU coordination: BeyondProcwise/AgentNick:extract (7B Q8)
# stays loaded for live extraction during the day, but full-precision
# fine-tuning needs ~14 GiB GPU which collides with the procwise service.
# The cron window matches the documented downtime — procwise should be
# stopped before this runs, or the QLoRA loader will OOM. The wrapper
# logs procwise state and continues; the underlying script either
# succeeds or fails with a clean log, never silently corrupting state.

set -uo pipefail

BP_ROOT="/home/muthu/PycharmProjects/BP_Backend"
LOG_DIR="${BP_ROOT}/logs"
mkdir -p "${LOG_DIR}"

STAMP="$(date -u +%Y%m%d-%H%M%S)"
LOG="${LOG_DIR}/agentnick_finetune_${STAMP}.log"

{
  echo "=== AgentNick daily fine-tune ==="
  echo "Start (UTC): $(date -u)"
  echo "Start (IST): $(TZ=Asia/Kolkata date)"
  echo "Host: $(hostname)"
  echo

  echo "--- GPU state ---"
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv 2>&1 || echo "(nvidia-smi unavailable)"
  echo
  echo "--- procwise state ---"
  systemctl is-active procwise 2>&1 || true
  echo

  echo "--- Triggering run_overnight_finetune.sh (8h hard limit) ---"
  cd "${BP_ROOT}"
  timeout --signal=SIGTERM --kill-after=60s 8h bash "${BP_ROOT}/scripts/run_overnight_finetune.sh"
  rc=$?
  echo
  echo "--- Result ---"
  echo "exit code: ${rc}"
  echo "End (UTC): $(date -u)"
  echo "End (IST): $(TZ=Asia/Kolkata date)"

  if [ "${rc}" -eq 0 ]; then
    echo "STATUS: SUCCESS"
  elif [ "${rc}" -eq 124 ] || [ "${rc}" -eq 137 ]; then
    echo "STATUS: TIMEOUT (8h window exhausted)"
  else
    echo "STATUS: FAILED rc=${rc}"
  fi
} 2>&1 | tee -a "${LOG}"
