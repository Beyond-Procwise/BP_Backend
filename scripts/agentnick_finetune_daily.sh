#!/bin/bash
# Daily AgentNick fine-tune wrapper.
#
# Schedule: triggered by cron at 18:30 UTC (= 00:00 IST midnight) every day.
# Window:   8h30m — hard-killed at 03:00 UTC (= 08:30 IST).
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

  echo "--- GPU state (before) ---"
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv 2>&1 || echo "(nvidia-smi unavailable)"
  echo
  echo "--- procwise state (before) ---"
  systemctl is-active procwise 2>&1 || true
  echo

  # Stop procwise so the QLoRA loader has the GPU to itself. Without
  # this, full-precision fine-tune (~14 GiB) OOMs against the live
  # extractor's 9-10 GiB footprint. Sudoers entry at
  # /etc/sudoers.d/procwise-finetune permits passwordless stop/start
  # for user `muthu`.
  echo "--- Stopping procwise to free GPU ---"
  sudo /usr/bin/systemctl stop procwise || {
    echo "WARN: failed to stop procwise — fine-tune will likely OOM"
  }
  # Give CUDA + Ollama a few seconds to release VRAM cleanly
  sleep 10
  echo "--- GPU state (after stop) ---"
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv 2>&1 || true
  echo

  echo "--- Triggering run_overnight_finetune.sh (8h30m hard limit) ---"
  cd "${BP_ROOT}"
  # 8h30m = 30600s. Timeout's --kill-after=60s gives QLoRA a graceful
  # exit window if the budget is exceeded so adapter checkpoints flush.
  timeout --signal=SIGTERM --kill-after=60s 30600s bash "${BP_ROOT}/scripts/run_overnight_finetune.sh"
  rc=$?
  echo
  echo "--- Result ---"
  echo "exit code: ${rc}"
  echo "End (UTC): $(date -u)"
  echo "End (IST): $(TZ=Asia/Kolkata date)"

  if [ "${rc}" -eq 0 ]; then
    echo "STATUS: SUCCESS"
  elif [ "${rc}" -eq 124 ] || [ "${rc}" -eq 137 ]; then
    echo "STATUS: TIMEOUT (8h30m window exhausted)"
  else
    echo "STATUS: FAILED rc=${rc}"
  fi

  # ALWAYS bring procwise back up, even on fine-tune failure or
  # timeout. The live extraction service must not stay offline because
  # of a training-pipeline issue. systemctl start is idempotent.
  echo
  echo "--- Restarting procwise ---"
  sudo /usr/bin/systemctl start procwise || {
    echo "ERROR: failed to start procwise — manual intervention required"
  }
  sleep 5
  echo "--- procwise state (after) ---"
  systemctl is-active procwise 2>&1 || true
  echo "End-of-wrapper (UTC): $(date -u)"
} 2>&1 | tee -a "${LOG}"
