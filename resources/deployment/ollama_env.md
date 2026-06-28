# Ollama daemon environment (GPU-upgrade tuning)

Hardware: **NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM**.

> **2026-06-28 finding:** raising `OLLAMA_NUM_PARALLEL` 2→8 was **measured to give
> no extraction speed-up** (GPU-compute-bound; NP=2 ≈ NP=8 ≈ ~40 s/9-doc batch).
> The daemon was reverted to `NUM_PARALLEL=2`. The block below is retained only as
> a reference for *how* to set daemon env if a future, non-compute-bound workload
> ever benefits — it is NOT a recommended change today.

The Ollama *daemon* (systemd) carries its own environment, separate from the
app's `.env`. The app's `OLLAMA_NUM_PARALLEL` in `.env` does **not** reach the
daemon — only the systemd unit does. Set these on the daemon:

```ini
# /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_NUM_PARALLEL=8"
Environment="OLLAMA_MAX_LOADED_MODELS=3"
Environment="OLLAMA_KEEP_ALIVE=30m"
Environment="OLLAMA_FLASH_ATTENTION=1"
```

Apply:
```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf < the block above
sudo systemctl daemon-reload
sudo systemctl restart ollama
systemctl show ollama | grep -i Environment   # verify NUM_PARALLEL=8
```

Rationale:
- `OLLAMA_NUM_PARALLEL=8` — was `2`, a survival setting for the old contended
  card. The app-side cap (`OLLAMA_MAX_CONCURRENT`, default now 8 in
  `src/services/ollama_client.py`) matches this so the semaphore and the daemon
  agree.
- `OLLAMA_MAX_LOADED_MODELS=3` — extract (~8 GB) + unified/latest (~18 GB) +
  nuextract (~6 GB) ≈ 32 GB, all resident at once with large headroom on 96 GB.
- `OLLAMA_KEEP_ALIVE=30m` — was `5m`; VRAM is now ample, so keep models warm and
  stop paying reload latency between bursts.

The model itself loads fully onto the GPU via `Modelfile` `PARAMETER num_gpu -1`
(was `25`, a partial offload forced by the old small card).
