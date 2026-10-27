# EP64 — Distributed Training Explained

**Deep Learning Series · Episode 64 · Module 14 — Scaling Deep Learning**

> Your model does not fit. Here’s what the people who train giant models actually do — with real multi-process runs.

## What this episode measures

- **Real multi-process DDP** (gloo backend on CPU cores — same code works on GPUs with nccl)
- **Gradient accumulation** with `no_sync()`
- **FSDP2 memory** (fully_shard) — measured bytes per worker
- **Throughput** (samples/second of the whole group)
- **Ring all-reduce** replayed message-by-message and checked against `torch.distributed.all_reduce`
- **Pipeline schedule** (GPipe-style) replayed and checked against the formula `M / (M + S − 1)`

No number in the video is typed by hand. Everything comes from `ep64_results.json`.

## Quick start

```bash
pip install torch numpy matplotlib
python ep64.py
