"""EP64 - Distributed training: run REAL multi-process DDP and FSDP on your CPU cores, replay a ring all-reduce and a pipeline schedule step by step, and measure what scaling really costs.
    python ep64_distributed.py     # 2-4 minutes; writes the distributed/ folder (dashboard.png) and ep64_results.json
pip install torch numpy matplotlib
# the "GPUs" are processes on your CPU (gloo backend): the same code runs on real GPUs with the nccl backend. A missing feature is skipped and reported, not faked.
"""
import contextlib, json, os, pickle, socket, tempfile, time, warnings
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
warnings.filterwarnings("ignore")
# ── 1. CONFIG ─────────────────────────────────────────────────────────────
WORLDS = [1, 2, 4]                   # how many worker processes (the "GPUs")
STEPS = 5                            # training steps in the equivalence tests
BATCH = 64                           # global batch of the equivalence tests
HIDDEN = 512                         # width of the model
THR_BATCH, THR_STEPS, THR_REPS = 1024, 10, 3     # throughput test: global batch, timed steps, repeats (the median is used)
RING_WORKERS, RING_LEN = 4, 8        # the ring all-reduce we replay step by step: workers, numbers per worker
STAGES, MICROBATCHES = 4, [1, 2, 4, 8, 16]       # the pipeline we replay: stages, micro-batch counts
GPU_GB = 80                          # arithmetic only: the memory of one big GPU
OUT = Path(__file__).with_name("distributed")    # dashboard.png ends up here
# ── 2. MODEL + DATA: tiny on purpose, so the maths can be checked exactly ──
def make():
    torch.manual_seed(0); return nn.Sequential(nn.Linear(64, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 10))
def batch(step, n):                  # the same global batch for everyone: the seed is the step number
    g = torch.Generator().manual_seed(step); return torch.randn(n, 64, generator=g), torch.randint(0, 10, (n,), generator=g)
def flat(net): return torch.cat([p.detach().reshape(-1) for p in net.parameters()]).numpy().astype(np.float64)
def single(micro=1):                 # the reference: ONE process, the whole batch (optionally cut into `micro` pieces that are accumulated)
    net = make(); opt = torch.optim.SGD(net.parameters(), lr=0.1)
    for s in range(STEPS):
        x, y = batch(s, BATCH); opt.zero_grad()
        for xs, ys in zip(x.chunk(micro), y.chunk(micro)): (nn.functional.cross_entropy(net(xs), ys) / micro).backward()
        opt.step()
    return flat(net)
def nbytes(ts): return sum(t.numel() * t.element_size() for t in ts)
# ── 3. REAL PROCESSES: DDP, gradient accumulation, FSDP2 memory, throughput ──
def worker(rank, world, port, path):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port)); torch.set_num_threads(1); dist.init_process_group("gloo", rank=rank, world_size=world); out = {}
    from torch.nn.parallel import DistributedDataParallel as DDP
    def mine(x, y): lo, hi = rank * len(x) // world, (rank + 1) * len(x) // world; return x[lo:hi], y[lo:hi]     # this worker's slice of the global batch
    net = make(); ddp = DDP(net); opt = torch.optim.SGD(ddp.parameters(), lr=0.1)                              # 1. DDP: every worker trains on its slice, gradients are averaged
    for s in range(STEPS): xs, ys = mine(*batch(s, BATCH)); opt.zero_grad(); nn.functional.cross_entropy(ddp(xs), ys).backward(); opt.step()
    out["ddp"] = flat(net)
    net = make(); ddp = DDP(net); opt = torch.optim.SGD(ddp.parameters(), lr=0.1)                              # 2. DDP + gradient accumulation: skip the all-reduce on the first micro-batch
    for s in range(STEPS):
        xs, ys = mine(*batch(s, BATCH)); opt.zero_grad()
        for k, (a, b) in enumerate(zip(xs.chunk(2), ys.chunk(2))):
            with (ddp.no_sync() if k == 0 else contextlib.nullcontext()): (nn.functional.cross_entropy(ddp(a), b) / 2).backward()
        opt.step()
    out["accum"] = flat(net)
    v = torch.arange(RING_LEN, dtype=torch.float64) * (rank + 1) + rank; dist.all_reduce(v); out["allreduce"] = v.numpy()     # 3. the real all-reduce, to check the replay against
    net = make(); ddp = DDP(net); opt = torch.optim.Adam(ddp.parameters(), lr=1e-3); xs, ys = mine(*batch(0, BATCH))                # 4. memory of the replicated copy (Adam keeps two numbers per weight)
    nn.functional.cross_entropy(ddp(xs), ys).backward(); opt.step()
    out["mem_ddp"] = dict(params=nbytes(net.parameters()), grads=nbytes(p.grad for p in net.parameters()), opt=nbytes(t for st in opt.state.values() for t in st.values() if torch.is_tensor(t) and t.ndim > 0))
    try:                                                                                                           # 5. memory with FSDP2: each worker keeps only its shard
        try: from torch.distributed.fsdp import fully_shard
        except ImportError: from torch.distributed._composable.fsdp import fully_shard
        net = make()
        for m in net:
            if isinstance(m, nn.Linear): fully_shard(m)
        fully_shard(net); opt = torch.optim.Adam(net.parameters(), lr=1e-3); nn.functional.cross_entropy(net(xs), ys).backward(); opt.step()
        loc = lambda t: t.to_local() if hasattr(t, "to_local") else t
        out["mem_fsdp"] = dict(params=nbytes(loc(p) for p in net.parameters()), grads=nbytes(loc(p.grad) for p in net.parameters()), opt=nbytes(loc(t) for st in opt.state.values() for t in st.values() if torch.is_tensor(t) and t.ndim > 0))
    except Exception as e: out["mem_fsdp_err"] = f"{type(e).__name__}: {str(e)[:120]}"
    net = make(); ddp = DDP(net); opt = torch.optim.SGD(ddp.parameters(), lr=0.1); times = []                    # 6. throughput: samples per second of the whole group
    for rep in range(THR_REPS + 1):
        dist.barrier(); t = time.perf_counter()
        for s in range(THR_STEPS if rep else 2): xs, ys = mine(*batch(s, THR_BATCH)); opt.zero_grad(); nn.functional.cross_entropy(ddp(xs), ys).backward(); opt.step()
        dist.barrier(); times.append(time.perf_counter() - t) if rep else None
    out["sps"] = THR_BATCH * THR_STEPS / float(np.median(times))
    if rank == 0: pickle.dump(out, open(path, "wb"))                                                        # a file, not a queue: a queue would block on these big arrays
    dist.destroy_process_group()
def run_world(world):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close(); fd, path = tempfile.mkstemp(suffix=".pkl"); os.close(fd)
    mp.spawn(worker, args=(world, port, path), nprocs=world, join=True); out = pickle.load(open(path, "rb")); os.remove(path); return out
# ── 4. REPLAYS: the ring all-reduce and the pipeline schedule, event by event ──
def ring_allreduce(vs):              # every worker ends with the sum, and each one only ever talks to its right-hand neighbour
    W = len(vs); ch = [np.array_split(v.astype(np.float64), W) for v in vs]; sent = [0] * W; ev = []
    for phase in ("reduce", "gather"):
        for s in range(W - 1):       # reduce: neighbours add their chunk in; gather: neighbours pass the finished chunks round
            msgs = [(i, (i + 1) % W, (i - s) % W if phase == "reduce" else (i + 1 - s) % W) for i in range(W)]
            for i, j, c in msgs:
                data = ch[i][c].copy(); sent[i] += data.nbytes; ev.append(dict(phase=phase, step=s, src=i, dst=j, chunk=c, values=[float(x) for x in data]))
                ch[j][c] = ch[j][c] + data if phase == "reduce" else data
    return [np.concatenate(c) for c in ch], ev, sent
def naive_allreduce_bytes(W, n): return 2 * (W - 1) * n * 8           # everyone sends to one root, the root sends the sum back: the root's traffic
def pipeline(S, M, fwd=1, bwd=2):    # GPipe: all forward passes flow through the stages, then all backward passes flow back
    done, free, ev = {}, [0] * S, []
    jobs = [[("F", m) for m in range(M)] + [("B", m) for m in reversed(range(M))] for _ in range(S)]; todo = [list(j) for j in jobs]
    while any(todo):
        for s in range(S):
            if not todo[s]: continue
            k, m = todo[s][0]; dep = [("F", s - 1, m)] if k == "F" and s else [("B", s + 1, m)] if k == "B" and s < S - 1 else [("F", s, m)] if k == "B" else []
            if all(d in done for d in dep):
                t0 = max([free[s]] + [done[d] for d in dep]); t1 = t0 + (fwd if k == "F" else bwd); done[(k, s, m)] = t1; free[s] = t1; ev.append(dict(stage=s, kind=k, mb=m, start=t0, end=t1)); todo[s].pop(0)
    mk = max(e["end"] for e in ev); return ev, mk, sum(e["end"] - e["start"] for e in ev) / (S * mk)
# ── 5. DASHBOARD ──────────────────────────────────────────────────────────
def dashboard(R, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "#050815", "axes.facecolor": "#0d1117", "axes.edgecolor": "#6272a4", "text.color": "#f8f8f2", "axes.labelcolor": "#f8f8f2",
                         "xtick.color": "#f8f8f2", "ytick.color": "#f8f8f2", "axes.grid": True, "grid.alpha": 0.15}); fig, ax = plt.subplots(2, 2, figsize=(13, 7.4)); W = [str(w) for w in R["worlds"]]
    a = ax[0, 0]; a.bar(W, [R["thr"][w]["sps"] for w in W], color="#8be9fd"); a.plot(W, [R["thr"]["1"]["sps"] * int(w) for w in W], "--o", color="#50fa7b", label="perfect scaling")
    a.set_title(f"Throughput, samples/second ({R['cores']} CPU core{'s' if R['cores'] != 1 else ''})", fontsize=10); a.set_xlabel("workers"); a.legend(fontsize=7)
    a = ax[0, 1]; x = np.arange(len(W)); bot = np.zeros(len(W)); mem = R["mem"]
    for k, c in (("params", "#8be9fd"), ("grads", "#ffb86c"), ("opt", "#ff79c6")):
        v = np.array([mem[w]["fsdp"][k] / 1e6 if mem[w].get("fsdp") else np.nan for w in W]); a.bar(x + 0.2, v, 0.4, bottom=np.nan_to_num(bot), color=c, label=k); bot = bot + np.nan_to_num(v)
    bot = np.zeros(len(W))
    for k, c in (("params", "#8be9fd"), ("grads", "#ffb86c"), ("opt", "#ff79c6")): v = np.array([mem[w]["ddp"][k] / 1e6 for w in W]); a.bar(x - 0.2, v, 0.4, bottom=bot, color=c, alpha=0.55); bot = bot + v
    a.set_xticks(x, [f"{w}\nDDP | FSDP" for w in W], fontsize=8); a.set_title("Memory per worker, MB (left: DDP copies everything, right: FSDP shards)", fontsize=9); a.legend(fontsize=7)
    a = ax[1, 0]; ws = R["ring"]["scale"]["workers"]; a.plot(ws, [v / 1e3 for v in R["ring"]["scale"]["ring_kb"]], "-o", color="#50fa7b", label="ring: busiest worker"); a.plot(ws, [v / 1e3 for v in R["ring"]["scale"]["naive_kb"]], "-o", color="#ff5555", label="naive: the root")
    a.set_title("Bytes the busiest worker must move per all-reduce, KB", fontsize=10); a.set_xlabel("workers"); a.legend(fontsize=7)
    a = ax[1, 1]; ms_ = R["pipe"]["microbatches"]; a.plot(ms_, R["pipe"]["util"], "-o", color="#ffb86c", label="replayed schedule"); a.plot(ms_, [m / (m + R["pipe"]["stages"] - 1) for m in ms_], "x", color="#f8f8f2", ms=10, label="M / (M + S - 1)")
    a.set_ylim(0, 1.05); a.set_title(f"Pipeline of {R['pipe']['stages']} stages: share of time each stage is busy", fontsize=10); a.set_xlabel("micro-batches"); a.legend(fontsize=7)
    fig.suptitle("Distributed training: what scaling really costs", fontsize=14, color="#8be9fd"); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
# ── 6. RUN: real processes, two replays, one results file ─────────────────
def run():
    OUT.mkdir(exist_ok=True); torch.set_num_threads(1); ref1, ref2 = single(1), single(2); res = {w: run_world(w) for w in WORLDS}; err = {w: r["mem_fsdp_err"] for w, r in res.items() if "mem_fsdp_err" in r}
    rng = np.random.default_rng(0); vs = [rng.integers(1, 9, RING_LEN).astype(np.float64) for _ in range(RING_WORKERS)]; out, ev, sent = ring_allreduce(vs)
    real = [res[w]["allreduce"] for w in WORLDS]; vv = lambda w: [np.arange(RING_LEN) * (r + 1) + r for r in range(w)]; chk = max(float(np.abs(ring_allreduce(vv(w))[0][0] - res[w]["allreduce"]).max()) for w in WORLDS)
    scale = dict(workers=[2, 4, 8, 16, 32]); n = 100000; sc = [ring_allreduce([np.ones(n)] * w)[2] for w in scale["workers"]]
    scale.update(ring_kb=[max(s) for s in sc], naive_kb=[naive_allreduce_bytes(w, n) for w in scale["workers"]])
    pipes = {m: pipeline(STAGES, m) for m in MICROBATCHES}; bpp = 4 + 4 + 8                                       # bytes per weight in fp32 training with Adam: weight, gradient, two Adam numbers
    R = dict(ver={"torch": version("torch"), "numpy": version("numpy")}, worlds=WORLDS, cores=os.cpu_count(), steps=STEPS, batch=BATCH, hidden=HIDDEN, n_params=int(nbytes(make().parameters()) // 4),
             equiv={str(w): dict(ddp=float(np.abs(res[w]["ddp"] - ref1).max()), accum=float(np.abs(res[w]["accum"] - ref1).max())) for w in WORLDS}, accum_single=float(np.abs(ref2 - ref1).max()),
             allreduce_check=chk, thr={str(w): dict(sps=res[w]["sps"]) for w in WORLDS},
             mem={str(w): dict(ddp=res[w]["mem_ddp"], fsdp=res[w].get("mem_fsdp")) for w in WORLDS}, fsdp_status="ok" if not err else f"failed: {next(iter(err.values()))}",
             ring=dict(workers=RING_WORKERS, length=RING_LEN, inputs=[[int(x) for x in v] for v in vs], result=[int(x) for x in out[0]], correct=all(np.allclose(o, sum(vs)) for o in out), events=ev, sent=sent, scale=scale,
                       naive_root=naive_allreduce_bytes(RING_WORKERS, RING_LEN)),
             pipe=dict(stages=STAGES, microbatches=MICROBATCHES, util=[pipes[m][2] for m in MICROBATCHES], makespan=[pipes[m][1] for m in MICROBATCHES], demos={str(m): dict(events=pipes[m][0], makespan=pipes[m][1]) for m in (1, 4)}),
             arith=dict(bytes_per_param=bpp, gpu_gb=GPU_GB, max_params_ddp=GPU_GB * 1e9 / bpp, example_params=7e9, example_ddp_gb=7e9 * bpp / 1e9))
    dashboard(R, OUT / "dashboard.png"); Path(__file__).with_name("ep64_results.json").write_text(json.dumps(R)); return R

if __name__ == "__main__":
    R = run(); t = R["thr"]; print(f"{R['cores']} CPU core(s); real processes with the gloo backend; model of {R['n_params']:,} weights")
    for w, e in R["equiv"].items(): print(f"  {w} worker(s): DDP vs one process, largest weight difference {e['ddp']:.1e} | with gradient accumulation {e['accum']:.1e}")
    print(f"  one process, batch cut into 2 micro-batches vs whole batch: {R['accum_single']:.1e}")
    print("  throughput, samples/s:", ", ".join(f"{w}: {t[w]['sps']:.0f}" for w in t), "| replayed ring vs the real all-reduce, largest difference:", R["allreduce_check"])
    for w, m in R["mem"].items(): print(f"  memory per worker, {w} worker(s): DDP {sum(m['ddp'].values()) / 1e6:.2f} MB" + (f" | FSDP {sum(m['fsdp'].values()) / 1e6:.2f} MB" if m["fsdp"] else " | FSDP: " + R["fsdp_status"]))
    r = R["ring"]; print(f"  ring all-reduce of {r['workers']} workers: correct {r['correct']}, bytes sent per worker {r['sent']}, naive root traffic {r['naive_root']}")
    print("  pipeline utilization:", ", ".join(f"M={m}: {u:.2f}" for m, u in zip(R["pipe"]["microbatches"], R["pipe"]["util"])))
