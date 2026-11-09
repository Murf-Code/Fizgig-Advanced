"""Times PoLoRA's optimizer.step on realistic LoRA pair sets - timing only, PoLoRA itself is not changed.

Runs standalone with the Fizgig venv's Python (torch only, no other packages). It imports PoLoRA from the Fizgig
install it sits in (scripts/../src), from --src, or from an installed fizgig package, so it times whichever PoLoRA
code that install has.

    python scripts\\bench_polora_step.py                      # the default sweep (GPU if there is one)
    python scripts\\bench_polora_step.py --quick              # a short sweep (the box CPU)
    python scripts\\bench_polora_step.py --sets krea2 --ranks 16 32 --compile both --its 0.8

For each case: median step time and spread (CUDA events on a GPU, perf_counter on the CPU) after warm-up, the
compile cold start separately, peak allocated / reserved VRAM during the step (GPU), and a breakdown by phase
(sync timers around PoLoRA's kernels) plus an op-level profile (kernel launches, stack / copy / writeback ops,
bytes allocated per step). --its: your training it/s, to estimate PoLoRA's share of a step. Writes a JSON file too.
"""
import argparse
import inspect
import json
import os
import platform
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# LoRA target shapes (in_features x out_features: count), read from each model's own LoRA target list on a meta
# copy of the model (Krea 2, Anima, MOSS-Image) - SDXL's from its UNet's transformer blocks (diffusers config).
REAL = {
    "krea2": {"6144x6144": 86, "6144x1536": 56, "6144x16384": 56, "16384x6144": 28, "2560x2560": 20,
              "2560x6912": 8, "6912x2560": 4, "64x6144": 1, "256x6144": 1, "12x1": 1, "2560x6144": 1,
              "6144x64": 1, "6144x36864": 1},
    "anima": {"2048x2048": 168, "1024x2048": 56, "2048x8192": 28, "8192x2048": 28},
    "moss": {"3072x3072": 168, "3072x16384": 42, "8192x3072": 42},
    "sdxl": {"640x640": 70, "2048x640": 20, "640x5120": 10, "2560x640": 10,
             "1280x1280": 372, "2048x1280": 120, "1280x10240": 60, "5120x1280": 60},
}
# same-shape sets: small / medium / large pair counts
SYNTH = {"same-16": {"3072x3072": 16}, "same-128": {"3072x3072": 128}, "same-512": {"3072x3072": 512},
         "mixed-64": {"3072x3072": 16, "3072x8192": 16, "8192x3072": 16, "1024x3072": 16}}


def _import_polora(src):
    for p in [src, os.path.join(HERE, "..", "src")]:
        if p and os.path.isdir(os.path.join(p, "fizgig")) and p not in sys.path:
            sys.path.insert(0, os.path.abspath(p))
            break
    from fizgig.training.polora import optim as po
    return po


def _parse(shapes):
    return [(int(k.split("x")[0]), int(k.split("x")[1]), n) for k, n in shapes.items()]


class Clock:
    def __init__(self, dev):
        import torch
        self.torch, self.cuda = torch, dev.type == "cuda"

    def sync(self):
        if self.cuda:
            self.torch.cuda.synchronize()

    def time(self, fn):
        """ms for fn(), the device's work included."""
        t = self.torch
        if self.cuda:
            a, b = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)
            t.cuda.synchronize()
            a.record()
            fn()
            b.record()
            t.cuda.synchronize()
            return a.elapsed_time(b)
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) * 1000.0


def build(torch, shapes, rank, dev, dtype, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    pairs, grads = [], []
    for d_in, d_out, n in _parse(shapes):
        r = min(rank, d_in, d_out)
        for _ in range(n):
            A = torch.nn.Parameter((torch.randn(r, d_in, generator=g) / d_in ** 0.5).to(dev, dtype))
            B = torch.nn.Parameter((torch.randn(d_out, r, generator=g) * 1e-3).to(dev, dtype))
            pairs.append((A, B))
            grads.append(((torch.randn(r, d_in, generator=g) * 1e-3).to(dev, dtype),
                          (torch.randn(d_out, r, generator=g) * 1e-3).to(dev, dtype)))
    return pairs, grads


def fill(pairs, grads):
    for (A, B), (gA, gB) in zip(pairs, grads):
        if A.grad is None:
            A.grad, B.grad = gA.clone(), gB.clone()
        else:
            A.grad.copy_(gA)
            B.grad.copy_(gB)


def make_opt(po, pairs, compile_):
    kw = dict(pairs=pairs, lr=2e-2)
    sig = inspect.signature(po.Polora.__init__).parameters
    for k, v in (("compile", compile_), ("magnitude_scale", 3.0), ("curvature_beta", 0.99)):
        if k in sig:
            kw[k] = v
    return po.Polora(**kw)


def phases(torch, po, opt, pairs, grads, clock, n):
    """ms per step in PoLoRA's kernels (sync timers wrapped round them on this optimizer only; the rest is stacking,
    the curvature matmuls, the preconditioner update and the per-pair writeback)."""
    acc = {"inv_sqrt (C_A, C_B)": 0.0, "polar (zA, zB)": 0.0, "sigma_max (power iter)": 0.0}
    # PoLoRA's kernel hooks: _smax_fn is the batched sigma_max of the stacked-state PoLoRA; an older one has _smax_warm
    names = {"_invsqrt_fn": "inv_sqrt (C_A, C_B)", "_polar_fn": "polar (zA, zB)", "_smax_fn": "sigma_max (power iter)"}

    def wrap(fn, key):
        def w(*a, **k):
            clock.sync()
            t0 = time.perf_counter()
            out = fn(*a, **k)
            clock.sync()
            acc[key] += (time.perf_counter() - t0) * 1000.0
            return out
        return w
    saved = {a: getattr(opt, a) for a in names if hasattr(opt, a)}
    for a, key in names.items():
        if a in saved:
            setattr(opt, a, wrap(saved[a], key))
    smax = getattr(opt, "_smax_warm", None) if "_smax_fn" not in saved else None
    if smax is not None:
        opt._smax_warm = wrap(smax, "sigma_max (power iter)")
    total = 0.0
    for _ in range(n):
        fill(pairs, grads)
        clock.sync()
        t0 = time.perf_counter()
        opt.step()
        clock.sync()
        total += (time.perf_counter() - t0) * 1000.0
    for a, f in saved.items():
        setattr(opt, a, f)
    if smax is not None:
        del opt._smax_warm
    out = {k: v / n for k, v in acc.items()}
    out["rest (stack, curvature, precond, writeback)"] = max(0.0, total / n - sum(out.values()))
    out["_total_with_syncs"] = total / n
    return out


def profile(torch, opt, pairs, grads, dev):
    """One step's ops: kernel launches, the stack / copy / writeback ops and the bytes allocated."""
    from torch.profiler import ProfilerActivity, profile as prof
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
    fill(pairs, grads)
    with prof(activities=acts, profile_memory=True) as p:
        opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
    ev = p.key_averages()
    dev_attr = "self_device_time_total" if dev.type == "cuda" else "self_cpu_time_total"

    def t(e):
        return float(getattr(e, dev_attr, None) or getattr(e, "self_cuda_time_total", 0) or 0) / 1000.0
    groups = {"stack/cat": ("aten::stack", "aten::cat"), "copy_ (state writeback, .float())": ("aten::copy_",),
              "add_/zero_ (param writeback, grads)": ("aten::add_", "aten::zero_", "aten::fill_"),
              "matmul (bmm/mm)": ("aten::bmm", "aten::mm", "aten::matmul")}
    out = {"ops": int(sum(e.count for e in ev if e.key.startswith("aten::"))), "groups": {}}
    for g, keys in groups.items():
        sel = [e for e in ev if e.key in keys]
        out["groups"][g] = {"calls": int(sum(e.count for e in sel)), "ms": round(sum(t(e) for e in sel), 3)}
    alloc = sum(max(0, int(getattr(e, "self_device_memory_usage" if dev.type == "cuda" else "self_cpu_memory_usage",
                                   0) or 0)) for e in ev)
    out["bytes_allocated_per_step_MB"] = round(alloc / 2 ** 20, 1)
    top = sorted(ev, key=t, reverse=True)[:6]
    out["top"] = [(e.key, int(e.count), round(t(e), 3)) for e in top]
    return out


class _Notes:
    """PoLoRA's own log lines (why a compile fell back, say), kept for the case's result."""
    def __init__(self):
        import logging
        self.lines, self.h = [], logging.Handler()
        self.h.emit = lambda rec: self.lines.append(rec.getMessage())
        self.log = logging.getLogger("fizgig.polora")

    def __enter__(self):
        self.log.addHandler(self.h)
        self.level = self.log.level
        self.log.setLevel("INFO")
        return self.lines

    def __exit__(self, *a):
        self.log.removeHandler(self.h)
        self.log.setLevel(self.level)


def _mode(r):
    return "eager" if not r["compile"] else ("compiled" if r["compiled"] else "fallback")


def run_case(torch, po, name, shapes, rank, dev, dtype, compile_, warm, iters, clock, do_phases, do_profile):
    with _Notes() as notes:
        res = _run_case(torch, po, name, shapes, rank, dev, dtype, compile_, warm, iters, clock, do_phases, do_profile)
    if notes:
        res["polora_log"] = notes[:4]
    return res


def _run_case(torch, po, name, shapes, rank, dev, dtype, compile_, warm, iters, clock, do_phases, do_profile):
    pairs, grads = build(torch, shapes, rank, dev, dtype)
    opt = make_opt(po, pairs, compile_)
    compiled = bool(getattr(opt, "_compiled", False)) if compile_ else False
    res = {"set": name, "rank": rank, "pairs": len(pairs), "shape_groups": len(shapes), "compile": compile_,
           "compiled": compiled}
    fill(pairs, grads)
    res["first_step_ms"] = round(clock.time(opt.step), 2)        # the cold start (compile, allocator, caches)
    if compile_ and not getattr(opt, "_compiled", False):
        res["compiled"] = False                                   # it fell back at the first call
    for _ in range(max(0, warm - 1)):
        fill(pairs, grads)
        opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iters):
        fill(pairs, grads)
        times.append(clock.time(opt.step))
    times.sort()
    res["median_ms"] = round(statistics.median(times), 3)
    res["p10_ms"] = round(times[max(0, int(0.1 * len(times)))], 3)
    res["p90_ms"] = round(times[min(len(times) - 1, int(0.9 * len(times)))], 3)
    res["per_pair_us"] = round(1000.0 * res["median_ms"] / len(pairs), 1)
    if dev.type == "cuda":
        res["peak_alloc_during_step_MB"] = round((torch.cuda.max_memory_allocated() - base) / 2 ** 20, 1)
        res["reserved_MB"] = round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)
    if do_phases:
        res["phases_ms"] = {k: round(v, 3) for k, v in phases(torch, po, opt, pairs, grads, clock,
                                                               max(3, iters // 3)).items()}
    if do_profile:
        try:
            res["profile"] = profile(torch, opt, pairs, grads, dev)
        except Exception as e:                                    # an old torch without some profiler fields
            res["profile"] = {"error": str(e)[:200]}
    del opt, pairs, grads
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default="", help="the Fizgig src folder to take PoLoRA from (default: ../src)")
    ap.add_argument("--device", default="auto", help="auto, cuda or cpu")
    ap.add_argument("--dtype", default="float32", help="the LoRA weights' dtype (Fizgig trains them in float32)")
    ap.add_argument("--sets", nargs="*", default=None,
                    help=f"pair sets: {', '.join(list(REAL) + list(SYNTH))} (default: all)")
    ap.add_argument("--ranks", nargs="*", type=int, default=[8, 16, 32, 64])
    ap.add_argument("--compile", default="both", choices=("off", "on", "both"), help="eager, compile=True or both")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--no-phases", action="store_true", help="skip the per-phase breakdown")
    ap.add_argument("--no-profile", action="store_true", help="skip the op-level profile")
    ap.add_argument("--its", type=float, default=0.0, help="your training speed in it/s: PoLoRA's share of a step")
    ap.add_argument("--quick", action="store_true", help="a short sweep: ranks 16/32, fewer sets and iterations")
    ap.add_argument("--json", default="", help="where to write the JSON (default: polora_bench_<time>.json here)")
    a = ap.parse_args()
    import torch
    po = _import_polora(a.src)
    dev = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else
                       ("cpu" if a.device == "auto" else a.device))
    dtype = getattr(torch, a.dtype)
    sets = {**REAL, **SYNTH}
    names = a.sets or (["anima", "moss", "same-16", "same-128", "mixed-64"] if a.quick else list(sets))
    ranks = [16, 32] if a.quick and a.ranks == [8, 16, 32, 64] else a.ranks
    iters, warm = (8, 2) if a.quick else (a.iters, a.warmup)
    comps = {"off": [False], "on": [True], "both": [False, True]}[a.compile]
    clock = Clock(dev)
    info = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else
            platform.processor() or platform.machine(), "python": platform.python_version(),
            "polora_file": po.__file__, "threads": torch.get_num_threads(), "dtype": a.dtype}
    print(f"PoLoRA step benchmark - {info['device']}, torch {info['torch']}, {a.dtype}")
    print(f"PoLoRA from {info['polora_file']}")
    rows = []
    hdr = (f"{'set':<10}{'rank':>5}{'pairs':>6}{'mode':>9}{'first ms':>10}{'median ms':>11}{'p10-p90':>17}"
           f"{'us/pair':>9}" + ("  peak MB" if dev.type == "cuda" else "") + (f"{'share':>8}" if a.its else ""))
    print(hdr)
    print("-" * len(hdr))
    for name in names:
        for rank in ranks:
            for c in comps:
                r = run_case(torch, po, name, sets[name], rank, dev, dtype, c, warm, iters, clock,
                             not a.no_phases, not a.no_profile)
                if a.its:
                    step_ms = 1000.0 / a.its
                    r["share_of_step"] = round(r["median_ms"] / step_ms, 4)
                rows.append(r)
                mode = _mode(r)
                print(f"{name:<10}{rank:>5}{r['pairs']:>6}{mode:>9}{r['first_step_ms']:>10.1f}{r['median_ms']:>11.2f}"
                      f"{r['p10_ms']:>8.2f}-{r['p90_ms']:<8.2f}{r['per_pair_us']:>9.1f}"
                      + (f"{r.get('peak_alloc_during_step_MB', 0):>9.0f}" if dev.type == "cuda" else "")
                      + (f"{100 * r['share_of_step']:>7.1f}%" if a.its else ""), flush=True)
    if not a.no_phases:
        print("\nWhere the time goes (ms per step, sync timers - they add a little themselves):")
        for r in rows:
            ph = r["phases_ms"]
            tot = ph.pop("_total_with_syncs")
            parts = ", ".join(f"{k} {v:.2f} ({100 * v / max(tot, 1e-9):.0f}%)" for k, v in ph.items())
            ph["_total_with_syncs"] = tot
            print(f"  {r['set']} r{r['rank']} {_mode(r)}: {parts}")
    if not a.no_profile:
        print("\nOps per step (profiler): aten ops; calls / ms of the stack, copy, writeback and matmul ops; MB "
              "allocated in the step")
        for r in rows:
            p = r.get("profile") or {}
            if "error" in p:
                print(f"  {r['set']} r{r['rank']}: profiler unavailable ({p['error']})")
                continue
            g = "; ".join(f"{k} {v['calls']}x {v['ms']:.2f}ms" for k, v in p["groups"].items())
            print(f"  {r['set']} r{r['rank']} {_mode(r)}: {p['ops']} ops; {g}; "
                  f"{p['bytes_allocated_per_step_MB']} MB")
    falls = sorted({ln for r in rows if r["compile"] and not r["compiled"] for ln in r.get("polora_log", [])
                    if "compile" in ln})
    if falls:
        print("\ncompile=True fell back to eager here (the 'fallback' rows time eager + PoLoRA's fallback):")
        for ln in falls:
            print("  " + ln)
    if a.its:
        print(f"\nshare = PoLoRA's median step over one training step at {a.its} it/s ({1000 / a.its:.0f} ms).")
    out = a.json or os.path.join(os.getcwd(), f"polora_bench_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"info": info, "args": vars(a), "results": rows}, f, indent=1)
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
