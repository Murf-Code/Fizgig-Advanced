"""Lightweight synchronized timing, with no capture/hash/gradient/export observer I/O.

The normal trainer still writes its final adapter and standard logs. Timings
cover the loss forward/backward and optimizer update, excluding setup,
cache reads before begin(), checkpoint saves, and between-step UI work.
"""
import statistics
import time
from pathlib import Path
import torch
from fit_queue import atomic_json
from research_trace import Trace

LIMIT=16
WARMUP=4

class TimingTrace:
    seed=Trace.seed

    def start(self,net,out,seed):
        self.out=Path(out); self.steps=[]; self.device=net.device
        self.seed_value=seed

    def begin(self,batch,latents,cond,gen):
        self.keys=list(map(str,batch.get("item_keys") or []))
        torch.cuda.synchronize(self.device)
        self.started=time.perf_counter()

    def pre_update(self):
        pass

    def end(self,info,loss):
        torch.cuda.synchronize(self.device)
        seconds=time.perf_counter()-self.started
        self.steps.append({"step":len(self.steps)+1,"items":self.keys,
                           "timestep":float(info.get("t",.5)),"seconds":seconds})
        return len(self.steps)>=LIMIT

    def finish(self):
        timed=[r["seconds"] for r in self.steps[WARMUP:]]
        atomic_json(self.out/"timing.json",{"status":"completed" if timed else "insufficient_steps",
            "seed":self.seed_value,"scope":"synchronized loss forward/backward and optimizer, no heavy observer I/O",
            "warmup_steps":WARMUP,"completed_updates":len(self.steps),"timed_updates":len(timed),
            "median_seconds":statistics.median(timed) if timed else None,
            "mean_seconds":statistics.mean(timed) if timed else None,"steps":self.steps})
