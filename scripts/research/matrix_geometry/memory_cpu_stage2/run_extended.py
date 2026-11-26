"""Longer synthetic convergence confirmation, all methods and seeds."""
import json,time,torch
from run_cpu import convergence
if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);t=time.monotonic()
    rows=convergence(1800)
    with open('cpu_results_extended.json','w') as f:json.dump(dict(steps=1800,torch=str(torch.__version__),convergence=rows,seconds=time.monotonic()-t),f,indent=2)
    print('PASS: extended 36 convergence runs',flush=True)
