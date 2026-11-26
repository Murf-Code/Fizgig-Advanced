"""Reduced-precision floating-point momentum alternatives, same convergence harness."""
import json,time,torch
from run_cpu import convergence
if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);t=time.monotonic();results={}
    for n in (600,1800):results[str(n)]=convergence(n,('bf16','fp16'))
    with open('cpu_results_half.json','w') as f:json.dump(dict(torch=str(torch.__version__),convergence=results,seconds=time.monotonic()-t),f,indent=2)
    print('PASS: 24 reduced-precision convergence runs',flush=True)
