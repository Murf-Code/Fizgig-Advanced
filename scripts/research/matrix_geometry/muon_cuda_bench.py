"""Isolated Muon CUDA memory/parity gate; synthetic gradients, real Krea 2 shapes."""
import argparse
import gc
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch
from muon_me import MuonME, Reference

# Rank-16 factor shapes extracted from the trained Krea 2 checkpoint header.
INVENTORY = [(16,6144,200),(6144,16,117),(1536,16,56),(16,16384,28),
             (16384,16,56),(16,64,1),(64,16,1),(16,256,1),(36864,16,1),
             (16,2560,29),(2560,16,24),(16,6912,4),(6912,16,8),(16,12,1),(1,16,1)]
CASES = [('reference',None),('reference-repeat',None),('shape-local',0),
         ('chunk-128',128),('chunk-32',32)]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rank',type=int,choices=(16,32,64),default=64)
    ap.add_argument('--steps',type=int,default=6,help='Warmed measured steps after two warmups')
    ap.add_argument('--output',default='muon-cuda-report.json')
    ap.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    ap.add_argument('--smoke',action='store_true',help='Small synthetic shapes, not full Krea 2 inventory')
    args=ap.parse_args()
    if args.steps<1: ap.error('--steps must be positive')
    if args.device=='cpu' and not args.smoke: ap.error('CPU is supported only for --smoke')
    if args.device=='cuda' and not torch.cuda.is_available(): ap.error('CUDA unavailable; use Fizgig training Python')
    torch.set_num_threads(1 if args.smoke else min(4,torch.get_num_threads()))
    dev=torch.device(args.device)
    cuda=dev.type=='cuda'
    shapes=[(args.rank if a==16 else a,args.rank if b==16 else b) for a,b,n in INVENTORY for _ in range(n)]
    if args.smoke: shapes=[(8,64)]*8+[(64,8)]*5+[(8,3),(1,8)]
    sync=lambda: torch.cuda.synchronize() if cuda else None
    def mem():
        if not cuda:return None
        free,total=torch.cuda.mem_get_info(dev)
        return dict(allocated=torch.cuda.memory_allocated(dev),reserved=torch.cuda.memory_reserved(dev),driver_free=free,driver_total=total)
    report=dict(schema=1,scope='isolated optimizer; synthetic gradients; not diffusion training',rank=args.rank,
                smoke=args.smoke,steps=args.steps,warmups=2,torch=torch.__version__,cuda_version=torch.version.cuda,
                device=torch.cuda.get_device_name(dev) if cuda else 'cpu',platform=platform.platform(),
                dtype='float32',factor_count=len(shapes),factor_elements=sum(a*b for a,b in shapes),
                cases=[],status='running')
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    def save():
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
    save(); reference=None

    def run(name,budget,ref):
        gc.collect()
        if cuda:torch.cuda.empty_cache()
        rng=torch.Generator(device=dev).manual_seed(42)
        ps=[torch.nn.Parameter(torch.randn(s,device=dev,generator=rng)*.01) for s in shapes]
        opt=(Reference(ps,lr=.0002,weight_decay=0,adjust_lr_fn='match_rms_adamw') if budget is None else
             MuonME(ps,lr=.0002,weight_decay=0,adjust_lr_fn='match_rms_adamw',chunk_memory_mib=budget))
        # Preallocate grads; deterministic independent gradient sequence, same for each arm.
        for p in ps:p.grad=torch.empty_like(p)
        initial=[p.detach().cpu().clone() for p in ps] if name=='reference' else None
        sync(); records=[]
        for step in range(args.steps+2):
            for p in ps:p.grad.normal_(generator=rng)
            sync();before=mem()
            if cuda:torch.cuda.reset_peak_memory_stats(dev)
            t=time.perf_counter();opt.step();sync();ms=(time.perf_counter()-t)*1000
            after=mem(); rec=dict(step=step+1,ms=ms,before=before,after=after)
            if cuda:
                rec.update(peak_allocated=torch.cuda.max_memory_allocated(dev),peak_reserved=torch.cuda.max_memory_reserved(dev))
                rec['extra_allocated']=rec['peak_allocated']-before['allocated']
            records.append(rec)
        result=dict(name=name,budget_mib=budget,status='finished',records=records,
                    median_warm_ms=statistics.median(x['ms'] for x in records[2:]),
                    peak_extra_warm_allocated=max(x['extra_allocated'] for x in records[2:]) if cuda else None,
                    peak_total_allocated=max(x['peak_allocated'] for x in records) if cuda else None,
                    peak_total_reserved=max(x['peak_reserved'] for x in records) if cuda else None)
        # Compare after measurements so validation does not contaminate GPU peaks.
        finite=True; diff2=0.; disp2=0.; max_abs=0.; weight_equal=True; momentum_equal=True
        weights=[]; momenta=[]
        for i,p in enumerate(ps):
            w=p.detach().cpu(); m=opt.state[p]['momentum_buffer'].detach().cpu()
            finite=finite and bool(torch.isfinite(w).all()) and bool(torch.isfinite(m).all())
            if name=='reference':
                weights.append(w.clone());momenta.append(m.clone())
                disp2+=float((w-initial[i]).double().square().sum())
            elif ref is not None:
                rw,rm=ref['weights'][i],ref['momenta'][i]
                delta=w-rw;diff2+=float(delta.double().square().sum());max_abs=max(max_abs,float(delta.abs().max()))
                weight_equal=weight_equal and torch.equal(w,rw);momentum_equal=momentum_equal and torch.equal(m,rm)
        result['finite']=finite
        if name=='reference':ref=dict(weights=weights,momenta=momenta,disp2=disp2)
        elif ref is not None:
            result['parity']=dict(weight_bitwise_equal=weight_equal,momentum_bitwise_equal=momentum_equal,
                                  max_abs_weight_difference=max_abs,relative_weight_difference_to_reference_displacement=math.sqrt(diff2/max(ref['disp2'],1e-30)))
        return result,ref

    for name,budget in CASES:
        print('Running',name,flush=True)
        try:
            result,reference=run(name,budget,reference)
        except torch.OutOfMemoryError as exc:
            result=dict(name=name,budget_mib=budget,status='oom',error=str(exc))
        except Exception as exc:
            report['cases'].append(dict(name=name,budget_mib=budget,status='error',error=repr(exc)))
            report['status']='error';save();raise
        report['cases'].append(result);save()
        print({k:v for k,v in result.items() if k!='records'},flush=True)
        gc.collect()
        if cuda:torch.cuda.empty_cache()
    report['status']='complete' if all(x['status']=='finished' for x in report['cases']) else 'complete_with_oom'
    save();print('Report:',path.resolve(),flush=True)

if __name__=='__main__':main()
