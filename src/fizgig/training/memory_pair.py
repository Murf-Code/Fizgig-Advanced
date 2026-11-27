"""Opt-in diagnostic only. Reference tensors stay on local disk, outside report ZIP.
Inspired by LittleBit research_trace; hashes every actual input and initial factor.
Exact gradient and effective-displacement comparisons on a bounded module sample.
"""
import hashlib
import json
import math
from pathlib import Path
import torch


def fingerprint(value):
    h = hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            x = v.detach().cpu().contiguous()
            h.update(repr((str(x.dtype), tuple(x.shape))).encode()+b'\0')
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            h.update(b'dict'+str(len(v)).encode()+b'\0')
            for k in sorted(v):
                add(k); add(v[k])
        elif isinstance(v, (tuple, list)):
            h.update(b'list'+str(len(v)).encode()+b'\0')
            for item in v: add(item)
        elif isinstance(v, (str, int, float, bool, type(None))):
            h.update(repr(v).encode()+b'\0')
        else:
            raise TypeError(f'unsupported paired input: {type(v)}')
    add(value)
    return h.hexdigest()


def product_norm(terms):
    """Frobenius norm of sum B_i A_i, never construct the m x n product."""
    b = torch.cat([b.double() for b, _ in terms], dim=1)
    a = torch.cat([a.double() for _, a in terms], dim=0)
    rb = torch.linalg.qr(b, mode='r').R
    ra = torch.linalg.qr(a.T, mode='r').R
    return float(torch.linalg.vector_norm(rb @ ra.T))


def displacement(before, after):
    a, b = (t.double() for t in before)
    aa, bb = (t.double() for t in after)
    return [(bb-b, a), (bb, aa-a)]


def compare(before, after, grads, reference):
    rbefore, rafter, rgrads = reference
    gerr = math.sqrt(sum(float((g.double()-r.double()).square().sum()) for g,r in zip(grads,rgrads)))
    gnorm = math.sqrt(sum(float(r.double().square().sum()) for r in rgrads))
    d = displacement(before,after)
    rd = displacement(rbefore,rafter)
    dnorm = product_norm(rd)
    derr = product_norm(d+[(-b,a) for b,a in rd])
    actual = product_norm(d)
    return dict(grad_error=gerr, grad_reference=gnorm, grad_relative=gerr/gnorm if gnorm else None,
                update_error=derr, update_reference=dnorm, update_norm=actual,
                update_relative=derr/dnorm if dnorm else None)


class PairTrace:
    def __init__(self, targets, out, case, steps):
        self.targets = dict(sorted(targets.items()))
        if not self.targets or steps < 1: raise ValueError('empty pair trace')
        self.out = Path(out)
        self.case = case
        self.reference = case == 'pair-fp32-control'
        self.root = self.out.parent / 'pair-fp32-control' / '.pair-reference'
        self.horizons = sorted({1,min(2,steps),steps})
        names = list(self.targets)
        # Four evenly spaced targets plus the largest factor pair; deterministic and bounded.
        self.sample = sorted({names[i*(len(names)-1)//3] for i in range(4)} |
                             {max(names,key=lambda k: sum(p.numel() for p in self.targets[k][0]))})
        manifest = {}
        for k,(ps,scale) in self.targets.items():
            if len(ps)!=2 or any(p.ndim!=2 for p in ps) or ps[0].shape[0]!=ps[1].shape[1]:
                raise ValueError('expected standard down/up factor pair')
            if torch.count_nonzero(ps[1]).item(): raise ValueError('expected fresh zero-up LoRA')
            manifest[k] = fingerprint((float(scale), [fingerprint(p) for p in ps]))
        self.data = dict(initial_sha256=fingerprint(manifest), modules=len(names), sample=self.sample,
                         horizons=self.horizons, expected_steps=steps, steps=[])
        self.ref = None
        if self.reference:
            self.root.mkdir(parents=True, exist_ok=False)
        else:
            self.ref = json.loads((self.root/'complete.json').read_text(encoding='utf-8'))
            for k in ('initial_sha256','modules','sample','horizons','expected_steps'):
                if self.ref[k]!=self.data[k]: raise RuntimeError(f'paired initialization mismatch: {k}')
        self.pending = None

    def begin(self, step, keys, latents, cond, noise, t):
        if self.pending is not None or step!=len(self.data['steps'])+1:
            raise RuntimeError('unexpected pair trace step')
        sig = fingerprint(dict(keys=list(map(str,keys)), latents=latents, cond=cond, noise=noise,t=t))
        if self.ref is not None and self.ref['steps'][step-1]['input_sha256']!=sig:
            raise RuntimeError(f'paired actual input mismatch at update {step}')
        self.pending = dict(step=step, input_sha256=sig, timestep=float(t.mean()))

    def pre_update(self, loss):
        if self.pending is None: raise RuntimeError('missing paired training input')
        self.pending['loss']=float(loss.detach())
        if not math.isfinite(self.pending['loss']): raise RuntimeError('nonfinite paired loss')
        self.pre={}
        if self.pending['step'] in self.horizons:
            for k in self.sample:
                ps,_=self.targets[k]
                if any(p.grad is None for p in ps): raise RuntimeError(f'missing sampled gradient: {k}')
                before=[p.detach().to('cpu',copy=True) for p in ps]
                grads=[p.grad.detach().to('cpu',copy=True) for p in ps]
                if not all(torch.isfinite(x).all() for x in before+grads): raise RuntimeError('nonfinite sampled state')
                self.pre[k]=(before,grads)

    def end(self):
        if self.pending is None: raise RuntimeError('missing paired update')
        step=self.pending['step']
        metrics={}
        for i,k in enumerate(self.sample):
            if k not in self.pre: continue
            before,grads=self.pre[k]
            ps,scale=self.targets[k]
            after=[p.detach().to('cpu',copy=True) for p in ps]
            if not all(torch.isfinite(x).all() for x in after): raise RuntimeError('nonfinite sampled update')
            path=self.root/f'{step:04d}-{i}.pt'
            if self.reference:
                torch.save((before,after,grads),path)
                reference=(before,after,grads)
            else:
                reference=torch.load(path,map_location='cpu',weights_only=True)
            metric=compare(before,after,grads,reference)
            for field in ('update_error','update_reference','update_norm'):
                metric[field]*=abs(float(scale))
            metrics[k]=metric
        self.pending['metrics']=metrics
        self.data['steps'].append(self.pending)
        self.pending=None
        self.pre={}
        return self.data['steps'][-1]

    def finish(self):
        if self.pending is not None or len(self.data['steps'])!=self.data['expected_steps']:
            raise RuntimeError('incomplete paired trace')
        if self.reference:
            path=self.root/'complete.json'
            tmp=path.with_suffix('.tmp')
            tmp.write_text(json.dumps(self.data,allow_nan=False),encoding='utf-8')
            tmp.replace(path)
        return {k:v for k,v in self.data.items() if k!='steps'}
