"""Diagnostic only: repeated Krea2 backward at two fixed adapter states."""
import json
import math
from pathlib import Path
import torch
from .memory_pair import fingerprint

REPEATS = 3
STATES = ('zero_up', 'nonzero_up')


def gradient_metrics(actual, reference):
    e = r = a = dot = 0.
    for x, y in zip(actual, reference, strict=True):
        if x.shape != y.shape or x.dtype != y.dtype:
            raise RuntimeError('gradient shape/dtype mismatch')
        x, y = x.double(), y.double()
        e += float((x-y).square().sum())
        a += float(x.square().sum()); r += float(y.square().sum())
        dot += float((x*y).sum())
    if not all(map(math.isfinite, (e,r,a,dot))):
        raise RuntimeError('nonfinite gradient metrics')
    return dict(error=math.sqrt(e), reference=math.sqrt(r), norm=math.sqrt(a),
                relative=math.sqrt(e/r) if r else None,
                cosine=max(-1.,min(1.,dot/math.sqrt(a*r))) if a*r else None)


def allocator_boundary(device, params, optimizer):
    grad_bytes = sum(p.grad.numel()*p.grad.element_size() for p in params if p.grad is not None)
    out = dict(gradient_payload_bytes=grad_bytes, optimizer_state_entries=len(optimizer.state))
    if device.type == 'cuda':
        s = torch.cuda.memory_stats(device)
        for k in ('allocated_bytes.all.current','active_bytes.all.current','reserved_bytes.all.current',
                  'requested_bytes.all.current','inactive_split_bytes.all.current','num_alloc_retries','num_ooms'):
            out[k] = int(s[k]) if k in s else None
    return out


def run_fixed(driver, dit, targets, optimizer, latents, cond, keys, *, case, output_dir,
              seed, min_t, max_t, phase, emit):
    device = latents.device
    targets = dict(sorted(targets.items()))
    names = list(targets)
    if not names or optimizer.state:
        raise RuntimeError('fixed diagnostic requires fresh optimizer and factors')
    params = [p for ps,_ in targets.values() for p in ps]
    sample = sorted({names[i*(len(names)-1)//3] for i in range(4)} |
                    {max(names,key=lambda k:sum(p.numel() for p in targets[k][0]))})
    for ps,_ in targets.values():
        if len(ps)!=2 or any(p.ndim!=2 for p in ps) or ps[0].shape[0]!=ps[1].shape[1] or torch.count_nonzero(ps[1]).item():
            raise RuntimeError('fixed diagnostic requires fresh standard zero-up LoRA')
    def manifest():
        return fingerprint({k:(float(s),[fingerprint(p) for p in ps]) for k,(ps,s) in targets.items()})
    root = Path(output_dir).parent/'fixed-control'/'.fixed-reference'
    control = case == 'fixed-control'
    if control: root.mkdir(parents=True,exist_ok=False)
    else:
        if not (root/'complete.json').is_file(): raise RuntimeError('fixed reference incomplete')
    refmeta = None if control else json.loads((root/'complete.json').read_text())
    meta = dict(modules=len(names),sample=sample,states={},trials=0,optimizer_updates=0)
    state_gen = torch.Generator().manual_seed(int(seed)+98231)
    ab = getattr(dit,'activation_offload_budget',None)
    def sync():
        if device.type == 'cuda': torch.cuda.synchronize(device)
    try:
        for state in STATES:
            if state == 'nonzero_up':
                with torch.no_grad():
                    for ps,_ in targets.values():
                        ps[1].copy_(torch.randn(ps[1].shape,generator=state_gen,dtype=torch.float32)*.001)
            signature = manifest()
            meta['states'][state]=signature
            if refmeta is not None and signature!=refmeta['states'][state]:
                raise RuntimeError('fixed adapter state mismatch')
            for trial in range(1,REPEATS+1):
                step=meta['trials']+1
                optimizer.zero_grad(set_to_none=True)
                rec=dict(state=state,trial=trial,step=step,factor_sha256=signature)
                def capture(lat,c,noise,t):
                    rec['input_sha256']=fingerprint((list(map(str,keys)),lat,c,noise,t))
                def output(pred):
                    rec['output_sha256']=fingerprint(pred)
                driver._geometry_pair_input=capture
                driver._geometry_fixed_output=output
                gen=torch.Generator().manual_seed(int(seed))
                sync()
                rec['forward_start']=allocator_boundary(device,params,optimizer)
                devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(int(seed)+1927)
                    with phase('forward',step):
                        loss,_=driver.training_loss(dit,latents,cond,gen,min_t=min_t,max_t=max_t)
                        sync()
                    rec['forward_end']=allocator_boundary(device,params,optimizer)
                    with phase('backward',step):
                        loss.backward()
                        sync()
                driver._geometry_pair_input=None
                driver._geometry_fixed_output=None
                rec['backward_end']=allocator_boundary(device,params,optimizer)
                rec['loss']=float(loss.detach())
                if not math.isfinite(rec['loss']) or not rec.get('input_sha256') or not rec.get('output_sha256'):
                    raise RuntimeError('missing/nonfinite fixed forward evidence')
                hashes={}; sampled={}
                for k,(ps,_) in targets.items():
                    grads=[]
                    for p in ps:
                        if p.grad is None: raise RuntimeError('missing fixed gradient '+k)
                        g=p.grad.detach().to('cpu',copy=True)
                        if not torch.isfinite(g).all(): raise RuntimeError('nonfinite fixed gradient '+k)
                        grads.append(g)
                    hashes[k]=fingerprint(grads)
                    if k in sample: sampled[k]=grads
                rec['gradient_sha256']=fingerprint(hashes)
                file=root/(state+'.pt')
                if control and trial==1:
                    torch.save(dict(input_sha256=rec['input_sha256'],output_sha256=rec['output_sha256'],grads=sampled),file)
                reference=torch.load(file,map_location='cpu',weights_only=True)
                if reference['input_sha256']!=rec['input_sha256']: raise RuntimeError('fixed input mismatch')
                rec['output_matches_reference']=reference['output_sha256']==rec['output_sha256']
                rec['metrics']={k:gradient_metrics(sampled[k],reference['grads'][k]) for k in sample}
                if ab is not None:
                    if hasattr(ab,'drain'): ab.drain()
                    rec['activation']=ab.snapshot() if hasattr(ab,'snapshot') else {k:getattr(ab,k) for k in ('cap','live','peak','accepted','skipped','layout_skipped')}
                    if ab.live: raise RuntimeError('fixed diagnostic live activation lease')
                if optimizer.state: raise RuntimeError('unexpected optimizer state')
                emit('[geometry-fixed-trial]',rec)
                meta['trials']+=1
                del loss,sampled,reference,grads,g
            if manifest()!=signature: raise RuntimeError('fixed factors changed during backward trials')
        if control:
            temp=root/'complete.tmp';temp.write_text(json.dumps(meta,allow_nan=False));temp.replace(root/'complete.json')
        emit('[geometry-fixed-complete]',meta)
        return meta
    finally:
        driver._geometry_pair_input=None
        driver._geometry_fixed_output=None
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for ps,_ in targets.values(): ps[1].zero_()
