"""Opt-in Krea 2 development pilot, not a Selective policy or confirmation test.
Only called from a strict unrestricted research run before the optimizer step.
The existing held-out panel is development data; no preservation claim is made.
"""
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import torch

from .replay_forks import network_pairs, match, RNGSnapshot
from .replay_disk import disk_capture
from .replay_probe import verify_endpoint, rng_state, array
from .replay_diagnostics import digest, localized, depth_shares
from .replay_evaluator import Case, metrics


def _atomic_json(path, report):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name+'.', suffix='.part')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def _state(pairs, optimizer, generators, net, dit, extra):
    # Stream each model buffer rather than keeping a second complete base model.
    # Block swapping can move frozen buffers to different devices, which does not
    # change their contents. Adapter/optimizer layouts are checked separately.
    return digest({'factors': [(p.name,p.A,p.B,p.A.grad,p.B.grad) for p in pairs],
        'optimizer': optimizer.state_dict(),
        'group_order': [[id(p) for p in g['params']] for g in optimizer.param_groups],
        'rng': rng_state(generators), 'extra': extra(),
        'modes': [(n,m.training) for n,m in dit.named_modules()],
        'scales': [(n,dict(w.scales)) for n,w in sorted(net.wrapped.items())],
        'buffers': [(n,digest(v.detach().cpu())) for n,v in dit.named_buffers()]})


def _score(predict, panel, parent, base):
    rows=[]
    for i,(case,inputs,target) in enumerate(panel):
        if digest(inputs)!=case.input_hash: raise RuntimeError('locked input mutated')
        out = array(predict(inputs))
        if digest(inputs)!=case.input_hash: raise RuntimeError('locked input mutated')
        ref, origin, y = array(parent[i]), array(base[i]), array(target)
        rows.append({'unit':case.unit,'timestep':case.timestep,'input_hash':case.input_hash,
            'metrics':metrics(case,out,ref,origin,y),
            'changed_fraction':float((out!=ref).mean()), 'localized':localized(out-ref)})
    return rows


def run_pilot(*, driver, dit, net, optimizer, validator, device, swapped, generators,
              read_extra_state, output, step, pairs=None, patchify=None):
    """One common full-block proposal, uniform sensitivity sweep + matched allocations.
    Fail closed: any contamination/error must abort the parent training run.
    No semantic claim, statistical gate, or automatic production policy.
    `pairs` and `patchify` are test seams; real trainer passes neither.
    """
    from fizgig.families.lora import TRAINABLE
    if patchify is None:
        from fizgig.krea2.sampling import patchify_block as patchify
    if pairs is None: pairs=network_pairs(net)
    pairs=tuple(pairs); generators=tuple(generators)
    if len(pairs)!=224 or set(p.block for p in pairs)!=set(range(28)):
        raise ValueError('pilot requires all 224 standard Krea 2 modules')
    if not validator or not validator.ready or not validator.wanted:
        raise ValueError('pilot requires fully cached held-out development panel')
    if any(set(w.adapters)!={TRAINABLE} for w in net.wrapped.values()):
        raise ValueError('pilot requires no frozen/context adapters')
    scales={n:dict(w.scales) for n,w in net.wrapped.items()}
    if any(not math.isfinite(s[TRAINABLE]) or s[TRAINABLE]==0 for s in scales.values()):
        raise ValueError('pilot requires enabled trainable adapters')
    output=Path(output); scratch=output.parent
    # Payload ~1.4 GiB for warm rank 32; allow overhead and other ranks.
    factor_bytes=sum(p.numel()*p.element_size() for pair in pairs for p in (pair.A,pair.B))
    if shutil.disk_usage(scratch).free < max(3*1024**3,6*factor_bytes):
        raise RuntimeError('insufficient scratch disk for movement pilot')
    report={'schema':1,'step':int(step),'scope':'development_target_only','status':'started',
            'semantic_preservation':False,'probe_units':list(validator.wanted),
            'arms':{},'sweep':{},'matching':None}
    _atomic_json(output,report)
    initial_rng=RNGSnapshot(generators)
    modes=[(m,m.training) for m in dit.modules()]
    panel=[]
    try:
        before=_state(pairs,optimizer,generators,net,dit,read_extra_state)
        with torch.no_grad():
            for j,k in enumerate(validator.wanted):
                batch=validator.batches[k]
                lat=batch['latents'].to(device=device,dtype=torch.bfloat16)
                cond=driver.batch_cond(batch,device)
                if lat.shape[0]!=1 or not cond: raise ValueError('expected one cached picture and conditioning')
                for i,t in enumerate(validator.ts):
                    seed=int(validator.seed)*1000003+j*1009+i
                    g=torch.Generator().manual_seed(seed)
                    noise=torch.randn(lat.shape,generator=g).to(device=device,dtype=lat.dtype)
                    ts=torch.tensor([t],device=device,dtype=torch.float32)
                    weight=ts.view(-1,1,1,1).to(lat.dtype)
                    xt=(1.-weight)*lat+weight*noise
                    target=patchify(noise-lat,dit.config.patch)[0].detach().cpu()
                    inputs={'xt':xt.detach().cpu(),'t':ts.cpu(),'cond':{n:v.detach().cpu() for n,v in cond.items()}}
                    case=Case('target',str(k),'original',float(t),seed,digest(inputs))
                    panel.append((case,inputs,target))
        if len({c.input_hash for c,_,_ in panel})!=len(panel): raise ValueError('duplicate probe input')
        lock=digest(panel_values(panel))
        scores={b:math.sqrt(sum(float(p.grad.detach().double().square().sum().cpu()) for pair in pairs if pair.block==b for p in (pair.A,pair.B))/sum(p.numel() for pair in pairs if pair.block==b for p in (pair.A,pair.B))) for b in range(28)}
        chosen=min(scores,key=scores.get)
        alloc={'uniform':{b:1. for b in scores},'static26':{b:.5 if b==26 else 1. for b in scores},'adaptive_low_rms':{b:.5 if b==chosen else 1. for b in scores}}
        report.update({'gradient_norms':scores,'adaptive_block':chosen,'allocations':alloc,'panel_hash':lock})
        def predict(x):
            with torch.no_grad():
                return driver.predict(dit,{'xt':x['xt'].to(device),'t':x['t'].to(device)}, {n:v.to(device) for n,v in x['cond'].items()}).detach().cpu()
        def panel_guard(fn):
            h=_state(pairs,optimizer,generators,net,dit,read_extra_state)
            result=fn()
            if h!=_state(pairs,optimizer,generators,net,dit,read_extra_state) or lock!=digest(panel_values(panel)):
                raise RuntimeError('live pilot state or panel mutation')
            return result
        with disk_capture(pairs,optimizer,generators=generators,scratch=scratch) as fork:
            try:
                if swapped: driver.block_swap_mode(dit,inference=True)
                dit.eval()
                fork.reset_weights(); fork.rng.restore()
                parent=panel_guard(lambda:[predict(x) for _,x,_ in panel])
                repeat=panel_guard(lambda:[predict(x) for _,x,_ in panel])
                for n,w in net.wrapped.items():w.scales[TRAINABLE]=0.
                try:base=panel_guard(lambda:[predict(x) for _,x,_ in panel])
                finally:
                    for n,w in net.wrapped.items():w.scales.clear();w.scales.update(scales[n])
                report['reference_hash']=digest((parent,base))
                report['repeat_parent']=[{'unit':c.unit,'timestep':c.timestep,
                    'max_abs':float(abs(array(a)-array(b)).max()),
                    'mse':float(((array(a)-array(b))**2).mean()),
                    'bitwise_equal':digest(a)==digest(b)} for (c,_,_),a,b in zip(panel,parent,repeat)]
                for m in (.0,.25,.5,1.):
                    multipliers={b:m for b in scores}
                    achieved,shares=fork.achieved(multipliers);verify_endpoint(fork,multipliers)
                    rows=panel_guard(lambda:_score(predict,panel,parent,base))
                    report['sweep'][str(m)]={'achieved':achieved,'depth':depth_shares(shares),'rows':rows}
                report['repeat_exact']=all(r['bitwise_equal'] for r in report['repeat_parent'])
                matched=(match(fork,alloc,minimum=1e-12,atol=1e-12,rtol=.001,max_initial_dip=.01)
                         if report['repeat_exact'] else {'matched':False,'reason':'identical-parent replay was not exact'})
                report['matching']=matched
                if matched['matched']:
                    actuals=[]
                    for name,arm in matched['arms'].items():
                        achieved,shares=fork.achieved(arm['multipliers']);verify_endpoint(fork,arm['multipliers']);actuals.append(achieved)
                        if abs(achieved-matched['target'])>matched['tolerance'] or abs(achieved-arm['achieved'])>matched['tolerance']:
                            raise RuntimeError('matched endpoint outside tolerance')
                        report['arms'][name]={'achieved':achieved,'depth':depth_shares(shares),'rows':panel_guard(lambda:_score(predict,panel,parent,base))}
                    if max(actuals)-min(actuals)>matched['tolerance']:raise RuntimeError('pairwise movement mismatch')
            finally:
                for n,w in net.wrapped.items():w.scales.clear();w.scales.update(scales[n])
                if swapped: driver.block_swap_mode(dit,inference=False)
                for m,was in modes:m.training=was
        if before!=_state(pairs,optimizer,generators,net,dit,read_extra_state):raise RuntimeError('parent rollback audit failed')
        report['status']='completed'; report['parent_rollback']='exact_registered_state';_atomic_json(output,report)
        return report
    except BaseException as exc:
        report['status']='failed';report['error']=repr(exc);_atomic_json(output,report);raise
    finally:initial_rng.restore()


def panel_values(panel):
    return [(c.domain,c.unit,c.variant,c.timestep,c.noise,c.input_hash,x,y) for c,x,y in panel]
