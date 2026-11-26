"""6C.2k-B: direct decay isolation, no proposal capture or training-policy change."""
from contextlib import contextmanager
import math
import torch
import torch.nn.functional as F
from .replay_diagnostics import digest
from .replay_probe import array
from .replay_evaluator import metrics


def equal(a, b):
    return digest(a) == digest(b)


def movement(a, b, c, d, scale):
    # BA -> DC, with stable thin QR instead of subtracting dense matrices.
    a,b,c,d = [x.double() for x in (a,b,c,d)]
    r = torch.linalg.qr(torch.cat((b,d-b),dim=1),mode='r')[1]
    s = torch.linalg.qr(torch.cat((c-a,c),dim=0).T,mode='r')[1]
    return float((r@s.T).square().sum())*scale**2


@contextmanager
def precision(net, weight_only, fp32_lora):
    from fizgig.families.lora import TRAINABLE
    changed=[];missing=object();old_precision=torch.get_float32_matmul_precision()
    try:
        if fp32_lora:torch.set_float32_matmul_precision("highest")
        for w in net.wrapped.values():
            if TRAINABLE not in w.adapters: continue
            if weight_only:
                from fizgig.modules.int8_train import int8_weight_only_forward
                base=w.base
                if not getattr(base,'_is_int8',False): raise ValueError('expected INT8 frozen base')
                changed.append((base,base.__dict__.get("forward",missing)));base.forward=int8_weight_only_forward.__get__(base,type(base))
            if fp32_lora:
                def forward(x,w=w):
                    out=w.base(x);s=w.scales[TRAINABLE]
                    if not s:return out
                    a,b=w.adapters[TRAINABLE]
                    with torch.autocast(device_type=x.device.type,enabled=False):
                        update=F.linear(F.linear(x.float(),a.weight.float()),b.weight.float())*float(s)
                    return out+update.to(out.dtype)
                changed.append((w,w.__dict__.get("forward",missing)));w.forward=forward
        yield
    finally:
        for m,fn in reversed(changed):
            if fn is missing:del m.forward
            else:m.forward=fn
        if fp32_lora:torch.set_float32_matmul_precision(old_precision)


def compare(panel, candidate, parent, base=None):
    out=[]
    for i,(case,_,target) in enumerate(panel):
        p,c=array(parent[i]),array(candidate[i]);y=array(target)
        if p.shape!=c.shape or p.shape!=y.shape:raise ValueError("prediction shape mismatch")
        delta=c-p
        row={'unit':case.unit,'timestep':case.timestep,'input_hash':case.input_hash,
             'equal':equal(candidate[i],parent[i]),'mse':float((delta**2).mean()),
             'changed_fraction':float((delta!=0).mean()),'max_abs':float(abs(delta).max()),
             'target_gain':float(-(delta*(2*(p-y)+delta)).mean())}
        if base is not None:row['metrics']=metrics(case,c,p,array(base[i]),array(target))
        out.append(row)
    return out


@torch.no_grad()
def isolate(*,pairs,optimizer,net,dit,panel,predict,guard,report,save,precision_context=precision,trace=True):
    """Caller owns outer RNG, mode, swap and registered-state audit. Fail closed."""
    from fizgig.families.lora import TRAINABLE
    params=[p for pair in pairs for p in (pair.A,pair.B)]
    if len(optimizer.param_groups)!=1 or set(optimizer.param_groups[0]['params'])!=set(params):
        raise ValueError('isolation requires one full native optimizer group')
    g=optimizer.param_groups[0];decay=1.-float(g['lr'])*float(g['weight_decay'])
    if not 0<decay<1 or any(p.dtype!=torch.float32 or p.grad is None for p in params):
        raise ValueError('expected full FP32 native decay')
    parent={p:p.detach().cpu().clone() for p in params}
    scales={n:dict(w.scales) for n,w in net.wrapped.items()}
    report.update(experiment='6C.2k-B',decay=decay,integrity={},precision={},factor_rounding=[],trace=[])
    def restore():
        for p,v in parent.items():p.copy_(v)
    def apply_decay():
        restore();torch._foreach_mul_(params,decay)
        for p in params:
            expected=parent[p].to(p.device,copy=True).mul_(decay)
            if not equal(p.detach().cpu(),expected.cpu()):raise RuntimeError('independent decay endpoint mismatch')
    def forward(sub=panel,reverse=False):
        order=list(range(len(sub)))
        if reverse:order.reverse()
        def run():
            outputs=[None]*len(sub)
            for i in order:outputs[i]=predict(sub[i][1]).clone()
            return outputs
        return guard(run)
    def require(name,actual,ref,sub=panel):
        rows=compare(sub,actual,ref);report['integrity'][name]=rows;save()
        if not all(r['equal'] for r in rows):raise RuntimeError('execution integrity failed: '+name)
    try:
        p0=forward();require('repeat',forward(),p0)
        restore();require('exact_rewrite',forward(),p0)
        apply_decay();d1=forward();report['native_decay']=compare(panel,d1,p0);save()
        restore();require('after_decay',forward(),p0)
        for n,w in net.wrapped.items():
            if TRAINABLE in w.adapters:w.scales[TRAINABLE]=0.
        try:base=forward()
        finally:
            for n,w in net.wrapped.items():w.scales.clear();w.scales.update(scales[n])
        require('after_base',forward(),p0)
        apply_decay();d2=forward(reverse=True);require('repeat_decay_reversed',d2,d1)
        report['native_decay']=compare(panel,d1,p0,base)
        for pair in pairs:
            a,b=parent[pair.A],parent[pair.B];c,d=pair.A.detach().cpu(),pair.B.detach().cpu()
            qa,qb,qc,qd=[x.to(torch.bfloat16) for x in (a,b,c,d)]
            count=sum(int((x.view(torch.int16)!=y.view(torch.int16)).sum()) for x,y in ((qa,qc),(qb,qd)))
            report['factor_rounding'].append({'name':pair.name,'block':pair.block,'changed_bf16':count,
                'values':a.numel()+b.numel(),'fp32_squared':movement(a,b,c,d,pair.scale),
                'bf16_squared':movement(qa,qb,qc,qd,pair.scale)})
        restore();require('final_native_parent',forward(),p0)
        report['integrity_passed']=True;save()
        # Predeclared low-timestep subset: at most two different source units.
        low=min(c.timestep for c,_,_ in panel);idx=[i for i,(c,_,_) in enumerate(panel) if c.timestep==low][:2]
        sub=[panel[i] for i in idx]
        for label,wo,fp in [('weight_only',True,False),('fp32_lora',False,True),('combined',True,True)]:
            with precision_context(net,wo,fp):
                ref=forward(sub);apply_decay();cand=forward(sub);restore()
                report['precision'][label]=compare(sub,cand,ref)
                require(label+'_restored',forward(sub),ref,sub)
        require('after_precision_modes',forward(sub),[p0[i] for i in idx],sub)
        for n,w in net.wrapped.items():
            if TRAINABLE in w.adapters:w.scales[TRAINABLE]=scales[n][TRAINABLE]*decay**2
        try:report['strength_only']=compare(sub,forward(sub),[p0[i] for i in idx])
        finally:
            for n,w in net.wrapped.items():w.scales.clear();w.scales.update(scales[n])
        require('after_strength',forward(sub),[p0[i] for i in idx],sub)
        if trace:
            # Synchronous instrumentation can change stream timing. Require identical final predictions.
            for i in idx:
                traces=[];bin_traces=[];outs=[]
                for decayed in (False,True):
                    if decayed:apply_decay()
                    else:restore()
                    rows=[];bins={};handles=[]
                    def hook(block):
                        def record(m,args,out):
                            x=out.detach().cpu();flat=x.reshape(-1);sample=flat[::max(1,math.ceil(flat.numel()/4096))].float()
                            rows.append((block,digest(x),sample))
                        return record
                    def bin_hook(name):
                        def record(m,args):
                            from fizgig.modules.int8_train import _quant_per_token
                            x=args[0].reshape(-1,args[0].shape[-1]);q,scale=_quant_per_token(x)
                            q=q.reshape(-1);scale=scale.reshape(-1)
                            bins[name]=(q[::max(1,math.ceil(q.numel()/4096))].cpu(),scale.cpu())
                        return record
                    try:
                        for pair in pairs:handles.append(net.wrapped[pair.name].base.register_forward_pre_hook(bin_hook(pair.name)))
                        for b,m in enumerate(dit.blocks):handles.append(m.register_forward_hook(hook(b)))
                        outs.append(forward([panel[i]])[0]);traces.append(rows);bin_traces.append(bins)
                    finally:
                        for h in handles:h.remove()
                if [r[0] for r in traces[0]]!=list(range(28)) or [r[0] for r in traces[1]]!=list(range(28)):
                    raise RuntimeError('incomplete block trace')
                if any(set(x)!={p.name for p in pairs} for x in bin_traces):raise RuntimeError('incomplete activation-bin trace')
                report['trace'].append({'unit':panel[i][0].unit,'timestep':low,
                    'matches_uninstrumented':equal(outs[0],p0[i]) and equal(outs[1],d1[i]),
                    'activation_bins':[{'name':p.name,'block':p.block,
                        'sample_changed_fraction':float((bin_traces[0][p.name][0]!=bin_traces[1][p.name][0]).float().mean()),
                        'scale_changed_fraction':float((bin_traces[0][p.name][1]!=bin_traces[1][p.name][1]).float().mean())} for p in pairs],
                    'blocks':[{'block':b,'equal':h==k,'sample_mse':float((x.double()-y.double()).square().mean())}
                              for (b,h,x),(_,k,y) in zip(*traces)]})
                restore();save()
            require('after_trace',forward(sub),[p0[i] for i in idx],sub)
        save()
    finally:
        restore()
        for n,w in net.wrapped.items():w.scales.clear();w.scales.update(scales[n])
    return report
