"""6F.2: eight-update historical factor reversion, development-only, forward-only.
No optimizer step, routing, momentum or EMA changes are made by this module.
"""
from contextlib import nullcontext
import logging
import math
import torch

from .replay_forks import network_pairs, MovementCurve
from .replay_diagnostics import digest
from .replay_isolation import movement, precision, compare

log = logging.getLogger(__name__)
FOCUS = (13, 17, 26)


@torch.no_grad()
def capture_history(net, step):
    pairs = network_pairs(net)
    if step != 3 or len(pairs) != 224 or set(p.block for p in pairs) != set(range(28)):
        raise ValueError("historical capture requires update 3 and all standard LoRA blocks")
    factors = {p.name: (p.A.detach().to("cpu", copy=True), p.B.detach().to("cpu", copy=True),
                        p.block, p.scale) for p in pairs}
    if len(factors) != len(pairs) or any(a.dtype != torch.float32 or b.dtype != torch.float32
                                        for a,b,_,_ in factors.values()):
        raise ValueError("historical capture needs unique FP32 LoRA pairs")
    return {"step": step, "factors": factors, "hash": digest(factors)}


def core(pair, parent, earlier):
    """Exact effective-weight movement polynomial for factor interpolation.
    QR compresses the concatenated factors without materializing dense BA.
    """
    a,b = (x.double() for x in parent)
    c,d = (x.double() for x in earlier)
    da,db = c-a,d-b
    rb = torch.linalg.qr(torch.cat((b,db),dim=1),mode="r")[1]
    ra = torch.linalg.qr(torch.cat((a,da),dim=0).T,mode="r")[1].T
    r = a.shape[0]
    b0,bd = rb[:,:r],rb[:,r:]
    a0,ad = ra[:r],ra[r:]
    l,q = (b0@ad+bd@a0)*pair.scale,(bd@ad)*pair.scale
    return (pair.block,torch.zeros_like(l).numpy(),l.numpy(),q.numpy())


@torch.no_grad()
def evaluate_history(*, pairs, net, history, panel, predict, guard, report, save,
                     precision_context=precision):
    from fizgig.families.lora import TRAINABLE
    pairs = tuple(pairs)
    blocks = set(range(28))
    if (history.get("step") != 3 or report.get("step") != 12 or set(p.block for p in pairs) != blocks
            or set(history["factors"]) != {p.name for p in pairs}
            or history["hash"] != digest(history["factors"])):
        raise ValueError("historical source identity mismatch")
    earlier = history["factors"]
    parent = {p.name:(p.A.detach().to("cpu",copy=True),p.B.detach().to("cpu",copy=True)) for p in pairs}
    for p in pairs:
        a,b,block,scale = earlier[p.name]
        if block != p.block or scale != p.scale or any(x.shape != y.shape or x.dtype != y.dtype
                 for x,y in zip((a,b),parent[p.name])):
            raise ValueError("historical tensor layout or scale mismatch")
    parent_hash = digest(parent)
    scales = {n:dict(w.scales) for n,w in net.wrapped.items()}
    report.update(experiment="6F.2",source_step=3,parent_step=11,source_hash=history["hash"],
                  parent_factor_hash=parent_hash,geometry={},matching={},evaluators={},integrity_passed=False)

    def endpoint():
        return digest({p.name:(p.A,p.B) for p in pairs})

    def restore_scales():
        for n,w in net.wrapped.items():
            w.scales.clear();w.scales.update(scales[n])

    def restore():
        for p in pairs:
            a,b=parent[p.name]
            p.A.copy_(a);p.B.copy_(b)
        if endpoint()!=parent_hash:
            raise RuntimeError("historical factor restore mismatch")

    def apply(ms):
        restore()
        expected={}
        for p in pairs:
            a,b=parent[p.name]
            c,d=earlier[p.name][:2]
            m=ms[p.block]
            if m==1.:
                a,b=c,d
            elif m!=0.:
                a,b=torch.lerp(a,c,m),torch.lerp(b,d,m)
            p.A.copy_(a);p.B.copy_(b)
            expected[p.name]=(a,b)
        h=endpoint()
        if h!=digest(expected):
            raise RuntimeError("historical endpoint mismatch")
        return h

    def geometry(ms):
        h=apply(ms)
        fp=bf=0.
        for p in pairs:
            if ms[p.block]==0.:continue
            a,b=parent[p.name]
            c,d=p.A.detach().cpu(),p.B.detach().cpu()
            fp+=movement(a,b,c,d,p.scale)
            bf+=movement(*(x.bfloat16() for x in (a,b,c,d)),p.scale)
        if not math.isfinite(fp) or not math.isfinite(bf):
            raise ValueError("nonfinite historical movement")
        return {"fp32":math.sqrt(fp),"bf16":math.sqrt(bf),"endpoint_hash":h,"multipliers":ms}

    def forward(label, reverse=False):
        log.info("[historical-reversion] %s (%d cases)",label,len(panel))
        def run():
            out=[None]*len(panel)
            for i in (reversed(range(len(panel))) if reverse else range(len(panel))):
                out[i]=predict(panel[i][1]).detach().cpu().clone()
            return out
        return guard(run)

    def require(a,b,label):
        if digest(a)!=digest(b):raise RuntimeError("historical prediction restore mismatch: "+label)

    def allocation(block=None, fraction=1.):
        return {b:fraction if block is None or b==block else 0. for b in sorted(blocks)}

    try:
        candidates={f"block{b:02d}":allocation(b) for b in sorted(blocks)}
        candidates["all_early"]=allocation()
        candidates.update({f"half{b:02d}":allocation(b,.5) for b in FOCUS})
        log.info("[historical-reversion] measuring %d predeclared reversion endpoints",len(candidates))
        for name,ms in candidates.items():
            report["geometry"][name]=geometry(ms)
        cores=[core(p,parent[p.name],earlier[p.name][:2]) for p in pairs]
        curve=MovementCurve(cores,{b:1. for b in blocks})
        # Search a bracket on the continuous polynomial, then verify actual FP32 endpoint.
        # No validation losses are involved in determining the control.
        for name in [f"{kind}{b:02d}" for kind in ("block","half") for b in FOCUS]:
            target=report["geometry"][name]["fp32"]
            tol=max(1e-12,target*.001)
            if target<=1e-12 or curve.norm(1.)+tol<target:
                raise ValueError("historical control movement target unreachable: "+name)
            lo,hi=0.,1.
            for _ in range(60):
                mid=(lo+hi)/2
                if curve.norm(mid)<target:lo=mid
                else:hi=mid
            label="uniform_"+name
            candidates[label]=allocation(fraction=(lo+hi)/2)
            g=geometry(candidates[label]);report["geometry"][label]=g
            gap=abs(g["fp32"]-target)
            report["matching"][label]={"reference":name,"target":target,"gap":gap,
                                       "tolerance":tol,"matched":gap<=tol}
            save()
            if gap>tol:raise RuntimeError("historical movement match failed: "+label)
        restore();native_parent=None
        for mode,wo in (("native",False),("weight_only",True)):
            with (precision_context(net,True,False) if wo else nullcontext()):
                ref=forward(mode+" parent")
                require(forward(mode+" repeat"),ref,mode+" repeat")
                for n,w in net.wrapped.items():
                    if TRAINABLE in w.adapters:w.scales[TRAINABLE]=0.
                try:base=forward(mode+" base")
                finally:restore_scales()
                require(forward(mode+" after base"),ref,mode+" after base")
                v={"reference_hash":digest((ref,base)),"arms":{},"restored":[],"repeat_exact":True}
                report["evaluators"][mode]=v;save()
                for name,ms in candidates.items():
                    if apply(ms)!=report["geometry"][name]["endpoint_hash"]:
                        raise RuntimeError("historical precision endpoint mismatch")
                    out=forward(mode+" "+name)
                    v["arms"][name]={"rows":compare(panel,out,ref,base),"prediction_hash":digest(out)}
                    restore();require(forward(mode+" restore "+name),ref,mode+" "+name)
                    v["restored"].append(name);save()
                apply(candidates["block26"])
                if digest(forward(mode+" reversed block26",True))!=v["arms"]["block26"]["prediction_hash"]:
                    raise RuntimeError("historical block26 repeat mismatch")
                restore();require(forward(mode+" final parent"),ref,mode+" final parent")
                v["reversion_repeat_exact"]=True;save()
                if not wo:native_parent=ref
        require(forward("native after precision switch"),native_parent,"native after precision switch")
        if history["hash"]!=digest(earlier):raise RuntimeError("historical source mutated")
        report["integrity_passed"]=True;save()
        return report
    finally:
        restore_scales();restore()
