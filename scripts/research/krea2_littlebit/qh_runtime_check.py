"""Small actual packed-runtime smoke: native install, backward, checkpoint, LoRA."""
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import native_codec as codec
import qh_runtime as qh
import geometry_codec as geo
import hadamard_scale_core as hs


def check(device):
    torch.manual_seed(42)
    results=[]
    for name in sorted(qh.NAMES):
        for m,n in ((41,128),(259,256)):
            w=torch.randn(m,n).bfloat16()
            w[0]=0
            p,s=codec.encode(w,name,128,64)
            codec.validate(p,s)
            z=geo.hadamard(w.float())
            c2,s2=hs.initial_codes(z,2)
            c3,s3=hs.initial_codes(z,3)
            d2,d3=hs.decode(c2,s2,2),hs.decode(c3,s3,3)
            if name=="qh2": ref=d2
            elif name=="qh3": ref=d3
            else:
                delta=(d3.float()-z).square().sum(1)-(d2.float()-z).square().sum(1)
                mask=torch.zeros(m,dtype=torch.bool)
                mask[delta.argsort(stable=True)[:s["q3_rows"]]]=True
                ref=torch.where(mask[:,None],d3,d2)
            state=qh.prepare(p,s,codec._unpack)
            actual=torch.cat([qh.rows(state,s,i,min(i+qh.TILE,m),codec._unpack) for i in range(0,m,qh.TILE)])
            if not torch.equal(actual,ref): raise AssertionError("screen decode mismatch")
            if p.nbytes!=sum(qh.lengths(s)): raise AssertionError("payload mismatch")
            module=torch.nn.Linear(n,m,bias=False,device="meta")
            codec.install(module,p,s)
            module.to(device)
            x=torch.randn(2,17,n,device=device,dtype=torch.bfloat16).requires_grad_()
            dy=torch.randn(2,17,m,device=device,dtype=torch.bfloat16)
            a=torch.randn(4,n,device=device,dtype=torch.float32,requires_grad=True)
            b=torch.randn(m,4,device=device,dtype=torch.float32,requires_grad=True)
            ref=ref.to(device)
            def reference(t):
                z=geo.hadamard(t)
                return torch.cat([F.linear(z,ref[i:i+qh.TILE]) for i in range(0,m,qh.TILE)],dim=-1)
            def ref_backward(g):
                d=torch.zeros((*g.shape[:-1],n),device=device,dtype=torch.float32)
                for i in range(0,m,qh.TILE):
                    d.add_((g[...,i:i+qh.TILE]@ref[i:i+qh.TILE]).float())
                return geo.hadamard(d).bfloat16()
            y=module(x)
            dx,=torch.autograd.grad(y,x,dy)
            if not torch.equal(y,reference(x)) or not torch.equal(dx,ref_backward(dy)):
                raise AssertionError("packed action parity failure")
            def fn(t,aa,bb): return module(t)+F.linear(F.linear(t.float(),aa),bb).bfloat16()
            direct=fn(x,a,b)
            grads=torch.autograd.grad(direct,(x,a,b),dy)
            replay=checkpoint(fn,x,a,b,use_reentrant=False)
            replay_grads=torch.autograd.grad(replay,(x,a,b),dy)
            if not torch.equal(direct,replay) or not all(torch.equal(g,h) for g,h in zip(grads,replay_grads)):
                raise AssertionError("checkpointed adapter parity failure")
            if not all(torch.isfinite(g).all() for g in grads) or module.weight.requires_grad or module.weight.grad is not None:
                raise AssertionError("invalid frozen/adapter gradients")
            if module.weight.dtype!=torch.uint8 or module.weight.nbytes!=p.nbytes:
                raise AssertionError("persistent dense weight")
            # A CPU round trip models the packed parameter transfer, not the actual swap engine.
            module.cpu().to(device)
            if not torch.equal(module(x),y): raise AssertionError("packed device roundtrip failure")
            results.append({"codec":name,"shape":[m,n],"bytes":p.nbytes,"bits":8*p.nbytes/(m*n),"device":str(device),"status":"passed"})
    return results


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device",choices=("cpu","cuda"),default="cuda" if torch.cuda.is_available() else "cpu")
    a=ap.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(check(torch.device(a.device)),indent=2))


if __name__=="__main__":
    main()
