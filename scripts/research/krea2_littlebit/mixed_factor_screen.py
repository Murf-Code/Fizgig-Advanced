"""LB-D3: matched maximum packed bytes, mixed precision factor screen.

All captures previously inspected; exploratory only. No training/runtime edits.
"""
import math
import torch
import torch.nn.functional as F
import native_codec as codec
import geometry_codec as geo
import private_correction as pc
import binary_rank_core as core
import binary_rank_screen as base
import mixed_factor_core as mix
import joint_latent as joint

PAIRS=((1,1),(1,2),(2,1),(2,2),(1,4),(4,1),(4,4),(8,8),(16,16))
CAPS=(.5,1.,2.25,3.25,4.25)


def direct_lowbit(w,bits):
    """Offline direct group64 Hadamard control; odd-level codebook, fitted scales."""
    m,n=w.shape
    if n%64: raise ValueError("direct control needs group64")
    tiles=[]; total=0; num=den=0.
    for i in range(0,m,geo.TILE):
        j=min(m,i+geo.TILE)
        z=geo.hadamard(w[i:j].float())
        d,nb=mix.quant_rows(z.reshape(-1,64),bits,codec._pack,codec._unpack)
        d=d.reshape(j-i,n)
        tiles.append(d)
        total+=nb
        num+=float((z-d.float()).square().sum(dtype=torch.float64))
        den+=float(z.square().sum(dtype=torch.float64))
    if total!=m*(n*bits//8+2*(n//64)): raise ValueError("direct bytes mismatch")
    def op(x,g):
        x=geo.hadamard(x.bfloat16()); g=g.bfloat16()
        ys=[]; dx=torch.zeros((len(g),n),dtype=torch.float32)
        for i,d in enumerate(tiles):
            start=i*geo.TILE
            ys.append(F.linear(x,d))
            dx.add_((g[:,start:start+len(d)]@d).float())
        return torch.cat(ys,dim=-1),geo.hadamard(dx).bfloat16()
    return op,total,math.sqrt(num/den) if den else None


def run(w,key,args,stock,trace,split,int8,archive,row,save):
    m,n=w.shape
    native=codec.stock_layer(w,int8)
    panels={label:[base.read_capture(stock,key,r,trace) for r in split[source]]
            for label,source in (("prior_development","development"),("prior_confirmation","confirmation"))}
    refs={label:[base.refs(w,c,native) for c in cs] for label,cs in panels.items()}
    row.update(shape=[m,n],candidates={},bit_caps=list(CAPS),
               scope="paired exploratory panels; no independent confirmation or fitting")
    def add(name,op,nb,rank,err,kind,precisions=None,scales=False):
        rec={"kind":kind,"rank":rank,"payload_bytes":nb,"effective_bits":8*nb/(m*n),
             "weight_error":err,"factor_bits":precisions,
             "outer_scales":scales,
             "decoded_bf16_factor_bytes":2*rank*(m+n)+(2*(m+n) if scales else 0) if rank else None,
             "dense_equivalent_macs_per_token":rank*(m+n) if rank else m*n}
        for label,cs in panels.items():
            scores=[base.score(op,c,r) for c,r in zip(cs,refs[label])]
            rec[label]=scores; rec[label+"_objective"]=base.objective(scores)
        row["candidates"][name]=rec
        print(key,name,"bits",round(rec["effective_bits"],4),"weight error",round(err,6) if err is not None else None,
              "prior dev",rec["prior_development_objective"],flush=True)
        save()
    for bits in (2,3):
        op,nb,err=direct_lowbit(w,bits)
        add(f"direct-q{bits}-odd-had",op,nb,0,err,"direct-odd-control")
        del op
    for name in ("q4-had","q6-had"):
        if name=="q6-had" and archive is not None:
            p=archive.get_tensor(key); s=codec.manifest(archive)[key]
            if s["codec"]!=name or s["shape"]!=[m,n]: raise ValueError("Q6 archive mismatch")
            codec.validate(p,s)
        else:
            p,s=codec.encode(w,name)
        add(name,lambda x,g:pc.operator(p,s,x,g),p.nbytes,0,pc.weight_error(w,p,s),"uniform-control")
        del p,s
    top=min(m,n,args.max_rank)
    print(key,"SVD sketch",top,flush=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        u,s,v=torch.svd_lowrank(w.float(),q=min(top+16,m,n),niter=4)
    # Map each method and cap to its largest admissible rank. Deduplicate
    # identical capped ranks; a candidate can be reused across cap panels.
    plans={}
    for cap in CAPS:
        limit=int(cap*m*n//8)
        for ba,bb in PAIRS:
            r=mix.budget_rank(m,n,ba,bb,limit,top)
            if r:
                plans.setdefault((ba,bb,r,False),[]).append(cap)
        # Conservative two-vector charge for selecting the rank, actual
        # single-vector and outer scales charged for joint binary candidate.
        r=mix.budget_rank(m,n,1,1,limit-2*(m+n),top)
        if r: plans.setdefault((1,1,r,True),[]).append(cap)
    for (ba,bb,r,rotated),caps in sorted(plans.items(),key=lambda kv:(kv[0][2],kv[0])):
        a,b=core.balanced(u,s,v,r)
        if rotated:
            a,b=core.rotate(a,b)
            pa,pb=codec._pack(a>=0,1),codec._pack(b>=0,1)
            da=(2*codec._unpack(pa,r*n,1).reshape(r,n)-1).bfloat16()
            sb=(2*codec._unpack(pb,m*r,1).reshape(m,r)-1).bfloat16()
            c=joint.fit_latent(w,da,sb)
            rr=torch.ones(m,dtype=torch.bfloat16); cc=torch.ones(n,dtype=torch.bfloat16)
            for _ in range(3):
                rr,cc=joint.fit_outer(w,da,sb,c,rr,cc)
                c=joint.fit_latent(w,da,sb,rr,cc)
            db=sb*c[None,:]
            nb=pa.nbytes+pb.nbytes+c.nbytes+rr.nbytes+cc.nbytes
            if nb!=joint.payload_bytes(m,n,r,True): raise ValueError("binary joint bytes mismatch")
            name=f"joint-binary-r{r}"
            add(name,lambda x,g:core.actions(x,g,da,db,rr,cc),nb,r,
                core.weight_error(w,da,db,rr,cc),"rotated-joint-control",[1,1],True)
            del pa,pb,sb,c,rr,cc
        else:
            da,na=mix.quant_rows(a,ba,codec._pack,codec._unpack)
            bt,nbb=mix.quant_rows(b.T,bb,codec._pack,codec._unpack)
            db=bt.T.contiguous(); del bt
            nb=na+nbb
            if nb!=mix.pair_bytes(m,n,r,ba,bb): raise ValueError("factor bytes mismatch")
            name=f"svd-a{ba}-b{bb}-r{r}"
            add(name,lambda x,g:core.actions(x,g,da,db),nb,r,core.weight_error(w,da,db),
                "mixed-svd",[ba,bb])
        rec=row["candidates"][name]
        rec["planned_caps"]=caps
        rec["cap_utilization"]={str(cap):nb/(cap*m*n/8) for cap in caps}
        if any(nb>int(cap*m*n//8) for cap in caps): raise ValueError("budget exceeded")
        rec["rank_cap_reached"]=r==top
        save()
        del a,b,da,db
    # All exploratory results kept; rank limits and unused budget visible.
    # No ranking based on second panel. No independent quality claim.
    row["cap_comparisons"]={}
    for cap in CAPS:
        limit=int(cap*m*n//8)
        eligible=[(name,rec) for name,rec in row["candidates"].items()
                  if rec["payload_bytes"]<=limit and rec["prior_development_objective"] is not None]
        row["cap_comparisons"][str(cap)]=[name for name,rec in sorted(eligible,
            key=lambda item:(item[1]["prior_development_objective"],item[1]["payload_bytes"],item[0]))]
    row["status"]="finished"
    save()


if __name__=="__main__":
    base.main(run_fn=run,protocol="mixed-factor-v1")
