"""LB-D1/D2: joint coefficients and alternating outer scales.

Reuses the prior capture split for paired exploratory comparison only.
Previously inspected confirmation capture is NOT independent confirmation.
No activation fitting, archive writing or trainer changes.
"""
import torch
import native_codec as codec
import private_correction as pc
import binary_rank_core as core
import binary_rank_screen as base
import joint_latent as joint


def run(w,key,args,stock,trace,split,int8,archive,row,save):
    m,n=w.shape
    native=codec.stock_layer(w,int8)
    panels={label:[base.read_capture(stock,key,r,trace) for r in split[source]]
            for label,source in (("prior_development","development"),("prior_confirmation","confirmation"))}
    refs={label:[base.refs(w,c,native) for c in captures] for label,captures in panels.items()}
    row.update(shape=[m,n],candidates={},scope="paired exploratory panels; previously inspected, not independent")
    def add(name,op,bytes_,rank,err,kind):
        rec={"kind":kind,"rank":rank,"payload_bytes":bytes_,"effective_bits":8*bytes_/(m*n),
             "weight_error":err,"dense_equivalent_macs_per_token":rank*(m+n) if rank else m*n}
        for label,captures in panels.items():
            scores=[base.score(op,c,r) for c,r in zip(captures,refs[label])]
            rec[label]=scores
            rec[label+"_objective"]=base.objective(scores)
        row["candidates"][name]=rec
        print(key,name,"bits",round(rec["effective_bits"],4),
              "weight error",round(err,6) if err is not None else None,
              "prior dev",rec["prior_development_objective"],flush=True)
        save()
    for name in ("q4-had","q6-had"):
        if name=="q6-had" and archive is not None:
            p=archive.get_tensor(key); s=codec.manifest(archive)[key]
            if s["codec"]!=name or s["shape"]!=[m,n]: raise ValueError("Q6 archive mismatch")
            codec.validate(p,s)
        else:
            p,s=codec.encode(w,name)
        add(name,lambda x,g:pc.operator(p,s,x,g),p.nbytes,0,pc.weight_error(w,p,s),"uniform-control")
        del p,s
    # Same sketch size/seed as LB-A/B/C, but only two or three diagnostic ranks.
    top=min(m,n,args.max_rank)
    ranks=sorted({min(r,top) for r in ((128,512,1536) if ".attn." in key else (128,512))})
    row["ranks"]=ranks
    print(key,"SVD sketch",top,flush=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        u,s,v=torch.svd_lowrank(w.float(),q=min(top+16,m,n),niter=4)
    for rank in ranks:
        a,b=core.balanced(u,s,v,rank)
        a,b=core.rotate(a,b)
        da,db,oldbytes=core.binarize(a,b,codec._pack,codec._unpack)
        prefix=f"r{rank}-"
        add(prefix+"previous",lambda x,g:core.actions(x,g,da,db),oldbytes,rank,
            core.weight_error(w,da,db),"previous-two-latent-scales")
        rr,cc=core.scale_fit(w,da,db)
        add(prefix+"previous-rowcol",lambda x,g:core.actions(x,g,da,db,rr,cc),
            oldbytes+rr.nbytes+cc.nbytes,rank,core.weight_error(w,da,db,rr,cc),"previous-rowcol")
        del da,db
        pa,pb=codec._pack(a>=0,1),codec._pack(b>=0,1)
        sa=(codec._unpack(pa,rank*n,1).reshape(rank,n)*2-1).bfloat16()
        sb=(codec._unpack(pb,m*rank,1).reshape(m,rank)*2-1).bfloat16()
        del a,b
        c=joint.fit_latent(w,sa,sb)
        if pa.nbytes+pb.nbytes+c.nbytes!=joint.payload_bytes(m,n,rank):
            raise ValueError("joint bytes mismatch")
        db=sb*c[None,:]
        add(prefix+"joint",lambda x,g:core.actions(x,g,sa,db),
            joint.payload_bytes(m,n,rank),rank,core.weight_error(w,sa,db),"joint-latent")
        del db
        rr=torch.ones(m,dtype=torch.bfloat16); cc=torch.ones(n,dtype=torch.bfloat16)
        # Fixed iteration count; retain every measurement, no capture-selected winner.
        for i in range(1,6):
            rr,cc=joint.fit_outer(w,sa,sb,c,rr,cc)
            c=joint.fit_latent(w,sa,sb,rr,cc)
            db=sb*c[None,:]
            if pa.nbytes+pb.nbytes+c.nbytes+rr.nbytes+cc.nbytes!=joint.payload_bytes(m,n,rank,True):
                raise ValueError("joint outer bytes mismatch")
            add(prefix+f"joint-outer-{i}",lambda x,g:core.actions(x,g,sa,db,rr,cc),
                joint.payload_bytes(m,n,rank,True),rank,core.weight_error(w,sa,db,rr,cc),"joint-alternating")
            del db
        del sa,sb,c,pa,pb,rr,cc
    row["status"]="finished"
    save()


if __name__=="__main__":
    base.main(run_fn=run,protocol="binary-joint-v1")
