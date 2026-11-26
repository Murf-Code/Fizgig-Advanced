"""Uniform-family mixed precision: an explicitly heuristic bytes/error frontier."""
import math


def choose(families, target_fraction=.70):
    """families maps family -> list of codec records (including INT8).
    Minimize a local relative-error proxy greedily under a nominal eligible-weight
    byte budget; report achieved budget and all choices, not guaranteed optimum.
    """
    if not 0<target_fraction<1 or not families:
        raise ValueError("invalid mixed budget")
    opts={}; chosen={}
    for name,rows in sorted(families.items()):
        stock=[r for r in rows if r["codec"]=="int8"]
        if len(stock)!=1 or stock[0]["bytes"]<=0:
            raise ValueError("need exactly one native INT8 record")
        s=stock[0]
        valid=[r for r in rows if 0<r["bytes"]<=s["bytes"] and math.isfinite(r["error"]) and r["error"]>=0]
        # If a row is worse in both axes it cannot help the local proxy.
        valid=[r for r in valid if not any(q["bytes"]<=r["bytes"] and q["error"]<=r["error"]
              and (q["bytes"]<r["bytes"] or q["error"]<r["error"]) for q in valid)]
        opts[name]=valid
        chosen[name]=s
    baseline=sum(r["bytes"] for r in chosen.values())
    target=math.floor(baseline*target_fraction)
    total=baseline
    while total>target:
        moves=[]
        for name,cur in chosen.items():
            for r in opts[name]:
                saved=cur["bytes"]-r["bytes"]
                if saved>0:
                    moves.append(((r["error"]-cur["error"])*families[name][0]["bytes"]/saved,
                                  -saved,name,r["codec"],r))
        if not moves: break
        _,neg,name,_,r=min(moves,key=lambda x:x[:4])
        chosen[name]=r; total+=neg
    return {"target_fraction":target_fraction,"target_bytes":target,"baseline_bytes":baseline,
            "chosen_bytes":total,"achieved_fraction":total/baseline,"budget_met":total<=target,
            "families":chosen,"objective":"greedy bytes versus mean squared local relative forward/input-gradient error; no retention threshold"}
