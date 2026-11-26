"""Paired development metrics; no fixed acceptance target or independent-timestep CI."""
import math
import random
import statistics


def paired(scores, replicates=2000, seed=42):
    keys = sorted(scores["no_adapter"]["per_image"])
    if not keys:
        raise ValueError("empty paired evaluation")
    means = {}
    for name, row in scores.items():
        values = row["per_image"]
        if sorted(values)!=keys:
            raise ValueError("different evaluation images")
        means[name] = {}
        for k in keys:
            if not values[k] or len(values[k])!=len(scores["no_adapter"]["per_image"][k]) or not all(math.isfinite(v) for v in values[k]):
                raise ValueError("invalid paired timestep losses")
            means[name][k] = statistics.mean(values[k])
    s=[means["no_adapter"][k]-means["stock_trained"][k] for k in keys]
    c=[means["no_adapter"][k]-means["compressed_trained"][k] for k in keys]
    sg,cg=statistics.mean(s),statistics.mean(c)
    result={"stock_gain":sg,"candidate_gain":cg,"gain_difference":cg-sg,
            "retention_descriptive":cg/sg if sg>0 and math.isfinite(cg/sg) else None,
            "image_clusters":len(keys),"uncertainty":"descriptive, single training seed; image-cluster bootstrap only"}
    if len(keys)<4:
        result.update(stock_gain_interval=None,gain_difference_interval=None,retention_interval=None,
                      retention_interpretable=False,reason="fewer than four image clusters")
        return result
    rng=random.Random(seed)
    bs=[]; bd=[]; br=[]
    for _ in range(replicates):
        idx=rng.choices(range(len(keys)),k=len(keys))
        a=statistics.mean(s[i] for i in idx); b=statistics.mean(c[i] for i in idx)
        bs.append(a); bd.append(b-a)
        if a>0 and math.isfinite(b/a): br.append(b/a)
    def interval(v):
        v=sorted(v); return [v[int(.025*(len(v)-1))],v[int(.975*(len(v)-1))]]
    si=interval(bs)
    valid=sg>0 and si[0]>0 and len(br)==replicates
    result.update(stock_gain_interval=si,gain_difference_interval=interval(bd),
                  retention_interval=interval(br) if valid else None,retention_interpretable=valid)
    return result
