"""Research-only effective LoRA displacement/identifiable-gradient alignment.

Exact for compatible real-arithmetic factor gradients. Mixed-precision training
provides approximate gradients; overlap/null-space checks are diagnostics, not
proof of exactness. All scratch is CPU FP64, one pair at a time.
"""
import math
import torch


def _basis(x):
    q, r = torch.linalg.qr(x, mode="reduced")
    u, sig, vh = torch.linalg.svd(r, full_matrices=False)
    keep = sig > (sig.max() * 1e-12)
    return q @ u[:, keep], sig[keep], vh[keep].T


@torch.no_grad()
def pair_terms(a, b, a0, b0, ga, gb, scale, *, overlap_tol=.02):
    """Return dot, squared norms, visible displacement energy, consistency.
    Shapes: A [r,n], B [m,r]. No m-by-n matrix is materialized.
    """
    a, b, a0, b0, ga, gb = [x.detach().to(device="cpu", dtype=torch.float64)
                            for x in (a,b,a0,b0,ga,gb)]
    if not math.isfinite(scale) or scale == 0:
        return None
    if not all(bool(torch.isfinite(x).all()) for x in (a,b,a0,b0,ga,gb)):
        return None
    u, sb, rb = _basis(b)
    v, sa, la = _basis(a.T)
    h = (gb @ la) / sa / scale
    k = (rb.T @ ga) / sb[:, None] / scale
    m1, m2 = u.T @ h, k @ v
    overlap = float((m1-m2).norm()) / max(float(h.norm()), float(k.norm()), 1e-300)
    null_b = float((gb - (gb @ la) @ la.T).norm()) / max(float(gb.norm()),1e-300)
    null_a = float((ga - rb @ (rb.T @ ga)).norm()) / max(float(ga.norm()),1e-300)
    error = max(overlap, null_a, null_b)
    if not math.isfinite(error):
        return None
    # Orthogonal pieces; average overlapping estimates only for approximate
    # compatible gradients. The acceptance threshold is experimental.
    hp, kp = h-u@m1, k-m2@v.T
    m = (m1+m2)*.5
    ql = torch.linalg.qr(torch.cat((b,b0),dim=1),mode="reduced").Q
    qr = torch.linalg.qr(torch.cat((a.T,a0.T),dim=1),mode="reduced").Q
    c = scale*((ql.T@(b-b0))@(a0@qr)+(ql.T@b)@((a-a0)@qr))
    uc, vc = ql.T@u, qr.T@v
    zc = (ql.T@hp)@vc.T + uc@(kp@qr) + uc@m@vc.T
    pc = uc@(uc.T@c) + (c@vc)@vc.T - uc@((uc.T@c)@vc)@vc.T
    d2 = float(c.square().sum())
    ref = abs(scale)*(float(((ql.T@b)@(a@qr)).norm())+
                      float(((ql.T@b0)@(a0@qr)).norm()))
    if math.sqrt(d2) <= 1e-12*max(ref,1e-300):
        d2, dot, p2 = 0., 0., 0.
    else:
        dot, p2 = float((zc*c).sum()), float(pc.square().sum())
    z2 = float(hp.square().sum()+kp.square().sum()+m.square().sum())
    if not all(math.isfinite(x) for x in (dot,d2,z2,p2)):
        return None
    return {"dot":dot,"d2":d2,"z2":z2,"p2":p2,
            "consistency_error":error,"compatible":error<=overlap_tol}


def block_score(terms):
    """Direct sum, not an average of pair cosines. Unknown is distinct from 0."""
    if not terms or any(t is None for t in terms):
        return {"score":None,"observability":None,"reason":"invalid_pair"}
    err = max(t["consistency_error"] for t in terms)
    d2,z2,p2,dot = [math.fsum(t[key] for t in terms) for key in ("d2","z2","p2","dot")]
    rho = min(1.,math.sqrt(max(0.,p2)/d2)) if d2>0 else None
    result = {"score":None,"observability":rho,"consistency_error":err,
              "displacement_norm":math.sqrt(d2),"projected_gradient_norm":math.sqrt(z2),
              "pair_count":len(terms)}
    if not all(t["compatible"] for t in terms):
        return {**result,"reason":"inconsistent_factor_gradients"}
    if d2<=1e-40 or z2<=1e-40:
        return {**result,"reason":"unresolved_signal"}
    return {**result,"score":max(-1.,min(1.,-dot/(math.sqrt(d2)*math.sqrt(z2)))),
            "reason":None}
