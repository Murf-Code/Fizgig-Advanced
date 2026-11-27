"""Fixed-sign least squares, offline only; no trainer or codec changes."""
import torch


def fit_latent(w, a, b, row=None, col=None, tile=256):
    """a: [r,n], b: [m,r] signs; solve for one BF16 coefficient per rank.

    Gram of the implicit outer-product design is the Hadamard product of
    the two factor Grams. Never construct an (m*n,r) design matrix.
    """
    af, bf = a.float(), b.float()
    if col is not None:
        af = af * col.float()[None, :]
    if row is not None:
        bf = bf * row.float()[:, None]
    gram = (bf.T @ bf).double() * (af @ af.T).double()
    rhs = torch.zeros(a.shape[0], dtype=torch.float64, device=a.device)
    for i in range(0, w.shape[0], tile):
        j = min(i+tile, w.shape[0])
        rhs += ((w[i:j].float() @ af.T) * bf[i:j]).sum(0, dtype=torch.float64)
    gram = (gram + gram.T) * .5
    # Small ridge relative to the average diagonal stabilizes duplicate signs.
    ridge = max(float(gram.diag().mean()) * 1e-7, 1e-30)
    gram.diagonal().add_(ridge)
    c = torch.linalg.solve(gram, rhs).bfloat16()
    if not torch.isfinite(c).all():
        raise ValueError('nonfinite latent coefficients')
    return c


def fit_outer(w, a, b, c, row, col, tile=256):
    """Alternating signed row and column LS; actual BF16 scales."""
    af, bf = a.float(), b.float()*c.float()[None, :]
    row = row.clone()
    for i in range(0, w.shape[0], tile):
        j = min(i+tile, w.shape[0])
        z = (bf[i:j] @ af)*col.float()[None, :]
        den = z.square().sum(1)
        num = (z*w[i:j].float()).sum(1)
        row[i:j] = torch.where(den>0, num/den.clamp_min(1e-30), row[i:j].float()).bfloat16()
    num = torch.zeros_like(col, dtype=torch.float32)
    den = torch.zeros_like(num)
    for i in range(0, w.shape[0], tile):
        j = min(i+tile, w.shape[0])
        z = (bf[i:j] @ af)*row[i:j,None].float()
        num += (z*w[i:j].float()).sum(0)
        den += z.square().sum(0)
    col = torch.where(den>0, num/den.clamp_min(1e-30), col.float()).bfloat16()
    if not torch.isfinite(row).all() or not torch.isfinite(col).all():
        raise ValueError('nonfinite outer scales')
    return row, col


def payload_bytes(m,n,r,outer=False):
    return (r*n+7)//8+(r*m+7)//8+2*r+(2*(m+n) if outer else 0)
