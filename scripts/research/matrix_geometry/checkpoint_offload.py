"""Narrow nonreentrant checkpoint input offload candidate.
No async/pinned pool yet. Never wrap entire model forward with hooks.
"""
import torch
from torch.utils.checkpoint import checkpoint
from contextlib import nullcontext

def checkpoint_block(block,*args,offload=False):
    # PyTorch non-reentrant checkpoint saves boundary inputs in outer scope;
    # checkpoint's own internal saved-tensor hooks handle recompute tensors.
    ctx=torch.autograd.graph.save_on_cpu(pin_memory=False) if offload else nullcontext()
    with ctx:
        return checkpoint(block,*args,use_reentrant=False,preserve_rng_state=True)

def checkpoint_activation_only(block,activation,*args,offload=False,stats=None):
    """Offload only first checkpoint boundary tensor, not repeated shared
    conditioning. Synchronous transfer reference, CPU copies in CPU tests.
    Requires immutable checkpoint inputs; no torch.compile support claimed.
    """
    if not offload:return checkpoint(block,activation,*args,use_reentrant=False,preserve_rng_state=True)
    target=id(activation)
    def pack(x):
        if id(x)==target:
            if stats is not None:
                stats['count']=stats.get('count',0)+1
                stats['bytes']=stats.get('bytes',0)+x.numel()*x.element_size()
            return (x.device,x.detach().to('cpu',copy=True))
        return (None,x.detach())
    def unpack(p):
        dev,x=p
        return x if dev is None else x.to(dev)
    with torch.autograd.graph.saved_tensors_hooks(pack,unpack):
        return checkpoint(block,activation,*args,use_reentrant=False,preserve_rng_state=True)
