"""Blockwise signed symmetric INT8/INT4 momentum research. FP32 computation.
INT4 is actually packed (two nibbles/byte); codebook -7..7, unused -8.
No bitsandbytes dependency. Not a fused kernel; explicit temporary buffers.
"""
import torch
from muon_me import MuonME, ns, adjusted_lr, scratch_per_matrix

@torch.no_grad()
def pack(x,bits=8,block=256):
    if bits not in (4,8) or block<=0 or block%2: raise ValueError('bits/block')
    f=x.float().reshape(-1); n=f.numel(); pad=(-n)%block
    if n==0: raise ValueError('empty')
    if pad: f=torch.nn.functional.pad(f,(0,pad))
    f=f.reshape(-1,block); qmax=(1<<(bits-1))-1
    s=f.abs().amax(1)/qmax
    # zero blocks encode exact zero with scale 1
    s=torch.where(s==0,torch.ones_like(s),s.clamp_min(torch.finfo(torch.float32).tiny))
    q=(f/s[:,None]).round().clamp(-qmax,qmax).to(torch.int8).flatten()
    if bits==4:
        q=(q.to(torch.int16)+8).to(torch.uint8)
        q=q[::2] | (q[1::2]<<4)
    return q,s

@torch.no_grad()
def unpack(q,s,shape,bits=8,block=256):
    if bits==4:
        q=torch.stack(((q&15).to(torch.int16)-8,(q>>4).to(torch.int16)-8),dim=1).flatten()
    n=1
    for d in shape:n*=d
    return (q.reshape(-1,block).float()*s[:,None]).flatten()[:n].reshape(shape)

class QuantMuon(MuonME):
    """Research subclass for FP32 factors, no shared storage. Decompress one
    chunk, update momentum and use unquantized current buffer for NS, then pack
    for the next step. No residual/error-feedback memory. Not trainer ready.
    """
    def __init__(self,params,*,bits=8,block=256,**kwargs):
        super().__init__(params,**kwargs)
        if bits not in (4,8) or block<=0 or block%2: raise ValueError('bits/block')
        if any(p.dtype!=torch.float32 for g in self.param_groups for p in g['params']):raise ValueError('FP32 factors only')
        self.bits=bits; self.block=block
        for g in self.param_groups:g.update(state_bits=bits,state_block=block)
    def load_state_dict(self,state_dict):
        # Torch normally casts optimizer tensors to parameter dtype. Restore
        # packed state from original payload, not the cast temporary copy.
        for g in state_dict['param_groups']:
            if g.get('state_bits')!=self.bits or g.get('state_block')!=self.block:raise ValueError('state format mismatch')
        super().load_state_dict(state_dict)
        for saved,live in zip(state_dict['param_groups'],self.param_groups):
            for k,p in zip(saved['params'],live['params']):
                old=state_dict['state'].get(k,{})
                if 'q' in old:
                    self.state[p]['q']=old['q'].to(device=p.device,copy=True)
                    self.state[p]['scale']=old['scale'].to(device=p.device,dtype=torch.float32,copy=True)
    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():loss=closure()
        for g in self.param_groups:
            by={}; lr,wd,mom=g['lr'],g['weight_decay'],g['momentum']
            for p in g['params']:
                if p.grad is None:continue
                if p.grad.is_sparse:raise RuntimeError('sparse gradient')
                by.setdefault((tuple(p.shape),p.device),[]).append(p)
            for (shape,dev),ps in by.items():
                # include conservative extra scratch for dequantization & packing
                cost=scratch_per_matrix(shape,4,g['nesterov'])+32*(shape[0]*shape[1]+self.block)
                budget=g['chunk_memory_mib']; chunk=len(ps) if budget==0 else max(1,min(len(ps),int(budget*2**20)//cost))
                for start in range(0,len(ps),chunk):
                    sub=ps[start:start+chunk]; grads=[p.grad for p in sub]; bufs=[]
                    for p in sub:
                        st=self.state[p]; buf=unpack(st['q'],st['scale'],p.shape,self.bits,self.block) if 'q' in st else torch.zeros_like(p)
                        buf.lerp_(p.grad,1-mom); bufs.append(buf)
                        st['q'],st['scale']=pack(buf,self.bits,self.block)
                    ups=torch._foreach_lerp(grads,bufs,mom) if g['nesterov'] else bufs
                    if wd:torch._foreach_mul_(sub,1-lr*wd)
                    ortho=ns(torch.stack(ups),g['ns_coefficients'],g['ns_steps'],g['eps'])
                    torch._foreach_add_(sub,list(ortho.float().unbind(0)),alpha=-adjusted_lr(lr,g['adjust_lr_fn'],shape))
                    del ortho,ups,bufs,buf
        return loss
