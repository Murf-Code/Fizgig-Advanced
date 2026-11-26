"""Disk-backed native fork, restricted to native Muon state. Trusted scratch only."""
from contextlib import contextmanager
from collections.abc import MutableMapping
from pathlib import Path
import tempfile, copy, hashlib
import torch
from fizgig.training.replay_forks import Fork,RNGSnapshot,_cpu
from fizgig.training.muon import Muon,BatchedMuon

class DiskMap(MutableMapping):
    def __init__(self,root):self.root=Path(root);self.root.mkdir();self.index={};self.seq=0
    def __setitem__(self,k,v):
        if k not in self.index:
            self.index[k]=self.root/f'{self.seq}.pt';self.seq+=1
        torch.save(v,self.index[k])
    def __getitem__(self,k):return torch.load(self.index[k],map_location='cpu',weights_only=True)
    def __delitem__(self,k):self.index.pop(k).unlink()
    def __iter__(self):return iter(self.index)
    def __len__(self):return len(self.index)
    def __contains__(self,k):return k in self.index

def bits(p):
    return hashlib.sha256(memoryview(p.detach().cpu().contiguous().view(torch.uint8).numpy()).cast('B')).digest()

@contextmanager
def disk_capture(pairs,opt,*,generators=(),scratch=None):
    if type(opt) not in (Muon,BatchedMuon):raise TypeError('native Muon only')
    rng=RNGSnapshot(generators);pairs=tuple(pairs);ps=[p for x in pairs for p in (x.A,x.B)]
    if not set(opt.state)<=set(ps):raise ValueError('unexpected state keys')
    for p,s in opt.state.items():
        if set(s)-{'momentum_buffer'}:raise ValueError('unexpected optimizer state')
        if 'momentum_buffer' in s and (s['momentum_buffer'].dtype!=p.dtype or s['momentum_buffer'].device!=p.device or s['momentum_buffer'].shape!=p.shape):raise ValueError('unsupported buffer layout')
    groups=[{k:list(v) if k=='params' else copy.deepcopy(v) for k,v in g.items()} for g in opt.param_groups]
    with tempfile.TemporaryDirectory(dir=scratch,prefix='fizgig-fork-') as root:
        f=Fork(pairs,opt,rng,compact=True)
        parent=DiskMap(Path(root)/'parent')
        for p,v in f.parent.items():parent[p]=v
        del v
        f.parent=parent;f.decayed=DiskMap(Path(root)/'decayed');f.direction=DiskMap(Path(root)/'direction')
        incoming=DiskMap(Path(root)/'momentum')
        for p,s in opt.state.items():incoming[p]={k:_cpu(v) for k,v in s.items()}
        try:
            opt.step(proposal_sink=f.observe)
            if set(f.direction)!=set(f.parent):raise RuntimeError('incomplete proposal')
            native={p:bits(p) for p in ps}
            f.apply({b:1. for b in set(f.block.values())})
            if any(bits(p)!=native[p] for p in ps):raise RuntimeError('native parity failed')
            f.reset_weights();rng.restore();yield f
        finally:
            f.live=False
            try:
                f.reset_weights();opt.state.clear();opt.param_groups[:]=groups
                for p in incoming:opt.state[p]={k:v.to(p.device) for k,v in incoming[p].items()}
            finally:rng.restore()
