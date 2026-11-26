"""Strict research checkpoint at an optimizer boundary. Explicit registration, no resume fallback.
Not yet wired to families/train.py. Only load trusted, locally generated files.
"""
import os, copy, random, tempfile
from pathlib import Path
import numpy as np
import torch
from .replay_forks import _clone_cpu

class BoundaryCheckpoint:
    def __init__(self, params, optimizer, *, buffers, components, generators, metadata):
        self.params=dict(params);self.buffers=dict(buffers);self.optimizer=optimizer
        self.components=dict(components);self.generators=dict(generators);self.metadata=copy.deepcopy(metadata)
        names={id(p):n for n,p in self.params.items()}
        if len(names)!=len(self.params) or not self.params:raise ValueError('invalid named parameters')
        self.groups=[[names[id(p)] for p in g['params']] for g in optimizer.param_groups]
        flat=[n for g in self.groups for n in g]
        if len(flat)!=len(names) or set(flat)!=set(self.params):raise ValueError('optimizer parameter layout mismatch')
        if not all(hasattr(v,'state_dict') and hasattr(v,'load_state_dict') for v in self.components.values()):raise ValueError('stateful components required')
    def _check_layout(self):
        names={id(p):n for n,p in self.params.items()}
        if [[names.get(id(p)) for p in g['params']] for g in self.optimizer.param_groups]!=self.groups:
            raise ValueError('live optimizer parameter order changed')
    def snapshot(self):
        self._check_layout()
        if any(p.grad is not None for p in self.params.values()):raise ValueError('checkpoint requires cleared gradients at optimizer boundary')
        ns=np.random.get_state()
        return _clone_cpu({'version':1,'metadata':self.metadata,'groups':self.groups,
            'params':self.params,'buffers':self.buffers,'optimizer':self.optimizer.state_dict(),
            'optimizer_class':type(self.optimizer).__qualname__,
            'components':{n:c.state_dict() for n,c in self.components.items()},
            'generators':{n:g.get_state() for n,g in self.generators.items()},
            'rng':{'python':random.getstate(),'numpy':(ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]),
                   'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}})
    def validate(self,s):
        if s['version']!=1 or s['metadata']!=self.metadata or s['groups']!=self.groups or s['optimizer_class']!=type(self.optimizer).__qualname__:raise ValueError('checkpoint identity or parameter order mismatch')
        for key,live in [('params',self.params),('buffers',self.buffers)]:
            if set(s[key])!=set(live):raise ValueError('tensor names mismatch')
            for n,p in live.items():
                q=s[key][n]
                if q.shape!=p.shape or q.dtype!=p.dtype:raise ValueError('tensor layout mismatch')
        if set(s['components'])!=set(self.components) or set(s['generators'])!=set(self.generators):raise ValueError('component or generator mismatch')
        if (s['rng']['cuda'] is None)!= (not torch.cuda.is_initialized()):raise ValueError('CUDA RNG topology mismatch')
        if s['rng']['cuda'] is not None and len(s['rng']['cuda'])!=torch.cuda.device_count():raise ValueError('CUDA device count mismatch')
        gs=s['optimizer']['param_groups'];ids=[i for g in gs for i in g['params']]
        if [len(g['params']) for g in gs]!=[len(g) for g in self.groups] or len(set(ids))!=len(ids) or not set(s['optimizer']['state'])<=set(ids):raise ValueError('optimizer state layout mismatch')
        # Muon buffers must match the named parameter, never silently cast.
        for i,n in zip(ids,[n for g in self.groups for n in g]):
            buf=s['optimizer']['state'].get(i,{}).get('momentum_buffer')
            if buf is not None and (buf.shape!=self.params[n].shape or buf.dtype!=self.params[n].dtype):raise ValueError('momentum layout mismatch')
    @torch.no_grad()
    def _apply(self,s):
        for key,live in [('params',self.params),('buffers',self.buffers)]:
            for n,p in live.items():p.copy_(s[key][n])
        self.optimizer.load_state_dict(copy.deepcopy(s['optimizer']))
        for n,c in self.components.items():c.load_state_dict(copy.deepcopy(s['components'][n]))
        for n,g in self.generators.items():g.set_state(s['generators'][n])
        r=s['rng'];random.setstate(r['python']);ns=r['numpy'];np.random.set_state((ns[0],np.array(ns[1],dtype=np.uint32),*ns[2:]));torch.set_rng_state(r['torch'])
        if r['cuda'] is not None:torch.cuda.set_rng_state_all(r['cuda'])
    def restore(self,s):
        self._check_layout();self.validate(s);old=self.snapshot()
        try:self._apply(s)
        except BaseException:
            self._apply(old);raise
    def save(self,path):
        path=Path(path);fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.',suffix='.part');os.close(fd)
        try:
            torch.save(self.snapshot(),tmp);os.replace(tmp,path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
    def load(self,path):self.restore(torch.load(path,map_location='cpu',weights_only=True))
