"""Research-only movement fork evaluator. Explicit live-state audit; no trainer hook."""
import math
from dataclasses import dataclass
import torch
from .replay_forks import match, RNGSnapshot
from .replay_evaluator import Evaluator, compare_candidates
from .replay_diagnostics import digest, guarded_forward, localized, depth_shares, _array

@dataclass(frozen=True)
class Probe:
    case: object
    inputs: object
    parent: object
    base: object
    target: object = None


def rng_state(generators):
    r=RNGSnapshot(generators)
    return (r.python,r.numpy,r.cpu,r.cuda,[state for _,state in r.generators])


def array(x):
    if isinstance(x,torch.Tensor):
        return x.detach().cpu().double().numpy().copy()
    return _array(x).copy()


@torch.no_grad()
def verify_endpoint(fork, multipliers):
    """Independently rebuild each expected factor with the native add kernel.
    This checks allocation identity, not merely a movement norm. One factor at a time.
    """
    for ps, alpha, foreach in fork.groups:
        for p in ps:
            expected=fork.decayed[p].to(device=p.device,copy=True)
            update=fork.direction[p].to(device=p.device,dtype=p.dtype)
            rate=alpha*float(multipliers[fork.block[p]])
            if foreach:torch._foreach_add_([expected],[update],alpha=rate)
            else:expected.add_(update,alpha=rate)
            if not torch.equal(expected.contiguous().view(torch.uint8),p.detach().contiguous().view(torch.uint8)):
                raise RuntimeError('committed factor endpoint mismatch')


def evaluate_fork(fork, allocations, probes, predict, read_extra_state, *, minimum, atol, rtol,
                  max_initial_dip=0., order=None):
    """Match first, audit endpoint and every forward; return no scores on contamination.

    References must be independently generated from identical locked inputs. Caller
    must supply all mutable non-adapter trainer/model state and abort on any error.
    Only weights/RNG and native optimizer state are restored by the outer capture.
    Does not authorize continued training after a contamination error.
    """
    probes=tuple(probes); cases=[p.case for p in probes]; Evaluator(cases)
    names=list(allocations) if order is None else list(order)
    if len(names)!=len(allocations) or set(names)!=set(allocations):raise ValueError('invalid arm order')
    if not callable(read_extra_state):raise TypeError('explicit extra-state reader required')
    generators=[g for g,_ in fork.rng.generators]
    def live():
        return {'weights':[(p.name,p.A,p.B,p.A.grad,p.B.grad) for p in fork.pairs],
                'optimizer':fork.optimizer.state_dict(),'rng':rng_state(generators),'extra':read_extra_state()}
    for p in probes:
        if digest(p.inputs)!=p.case.input_hash:raise ValueError('input manifest mismatch')
    refs={p.case:digest((p.parent,p.base,p.target)) for p in probes}
    try:
        result=match(fork,allocations,minimum=minimum,atol=atol,rtol=rtol,max_initial_dip=max_initial_dip)
        if not result['matched']:return {'status':'unmatched','movement':result}
        outcomes={};actuals=[]
        for name in names:
            fork.rng.restore(); row=result['arms'][name]
            actual,shares=fork.achieved(row['multipliers']);actuals.append(actual)
            verify_endpoint(fork,row['multipliers'])
            tol=result['tolerance']
            if (not math.isfinite(actual) or abs(actual-row['achieved'])>tol or abs(actual-result['target'])>tol
                or actual<=minimum or set(shares)!=set(row['block_squared_movement'])
                or any(not math.isclose(v,row['block_squared_movement'][b],rel_tol=1e-10,abs_tol=1e-15) for b,v in shares.items())):
                raise RuntimeError('committed endpoint mismatch')
            e=Evaluator(cases);local=[]
            for p in probes:
                if digest((p.parent,p.base,p.target))!=refs[p.case]:raise RuntimeError('reference mutation')
                def forward(inputs):
                    with torch.no_grad():return array(predict(inputs))
                out=guarded_forward(forward,p.inputs,expected_input_hash=p.case.input_hash,read_state=live)
                if digest((p.parent,p.base,p.target))!=refs[p.case]:raise RuntimeError('reference mutation')
                parent,base=array(p.parent),array(p.base)
                e.add(p.case,candidate=out,parent=parent,base=base,target=None if p.target is None else array(p.target))
                local.append({'case':p.case.input_hash,'parent':localized(out-parent),'base':localized(out-base)})
            outcomes[name]={'metrics':e.finish(),'local':local,'achieved':actual,'depth':depth_shares(shares)}
        if max(actuals)-min(actuals)>result['tolerance']:raise RuntimeError('pairwise endpoint mismatch')
        # Catch reference mismatches even if caller accidentally supplied different panels.
        for name in names[1:]:compare_candidates(outcomes[name]['metrics'],outcomes[names[0]]['metrics'])
        return {'status':'evaluated','movement':result,'arms':outcomes}
    finally:
        fork.reset_weights();fork.rng.restore()
