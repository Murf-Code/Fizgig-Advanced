"""Round-four experimental guard wrapper; trusted callbacks, no trainer sandbox.
Requires regenerated reference predictions from actual parent and explicit base provider.
"""
from dataclasses import asdict
from collections import defaultdict
import math,copy
import torch
from fizgig.training.replay_diagnostics import digest,guarded_forward
from fizgig.training.replay_probe import evaluate_fork,array,rng_state
from fizgig.training.replay_evaluator import Case,freeze,aggregate,compare_candidates,paired_summary

def audit_manifest(cases):
    cases,h=freeze(cases);owners=defaultdict(set)
    for c in cases:owners[c.input_hash].add((c.domain,c.unit))
    if any(len(v)>1 for v in owners.values()):raise ValueError('duplicate input across source units')
    return cases,h

def audited_compare(a,b):
    for result in (a,b):
        rows=result['rows'];cases=[Case(**r['case']) for r in rows];_,h=audit_manifest(cases)
        if h!=result['manifest_hash']:raise ValueError('manifest mismatch')
        reconstructed=aggregate({c:{k:v for k,v in row.items() if k not in ('case','reference_hash')} for c,row in zip(cases,rows)})
        if digest(reconstructed)!=digest(result['domains']):raise ValueError('aggregate mismatch')
    return compare_candidates(a,b)

def named_summary(rows):
    # rows = (training_seed_id, paired scalar). Named IDs must represent actual independent runs.
    ids=[i for i,_ in rows]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate seed IDs')
    return paired_summary([v for _,v in rows])

def seal(snapshot):return {'snapshot':snapshot,'sha256':digest(snapshot)}
def restore_sealed(cp,envelope):
    if digest(envelope['snapshot'])!=envelope['sha256']:raise ValueError('checkpoint checksum mismatch')
    cp.restore(envelope['snapshot'])

def audited_evaluate(fork,allocations,probes,predict,predict_base,read_extra_state,**kwargs):
    probes=tuple(probes);audit_manifest([p.case for p in probes]);gens=[g for g,_ in fork.rng.generators]
    # A caller must implement predict_base without persistent changes, and register all relevant state.
    def live():return {'params':[(p.name,p.A,p.B,p.A.grad,p.B.grad) for p in fork.pairs],
        'optimizer':fork.optimizer.state_dict(),'group_order':[[id(p) for p in g['params']] for g in fork.optimizer.param_groups],
        'rng':rng_state(gens),'extra':read_extra_state()}
    def allrefs():return digest([(p.inputs,p.parent,p.base,p.target) for p in probes])
    locked=allrefs();fork.reset_weights();fork.rng.restore()
    try:
        for p in probes:
            for callback,ref in ((predict,p.parent),(predict_base,p.base)):
                def f(inputs):
                    with torch.no_grad():return array(callback(inputs))
                actual=guarded_forward(f,p.inputs,expected_input_hash=p.case.input_hash,read_state=live)
                if digest(actual)!=digest(array(ref)):raise ValueError('reference provenance mismatch')
            if allrefs()!=locked:raise RuntimeError('panel mutation')
        def checked(x):
            y=predict(x)
            if allrefs()!=locked:raise RuntimeError('panel mutation')
            return y
        result=evaluate_fork(fork,allocations,probes,checked,read_extra_state,**kwargs)
        if allrefs()!=locked:raise RuntimeError('panel mutation')
        if result['status']=='evaluated':
            values=list(result['arms'].values())
            for v in values:audited_compare(v['metrics'],values[0]['metrics'])
        return result
    finally:fork.reset_weights();fork.rng.restore()
