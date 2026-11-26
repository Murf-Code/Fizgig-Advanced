"""6C.2k-C: common proposal, two diagnostic forward modes; target development only."""
from contextlib import nullcontext
import logging
import math
import torch
from .replay_forks import match
from .replay_probe import verify_endpoint
from .replay_diagnostics import digest, depth_shares
from .replay_isolation import movement, precision, compare

log = logging.getLogger(__name__)


def allocations(scores, shares):
    if set(scores)!=set(range(28)) or set(shares)!=set(scores):raise ValueError('invalid block map')
    if any(not math.isfinite(v) or v<0 for v in (*scores.values(),*shares.values())):raise ValueError('invalid block scores')
    ratio={b:math.sqrt(shares[b])/max(scores[b],1e-12) for b in scores}
    picks={'uniform':None,'static26':26,'static13':13,
           'adaptive_low_rms':min(scores,key=lambda b:(scores[b],b)),
           'adaptive_movement':max(ratio,key=lambda b:(ratio[b],-b))}
    return {n:{b:.5 if b==chosen else 1. for b in scores} for n,chosen in picks.items()},picks,ratio


@torch.no_grad()
def evaluate_dual(fork, net, scores, panel, predict, guard, report, save, precision_context=precision):
    from fizgig.families.lora import TRAINABLE
    scales={n:dict(w.scales) for n,w in net.wrapped.items()}
    def restore_scales():
        for n,w in net.wrapped.items():w.scales.clear();w.scales.update(scales[n])
    def restore():fork.reset_weights();fork.rng.restore()
    def endpoint():return digest([(p.name,p.A,p.B) for p in fork.pairs])
    def forward(label,reverse=False):
        log.info('[movement-dual] %s (%d cases)',label,len(panel))
        def run():
            out=[None]*len(panel)
            for i in (reversed(range(len(panel))) if reverse else range(len(panel))):
                out[i]=predict(panel[i][1]).detach().cpu().clone()
            return out
        return guard(run)
    def require(a,b,label):
        if digest(a)!=digest(b):raise RuntimeError('dual restoration failed: '+label)
    report.update(experiment='6C.2k-C',evaluators={},endpoint_geometry={},aliases={},integrity_passed=False)
    try:
        log.info('[movement-dual] native proposal geometry and matched allocation')
        _,shares=fork.achieved({b:1. for b in scores})
        alloc,picks,ratio=allocations(scores,shares)
        report.update(allocations=alloc,selected_blocks=picks,movement_gradient_ratio=ratio,native_block_squared=shares)
        matched=match(fork,alloc,minimum=1e-12,atol=1e-12,rtol=.001,max_initial_dip=.01)
        report['matching']=matched;save()
        if not matched['matched']:raise RuntimeError('dual budget unmatched: '+str(matched['reason']))
        candidates={n:{b:m for b in scores} for n,m in [('decay',0.),('quarter',.25),('half',.5),('full',1.)]}
        candidates.update({n:r['multipliers'] for n,r in matched['arms'].items()})
        keys={};unique={}
        for n,ms in candidates.items():
            key=tuple(ms[b] for b in sorted(scores))
            if key in keys:report['aliases'][n]=keys[key]
            else:keys[key]=n;unique[n]=ms;report['aliases'][n]=n
        native_parent=None
        for mode,wo in [('native',False),('weight_only',True)]:
            restore()
            with (precision_context(net,True,False) if wo else nullcontext()):
                parent=forward(mode+' parent');require(forward(mode+' repeat'),parent,mode+' repeat')
                for n,w in net.wrapped.items():
                    if TRAINABLE in w.adapters:w.scales[TRAINABLE]=0.
                try:base=forward(mode+' base')
                finally:restore_scales()
                restore();require(forward(mode+' after base'),parent,mode+' after base')
                result={'reference_hash':digest((parent,base)), 'arms':{},'restored':[]}
                report['evaluators'][mode]=result;save()
                for name,ms in unique.items():
                    restore()
                    if not wo:
                        achieved,block=fork.achieved(ms);verify_endpoint(fork,ms)
                        squared=0.;changed=0;total=0
                        for p in fork.pairs:
                            a,b=fork.parent[p.A].bfloat16(),fork.parent[p.B].bfloat16()
                            c,d=p.A.detach().cpu().bfloat16(),p.B.detach().cpu().bfloat16()
                            squared+=movement(a,b,c,d,p.scale)
                            changed+=sum(int((x.view(torch.int16)!=y.view(torch.int16)).sum()) for x,y in ((a,c),(b,d)))
                            total+=a.numel()+b.numel()
                        report['endpoint_geometry'][name]={'fp32':achieved,'bf16':math.sqrt(squared),
                            'changed_bf16':changed,'factor_values':total,'depth':depth_shares(block),'endpoint_hash':endpoint()}
                    else:
                        fork.apply(ms);verify_endpoint(fork,ms)
                        if endpoint()!=report['endpoint_geometry'][name]['endpoint_hash']:raise RuntimeError('precision endpoints differ')
                    if name in matched['arms'] and abs(report['endpoint_geometry'][name]['fp32']-matched['target'])>matched['tolerance']:
                        raise RuntimeError('matched movement outside tolerance')
                    out=forward(mode+' '+name)
                    result['arms'][name]={'rows':compare(panel,out,parent,base),'prediction_hash':digest(out)}
                    restore();require(forward(mode+' restore '+name),parent,mode+' '+name)
                    result['restored'].append(name);save()
                # Independent reversed decay replay after the other candidates.
                fork.apply(unique['decay']);verify_endpoint(fork,unique['decay'])
                if digest(forward(mode+' reversed decay',True))!=result['arms']['decay']['prediction_hash']:
                    raise RuntimeError('dual decay repeat differs: '+mode)
                restore();require(forward(mode+' final parent'),parent,mode+' final parent')
                result['decay_repeat_exact']=True;save()
                if not wo:native_parent=parent
        restore();require(forward('native after precision switch'),native_parent,'native after precision switch')
        report['integrity_passed']=True;save();return report
    finally:
        restore_scales();restore()
